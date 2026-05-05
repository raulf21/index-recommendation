"""
feature_extractor.py
--------------------
Builds ML-ready numeric feature rows for candidate indexes.

Pipeline position:
    workload_parser -> candidate_generator -> feature_extractor -> hypopg_labeler -> ml_model

Input:
    - candidates from candidate_generator.py
      Expected keys:
          table, columns, type, range_scan_candidate,
          source_queries, access_pattern, cost_impact
    - workload rows from workload_parser.py
    - PostgreSQL catalog statistics and EXPLAIN summaries

Output:
    - List of dicts, one row per (query_name, candidate) pair.
      These rows are later joined with HypoPG labels in training_dataset.py.

Important design choices:
    - This file does not decide which indexes are good. It only describes them.
    - Candidate selection/pruning belongs in candidate_generator.py.
    - Ground-truth benefit labels belong in hypopg_labeler.py.
    - The old clustered_candidate field is intentionally removed because HypoPG
      simulates normal indexes, not PostgreSQL physical CLUSTER behavior.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from db_utils import get_connection, explain_query_json
from workload_parser import load_queries, parse_workload

QUERIES_DIR = "queries"

# Keep this list fixed so the ML dataset has stable columns even when a
# particular run does not generate every access pattern.
KNOWN_ACCESS_PATTERNS: Tuple[str, ...] = (
    "equality",
    "range",
    "prefix_like",
    "sort_group",
    "equality_equality",
    "equality_range",
    "equality_prefix_like",
    "range_range",
    "range_prefix_like",
    "prefix_like_prefix_like",
)


def _parse_pg_array(value: Any) -> Optional[List[Any]]:
    """Parse PostgreSQL array text or pass through a list from psycopg2."""
    if value is None:
        return None

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        if value == "{}":
            return []

        inner = value.strip()
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1]

        if not inner:
            return []

        # Split on commas that are not inside parentheses.
        parts = re.split(r",(?![^()]*\))", inner)
        return [p.strip().strip('"') for p in parts if p.strip()]

    return None


def _histogram_summary(bounds: Any) -> Dict[str, float]:
    """Turn pg_stats.histogram_bounds into a compact numeric summary."""
    arr = _parse_pg_array(bounds)
    if not arr:
        return {"hist_n_buckets": 0.0, "hist_span": 0.0}

    n = len(arr)
    try:
        first = float(arr[0])
        last = float(arr[-1])
        span = abs(last - first)
    except (TypeError, ValueError):
        # Text/date histograms are still useful through bucket count; span is a
        # rough symbolic proxy when numeric conversion is unavailable.
        span = float(n)

    return {
        "hist_n_buckets": float(max(0, n - 1)),
        "hist_span": float(span),
    }


def _mcv_top_freq(freqs: Any) -> float:
    """Return the largest most-common-value frequency from pg_stats."""
    arr = _parse_pg_array(freqs)
    if not arr:
        return 0.0

    best = 0.0
    for x in arr:
        try:
            best = max(best, float(x))
        except (TypeError, ValueError):
            continue
    return best


def fetch_pg_stats_row(
    conn,
    table: str,
    column: str,
    schema: str = "public",
) -> Optional[Dict[str, Any]]:
    """Fetch one pg_stats row for (schema, table, column)."""
    q = """
        SELECT
            null_frac,
            avg_width,
            n_distinct,
            correlation,
            most_common_freqs,
            histogram_bounds
        FROM pg_stats
        WHERE schemaname = %s
          AND tablename = %s
          AND attname = %s
    """

    with conn.cursor() as cur:
        cur.execute(q, (schema, table, column))
        row = cur.fetchone()

    if not row:
        return None

    null_frac, avg_width, n_distinct, correlation, mcf, hb = row
    hist = _histogram_summary(hb)

    # PostgreSQL uses negative n_distinct values to represent a fraction of the
    # table. abs() avoids teaching the model that high-cardinality columns have
    # negative cardinality.
    nd = abs(float(n_distinct)) if n_distinct is not None else 0.0
    corr = float(correlation) if correlation is not None else 0.0

    return {
        "null_frac": float(null_frac or 0.0),
        "avg_width": float(avg_width or 0.0),
        "n_distinct": nd,
        "correlation": corr,
        "mcv_top_freq": _mcv_top_freq(mcf),
        **hist,
    }


def fetch_pg_stats_for_columns(
    conn,
    table_columns: Iterable[Tuple[str, str]],
    schema: str = "public",
) -> Dict[str, Dict[str, float]]:
    """Batch-fetch pg_stats for unique (table, column) pairs."""
    seen: Set[Tuple[str, str]] = set()
    out: Dict[str, Dict[str, float]] = {}

    for table, col in table_columns:
        key_t = (table, col)
        if key_t in seen:
            continue
        seen.add(key_t)

        row = fetch_pg_stats_row(conn, table, col, schema)
        key = f"{table}.{col}"
        if row:
            out[key] = row
        else:
            out[key] = {
                "null_frac": 0.0,
                "avg_width": 0.0,
                "n_distinct": 0.0,
                "correlation": 0.0,
                "mcv_top_freq": 0.0,
                "hist_n_buckets": 0.0,
                "hist_span": 0.0,
            }

    return out


def aggregate_column_stats(
    stats_by_col: Mapping[str, Mapping[str, float]],
    columns: Sequence[str],
    table: str,
) -> Dict[str, float]:
    """Aggregate per-column pg_stats into fixed-width candidate features."""
    keys = [f"{table}.{c}" for c in columns]
    rows = [stats_by_col[k] for k in keys if k in stats_by_col]

    if not rows:
        return {
            "colstats_mean_null_frac": 0.0,
            "colstats_mean_avg_width": 0.0,
            "colstats_mean_corr": 0.0,
            "colstats_min_ndistinct": 0.0,
            "colstats_max_ndistinct": 0.0,
            "colstats_max_mcv_top": 0.0,
            "colstats_mean_hist_buckets": 0.0,
            "colstats_mean_hist_span": 0.0,
        }

    def mean(name: str) -> float:
        return sum(float(r[name]) for r in rows) / len(rows)

    return {
        "colstats_mean_null_frac": mean("null_frac"),
        "colstats_mean_avg_width": mean("avg_width"),
        "colstats_mean_corr": mean("correlation"),
        "colstats_min_ndistinct": min(float(r["n_distinct"]) for r in rows),
        "colstats_max_ndistinct": max(float(r["n_distinct"]) for r in rows),
        "colstats_max_mcv_top": max(float(r["mcv_top_freq"]) for r in rows),
        "colstats_mean_hist_buckets": mean("hist_n_buckets"),
        "colstats_mean_hist_span": mean("hist_span"),
    }


def estimate_write_penalty(
    conn,
    table: str,
    columns: Sequence[str],
    schema: str = "public",
) -> Dict[str, float]:
    """Approximate index maintenance/storage overhead.

    This is intentionally a proxy, not an exact size estimate. It gives the
    model a rough signal that wider indexes on larger tables are more expensive
    to maintain.
    """
    try:
        q = """
            SELECT c.reltuples
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relname = %s
        """
        with conn.cursor() as cur:
            cur.execute(q, (schema, table))
            row = cur.fetchone()

        if not row:
            return {"write_penalty_proxy": 0.0}

        (reltuples,) = row
        estimated_rows = max(0.0, float(reltuples))
        index_width_bytes = max(1.0, float(len(columns))) * 4.0
        penalty = (estimated_rows / 1000.0) * index_width_bytes
        return {"write_penalty_proxy": float(penalty)}

    except Exception:
        return {"write_penalty_proxy": 0.0}


def list_indexed_column_sets(
    conn,
    table: str,
    schema: str = "public",
) -> List[Tuple[str, ...]]:
    """Return ordered column tuples for each non-primary index on a table."""
    q = """
        SELECT ARRAY(
            SELECT a.attname
            FROM unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ord)
            JOIN pg_attribute a
              ON a.attrelid = ix.indrelid
             AND a.attnum = k.attnum
             AND NOT a.attisdropped
            ORDER BY k.ord
        ) AS cols
        FROM pg_class t
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_index ix ON t.oid = ix.indrelid
        WHERE n.nspname = %s
          AND t.relkind = 'r'
          AND t.relname = %s
          AND NOT ix.indisprimary
    """

    with conn.cursor() as cur:
        cur.execute(q, (schema, table))
        rows = cur.fetchall()

    out: List[Tuple[str, ...]] = []
    for (cols,) in rows:
        if cols:
            out.append(tuple(cols))
    return out


def existing_index_overlap_features(
    conn,
    table: str,
    columns: Sequence[str],
    schema: str = "public",
) -> Dict[str, float]:
    """Check whether a candidate is already covered by an existing index."""
    want = tuple(columns)
    idx_sets = list_indexed_column_sets(conn, table, schema)

    exact = 0.0
    prefix = 0.0
    for indexed_cols in idx_sets:
        if indexed_cols == want:
            exact = 1.0
        if len(indexed_cols) >= len(want) and indexed_cols[: len(want)] == want:
            prefix = 1.0

    return {
        "idx_already_exact": exact,
        "idx_already_prefix": max(exact, prefix),
        "n_existing_indexes_on_table": float(len(idx_sets)),
    }


def explain_workload(
    conn,
    queries: Optional[Mapping[str, str]] = None,
    queries_dir: str = QUERIES_DIR,
) -> Dict[str, Dict[str, float]]:
    """Summarize EXPLAIN output for each query name."""
    if queries is None:
        queries = load_queries(queries_dir)

    out: Dict[str, Dict[str, float]] = {}
    for name in sorted(queries.keys()):
        sql = queries[name]
        try:
            out[name] = explain_query_json(conn, sql)
        except Exception as ex:
            print(f"[warn] EXPLAIN failed for {name}: {ex}")
            out[name] = {
                "plan_total_cost": 0.0,
                "plan_startup_cost": 0.0,
                "plan_rows": 0.0,
                "n_seq_scan": 0.0,
                "n_index_scan": 0.0,
                "explain_error": 1.0,
                "explain_error_msg_len": float(len(str(ex))),
            }
        else:
            out[name]["explain_error"] = 0.0
            out[name]["explain_error_msg_len"] = 0.0

    return out


def workload_column_frequencies(workload: List[dict]) -> Dict[str, float]:
    """Cost-weighted total query impact per table.column."""
    freq: Dict[str, float] = {}
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        freq[key] = freq.get(key, 0.0) + float(item.get("query_cost", 1.0))
    return freq


def workload_clause_features(workload: List[dict]) -> Dict[str, Dict[str, float]]:
    """Cost-weighted GROUP BY and ORDER BY signals per table.column."""
    result: Dict[str, Dict[str, float]] = {}

    for item in workload:
        key = f"{item['table']}.{item['column']}"
        if key not in result:
            result[key] = {"group_by_cost": 0.0, "order_by_cost": 0.0}

        qcost = float(item.get("query_cost", 1.0))
        if item.get("in_group_by"):
            result[key]["group_by_cost"] += qcost
        if item.get("in_order_by"):
            result[key]["order_by_cost"] += qcost

    return result


def queries_touching_table(workload: List[dict], table: str) -> Set[str]:
    """Fallback query set: all workload queries referencing the candidate table."""
    return {item["query"] for item in workload if item["table"] == table}


def _candidate_query_set(
    cand: Mapping[str, Any],
    workload: List[dict],
    table: str,
) -> Set[str]:
    """Use candidate source_queries when available; otherwise use table touch set."""
    source_queries = cand.get("source_queries") or []
    if source_queries:
        return {str(q) for q in source_queries}
    return queries_touching_table(workload, table)


def _candidate_access_features(access_pattern: str) -> Dict[str, float]:
    """Encode candidate_generator access_pattern into stable numeric features."""
    pattern = (access_pattern or "unknown").strip()

    out: Dict[str, float] = {}
    for p in KNOWN_ACCESS_PATTERNS:
        out[f"cand_access_is_{p}"] = 1.0 if pattern == p else 0.0

    out["cand_access_unknown"] = 0.0 if pattern in KNOWN_ACCESS_PATTERNS else 1.0
    out["cand_access_has_equality"] = 1.0 if "equality" in pattern else 0.0
    out["cand_access_has_range"] = 1.0 if "range" in pattern else 0.0
    out["cand_access_has_prefix_like"] = 1.0 if "prefix_like" in pattern else 0.0
    out["cand_access_is_sort_group"] = 1.0 if pattern == "sort_group" else 0.0
    return out


def build_feature_rows(
    conn,
    candidates: List[dict],
    workload: List[dict],
    queries: Optional[Mapping[str, str]] = None,
    queries_dir: str = QUERIES_DIR,
    schema: str = "public",
) -> List[Dict[str, Any]]:
    """Build one ML feature row per (query_name, candidate) pair.

    Candidate metadata comes from candidate_generator.py. Query-level plan
    features come from EXPLAIN. Catalog features come from pg_stats and pg_index.
    """
    if queries is None:
        queries = load_queries(queries_dir)

    pairs: Set[Tuple[str, str]] = set()
    for cand in candidates:
        for col in cand["columns"]:
            pairs.add((cand["table"], col))

    stats_by_col = fetch_pg_stats_for_columns(conn, pairs, schema)
    explain_by_q = explain_workload(conn, queries=queries)
    freqs = workload_column_frequencies(workload)
    clause_feats = workload_clause_features(workload)

    rows: List[Dict[str, Any]] = []

    for cand in candidates:
        table = cand["table"]
        cols = list(cand["columns"])
        access_pattern = str(cand.get("access_pattern", "unknown"))

        colstats = aggregate_column_stats(stats_by_col, cols, table)
        idxmeta = existing_index_overlap_features(conn, table, cols, schema)
        write_penalty = estimate_write_penalty(conn, table, cols, schema)
        touch = _candidate_query_set(cand, workload, table)

        group_by_costs = [
            clause_feats.get(f"{table}.{c}", {}).get("group_by_cost", 0.0)
            for c in cols
        ]
        order_by_costs = [
            clause_feats.get(f"{table}.{c}", {}).get("order_by_cost", 0.0)
            for c in cols
        ]

        base: Dict[str, Any] = {
            "candidate_table": table,
            "candidate_cols": ",".join(cols),
            "candidate_type": cand.get("type", "unknown"),
            "n_index_columns": float(len(cols)),
            "is_composite": 1.0 if cand.get("type") == "composite" else 0.0,
            "range_scan_candidate": float(bool(cand.get("range_scan_candidate", False))),
            "candidate_cost_impact": float(cand.get("cost_impact", 0.0)),
            "candidate_source_query_count": float(len(cand.get("source_queries") or [])),
            **_candidate_access_features(access_pattern),
            **colstats,
            **idxmeta,
            **write_penalty,
        }

        # Cost-weighted workload impact of the candidate columns.
        base["workload_max_col_freq"] = max(
            (freqs.get(f"{table}.{c}", 0.0) for c in cols),
            default=0.0,
        )
        base["workload_sum_col_freq"] = sum(
            freqs.get(f"{table}.{c}", 0.0) for c in cols
        )

        # Cost-weighted sort/group signals.
        base["workload_max_group_by_freq"] = max(group_by_costs) if group_by_costs else 0.0
        base["workload_sum_group_by_freq"] = sum(group_by_costs)
        base["workload_max_order_by_freq"] = max(order_by_costs) if order_by_costs else 0.0
        base["workload_sum_order_by_freq"] = sum(order_by_costs)

        # Leading-column sort-elimination signals.
        base["first_col_in_group_by"] = 1.0 if group_by_costs and group_by_costs[0] > 0 else 0.0
        base["first_col_in_order_by"] = 1.0 if order_by_costs and order_by_costs[0] > 0 else 0.0

        for qname in sorted(touch):
            if qname not in explain_by_q:
                continue

            exp = explain_by_q[qname]
            row = {
                "query_name": qname,
                **base,
                **{f"q_{k}": v for k, v in exp.items()},
            }
            rows.append(row)

    return rows


def _validate_feature_rows(rows: List[Dict[str, Any]], candidates: List[dict]) -> None:
    """Sanity check for candidate_generator -> feature_extractor transition."""
    assert rows, "no feature rows produced"

    required = {
        "query_name",
        "candidate_table",
        "candidate_cols",
        "candidate_type",
        "n_index_columns",
        "is_composite",
        "range_scan_candidate",
        "candidate_cost_impact",
        "candidate_source_query_count",
        "workload_sum_col_freq",
        "q_plan_total_cost",
    }

    for row in rows:
        missing = required - set(row.keys())
        assert not missing, f"feature row missing fields: {missing}"
        assert "clustered_candidate" not in row, "old clustered_candidate feature still present"
        assert isinstance(row["candidate_cols"], str) and row["candidate_cols"], row
        assert row["candidate_type"] in {"single", "composite"}, row

    candidate_keys = {(c["table"], ",".join(c["columns"])) for c in candidates}
    row_keys = {(r["candidate_table"], r["candidate_cols"]) for r in rows}
    assert row_keys.issubset(candidate_keys), "feature rows contain unknown candidates"


if __name__ == "__main__":
    from pprint import pprint

    from candidate_generator import generate_candidates

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    queries_dir = os.path.join(repo_root, "queries")

    workload = parse_workload(queries_dir)
    candidates = generate_candidates(workload, min_cost_impact=50000.0)

    conn = get_connection()
    try:
        rows = build_feature_rows(
            conn,
            candidates,
            workload,
            queries_dir=queries_dir,
        )

        _validate_feature_rows(rows, candidates)

        print(f"Generated {len(rows)} feature rows from {len(candidates)} candidates.")
        print("Sanity check passed: candidate_generator -> feature_extractor transition is valid.")
        print("\nSample feature row:")
        pprint(rows[0])
    finally:
        conn.close()