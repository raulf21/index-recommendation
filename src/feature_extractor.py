"""
feature_extractor.py
--------------------
Pulls PostgreSQL catalog statistics (pg_stats) and optimizer estimates from
EXPLAIN (FORMAT JSON) for workload queries. Produces fixed-width numeric
summaries suitable for tree-based models.

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → ml_model

Requires: running PostgreSQL with TPC-H loaded (.env DB_* vars), same as the rest of the stack.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import psycopg2
from dotenv import load_dotenv

from workload_parser import load_queries, parse_workload

QUERIES_DIR = "queries"


def get_connection():
    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tpch"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def _parse_pg_array(value: Any) -> Optional[List[Any]]:
    """Parse PostgreSQL array text or pass through list from psycopg2."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value == "{}":
            return []
        # Rough parse for common histogram / mcv forms from pg_stats
        inner = value.strip()
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1]
        if not inner:
            return []
        parts = re.split(r",(?![^(]*\))", inner)
        return [p.strip().strip('"') for p in parts if p.strip()]
    return None


def _histogram_summary(bounds: Any) -> Dict[str, float]:
    """Turn histogram_bounds into a small numeric fingerprint."""
    arr = _parse_pg_array(bounds)
    if not arr:
        return {
            "hist_n_buckets": 0.0,
            "hist_span": 0.0,
        }
    n = len(arr)
    try:
        first = float(arr[0])
        last = float(arr[-1])
        span = abs(last - first)
    except (TypeError, ValueError):
        span = float(n)
    return {
        "hist_n_buckets": float(max(0, n - 1)),
        "hist_span": float(span),
    }


def _mcv_top_freq(freqs: Any) -> float:
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
    conn, table: str, column: str, schema: str = "public"
) -> Optional[Dict[str, Any]]:
    """
    One row from pg_stats for (schema, table, column).
    Returns None if missing (e.g. system column or no stats yet).
    """
    q = """
        SELECT
            null_frac,
            avg_width,
            n_distinct,
            correlation,
            most_common_freqs,
            histogram_bounds
        FROM pg_stats
        WHERE schemaname = %s AND tablename = %s AND attname = %s
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table, column))
        row = cur.fetchone()
    if not row:
        return None
    null_frac, avg_width, n_distinct, correlation, mcf, hb = row
    hist = _histogram_summary(hb)
    corr = correlation
    if corr is None:
        corr = 0.0
    nd = n_distinct
    if nd is None:
        nd = 0.0
    return {
        "null_frac": float(null_frac or 0.0),
        "avg_width": float(avg_width or 0.0),
        "n_distinct": float(nd),
        "correlation": float(corr),
        "mcv_top_freq": _mcv_top_freq(mcf),
        **hist,
    }


def fetch_pg_stats_for_columns(
    conn,
    table_columns: Iterable[Tuple[str, str]],
    schema: str = "public",
) -> Dict[str, Dict[str, float]]:
    """
    Batch-fetch pg_stats for unique (table, column) pairs.
    Keys are 'table.column' (same convention as candidate_generator frequencies).
    """
    seen: Set[Tuple[str, str]] = set()
    out: Dict[str, Dict[str, float]] = {}
    for table, col in table_columns:
        key_t = (table, col)
        if key_t in seen:
            continue
        seen.add(key_t)
        row = fetch_pg_stats_row(conn, table, col, schema)
        if row:
            out[f"{table}.{col}"] = row
        else:
            out[f"{table}.{col}"] = {
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
    stats_by_col: Mapping[str, Mapping[str, float]], columns: Sequence[str], table: str
) -> Dict[str, float]:
    """
    Combine per-column pg_stats for a multi-column candidate into one fixed vector
    using simple summaries (mean / min / max).
    """
    keys = [f"{table}.{c}" for c in columns]
    rows = [stats_by_col[k] for k in keys if k in stats_by_col]
    if not rows:
        return {
            "colstats_mean_null_frac": 0.0,
            "colstats_mean_corr": 0.0,
            "colstats_min_ndistinct": 0.0,
            "colstats_max_mcv_top": 0.0,
            "colstats_mean_hist_buckets": 0.0,
        }

    def mean(name: str) -> float:
        return sum(float(r[name]) for r in rows) / len(rows)

    return {
        "colstats_mean_null_frac": mean("null_frac"),
        "colstats_mean_corr": mean("correlation"),
        "colstats_min_ndistinct": min(float(r["n_distinct"]) for r in rows),
        "colstats_max_mcv_top": max(float(r["mcv_top_freq"]) for r in rows),
        "colstats_mean_hist_buckets": mean("hist_n_buckets"),
    }

def list_indexed_column_sets(conn, table: str, schema: str = "public") -> List[Tuple[str, ...]]:
    """Return list of column tuples for each index on the table (ordered)."""
    q = """
        SELECT ARRAY(
            SELECT a.attname
            FROM unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ord)
            JOIN pg_attribute a
              ON a.attrelid = ix.indrelid AND a.attnum = k.attnum AND NOT a.attisdropped
            ORDER BY k.ord
        ) AS cols
        FROM pg_class t
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_index ix ON t.oid = ix.indrelid
        WHERE n.nspname = %s AND t.relkind = 'r' AND t.relname = %s
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
    conn, table: str, columns: Sequence[str], schema: str = "public"
) -> Dict[str, float]:
    """
    Heuristic: does an index already cover this column set as a prefix?
    """
    want = tuple(columns)
    idx_sets = list_indexed_column_sets(conn, table, schema)
    exact = 0.0
    prefix = 0.0
    for icols in idx_sets:
        if icols == want:
            exact = 1.0
        if len(icols) >= len(want) and icols[: len(want)] == want:
            prefix = 1.0
    return {
        "idx_already_exact": exact,
        "idx_already_prefix": max(exact, prefix),
        "n_existing_indexes_on_table": float(len(idx_sets)),
    }

def _walk_plan(node: Mapping[str, Any], acc: Dict[str, float]) -> None:
    if not isinstance(node, dict):
        return
    nt = node.get("Node Type", "")
    if nt == "Seq Scan":
        acc["n_seq_scan"] += 1.0
    elif nt in ("Index Scan", "Index Only Scan", "Bitmap Index Scan"):
        acc["n_index_scan"] += 1.0
    for child in node.get("Plans") or []:
        _walk_plan(child, acc)


def summarize_explain_json(explain_parsed: Any) -> Dict[str, float]:
    """
    Reduce EXPLAIN JSON to a small numeric dict.
    explain_parsed is typically a list of one element from PostgreSQL.
    """
    acc = {
        "plan_total_cost": 0.0,
        "plan_startup_cost": 0.0,
        "plan_rows": 0.0,
        "n_seq_scan": 0.0,
        "n_index_scan": 0.0,
    }
    if not explain_parsed or not isinstance(explain_parsed, list):
        return acc
    root = explain_parsed[0].get("Plan")
    if not isinstance(root, dict):
        return acc
    acc["plan_total_cost"] = float(root.get("Total Cost") or 0.0)
    acc["plan_startup_cost"] = float(root.get("Startup Cost") or 0.0)
    acc["plan_rows"] = float(root.get("Plan Rows") or 0.0)
    _walk_plan(root, acc)
    return acc


def normalize_query_for_postgres(sql: str) -> str:
    """
    Normalize common Oracle-style TPC-H query variants so PostgreSQL can EXPLAIN them.
    """
    normalized = sql

    lower = normalized.lower()
    if "create view revenue0" in lower and "drop view revenue0" in lower:
        view_match = re.search(
            r"create\s+view\s+revenue0\s*\([^)]*\)\s*as\s*(select.*?);",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        create_end = lower.find("create view revenue0")
        if create_end != -1:
            create_stmt_end = lower.find(";", create_end)
        else:
            create_stmt_end = -1
        main_select_match = None
        if create_stmt_end != -1:
            main_select_match = re.search(
                r"\bselect\b.*?;",
                normalized[create_stmt_end + 1 :],
                flags=re.IGNORECASE | re.DOTALL,
            )

        if view_match and main_select_match:
            view_select = view_match.group(1).strip().rstrip(";")
            main_select = main_select_match.group(0).strip().rstrip(";")
            normalized = (
                "WITH revenue0 AS (\n"
                "    SELECT supplier_no, total_revenue\n"
                f"    FROM ({view_select}) AS revenue0_base(supplier_no, total_revenue)\n"
                ")\n"
                f"{main_select};"
            )
        else:
            end = lower.rfind("drop view revenue0")
            if end != -1:
                normalized = normalized[:end]

    normalized = re.sub(
        r"\n\s*where\s+rownum\s*<=\s*-?\d+\s*;\s*$",
        ";\n",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"interval\s*'(\d+)'\s*day\s*\(\d+\)",
        r"interval '\1 days'",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"interval\s*'(\d+)'\s*month\b",
        r"interval '\1 months'",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = normalized.strip()
    if not normalized.endswith(";"):
        normalized += ";"
    return normalized


def explain_query_json(conn, sql: str) -> Dict[str, float]:
    """Run EXPLAIN (FORMAT JSON) and return summarized optimizer features."""
    stripped = normalize_query_for_postgres(sql).strip().rstrip(";")
    with conn.cursor() as cur:
        cur.execute(f"EXPLAIN (FORMAT JSON) {stripped}")
        (raw,) = cur.fetchone()
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return summarize_explain_json(parsed)


def explain_workload(
    conn, queries: Optional[Mapping[str, str]] = None, queries_dir: str = QUERIES_DIR
) -> Dict[str, Dict[str, float]]:
    """
    Summarized EXPLAIN output per query name.
    """
    if queries is None:
        queries = load_queries(queries_dir)
    out: Dict[str, Dict[str, float]] = {}
    for name in sorted(queries.keys()):
        sql = queries[name]
        try:
            out[name] = explain_query_json(conn, sql)
        except Exception as ex:  
            conn.rollback()
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
    """Count queries per table.column."""
    freq: Dict[str, int] = {}
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        freq[key] = freq.get(key, 0) + 1
    return {k: float(v) for k, v in freq.items()}


def queries_touching_table(workload: List[dict], table: str) -> Set[str]:
    return {item["query"] for item in workload if item["table"] == table}


def build_feature_rows(
    conn,
    candidates: List[dict],
    workload: List[dict],
    queries: Optional[Mapping[str, str]] = None,
    queries_dir: str = QUERIES_DIR,
    schema: str = "public",
) -> List[Dict[str, Any]]:
    """
    Join candidate index metadata with pg_stats, existing-index flags, workload
    frequencies, and per-query EXPLAIN summaries for queries that touch the
    candidate's table.

    Each row: identifiers + numeric features for a (query_name, candidate) pair.
    """
    if queries is None:
        queries = load_queries(queries_dir)

    pairs: Set[Tuple[str, str]] = set()
    for c in candidates:
        for col in c["columns"]:
            pairs.add((c["table"], col))
    stats_by_col = fetch_pg_stats_for_columns(conn, pairs, schema)

    explain_by_q = explain_workload(conn, queries=queries)
    freqs = workload_column_frequencies(workload)

    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        table = cand["table"]
        cols = cand["columns"]
        colstats = aggregate_column_stats(stats_by_col, cols, table)
        idxmeta = existing_index_overlap_features(conn, table, cols, schema)
        touch = queries_touching_table(workload, table)

        base = {
            "candidate_table": table,
            "candidate_cols": ",".join(cols),
            "candidate_type": cand.get("type", "unknown"),
            "n_index_columns": float(len(cols)),
            **colstats,
            **idxmeta,
        }
        base["workload_max_col_freq"] = max(
            (freqs.get(f"{table}.{c}", 0.0) for c in cols), default=0.0
        )
        base["workload_sum_col_freq"] = sum(freqs.get(f"{table}.{c}", 0.0) for c in cols)

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


if __name__ == "__main__":
    from pprint import pprint

    from candidate_generator import generate_candidates

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _queries_dir = os.path.join(_repo_root, "queries")
    workload = parse_workload(_queries_dir)
    candidates = generate_candidates(workload, min_frequency=2)
    conn = get_connection()
    try:
        rows = build_feature_rows(
            conn,
            candidates[:5],
            workload,
            queries_dir=_queries_dir,
        )
        print(f"Sample feature rows (first of {len(rows)}): ")
        if rows:
            pprint(rows[0])
        else:
            print("(no rows — check DB and workload)")
    finally:
        conn.close()
