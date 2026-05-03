"""
hypopg_labeler.py
-----------------
Creates supervised labels for candidate indexes using HypoPG.

Important design choice:
    Labels are optimizer-estimated benefits, not measured runtime benefits.
    HypoPG indexes are hypothetical/planner-only, so the correct measurement is:

        label = baseline_EXPLAIN_cost - EXPLAIN_cost_with_hypothetical_index

    Do not replace this with EXPLAIN ANALYZE. EXPLAIN ANALYZE executes the query
    against real storage, where the hypothetical index does not physically exist.

Input:
    - workload rows from workload_parser.py
    - candidates from candidate_generator.py
    - SQL files from queries/

Output CSV format:
    query_name,candidate_table,candidate_cols,label,label_source

Pipeline:
    workload_parser -> candidate_generator -> feature_extractor -> hypopg_labeler
    -> training_dataset -> ml_model
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from candidate_generator import generate_candidates
from db_utils import get_connection, explain_query_json
from feature_extractor import build_feature_rows, queries_touching_table
from workload_parser import load_queries, parse_workload

REPO_ROOT = os.path.dirname(_SRC_DIR)
QUERIES_DIR = os.path.join(REPO_ROOT, "queries")
DEFAULT_LABELS_PATH = os.path.join(REPO_ROOT, "data", "labels.csv")
MIN_COST_IMPACT = 50000.0

REQUIRED_CANDIDATE_FIELDS = {
    "table",
    "columns",
    "type",
    "range_scan_candidate",
    "source_queries",
    "access_pattern",
    "cost_impact",
}

LABEL_COLUMNS = [
    "query_name",
    "candidate_table",
    "candidate_cols",
    "label",
    "label_source",
]


def _quote_ident(identifier: str) -> str:
    """Return a safely double-quoted SQL identifier."""
    return '"' + str(identifier).replace('"', '""') + '"'


def _qualified_table(table: str, schema: str = "public") -> str:
    if "." in table:
        return ".".join(_quote_ident(part) for part in table.split("."))
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def build_index_sql(candidate: dict, schema: str = "public") -> str:
    """Build CREATE INDEX SQL text for HypoPG from candidate metadata."""
    table = str(candidate["table"])
    columns = list(candidate["columns"])
    if not columns:
        raise ValueError(f"Candidate has no columns: {candidate}")

    quoted_cols = ", ".join(_quote_ident(c) for c in columns)
    return f"CREATE INDEX ON {_qualified_table(table, schema=schema)} ({quoted_cols})"


def hypopg_reset(conn) -> None:
    """Reset all hypothetical indexes in the current session."""
    with conn.cursor() as cur:
        cur.execute("SELECT hypopg_reset()")


def ensure_hypopg(conn) -> None:
    """Verify HypoPG is available in the current database/session."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'hypopg'")
        row = cur.fetchone()
    if not row:
        raise RuntimeError("HypoPG extension is not installed or not visible.")


def hypopg_create_index(conn, index_sql: str) -> None:
    """
    Create a hypothetical index through HypoPG.

    hypopg_create_index accepts the CREATE INDEX SQL as a string, so this uses a
    bound parameter rather than string-interpolating into the outer SELECT.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM hypopg_create_index(%s)", (index_sql,))


def get_explain_cost(conn, sql: str) -> float:
    """
    Return optimizer-estimated total plan cost for a query.

    This intentionally uses plain EXPLAIN, not EXPLAIN ANALYZE, because HypoPG
    indexes are planner-only and do not exist during real execution.
    """
    result = explain_query_json(conn, sql)
    return float(result["plan_total_cost"])


def compute_baseline_costs(conn, queries: Dict[str, str]) -> Dict[str, float]:
    """Compute no-hypothetical-index EXPLAIN costs for every query."""
    hypopg_reset(conn)
    return {query_name: get_explain_cost(conn, sql) for query_name, sql in sorted(queries.items())}


def candidate_cols_string(candidate: dict) -> str:
    """Stable string representation used by feature_extractor/training_dataset."""
    return ",".join(candidate["columns"])


def label_key(query_name: str, candidate: dict) -> Tuple[str, str, str]:
    return (str(query_name), str(candidate["table"]), candidate_cols_string(candidate))


def relevant_queries_for_candidate(candidate: dict, workload: List[dict]) -> Set[str]:
    """
    Use candidate source_queries when present so label rows align with patched
    feature_extractor.py. Fall back to all queries touching the table for legacy
    candidates.
    """
    source_queries = candidate.get("source_queries") or []
    if source_queries:
        return {str(q) for q in source_queries}
    return set(queries_touching_table(workload, candidate["table"]))


def label_candidate_individual(
    conn,
    candidate: dict,
    queries: Dict[str, str],
    baseline_costs: Dict[str, float],
    workload: List[dict],
    schema: str = "public",
) -> List[dict]:
    """
    Label one candidate in isolation.

    Returns one row per relevant query:
        query_name, candidate_table, candidate_cols, label, label_source
    """
    query_names = sorted(relevant_queries_for_candidate(candidate, workload))
    missing = [q for q in query_names if q not in queries]
    if missing:
        raise KeyError(f"Candidate references queries not found in workload files: {missing}")

    index_sql = build_index_sql(candidate, schema=schema)
    rows: List[dict] = []

    hypopg_reset(conn)
    try:
        hypopg_create_index(conn, index_sql)
        for query_name in query_names:
            baseline = baseline_costs[query_name]
            indexed = get_explain_cost(conn, queries[query_name])
            benefit = baseline - indexed
            rows.append(
                {
                    "query_name": query_name,
                    "candidate_table": candidate["table"],
                    "candidate_cols": candidate_cols_string(candidate),
                    "label": float(benefit),
                    "label_source": "individual",
                }
            )
    finally:
        hypopg_reset(conn)

    return rows


def label_all_candidates_individual(
    conn,
    candidates: Sequence[dict],
    queries: Dict[str, str],
    workload: List[dict],
    schema: str = "public",
) -> List[dict]:
    """Create individual HypoPG labels for all candidates."""
    baseline_costs = compute_baseline_costs(conn, queries)
    all_rows: List[dict] = []

    for i, candidate in enumerate(candidates, start=1):
        rows = label_candidate_individual(
            conn=conn,
            candidate=candidate,
            queries=queries,
            baseline_costs=baseline_costs,
            workload=workload,
            schema=schema,
        )
        all_rows.extend(rows)

        if i % 20 == 0 or i == len(candidates):
            print(f"Labeled {i}/{len(candidates)} candidates ({len(all_rows)} rows so far).")

    return all_rows


def write_labels_csv(rows: Sequence[dict], path: str) -> None:
    """Write minimal labels CSV expected by training_dataset.py."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in LABEL_COLUMNS})


def _candidate_key(candidate: dict) -> Tuple[str, Tuple[str, ...]]:
    return (str(candidate["table"]), tuple(candidate["columns"]))


def _feature_key(row: dict) -> Tuple[str, str, str]:
    return (str(row["query_name"]), str(row["candidate_table"]), str(row["candidate_cols"]))


def _label_row_key(row: dict) -> Tuple[str, str, str]:
    return (str(row["query_name"]), str(row["candidate_table"]), str(row["candidate_cols"]))


def sanity_check_candidate_system(candidates: Sequence[dict]) -> None:
    """Validate patched candidate_generator.py output contract."""
    if not candidates:
        raise AssertionError("No candidates generated.")

    seen = set()
    for c in candidates:
        missing = REQUIRED_CANDIDATE_FIELDS - set(c.keys())
        if missing:
            raise AssertionError(f"Candidate missing required fields {missing}: {c}")
        if "clustered_candidate" in c:
            raise AssertionError(f"Stale clustered_candidate field found: {c}")
        if not isinstance(c["columns"], list) or not c["columns"]:
            raise AssertionError(f"Candidate columns must be a non-empty list: {c}")
        if c["type"] not in {"single", "composite"}:
            raise AssertionError(f"Unexpected candidate type: {c}")
        if c["type"] == "single" and len(c["columns"]) != 1:
            raise AssertionError(f"Single candidate must have one column: {c}")
        if c["type"] == "composite" and len(c["columns"]) != 2:
            raise AssertionError(f"Composite candidate must have two columns: {c}")

        key = _candidate_key(c)
        if key in seen:
            raise AssertionError(f"Duplicate candidate found: {c}")
        seen.add(key)


def sanity_check_learning_alignment(
    feature_rows: Sequence[dict],
    label_rows: Sequence[dict],
) -> None:
    """
    Verify feature_extractor.py rows and hypopg_labeler.py rows align exactly
    on training_dataset.py's join key:
        query_name|candidate_table|candidate_cols
    """
    if not feature_rows:
        raise AssertionError("No feature rows generated.")
    if not label_rows:
        raise AssertionError("No label rows generated.")

    for row in feature_rows[:5]:
        if "clustered_candidate" in row:
            raise AssertionError("Feature rows still contain clustered_candidate.")
        if "range_scan_candidate" not in row:
            raise AssertionError("Feature rows missing range_scan_candidate.")

    feature_keys = [_feature_key(r) for r in feature_rows]
    label_keys = [_label_row_key(r) for r in label_rows]

    if len(feature_keys) != len(set(feature_keys)):
        raise AssertionError("Duplicate feature keys found.")
    if len(label_keys) != len(set(label_keys)):
        raise AssertionError("Duplicate label keys found.")

    feature_key_set = set(feature_keys)
    label_key_set = set(label_keys)

    missing_labels = sorted(feature_key_set - label_key_set)
    extra_labels = sorted(label_key_set - feature_key_set)

    if missing_labels:
        preview = missing_labels[:10]
        raise AssertionError(f"Feature rows without labels: {preview}")
    if extra_labels:
        preview = extra_labels[:10]
        raise AssertionError(f"Label rows without feature rows: {preview}")

    bad_sources = {row.get("label_source") for row in label_rows} - {"individual"}
    if bad_sources:
        raise AssertionError(f"Only individual labels are expected, found: {bad_sources}")

    for row in label_rows:
        try:
            float(row["label"])
        except Exception as e:
            raise AssertionError(f"Non-numeric label row: {row}") from e


def run_pipeline(
    repo_root: str,
    out_path: str,
    min_cost_impact: float = MIN_COST_IMPACT,
    schema: str = "public",
    check_features: bool = True,
) -> List[dict]:
    """End-to-end label generation with transition sanity checks."""
    queries_dir = os.path.join(repo_root, "queries")

    workload = parse_workload(queries_dir)
    candidates = generate_candidates(workload, min_cost_impact=min_cost_impact)
    sanity_check_candidate_system(candidates)

    queries = load_queries(queries_dir)

    conn = get_connection()
    try:
        conn.autocommit = True
        ensure_hypopg(conn)

        feature_rows: List[dict] = []
        if check_features:
            feature_rows = build_feature_rows(
                conn,
                candidates,
                workload,
                queries_dir=queries_dir,
                schema=schema,
            )

        label_rows = label_all_candidates_individual(
            conn=conn,
            candidates=candidates,
            queries=queries,
            workload=workload,
            schema=schema,
        )

        if check_features:
            sanity_check_learning_alignment(feature_rows, label_rows)
    finally:
        conn.close()

    write_labels_csv(label_rows, out_path)
    return label_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate individual HypoPG labels for candidate indexes."
    )
    parser.add_argument("--repo-root", default=REPO_ROOT)
    parser.add_argument("--out", default=DEFAULT_LABELS_PATH)
    parser.add_argument("--schema", default="public")
    parser.add_argument("--min-cost-impact", type=float, default=MIN_COST_IMPACT)
    parser.add_argument(
        "--skip-feature-check",
        action="store_true",
        help="Skip building feature rows and checking feature/label key alignment.",
    )
    args = parser.parse_args()

    rows = run_pipeline(
        repo_root=os.path.abspath(args.repo_root),
        out_path=os.path.abspath(args.out),
        min_cost_impact=args.min_cost_impact,
        schema=args.schema,
        check_features=not args.skip_feature_check,
    )

    print("\n" + "=" * 60)
    print(f"Wrote {len(rows)} label rows to {os.path.abspath(args.out)}")
    print("Label source: individual HypoPG optimizer-estimated cost deltas")
    print("Sanity check passed: candidate_system -> feature_extractor -> hypopg_labeler alignment is valid.")
    print("=" * 60)

    print("\nSample labels:")
    for row in rows[:10]:
        print(
            f"  {row['query_name']:>4}  "
            f"{row['candidate_table']}({row['candidate_cols']})  "
            f"label={float(row['label']):.2f}  "
            f"source={row['label_source']}"
        )


if __name__ == "__main__":
    main()