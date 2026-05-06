"""
evaluate_indexes.py
-------------------
Physical index evaluation: create top-k recommended indexes and compare workload
cost / optional wall time vs baseline.

Uses the same ranking path as ml_model.recommend (fresh features + trained model).

Default measurement: EXPLAIN (FORMAT JSON) total plan cost per query (fast,
aligned with HypoPG labeling). Optional --analyze runs EXPLAIN (ANALYZE, FORMAT
JSON) which executes each query (slow on TPC-H).

Usage:
    python src/evaluate_indexes.py --top-k 5
    python src/evaluate_indexes.py --top-k 5 --analyze
    python src/evaluate_indexes.py --top-k 3 --drop-after
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Sequence, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import xgboost as xgb

from db_utils import get_connection, normalize_query_for_postgres
from ml_model import (
    DEFAULT_MIN_COST_IMPACT,
    MODEL_PATH,
    build_recommendation_features,
    feature_cols_path_for_model,
    load_feature_columns,
    recommend,
)
from workload_parser import load_queries

_REPO_ROOT = os.path.dirname(_SRC_DIR)
INDEX_PREFIX = "ir_eval"


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def create_index_statement(name: str, table: str, cols_csv: str) -> str:
    cols = [_quote_ident(c.strip()) for c in str(cols_csv).split(",") if c.strip()]
    collist = ", ".join(cols)
    t = _quote_ident("public") + "." + _quote_ident(table)
    return f"CREATE INDEX IF NOT EXISTS {_quote_ident(name)} ON {t} ({collist})"


def drop_eval_indexes(conn, names: Sequence[str]) -> None:
    if not names:
        return
    with conn.cursor() as cur:
        for name in names:
            cur.execute(f"DROP INDEX IF EXISTS {_quote_ident(name)}")


def list_eval_indexes(conn) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public' AND indexname ~ %s
            ORDER BY indexname
            """,
            (f"^{re.escape(INDEX_PREFIX)}_[0-9]+$",),
        )
        return [r[0] for r in cur.fetchall()]


def hypopg_reset_safe(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT hypopg_reset()")
    except Exception:
        pass


def workload_planner_costs(conn, queries: Dict[str, str]) -> Tuple[Dict[str, float], float]:
    from db_utils import explain_query_json

    per: Dict[str, float] = {}
    total = 0.0
    for qn in sorted(queries.keys()):
        cost = float(explain_query_json(conn, queries[qn])["plan_total_cost"])
        per[str(qn)] = cost
        total += cost
    return per, total


def workload_analyze_times(conn, queries: Dict[str, str]) -> Tuple[Dict[str, float], float]:
    """Actual execution time (ms) from EXPLAIN (ANALYZE, FORMAT JSON) root plan."""
    per: Dict[str, float] = {}
    total = 0.0
    for qn in sorted(queries.keys()):
        sql = normalize_query_for_postgres(queries[qn]).strip().rstrip(";")
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
            (raw,) = cur.fetchone()
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        root = parsed[0].get("Plan") if parsed else None
        if not isinstance(root, dict):
            ms = 0.0
        else:
            v = root.get("Actual Total Time")
            ms = float(v) if v is not None else 0.0
        per[str(qn)] = ms
        total += ms
    return per, total


def run_evaluation(
    repo_root: str,
    model_path: str,
    top_k: int,
    min_cost_impact: float,
    run_analyze: bool,
    drop_after: bool,
    dry_run: bool,
) -> None:
    queries_dir = os.path.join(repo_root, "queries")
    queries = load_queries(queries_dir)
    cols_path = feature_cols_path_for_model(model_path)

    if not os.path.exists(model_path):
        raise SystemExit(f"Missing model: {model_path}")
    if not os.path.exists(cols_path):
        raise SystemExit(f"Missing feature columns file: {cols_path}")

    conn = get_connection()
    conn.autocommit = True
    hypopg_reset_safe(conn)

    existing = list_eval_indexes(conn)
    if existing:
        print(f"Removing prior eval indexes: {existing}")
        drop_eval_indexes(conn, existing)

    print("\n--- Baseline (no new indexes) ---")
    base_costs, base_cost_sum = workload_planner_costs(conn, queries)
    print(f"Sum of planner costs: {base_cost_sum:,.2f}")
    base_ms_sum = 0.0
    if run_analyze:
        print("Running EXPLAIN ANALYZE for baseline (executes workload; may take a while)...")
        t0 = time.time()
        _base_ms, base_ms_sum = workload_analyze_times(conn, queries)
        print(f"Sum of actual times: {base_ms_sum:,.2f} ms ({time.time() - t0:.1f}s wall)")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    feature_cols = load_feature_columns(cols_path)
    features_df = build_recommendation_features(
        repo_root=os.path.abspath(repo_root),
        min_cost_impact=min_cost_impact,
    )
    if features_df.empty:
        raise SystemExit("No feature rows for recommendation.")

    ranked = recommend(model, feature_cols, features_df, top_k=top_k, budget=None)
    if ranked.empty:
        raise SystemExit("No recommendations to apply.")

    index_names: List[str] = []
    stmts: List[str] = []
    tables: set[str] = set()

    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
        name = f"{INDEX_PREFIX}_{i:03d}"
        index_names.append(name)
        tables.add(str(row["candidate_table"]))
        stmts.append(create_index_statement(name, row["candidate_table"], row["candidate_cols"]))

    print(f"\n--- Applying {len(stmts)} physical indexes ---")
    for s in stmts:
        print(s + ";")

    if dry_run:
        print("\nDry run: skipping CREATE INDEX and remeasure.")
        return

    with conn.cursor() as cur:
        for s, iname in zip(stmts, index_names):
            t0 = time.time()
            cur.execute(s)
            print(f"Created in {time.time() - t0:.2f}s: {iname}")

    with conn.cursor() as cur:
        for tbl in sorted(tables):
            cur.execute(f"ANALYZE {_quote_ident('public')}.{_quote_ident(tbl)}")

    print("\n--- After indexes (+ ANALYZE on affected tables) ---")
    idx_costs, idx_cost_sum = workload_planner_costs(conn, queries)
    print(f"Sum of planner costs: {idx_cost_sum:,.2f}")
    delta_c = base_cost_sum - idx_cost_sum
    pct_c = (100.0 * delta_c / base_cost_sum) if base_cost_sum > 0 else 0.0
    print(f"Planner cost reduction: {delta_c:,.2f} ({pct_c:.2f}% lower than baseline)")

    if run_analyze:
        print("Running EXPLAIN ANALYZE again (indexed workload)...")
        t0 = time.time()
        idx_ms, idx_ms_sum = workload_analyze_times(conn, queries)
        print(f"Sum of actual times: {idx_ms_sum:,.2f} ms ({time.time() - t0:.1f}s wall)")
        delta_t = base_ms_sum - idx_ms_sum
        pct_t = (100.0 * delta_t / base_ms_sum) if base_ms_sum > 0 else 0.0
        print(f"Actual time reduction: {delta_t:,.2f} ms ({pct_t:.2f}% faster than baseline)")

    print("\nPer-query planner cost (baseline -> indexed):")
    for qn in sorted(queries.keys()):
        bc, ic = base_costs[qn], idx_costs[qn]
        print(f"  {qn:>4}  {bc:>14,.2f}  ->  {ic:>14,.2f}  ({(bc - ic):>+12,.2f})")

    if drop_after:
        print(f"\nDropping {index_names} ...")
        drop_eval_indexes(conn, index_names)
    else:
        print(f"\nIndexes left in place: {index_names}")
        print("Re-run with --drop-after to remove them, or DROP INDEX manually.")


def main() -> None:
    p = argparse.ArgumentParser(description="Create recommended indexes and measure workload improvement.")
    p.add_argument("--repo-root", default=_REPO_ROOT)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-cost-impact", type=float, default=DEFAULT_MIN_COST_IMPACT)
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Also run EXPLAIN ANALYZE (executes queries; slow).",
    )
    p.add_argument(
        "--drop-after",
        action="store_true",
        help="DROP eval indexes after measurement (keeps DB clean).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print SQL only; do not create indexes.")
    args = p.parse_args()

    run_evaluation(
        repo_root=os.path.abspath(args.repo_root),
        model_path=os.path.abspath(args.model_path),
        top_k=args.top_k,
        min_cost_impact=args.min_cost_impact,
        run_analyze=args.analyze,
        drop_after=args.drop_after,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()