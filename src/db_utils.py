"""
db_utils.py
-----------
Shared database utilities used across the pipeline.
Centralizes get_connection, normalize_query_for_postgres, and explain_query_json
so that workload_parser and feature_extractor can both import from here
without creating a circular dependency.
"""

import json
import os
import re

import psycopg2
from dotenv import load_dotenv


def get_connection():
    """Connect to Postgres using .env credentials."""
    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tpch'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'password')
    )
    conn.autocommit = True
    return conn


def normalize_query_for_postgres(sql: str) -> str:
    """
    Normalize common Oracle-style TPC-H query variants so PostgreSQL can EXPLAIN them.
    Handles: revenue0 view rewrite, rownum clauses, interval syntax.
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
        create_stmt_end = lower.find(";", create_end) if create_end != -1 else -1
        main_select_match = None
        if create_stmt_end != -1:
            main_select_match = re.search(
                r"\bselect\b.*?;",
                normalized[create_stmt_end + 1:],
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


def _walk_plan(node, acc):
    """Recursively walk an EXPLAIN JSON plan tree and count node types."""
    if not isinstance(node, dict):
        return
    nt = node.get("Node Type", "")
    if nt == "Seq Scan":
        acc["n_seq_scan"] += 1.0
    elif nt in ("Index Scan", "Index Only Scan", "Bitmap Index Scan"):
        acc["n_index_scan"] += 1.0
    for child in node.get("Plans") or []:
        _walk_plan(child, acc)


def summarize_explain_json(explain_parsed) -> dict:
    """Reduce EXPLAIN JSON to a small numeric dict."""
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


def explain_query_json(conn, sql: str) -> dict:
    """Run EXPLAIN (FORMAT JSON) and return summarized optimizer features."""
    stripped = normalize_query_for_postgres(sql).strip().rstrip(";")
    with conn.cursor() as cur:
        cur.execute(f"EXPLAIN (FORMAT JSON) {stripped}")
        (raw,) = cur.fetchone()
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return summarize_explain_json(parsed)