"""
workload_parser.py
------------------
Parses TPC-H SQL query files and extracts column references
used in WHERE, GROUP BY, and ORDER BY clauses.

Input:
    - Directory of .sql query files (queries/)

Output:
    - List of dicts, each representing a column reference:
      {
          "table":         "lineitem",
          "column":        "l_shipdate",
          "clause":        "WHERE",          # primary clause (WHERE > GROUP BY > ORDER BY)
          "predicate_type":"range",           # from WHERE occurrence; 'n/a' if only in sorting clauses
          "in_where":      True,             # appeared in WHERE / ON / HAVING
          "in_group_by":   False,            # appeared in GROUP BY
          "in_order_by":   True,             # appeared in ORDER BY
          "query":         "q1",
          "query_cost":    84530.0
      }

Changes from original:
    - Dedup key changed from (table, column, clause) to (table, column) so that
      query_cost is counted exactly once per column per query in candidate_generator,
      regardless of how many clauses that column appears in.
    - Added DB connection to fetch optimizer cost per query via EXPLAIN.
    - Imports from db_utils to avoid circular dependency with feature_extractor.
    - Clause membership preserved as in_where / in_group_by / in_order_by boolean
      flags on each merged row instead of dropping multi-clause columns.
      Previously a column in both WHERE and GROUP BY would lose its GROUP BY signal
      entirely. Now both are recorded. This allows feature_extractor to expose
      sort-elimination opportunities (B+ trees skip Sort nodes for GROUP BY / ORDER BY)
      as features for the model.

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → ml_model
"""

import os
import sqlparse

from db_utils import get_connection, normalize_query_for_postgres, explain_query_json

QUERIES_DIR = 'queries'

PREFIX_TO_TABLE = {
    'l': 'lineitem',
    'o': 'orders',
    'c': 'customer',
    's': 'supplier',
    'n': 'nation',
    'r': 'region',
    'p': 'part',
    'ps': 'partsupp'
}

# Clause priority for the 'clause' field when a column appears in multiple clauses.
# WHERE carries the most actionable predicate info, so it wins on ties.
_CLAUSE_PRIORITY = {'WHERE': 0, 'GROUP BY': 1, 'ORDER BY': 2, 'n/a': 3}


def get_operator(comparison) -> str:
    """
    Extract the comparison operator from a sqlparse Comparison token.
    Defaults to '=' if no operator token is found.
    """
    for token in comparison.tokens:
        if token.ttype in (sqlparse.tokens.Comparison, sqlparse.tokens.Token.Comparison):
            return token.value.strip()
    return '='


def get_predicate_type_from_expr(expr: str) -> str:
    """
    Classify a predicate using the full comparison text.

    Categories:
      equality      -> =, IN
      range         -> <, <=, >, >=, BETWEEN
      prefix_like   -> LIKE 'abc%'
      pattern_like  -> LIKE '%abc%'
      negative      -> !=, <>, NOT IN, NOT LIKE, NOT BETWEEN
      unknown       -> fallback
    """
    e = " ".join(expr.upper().split())

    if " NOT LIKE " in e or " NOT IN " in e or " NOT BETWEEN " in e:
        return "negative"

    if "<>" in e or "!=" in e:
        return "negative"

    if " LIKE " in e:
        # Leading-wildcard LIKE is not a normal B-tree equality/range predicate.
        # Example: p_name LIKE '%light%'
        if "LIKE '%" in e or 'LIKE "%' in e:
            return "pattern_like"
        return "prefix_like"

    if " BETWEEN " in e:
        return "range"

    if any(op in e for op in ["<=", ">=", "<", ">"]):
        return "range"

    if " IN " in e:
        return "equality"

    if "=" in e:
        return "equality"

    return "unknown"


def strip_alias(column_name: str) -> str:
    """
    Strip table alias from a column name.
    e.g. 'l1.l_suppkey' → 'l_suppkey', 'n1.n_name' → 'n_name'
    """
    if '.' in column_name:
        return column_name.split('.')[-1]
    return column_name


def load_queries(queries_dir: str) -> dict:
    """
    Load all .sql files from a directory into a dictionary.
    Returns dict mapping query name to SQL string.
    """
    queries = {}
    for filename in os.listdir(queries_dir):
        if filename.endswith('.sql'):
            query_name = os.path.splitext(filename)[0]
            with open(os.path.join(queries_dir, filename), 'r') as f:
                queries[query_name] = f.read()
    return queries


def extract_columns_from_token(token, clause, predicate_type='n/a'):
    """
    Recursively flattens an AST token and extracts any TPC-H columns.
    Bypasses all nesting issues (functions, math ops, double parentheses).
    """
    cols = []
    IGNORE_ALIASES = {'l_year', 'o_year', 'c_count'}

    for leaf in token.flatten():
        val = str(leaf).strip()
        if '_' in val and not val.startswith("'") and not val.startswith('"'):
            col_name = val.split('.')[-1].lower()
            if col_name in IGNORE_ALIASES:
                continue
            prefix = col_name.split('_')[0]
            if prefix in PREFIX_TO_TABLE:
                cols.append({
                    'table': PREFIX_TO_TABLE[prefix],
                    'column': col_name,
                    'clause': clause,
                    'predicate_type': predicate_type
                })
    return cols


def extract_columns(sql: str) -> list:
    """
    Parses a SQL query using a recursive walker to catch ALL nested columns
    in WHERE, ON, HAVING, GROUP BY, and ORDER BY clauses.

    Deduplicates on (table, column) — one output row per column per query —
    but preserves full clause membership via in_where / in_group_by / in_order_by
    flags. This means:

      - query_cost is still counted exactly once per column in candidate_generator
        (no double-counting from multi-clause columns).
      - feature_extractor can use in_group_by and in_order_by as features,
        allowing the model to learn sort-elimination opportunities: a B+ tree
        index on a GROUP BY or ORDER BY column lets the optimizer skip an
        expensive in-memory Sort node entirely.

    predicate_type is taken from the WHERE occurrence when available (most
    informative), falling back to 'n/a' for columns that only appear in
    sorting clauses.

    The primary 'clause' field reflects the highest-priority clause the
    column appears in: WHERE > GROUP BY > ORDER BY.
    """
    raw_results = []
    parsed_statements = sqlparse.parse(sql)
    current_clause = 'n/a'

    def walk(token_list):
        nonlocal current_clause
        idx = 0
        tokens = token_list.tokens

        while idx < len(tokens):
            token = tokens[idx]

            if token.is_whitespace:
                idx += 1
                continue

            # 1. Update context based on keywords
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML):
                val = token.value.upper()
                if val in ('WHERE', 'ON', 'HAVING'):
                    current_clause = 'WHERE'
                elif val == 'GROUP BY':
                    current_clause = 'GROUP BY'
                elif val == 'ORDER BY':
                    current_clause = 'ORDER BY'
                elif val in ('SELECT', 'FROM'):
                    current_clause = 'n/a'

            # 2. Extract from filtering clauses (WHERE, ON, HAVING)
            if current_clause == 'WHERE':
                if isinstance(token, sqlparse.sql.Comparison):
                    try:
                        ptype = get_predicate_type_from_expr(str(token))
                    except Exception:
                        ptype = 'unknown'

                    raw_results.extend(extract_columns_from_token(token.left, current_clause, ptype))

                    if not isinstance(token.right, sqlparse.sql.Parenthesis) or 'select' not in str(token.right).lower():
                        raw_results.extend(extract_columns_from_token(token.right, current_clause, ptype))

                elif isinstance(token, (sqlparse.sql.Identifier, sqlparse.sql.Function)):
                    peek_idx = idx + 1
                    while peek_idx < len(tokens) and tokens[peek_idx].is_whitespace:
                        peek_idx += 1

                    if peek_idx < len(tokens):
                        next_tok = tokens[peek_idx]
                        kw = next_tok.value.upper()

                        # Handle two-token operators like NOT IN / NOT LIKE / NOT BETWEEN
                        if kw == "NOT":
                            peek2_idx = peek_idx + 1
                            while peek2_idx < len(tokens) and tokens[peek2_idx].is_whitespace:
                                peek2_idx += 1
                            if peek2_idx < len(tokens):
                                kw2 = tokens[peek2_idx].value.upper()
                                combined = f"NOT {kw2}"
                                if combined in {"NOT IN", "NOT LIKE", "NOT BETWEEN"}:
                                    raw_results.extend(
                                        extract_columns_from_token(token, current_clause, "negative")
                                    )

                        elif kw == "IN":
                            raw_results.extend(
                                extract_columns_from_token(token, current_clause, "equality")
                            )

                        elif kw == "BETWEEN":
                            raw_results.extend(
                                extract_columns_from_token(token, current_clause, "range")
                            )

                        elif kw == "LIKE":
                            # Look ahead for the pattern literal.
                            pattern_idx = peek_idx + 1
                            while pattern_idx < len(tokens) and tokens[pattern_idx].is_whitespace:
                                pattern_idx += 1

                            pattern = str(tokens[pattern_idx]) if pattern_idx < len(tokens) else ""
                            if pattern.strip().startswith(("'%", '"%')):
                                ptype = "pattern_like"
                            else:
                                ptype = "prefix_like"

                            raw_results.extend(
                                extract_columns_from_token(token, current_clause, ptype)
                            )

            # 3. Extract from grouping/sorting clauses
            elif current_clause in ('GROUP BY', 'ORDER BY'):
                if isinstance(token, (sqlparse.sql.Identifier, sqlparse.sql.IdentifierList, sqlparse.sql.Function)):
                    raw_results.extend(extract_columns_from_token(token, current_clause, 'n/a'))

            # 4. Recurse into containers
            if hasattr(token, 'tokens'):
                prev_clause = current_clause
                if isinstance(token, sqlparse.sql.Parenthesis) and 'select' in str(token).lower():
                    current_clause = 'n/a'
                walk(token)
                current_clause = prev_clause

            idx += 1

    for parsed in parsed_statements:
        current_clause = 'n/a'
        walk(parsed)

    # Merge all occurrences of (table, column) into one row with clause flags.
    # This preserves GROUP BY / ORDER BY signal that was previously discarded,
    # while still emitting exactly one row per column so query_cost is counted
    # once in candidate_generator.
    merged: dict = {}  # (table, column) → merged row

    for r in raw_results:
        key = (r['table'], r['column'])
        if key not in merged:
            merged[key] = {
                'table':          r['table'],
                'column':         r['column'],
                'clause':         r['clause'],
                'predicate_type': r['predicate_type'],
                'in_where':       r['clause'] == 'WHERE',
                'in_group_by':    r['clause'] == 'GROUP BY',
                'in_order_by':    r['clause'] == 'ORDER BY',
            }
        else:
            existing = merged[key]
            # Update primary clause to highest priority seen
            if _CLAUSE_PRIORITY.get(r['clause'], 3) < _CLAUSE_PRIORITY.get(existing['clause'], 3):
                existing['clause'] = r['clause']
            # Update predicate_type from WHERE if we didn't have one yet
            if existing['predicate_type'] == 'n/a' and r['clause'] == 'WHERE':
                existing['predicate_type'] = r['predicate_type']
            # Set clause membership flags
            if r['clause'] == 'WHERE':
                existing['in_where'] = True
            elif r['clause'] == 'GROUP BY':
                existing['in_group_by'] = True
            elif r['clause'] == 'ORDER BY':
                existing['in_order_by'] = True

    return list(merged.values())


def parse_workload(queries_dir: str) -> list:
    """
    Main entry point — load all queries, get their estimated optimizer cost,
    and extract column references with cost attached to each row.
    """
    queries = load_queries(queries_dir)
    all_results = []

    try:
        conn = get_connection()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        conn = None

    for query_name, sql in sorted(queries.items()):
        cost = 1.0
        if conn:
            try:
                explain_result = explain_query_json(conn, sql)
                cost = explain_result['plan_total_cost']
            except Exception as e:
                print(f"Warning: Could not get cost for {query_name}, defaulting to 1.0. Error: {e}")

        columns = extract_columns(sql)
        for col in columns:
            col['query'] = query_name
            col['query_cost'] = cost
            col['query_sql'] = sql.strip()

        all_results.extend(columns)

    if conn:
        conn.close()

    return all_results


if __name__ == '__main__':
    workload = parse_workload(QUERIES_DIR)

    with open('query_analysis.txt', 'w') as f:
        total = len(workload)
        f.write(f"Parsed {total} column references with cost data attached.\n\n")

        current_query = ""

        for item in workload:
            if item['query'] != current_query:
                current_query = item['query']

                f.write(f"\n{'='*80}\n")
                f.write(f"QUERY: {current_query} (Estimated Cost: {item['query_cost']})\n")
                f.write(f"{'='*80}\n\n")

                f.write("ORIGINAL SQL:\n")
                f.write("-" * 80 + "\n")
                f.write(item.get("query_sql", "").strip())
                f.write("\n")
                f.write("-" * 80 + "\n\n")

                f.write("EXTRACTED COLUMNS:\n")

            flags = []
            if item.get('in_where'):
                flags.append('WHERE')
            if item.get('in_group_by'):
                flags.append('GROUP BY')
            if item.get('in_order_by'):
                flags.append('ORDER BY')

            clause_str = ' + '.join(flags) if flags else 'n/a'
            predicate = item.get('predicate_type', 'n/a')

            f.write(
                f"  {clause_str:30} "
                f"{predicate:10} "
                f"{item['table']}.{item['column']}\n"
            )

    print("Written to query_analysis.txt")