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
          "table": "lineitem",
          "column": "l_shipdate",
          "clause": "WHERE",
          "predicate_type": "range",
          "query": "q1"
      }

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → ml_model
"""

import sqlparse
import os

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


def get_operator(comparison) -> str:
    """
    Extract the comparison operator from a sqlparse Comparison token.
    Defaults to '=' if no operator token is found.
    """
    for token in comparison.tokens:
        if token.ttype in (sqlparse.tokens.Comparison, sqlparse.tokens.Token.Comparison):
            return token.value.strip()
    return '='


def get_predicate_type(operator: str) -> str:
    """
    Classify a SQL comparison operator as equality or range.
    """
    operator = operator.strip().upper()
    if operator in ['=', '!=', '<>', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE']:
        return 'equality'
    elif operator in ['<', '<=', '>', '>=', 'BETWEEN', 'NOT BETWEEN']:
        return 'range'
    else:
        return 'equality'


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


def extract_columns(sql: str) -> list:
    """
    Parse a single SQL query and extract all column references from
    WHERE, GROUP BY, and ORDER BY clauses.

    Handles:
    1. Direct comparisons in WHERE: col = val, col > val
    2. col IN (SELECT ...) and col BETWEEN — captured with correct predicate type
    3. Parenthesized grouped conditions: (col = val OR col = val)
    4. Subqueries in WHERE: col > (SELECT ...)
    5. Subqueries in FROM: SELECT ... FROM (SELECT ... WHERE ...)
    6. Table aliases: l1.l_suppkey, n1.n_name etc.

    Only columns with a recognized TPC-H prefix are included.
    Duplicate references (same table/column/clause) are removed.
    """
    results = []
    parsed = sqlparse.parse(sql)[0]

    for token in parsed.tokens:
        if token.is_whitespace:
            continue

        # -- WHERE clause --
        if isinstance(token, sqlparse.sql.Where):
            for item in token.tokens:
                if item.is_whitespace:
                    continue

                # direct comparison: l_shipdate <= date '1998-12-01'
                if isinstance(item, sqlparse.sql.Comparison):
                    column_name = strip_alias(str(item.left).strip())
                    prefix = column_name.split('_')[0]
                    if prefix in PREFIX_TO_TABLE:
                        results.append({
                            'table': PREFIX_TO_TABLE[prefix],
                            'column': column_name,
                            'clause': 'WHERE',
                            'predicate_type': get_predicate_type(get_operator(item))
                        })
                    # check if right side is a subquery
                    # e.g. ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem WHERE ...)
                    if isinstance(item.right, sqlparse.sql.Parenthesis):
                        right_str = str(item.right)[1:-1]
                        if 'select' in right_str.lower():
                            results.extend(extract_columns(right_str))

                # col IN (...) or col BETWEEN ... AND ...
                # sqlparse splits these into: Identifier(col) → Keyword → values
                elif isinstance(item, sqlparse.sql.Identifier):
                    idx = token.tokens.index(item)
                    for k in range(idx + 1, len(token.tokens)):
                        if not token.tokens[k].is_whitespace:
                            next_t = token.tokens[k]
                            if next_t.ttype is sqlparse.tokens.Keyword:
                                normalized = next_t.normalized.upper()
                                if normalized == 'IN':
                                    predicate = 'equality'
                                elif normalized == 'BETWEEN':
                                    predicate = 'range'
                                else:
                                    break
                                column_name = strip_alias(str(item).strip())
                                prefix = column_name.split('_')[0]
                                if prefix in PREFIX_TO_TABLE:
                                    results.append({
                                        'table': PREFIX_TO_TABLE[prefix],
                                        'column': column_name,
                                        'clause': 'WHERE',
                                        'predicate_type': predicate
                                    })
                            break

                # parenthesis — subquery or grouped conditions
                elif isinstance(item, sqlparse.sql.Parenthesis):
                    subquery_str = str(item)[1:-1]
                    if 'select' in subquery_str.lower():
                        results.extend(extract_columns(subquery_str))
                    else:
                        # grouped condition like q19: (col = val OR col = val)
                        for subitem in item.tokens:
                            if isinstance(subitem, sqlparse.sql.Comparison):
                                column_name = strip_alias(str(subitem.left).strip())
                                prefix = column_name.split('_')[0]
                                if prefix in PREFIX_TO_TABLE:
                                    results.append({
                                        'table': PREFIX_TO_TABLE[prefix],
                                        'column': column_name,
                                        'clause': 'WHERE',
                                        'predicate_type': get_predicate_type(get_operator(subitem))
                                    })

        # -- ORDER BY clause --
        if token.ttype is sqlparse.tokens.Keyword and token.normalized == 'ORDER BY':
            idx = parsed.tokens.index(token)
            for j in range(idx + 1, len(parsed.tokens)):
                if not parsed.tokens[j].is_whitespace:
                    next_token = parsed.tokens[j]
                    if isinstance(next_token, sqlparse.sql.IdentifierList):
                        for identifier in next_token.get_identifiers():
                            column_name = strip_alias(str(identifier).strip().split()[0])
                            prefix = column_name.split('_')[0]
                            if prefix in PREFIX_TO_TABLE:
                                results.append({
                                    'table': PREFIX_TO_TABLE[prefix],
                                    'column': column_name,
                                    'clause': 'ORDER BY'
                                })
                    elif isinstance(next_token, sqlparse.sql.Identifier):
                        column_name = strip_alias(str(next_token).strip().split()[0])
                        prefix = column_name.split('_')[0]
                        if prefix in PREFIX_TO_TABLE:
                            results.append({
                                'table': PREFIX_TO_TABLE[prefix],
                                'column': column_name,
                                'clause': 'ORDER BY'
                            })
                    break

        # -- GROUP BY clause --
        if token.ttype is sqlparse.tokens.Keyword and token.normalized == 'GROUP BY':
            idx = parsed.tokens.index(token)
            for j in range(idx + 1, len(parsed.tokens)):
                if not parsed.tokens[j].is_whitespace:
                    next_token = parsed.tokens[j]
                    if isinstance(next_token, sqlparse.sql.IdentifierList):
                        for identifier in next_token.get_identifiers():
                            column_name = strip_alias(str(identifier).strip().split()[0])
                            prefix = column_name.split('_')[0]
                            if prefix in PREFIX_TO_TABLE:
                                results.append({
                                    'table': PREFIX_TO_TABLE[prefix],
                                    'column': column_name,
                                    'clause': 'GROUP BY'
                                })
                    elif isinstance(next_token, sqlparse.sql.Identifier):
                        column_name = strip_alias(str(next_token).strip().split()[0])
                        prefix = column_name.split('_')[0]
                        if prefix in PREFIX_TO_TABLE:
                            results.append({
                                'table': PREFIX_TO_TABLE[prefix],
                                'column': column_name,
                                'clause': 'GROUP BY'
                            })
                    break

    # -- FROM clause subqueries --
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.Identifier):
            for subtoken in token.tokens:
                if isinstance(subtoken, sqlparse.sql.Parenthesis):
                    subquery_str = str(subtoken)[1:-1]
                    if 'select' in subquery_str.lower():
                        results.extend(extract_columns(subquery_str))

    # -- Deduplicate --
    seen = set()
    unique_results = []
    for r in results:
        key = (r['table'], r['column'], r['clause'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results


def parse_workload(queries_dir: str) -> list:
    """
    Main entry point — load all queries and extract column references.
    """
    queries = load_queries(queries_dir)
    all_results = []
    for query_name, sql in sorted(queries.items()):
        columns = extract_columns(sql)
        for col in columns:
            col['query'] = query_name
        all_results.extend(columns)
    return all_results


if __name__ == '__main__':
    queries = load_queries(QUERIES_DIR)

    with open('query_analysis.txt', 'w') as f:
        total = 0
        for name, sql in sorted(queries.items()):
            result = extract_columns(sql)
            f.write(f"{'='*60}\n")
            f.write(f"QUERY: {name}\n")
            f.write(f"{'='*60}\n")
            f.write(sql)
            f.write(f"\n{'─'*60}\n")
            f.write(f"EXTRACTED COLUMNS ({len(result)}):\n")
            for r in result:
                predicate = r.get('predicate_type', 'n/a')
                f.write(f"  {r['clause']:10} {predicate:10} {r['table']}.{r['column']}\n")
            f.write('\n')
            total += len(result)
        f.write(f"{'='*60}\n")
        f.write(f"TOTAL COLUMN REFERENCES: {total}\n")

    print("Written to query_analysis.txt")