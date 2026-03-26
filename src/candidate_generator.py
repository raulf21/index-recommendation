"""
Generates candidate indexes from TPC-H workload by analyzing column usage patterns in queries.

Takes flatten list of column references and produces a set of index candidates, 
by countin column frequency across queries and pairing frequently used columns together (especially those that appear in the same queries).

Composite index columns ordering follows standard heauristics: columns with equality predicates first, followed by range predicates.
This ensures composite index is usable for widest range of queries. 

Input:
- Workload list from workload_parser, where each item is a dict with keys: table, column, clause (WHERE, JOIN, etc), predicate_type (equality, range)
Output:
- List of candidate indexes, where each candidate is a dict with keys: table, columns (list), type (single or composite)
"""



import collections
from workload_parser import parse_workload

QUERIES_DIR = 'queries/'

def count_column_frequency(workload: list) -> dict:
    """
    Count how many queries each column appear in across the workload.
    Uses table.column as the key to distinguish columns with the same name in different tables.

    Args:
        - workload: List of dicts, each dict has keys: table, column, clause, predicate_type
    Returns:
        - a dict:("table.column": count) representing how many queries reference each column
    """
    frequencies = collections.defaultdict(int)
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        frequencies[key] += 1
    return frequencies

def get_column_predicate_types(workload: list) -> dict:
    """
    Determine the dominant predicate type (equality or range) for each column based on the workload.
    
    A column might appear as both equality and range predicate in different queries, 
    but we want to identify which type is more common for that column to inform index design. 
    This is also used for ordering columns in composite indexes (equality predicates should come before range predicates).

    Args:
        - workload: List of dicts, each dict has keys: table, column, clause, predicate_type
    Returns:
        - a dict:("table.column": "equality" or "range") representing the dominant predicate type for each column

    """
    # track equality vs range count for each column
    counts = collections.defaultdict(lambda:{'equality':0, 'range':0})
    for item in workload:
        # only consider predicates in WHERE clause for this analysis, since those are the ones that benefit most from indexing
        if item['clause'] in 'WHERE' and 'predicate_type' in item:
            key = f"{item['table']}.{item['column']}"
            counts[key][item['predicate_type']] += 1
    
    # pick the dominant type for each column
    result = {}
    for key, type_counts in counts.items():
        if type_counts['range'] > type_counts['equality']:
            result[key] = 'range'
        else:
            result[key] = 'equality' # if tie, default to equality since that is more common and more beneficial for indexing
    return result

def generate_candidates(workload: list, min_frequency: int=2) -> list:
    """
    Generate candidate columns for indexing based on frequency.
    Steps: 
        1. Count column frequency across queries and filter out columns that don't meet the minimum frequency threshold.
        2. Build single column candidates from the remaining frequent columns.
        3. Group frequent columns by table and identify pairs of columns that appear together in queries to build composite index candidates.

    For composite indexes, we also determine the ordering of columns based on their predicate types (equality predicates should come before range predicates) to ensure the index is usable for a wider range of queries.
    Args:
        - workload: List of dicts, each dict has keys: table, column, clause, predicate_type
        - min_frequency: Minimum number of queries a column must appear in to be considered for indexing
    Returns:
        - List of candidate indexes, where each candidate is a dict with keys: table, columns (list), type (single or composite)
    """
    frequencies = count_column_frequency(workload)
    predicate_types = get_column_predicate_types(workload)
    candidates = []

    # Keep only columns that appear in at least min_frequency queries to focus on the most impactful candidates
    # Low frequency columns are less likely to benefit from indexing and can be filtered out to reduce the candidate set size
    # justify the overhead of index maintenance and storage with the potential query performance benefits
    frequent_columns = {k: v for k,v in frequencies.items() if v >= min_frequency}
    for key in frequent_columns:
        table, column = key.split('.',1)
        candidates.append({
            'table': table,
            'columns': [column],
            'type': 'single'
        })
    # build multi-column candidates (pairs of columns that appear together in queries)
    # group frequent columns by table first
    # then pair up columns from the same table that appear together in queries
    table_to_columns = collections.defaultdict(set)
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        if key in frequent_columns:
            table_to_columns[item['table']].add(item['column'])
    # build composite candidates with ordering (equality predicates first, then range predicates)
    for table, columns in table_to_columns.items():
        columns = list(columns)
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col_a = columns[i]
                col_b = columns[j]
                key_a = f"{table}.{col_a}"
                key_b = f"{table}.{col_b}"
                type_a = predicate_types.get(key_a, 'equality')
                type_b = predicate_types.get(key_b, 'equality')

                # equality first, then range
                if type_a == 'equality' and type_b == 'range':
                    ordered = [col_a, col_b]
                elif type_a == 'range' and type_b == 'equality':
                    ordered = [col_b, col_a]
                else:
                    # both same type, order doesn't matter'
                    # both ordering since either could be optimal depending
                    # on which column has higher selectivity or is more commonly used in queries, so we want to include both orderings as candidates
                    candidates.append({
                        'table': table,
                        'columns': [col_b, col_a],
                        'type': 'composite'
                    })
                    ordered = [col_a, col_b]
                candidates.append({
                    'table': table,
                    'columns': ordered,
                    'type': 'composite'
                })
    return candidates
if __name__ == '__main__':
    workload = parse_workload(QUERIES_DIR)
    candidates = generate_candidates(workload)
    print(f"Total candidates: {len(candidates)}")
    for c in candidates:
        print(c)