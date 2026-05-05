"""
candidate_generator.py
----------------------
Generates candidate B-tree indexes from parsed TPC-H workload signals.

Input:
    Workload list from workload_parser, with keys such as:
        table, column, clause, predicate_type, query, query_cost,
        in_where, in_group_by, in_order_by

Output:
    List of candidate index dicts:
        {
            "table": "lineitem",
            "columns": ["l_orderkey", "l_shipdate"],
            "type": "single" | "composite",
            "range_scan_candidate": bool,
            "source_queries": ["q3", "q20"],
            "access_pattern": "equality" | "range" | "prefix_like" | ...,
            "cost_impact": float
        }

Pipeline:
    workload_parser -> candidate_generator -> feature_extractor -> hypopg_labeler -> ml_model

Design notes:
    - Composite candidates are generated only from same-query, same-table
      co-occurrences.
    - Equality columns are placed before range/prefix-like columns for B-tree
      prefix traversal.
    - Negative predicates and leading-wildcard LIKE predicates are skipped as
      normal B-tree key candidates.
    - This module uses range_scan_candidate instead of clustered_candidate.
      PostgreSQL CLUSTER is a physical table reordering operation, while HypoPG
      labels normal hypothetical indexes only.
"""

from __future__ import annotations

import collections
from itertools import combinations
from typing import DefaultDict, Dict, Iterable, List, MutableMapping, Sequence, Set, Tuple

from workload_parser import parse_workload

QUERIES_DIR = "queries/"

# Predicate types emitted by the fixed workload_parser.
SARGABLE_PREDICATES = {"equality", "range", "prefix_like"}
SKIP_AS_BTREE_KEY = {"negative", "pattern_like", "unknown", "non_sargable"}
RANGE_LIKE_PREDICATES = {"range", "prefix_like"}


def _column_key(table: str, column: str) -> str:
    return f"{table}.{column}"


def _candidate_key(candidate: MutableMapping) -> Tuple[str, Tuple[str, ...]]:
    return candidate["table"], tuple(candidate["columns"])


def _is_sort_group_signal(item: MutableMapping) -> bool:
    """True when a column appears only as a GROUP BY / ORDER BY signal."""
    return bool(item.get("in_group_by") or item.get("in_order_by"))


def _is_useful_single_signal(item: MutableMapping) -> bool:
    """
    Decide whether a workload row can create a single-column index candidate.

    We keep equality/range/prefix_like predicates and pure GROUP BY / ORDER BY
    signals. We skip negative predicates and leading-wildcard LIKE because they
    are weak normal B-tree key signals.
    """
    predicate_type = item.get("predicate_type", "n/a")

    if item.get("in_where") and predicate_type in SARGABLE_PREDICATES:
        return True

    if predicate_type in SKIP_AS_BTREE_KEY:
        return False

    # Columns that are only used for grouping/sorting can still be useful for
    # avoiding Sort/GroupAggregate work, so keep them as lower-priority singles.
    if _is_sort_group_signal(item) and not item.get("in_where"):
        return True

    return False


def _is_useful_composite_signal(item: MutableMapping) -> bool:
    """
    Decide whether a workload row can participate in a composite index.

    For now, composites are based on WHERE/JOIN-like access only. GROUP BY and
    ORDER BY can become features later, but they should not create arbitrary
    composite key pairs here.
    """
    predicate_type = item.get("predicate_type", "n/a")
    return bool(item.get("in_where") and predicate_type in SARGABLE_PREDICATES)


def _dominant_access_pattern(type_costs: MutableMapping[str, float]) -> str:
    """Return the cost-weighted dominant access pattern for one column."""
    if not type_costs:
        return "sort_group"

    # Prefer the highest cost. Tie-breaker favors equality because it is the
    # best leading-column signal for B-tree composites.
    priority = {"equality": 0, "range": 1, "prefix_like": 2, "sort_group": 3}
    return sorted(
        type_costs.items(),
        key=lambda kv: (-kv[1], priority.get(kv[0], 99), kv[0]),
    )[0][0]


def count_column_frequency(workload: List[dict]) -> Dict[str, float]:
    """
    Sum query_cost per useful candidate column across the workload.

    This is cost-weighted rather than raw-count-weighted: a column in one
    expensive query can outrank a column that appears in many cheap queries.
    workload_parser deduplicates (query, table, column), so each query's cost
    contributes at most once per column.
    """
    frequencies: DefaultDict[str, float] = collections.defaultdict(float)

    for item in workload:
        if not _is_useful_single_signal(item):
            continue

        key = _column_key(item["table"], item["column"])
        frequencies[key] += float(item.get("query_cost", 1.0))

    return dict(frequencies)


def get_column_predicate_types(workload: List[dict]) -> Dict[str, str]:
    """
    Determine the cost-weighted dominant access pattern per column.

    Returned values may include:
        equality, range, prefix_like, sort_group

    Negative and pattern-like predicates are intentionally excluded from normal
    B-tree candidate generation.
    """
    type_costs: DefaultDict[str, DefaultDict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )

    for item in workload:
        if not _is_useful_single_signal(item):
            continue

        key = _column_key(item["table"], item["column"])
        cost = float(item.get("query_cost", 1.0))
        predicate_type = item.get("predicate_type", "n/a")

        if item.get("in_where") and predicate_type in SARGABLE_PREDICATES:
            type_costs[key][predicate_type] += cost
        elif _is_sort_group_signal(item) and not item.get("in_where"):
            type_costs[key]["sort_group"] += cost

    return {key: _dominant_access_pattern(costs) for key, costs in type_costs.items()}


def _source_query_info(workload: List[dict]) -> Dict[str, dict]:
    """
    Build per-column provenance for candidate metadata.

    Output maps "table.column" to:
        {
            "queries": set[str],
            "cost_impact": float,
            "type_costs": {access_pattern: cost}
        }
    """
    info: Dict[str, dict] = {}

    for item in workload:
        if not _is_useful_single_signal(item):
            continue

        key = _column_key(item["table"], item["column"])
        query = item.get("query", "unknown")
        cost = float(item.get("query_cost", 1.0))
        predicate_type = item.get("predicate_type", "n/a")

        if key not in info:
            info[key] = {
                "queries": set(),
                "cost_impact": 0.0,
                "type_costs": collections.defaultdict(float),
            }

        # workload_parser emits one row per (query, table, column), so this is
        # not expected to double-count query cost for a column.
        info[key]["queries"].add(query)
        info[key]["cost_impact"] += cost

        if item.get("in_where") and predicate_type in SARGABLE_PREDICATES:
            info[key]["type_costs"][predicate_type] += cost
        elif _is_sort_group_signal(item) and not item.get("in_where"):
            info[key]["type_costs"]["sort_group"] += cost

    return info


def _build_cooccurrence_pairs(
    workload: List[dict], frequent_columns: Set[str]
) -> Dict[Tuple[str, str, str], dict]:
    """
    Return same-query, same-table column pairs that can form composites.

    Output key:
        (table, col_a, col_b), with col_a/col_b sorted alphabetically

    Output value:
        {
            "queries": set[str],
            "cost_impact": float,
            "column_patterns": {col: access_pattern}
        }

    A composite index on (A, B) can only help a query that filters on both A and
    B simultaneously. This prevents global same-table pairing.
    """
    per_query: DefaultDict[str, DefaultDict[str, Dict[str, dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )

    for item in workload:
        if not _is_useful_composite_signal(item):
            continue

        table = item["table"]
        column = item["column"]
        key = _column_key(table, column)
        if key not in frequent_columns:
            continue

        query = item.get("query", "unknown")
        per_query[query][table][column] = {
            "pattern": item.get("predicate_type", "unknown"),
            "cost": float(item.get("query_cost", 1.0)),
        }

    pairs: Dict[Tuple[str, str, str], dict] = {}

    for query, tables in per_query.items():
        for table, col_map in tables.items():
            cols = sorted(col_map)
            for col_a, col_b in combinations(cols, 2):
                pair_key = (table, col_a, col_b)
                if pair_key not in pairs:
                    pairs[pair_key] = {
                        "queries": set(),
                        "cost_impact": 0.0,
                        "column_patterns": collections.defaultdict(lambda: collections.defaultdict(float)),
                    }

                # Count the query cost once for the pair. Each item in the pair
                # came from the same query, so either column's cost is enough.
                query_cost = max(col_map[col_a]["cost"], col_map[col_b]["cost"])
                pairs[pair_key]["queries"].add(query)
                pairs[pair_key]["cost_impact"] += query_cost
                pairs[pair_key]["column_patterns"][col_a][col_map[col_a]["pattern"]] += query_cost
                pairs[pair_key]["column_patterns"][col_b][col_map[col_b]["pattern"]] += query_cost

    return pairs


def _pattern_rank(pattern: str) -> int:
    """
    Lower rank means better leading position in a composite B-tree index.

    Equality should lead. Range and prefix_like are range-like and normally come
    after equality. sort_group is not used in composites here.
    """
    return {
        "equality": 0,
        "range": 1,
        "prefix_like": 1,
        "sort_group": 2,
    }.get(pattern, 9)


def _ordered_composite_permutations(
    col_a: str,
    pattern_a: str,
    col_b: str,
    pattern_b: str,
) -> List[List[str]]:
    """Return deterministic column orders to evaluate for a two-column index."""
    rank_a = _pattern_rank(pattern_a)
    rank_b = _pattern_rank(pattern_b)

    if rank_a < rank_b:
        return [[col_a, col_b]]
    if rank_b < rank_a:
        return [[col_b, col_a]]

    # Equal/equal and range/range order can matter, and selectivity is not known
    # in candidate_generator. Emit both deterministic permutations.
    first, second = sorted([col_a, col_b])
    return [[first, second], [second, first]]


def _access_pattern_for_columns(patterns: Sequence[str]) -> str:
    return "_".join(patterns)


def _add_candidate(candidates: List[dict], seen: Set[Tuple[str, Tuple[str, ...]]], candidate: dict) -> None:
    key = _candidate_key(candidate)
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def generate_candidates(workload: List[dict], min_cost_impact: float = 50000.0) -> List[dict]:
    """
    Generate single-column and two-column composite index candidates.

    Single-column candidates:
        Generated for useful columns whose total cost impact clears
        min_cost_impact.

    Composite candidates:
        Generated only for pairs that co-occur in the same query and same table
        as sargable WHERE/JOIN-like predicates, and whose pair-level cost
        impact also clears min_cost_impact.

    min_cost_impact is scale-factor dependent; tune per dataset size.
    """
    frequencies = count_column_frequency(workload)
    predicate_types = get_column_predicate_types(workload)
    source_info = _source_query_info(workload)

    frequent_columns: Set[str] = {
        key for key, cost in frequencies.items() if cost >= min_cost_impact
    }

    candidates: List[dict] = []
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()

    # Single-column candidates.
    for key in sorted(frequent_columns):
        table, column = key.split(".", 1)
        access_pattern = predicate_types.get(key, "sort_group")
        info = source_info.get(key, {})

        _add_candidate(
            candidates,
            seen,
            {
                "table": table,
                "columns": [column],
                "type": "single",
                "range_scan_candidate": access_pattern in RANGE_LIKE_PREDICATES,
                "source_queries": sorted(info.get("queries", [])),
                "access_pattern": access_pattern,
                "cost_impact": float(info.get("cost_impact", frequencies.get(key, 0.0))),
            },
        )

    # Composite candidates.
    cooccur_pairs = _build_cooccurrence_pairs(workload, frequent_columns)

    for (table, col_a, col_b), pair_info in sorted(cooccur_pairs.items()):
        # Apply the same pruning threshold to the composite candidate's own
        # pair-level workload impact. A pair can be made of globally frequent
        # columns but only co-occur in a low-cost query; that should not become
        # a composite candidate under this pruning rule.
        if float(pair_info.get("cost_impact", 0.0)) < min_cost_impact:
            continue

        pattern_a = _dominant_access_pattern(pair_info["column_patterns"][col_a])
        pattern_b = _dominant_access_pattern(pair_info["column_patterns"][col_b])

        for ordered_cols in _ordered_composite_permutations(col_a, pattern_a, col_b, pattern_b):
            ordered_patterns = [pattern_a if c == col_a else pattern_b for c in ordered_cols]
            access_pattern = _access_pattern_for_columns(ordered_patterns)

            _add_candidate(
                candidates,
                seen,
                {
                    "table": table,
                    "columns": ordered_cols,
                    "type": "composite",
                    "range_scan_candidate": any(p in RANGE_LIKE_PREDICATES for p in ordered_patterns),
                    "source_queries": sorted(pair_info["queries"]),
                    "access_pattern": access_pattern,
                    "cost_impact": float(pair_info["cost_impact"]),
                },
            )

    return candidates


def _validate_transition(workload: List[dict], candidates: List[dict], min_cost_impact: float) -> None:
    """
    Sanity-check the workload_parser -> candidate_generator transition.

    Raises AssertionError on failures. This is intentionally lightweight and can
    run whenever candidate_generator.py is executed directly.
    """
    assert isinstance(workload, list), "workload must be a list"
    assert workload, "workload_parser returned no rows"
    assert isinstance(candidates, list), "candidates must be a list"
    assert candidates, "candidate_generator returned no candidates"

    required_workload_keys = {
        "table",
        "column",
        "query",
        "query_cost",
        "in_where",
        "in_group_by",
        "in_order_by",
        "predicate_type",
    }
    missing_workload = [
        (i, required_workload_keys - set(row))
        for i, row in enumerate(workload[:20])
        if required_workload_keys - set(row)
    ]
    assert not missing_workload, f"workload rows missing keys: {missing_workload[:3]}"

    required_candidate_keys = {
        "table",
        "columns",
        "type",
        "range_scan_candidate",
        "source_queries",
        "access_pattern",
        "cost_impact",
    }
    missing_candidates = [
        (i, required_candidate_keys - set(row))
        for i, row in enumerate(candidates)
        if required_candidate_keys - set(row)
    ]
    assert not missing_candidates, f"candidate rows missing keys: {missing_candidates[:3]}"

    assert all("clustered_candidate" not in c for c in candidates), (
        "clustered_candidate should not appear; use range_scan_candidate instead"
    )

    duplicate_keys = [
        key
        for key, count in collections.Counter(_candidate_key(c) for c in candidates).items()
        if count > 1
    ]
    assert not duplicate_keys, f"duplicate candidates found: {duplicate_keys[:5]}"

    bad_patterns = {"negative", "pattern_like", "unknown", "non_sargable"}
    bad_candidates = [
        c
        for c in candidates
        if any(part in bad_patterns for part in c.get("access_pattern", "").split("_"))
    ]
    assert not bad_candidates, f"bad predicate candidates found: {bad_candidates[:5]}"

    # Build actual same-query, same-table co-occurrence from workload for validation.
    per_query_table_cols: DefaultDict[str, DefaultDict[str, Set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    for item in workload:
        if _is_useful_composite_signal(item):
            per_query_table_cols[item["query"]][item["table"]].add(item["column"])

    for c in candidates:
        assert c["type"] in {"single", "composite"}, f"unknown candidate type: {c}"
        assert isinstance(c["columns"], list) and c["columns"], f"bad columns: {c}"
        assert isinstance(c["source_queries"], list), f"source_queries must be list: {c}"
        assert c["cost_impact"] >= min_cost_impact, (
            f"candidate below min_cost_impact={min_cost_impact}: {c}"
        )

        if c["type"] == "single":
            assert len(c["columns"]) == 1, f"single candidate has != 1 column: {c}"
        else:
            assert len(c["columns"]) == 2, f"composite candidate has != 2 columns: {c}"
            table = c["table"]
            col_a, col_b = c["columns"]
            found_cooccur = any(
                col_a in per_query_table_cols[q][table]
                and col_b in per_query_table_cols[q][table]
                for q in c["source_queries"]
            )
            assert found_cooccur, f"composite does not co-occur in its source queries: {c}"


def _print_candidate_summary(workload: List[dict], candidates: List[dict], min_cost_impact: float) -> None:
    frequencies = count_column_frequency(workload)

    print("Top 10 useful columns by total query-cost impact:")
    for key, cost in sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))[:10]:
        print(f"  {cost:14.2f}  {key}")

    total_candidates = len(candidates)
    single_count = sum(1 for c in candidates if c["type"] == "single")
    composite_count = total_candidates - single_count
    range_scan_count = sum(1 for c in candidates if c.get("range_scan_candidate"))

    print(f"\n{'=' * 56}")
    print(f"TOTAL CANDIDATES GENERATED: {total_candidates}")
    print(f"  Single-column indexes:      {single_count}")
    print(f"  Composite indexes:          {composite_count}")
    print(f"  Range-scan candidates:      {range_scan_count}")
    print(f"  min_cost_impact:            {min_cost_impact:.2f}")
    print(f"{'=' * 56}")

    print("\nSample candidates:")
    for c in candidates[:15]:
        cols = ", ".join(c["columns"])
        queries = ",".join(c["source_queries"][:5])
        if len(c["source_queries"]) > 5:
            queries += ",..."
        print(
            f"  {c['type']:9} {c['table']}({cols}) "
            f"pattern={c['access_pattern']:<18} "
            f"cost={c['cost_impact']:12.2f} "
            f"queries=[{queries}]"
        )


if __name__ == "__main__":
    MIN_COST_IMPACT = 50000.0

    workload = parse_workload(QUERIES_DIR)
    candidates = generate_candidates(workload, min_cost_impact=MIN_COST_IMPACT)

    _validate_transition(workload, candidates, min_cost_impact=MIN_COST_IMPACT)
    _print_candidate_summary(workload, candidates, min_cost_impact=MIN_COST_IMPACT)

    print("\nSanity check passed: workload_parser -> candidate_generator transition is valid.")
