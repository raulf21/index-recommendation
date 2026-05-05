"""
training_dataset.py
-------------------
Builds the supervised ML dataset for learned index-benefit prediction.

Pipeline position:
    workload_parser -> candidate_generator -> feature_extractor -> hypopg_labeler
    -> training_dataset -> ml_model

Input:
    - Feature rows from feature_extractor.py
    - Per-query HypoPG labels from data/labels.csv

Output:
    - data/training/train.csv
    - data/training/val.csv
    - data/training/test.csv
    - data/training/all.csv
    - data/training/all_debug_with_raw_labels.csv

Important design choices:
    - Requires exact per-query labels keyed by:
          query_name | candidate_table | candidate_cols
    - Does NOT support legacy candidate-level label broadcasting.
    - Uses only individual HypoPG optimizer-estimated labels.
    - Removes label_raw and label_source from train/val/test exports to avoid leakage.
    - Splits by query_name, not by random row, to reduce query-template leakage.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from workload_parser import parse_workload
from candidate_generator import generate_candidates
from feature_extractor import build_feature_rows
from db_utils import get_connection


QUERIES_DIR = "queries"
DEFAULT_LABELS_PATH = "data/labels.csv"
DEFAULT_OUTPUT_DIR = "data/training"
DEFAULT_MIN_COST_IMPACT = 50000.0
DEFAULT_SEED = 42

# Metadata columns are allowed in the exported CSVs, but should never be model
# features. ml_model.py should exclude these explicitly when selecting features.
ID_METADATA_COLUMNS: Tuple[str, ...] = (
    "example_id",
    "query_name",
    "candidate_table",
    "candidate_cols",
    "candidate_type",
)

# Columns that contain label-generation information and must not enter model
# features. We keep them only in all_debug_with_raw_labels.csv for auditing.
LEAKAGE_OR_DEBUG_COLUMNS: Tuple[str, ...] = (
    "label_raw",
    "label_source",
    "is_marginal",
)


def normalize_candidate_cols(value: object) -> str:
    """Canonicalize candidate_cols while preserving column order."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value]
    else:
        parts = [p.strip() for p in str(value).split(",")]
    return ",".join(p for p in parts if p)


def make_example_id(query_name: object, table: object, cols: object) -> str:
    """Build the shared join key used by features and labels."""
    return f"{str(query_name)}|{str(table)}|{normalize_candidate_cols(cols)}"


def signed_log1p(x: float) -> float:
    """Signed log transform for cost-delta labels with large dynamic range."""
    if pd.isna(x):
        return float("nan")
    x = float(x)
    return math.copysign(math.log1p(abs(x)), x)


def build_features(min_cost_impact: float = DEFAULT_MIN_COST_IMPACT) -> pd.DataFrame:
    """Run workload_parser -> candidate_generator -> feature_extractor."""
    workload = parse_workload(QUERIES_DIR)
    candidates = generate_candidates(workload, min_cost_impact=min_cost_impact)

    conn = get_connection()
    try:
        feature_rows = build_feature_rows(conn, candidates, workload)
    finally:
        conn.close()

    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise RuntimeError("feature_extractor produced no feature rows")

    required = {"query_name", "candidate_table", "candidate_cols"}
    missing = required - set(features.columns)
    if missing:
        raise RuntimeError(f"feature rows missing required columns: {sorted(missing)}")

    features["candidate_cols"] = features["candidate_cols"].map(normalize_candidate_cols)
    features["example_id"] = features.apply(
        lambda r: make_example_id(r["query_name"], r["candidate_table"], r["candidate_cols"]),
        axis=1,
    )

    return features


def load_labels(labels_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load exact per-query HypoPG labels and build example_id."""
    labels = pd.read_csv(labels_path)

    required = {"query_name", "candidate_table", "candidate_cols", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise RuntimeError(
            "labels.csv must contain exact per-query labels with columns "
            f"{sorted(required)}; missing {sorted(missing)}"
        )

    labels["candidate_cols"] = labels["candidate_cols"].map(normalize_candidate_cols)
    labels["example_id"] = labels.apply(
        lambda r: make_example_id(r["query_name"], r["candidate_table"], r["candidate_cols"]),
        axis=1,
    )

    labels["label"] = pd.to_numeric(labels["label"], errors="raise")

    if "label_source" in labels.columns:
        sources = set(labels["label_source"].dropna().astype(str).unique())
        if sources != {"individual"}:
            raise RuntimeError(
                "training_dataset expects only individual HypoPG labels. "
                f"Found label_source values: {sorted(sources)}"
            )
    else:
        labels["label_source"] = "individual"

    return labels[["example_id", "query_name", "candidate_table", "candidate_cols", "label", "label_source"]]


def validate_exact_alignment(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Fail fast if feature rows and label rows do not match exactly."""
    if features["example_id"].duplicated().any():
        dupes = features.loc[features["example_id"].duplicated(), "example_id"].head(10).tolist()
        raise RuntimeError(f"duplicate feature example_id values, examples: {dupes}")

    if labels["example_id"].duplicated().any():
        dupes = labels.loc[labels["example_id"].duplicated(), "example_id"].head(10).tolist()
        raise RuntimeError(f"duplicate label example_id values, examples: {dupes}")

    feature_keys = set(features["example_id"])
    label_keys = set(labels["example_id"])

    missing_labels = sorted(feature_keys - label_keys)
    extra_labels = sorted(label_keys - feature_keys)

    if missing_labels or extra_labels:
        msg = [
            "feature-label alignment failed",
            f"feature rows: {len(feature_keys)}",
            f"label rows:   {len(label_keys)}",
        ]
        if missing_labels:
            msg.append(f"missing labels for first examples: {missing_labels[:10]}")
        if extra_labels:
            msg.append(f"labels without features for first examples: {extra_labels[:10]}")
        raise RuntimeError("\n".join(msg))

    forbidden_feature_cols = {"clustered_candidate", "is_marginal"}
    present = forbidden_feature_cols & set(features.columns)
    if present:
        raise RuntimeError(f"feature rows contain stale/leaky columns: {sorted(present)}")


def build_training_dataset(
    labels_path: str | os.PathLike[str] = DEFAULT_LABELS_PATH,
    min_cost_impact: float = DEFAULT_MIN_COST_IMPACT,
) -> pd.DataFrame:
    """Build exact joined dataset and transform labels."""
    features = build_features(min_cost_impact=min_cost_impact)
    labels = load_labels(labels_path)

    validate_exact_alignment(features, labels)

    labels_for_join = labels[["example_id", "label", "label_source"]].rename(
        columns={"label": "label_raw"}
    )
    dataset = features.merge(labels_for_join, on="example_id", how="inner", validate="one_to_one")

    if dataset["label_raw"].isna().any():
        raise RuntimeError("merged dataset contains missing label_raw values")

    dataset["label"] = dataset["label_raw"].map(signed_log1p)

    if dataset["label"].isna().any():
        raise RuntimeError("label transform produced NaN values")

    return dataset


def split_queries(
    query_names: Sequence[str],
    seed: int = DEFAULT_SEED,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[set, set, set]:
    """Deterministically split query templates into train/val/test groups."""
    queries = sorted(set(str(q) for q in query_names))
    if len(queries) < 3:
        raise RuntimeError("Need at least 3 distinct query_name values for train/val/test split")

    rng = random.Random(seed)
    rng.shuffle(queries)

    n = len(queries)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))

    # Ensure test split is non-empty.
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_q = set(queries[:n_train])
    val_q = set(queries[n_train:n_train + n_val])
    test_q = set(queries[n_train + n_val:])

    if not train_q or not val_q or not test_q:
        raise RuntimeError(
            f"invalid query split sizes: train={len(train_q)}, val={len(val_q)}, test={len(test_q)}"
        )

    return train_q, val_q, test_q


def remove_leakage_columns_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop debug/label-generation columns from training exports.

    This prevents the current or future ml_model.py from accidentally using
    raw labels or label-source metadata as numeric features.
    """
    drop_cols = [c for c in LEAKAGE_OR_DEBUG_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)


def save_splits(
    dataset: pd.DataFrame,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Path]:
    """Save all/debug/train/val/test CSV files using query-template split."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_q, val_q, test_q = split_queries(dataset["query_name"].tolist(), seed=seed)

    train = dataset[dataset["query_name"].astype(str).isin(train_q)].copy()
    val = dataset[dataset["query_name"].astype(str).isin(val_q)].copy()
    test = dataset[dataset["query_name"].astype(str).isin(test_q)].copy()

    validate_split_no_query_leakage(train, val, test)

    # Debug file preserves raw labels for human inspection only.
    debug_path = out / "all_debug_with_raw_labels.csv"
    dataset.to_csv(debug_path, index=False)

    # Training-facing files drop raw labels and label source metadata.
    all_clean = remove_leakage_columns_for_training(dataset)
    train_clean = remove_leakage_columns_for_training(train)
    val_clean = remove_leakage_columns_for_training(val)
    test_clean = remove_leakage_columns_for_training(test)

    paths = {
        "all": out / "all.csv",
        "train": out / "train.csv",
        "val": out / "val.csv",
        "test": out / "test.csv",
        "debug": debug_path,
    }

    all_clean.to_csv(paths["all"], index=False)
    train_clean.to_csv(paths["train"], index=False)
    val_clean.to_csv(paths["val"], index=False)
    test_clean.to_csv(paths["test"], index=False)

    return paths


def validate_split_no_query_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Ensure no query template appears in more than one split."""
    train_q = set(train["query_name"].astype(str))
    val_q = set(val["query_name"].astype(str))
    test_q = set(test["query_name"].astype(str))

    overlaps = {
        "train_val": train_q & val_q,
        "train_test": train_q & test_q,
        "val_test": val_q & test_q,
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f"query-template leakage across splits: {bad}")


def validate_training_exports(paths: Dict[str, Path]) -> None:
    """Sanity-check exported CSVs for leakage-prone columns."""
    for split_name in ("all", "train", "val", "test"):
        df = pd.read_csv(paths[split_name])
        forbidden = {"label_raw", "is_marginal", "clustered_candidate"} & set(df.columns)
        if forbidden:
            raise RuntimeError(f"{paths[split_name]} contains forbidden columns: {sorted(forbidden)}")
        if "label" not in df.columns:
            raise RuntimeError(f"{paths[split_name]} missing transformed label column")
        if df["label"].isna().any():
            raise RuntimeError(f"{paths[split_name]} contains NaN labels")

    debug = pd.read_csv(paths["debug"])
    if "label_raw" not in debug.columns:
        raise RuntimeError("debug dataset should contain label_raw for auditing")


def print_summary(dataset: pd.DataFrame, paths: Dict[str, Path]) -> None:
    """Human-readable run summary."""
    train = pd.read_csv(paths["train"])
    val = pd.read_csv(paths["val"])
    test = pd.read_csv(paths["test"])

    print("\n" + "=" * 60)
    print("Training dataset built successfully")
    print(f"Total joined rows:       {len(dataset)}")
    print(f"Distinct queries:        {dataset['query_name'].nunique()}")
    print(f"Distinct candidates:     {dataset[['candidate_table', 'candidate_cols']].drop_duplicates().shape[0]}")
    print(f"Train rows / queries:    {len(train)} / {train['query_name'].nunique()}")
    print(f"Val rows / queries:      {len(val)} / {val['query_name'].nunique()}")
    print(f"Test rows / queries:     {len(test)} / {test['query_name'].nunique()}")
    print("\nWrote:")
    for name, path in paths.items():
        print(f"  {name:>5}: {path}")
    print("\nSanity check passed: feature rows + HypoPG labels are exactly aligned,")
    print("and train/val/test are split by query_name to reduce template leakage.")
    print("=" * 60)

    preview_cols = [
        "example_id",
        "query_name",
        "candidate_table",
        "candidate_cols",
        "candidate_type",
        "label",
    ]
    debug_cols = [c for c in ["label_raw", "label_source"] if c in dataset.columns]
    sample_cols = [c for c in preview_cols + debug_cols if c in dataset.columns]
    print("\nSample rows:")
    print(dataset[sample_cols].head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML training dataset from features and HypoPG labels.")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Path to labels.csv from hypopg_labeler.py")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for train/val/test CSVs")
    parser.add_argument("--min-cost-impact", type=float, default=DEFAULT_MIN_COST_IMPACT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_training_dataset(labels_path=args.labels, min_cost_impact=args.min_cost_impact)
    paths = save_splits(dataset, output_dir=args.output_dir, seed=args.seed)
    validate_training_exports(paths)
    print_summary(dataset, paths)


if __name__ == "__main__":
    main()