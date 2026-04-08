"""
training_dataset.py
-------------------
Assemble ML-ready tables from `feature_extractor` output, optionally merge labels
produced by `hypopg_labeler`, and export train / validation / test splits.

Join key (stable across the pipeline):
    example_id = f"{query_name}|{candidate_table}|{candidate_cols}"
where `candidate_cols` is the comma-separated column list (same string as in feature rows).

Expected labels file (CSV): must include `example_id`, or the triple
(`query_name`, `candidate_table`, `candidate_cols`). Any additional numeric columns
are kept as targets (e.g. `label`, `label_cost_ratio`).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from candidate_generator import generate_candidates
from feature_extractor import build_feature_rows, get_connection
from workload_parser import parse_workload

# Columns not used as model inputs (identifiers + string metadata).
ID_METADATA_COLUMNS: Tuple[str, ...] = (
    "example_id",
    "query_name",
    "candidate_table",
    "candidate_cols",
    "candidate_type",
)


def example_id(query_name: str, candidate_table: str, candidate_cols: str) -> str:
    """Stable row id for merging features with hypopg_labeler output."""
    return f"{query_name}|{candidate_table}|{candidate_cols}"


def add_example_ids(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["example_id"] = example_id(
            str(r["query_name"]),
            str(r["candidate_table"]),
            str(r["candidate_cols"]),
        )
        out.append(r)
    return out


def build_feature_dataframe(
    conn,
    candidates: List[dict],
    workload: List[dict],
    queries_dir: str,
    schema: str = "public",
) -> pd.DataFrame:
    rows = build_feature_rows(
        conn,
        candidates,
        workload,
        queries_dir=queries_dir,
        schema=schema,
    )
    rows = add_example_ids(rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_labels_csv(path: str) -> pd.DataFrame:
    """
    Load labels from CSV. Requires `example_id` or
    (`query_name`, `candidate_table`, `candidate_cols`).
    """
    df = pd.read_csv(path)
    if "example_id" not in df.columns:
        missing = {
            c
            for c in ("query_name", "candidate_table", "candidate_cols")
            if c not in df.columns
        }
        if missing:
            raise ValueError(
                f"Labels CSV must have 'example_id' or all of query_name, "
                f"candidate_table, candidate_cols. Missing: {sorted(missing)}"
            )
        df = df.copy()
        df["example_id"] = [
            example_id(str(q), str(t), str(c))
            for q, t, c in zip(
                df["query_name"], df["candidate_table"], df["candidate_cols"]
            )
        ]
    return df


def merge_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Left-join labels onto features on `example_id`.
    If label_columns is None, use every column in labels except merge keys.
    """
    if features.empty:
        return features.copy()

    key = "example_id"
    label_keys = {key}
    if all(c in labels.columns for c in ("query_name", "candidate_table", "candidate_cols")):
        label_keys.update({"query_name", "candidate_table", "candidate_cols"})

    if label_columns is None:
        label_columns = [
            c
            for c in labels.columns
            if c not in label_keys
        ]
    else:
        for c in label_columns:
            if c not in labels.columns:
                raise ValueError(f"Label column {c!r} not in labels dataframe")

    merge_df = labels[[key] + list(label_columns)].drop_duplicates(subset=[key])
    return features.merge(merge_df, on=key, how="left")


def infer_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """All numeric columns except ids and obvious label prefixes."""
    skip = set(ID_METADATA_COLUMNS)
    cols: List[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if c.startswith("label_") or c == "label":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def train_val_test_split_dataframe(
    df: pd.DataFrame,
    label_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split rows into train / val / test. Rows with NaN in `label_column` are dropped
    before splitting (callers can merge with how='inner' earlier if preferred).
    """
    work = df.dropna(subset=[label_column]).copy()
    if work.empty:
        raise ValueError("No rows with non-null labels; cannot split.")

    strat = None
    if stratify:
        y = work[label_column]
        if y.nunique() < 2:
            strat = None
        else:
            vc = y.value_counts()
            if vc.min() < 2:
                strat = None
            else:
                strat = y

    idx = np.arange(len(work))
    if strat is not None:
        i_trainval, i_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    else:
        i_trainval, i_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
        )

    trainval = work.iloc[i_trainval]
    test = work.iloc[i_test]

    val_ratio = val_size / max(1e-9, (1.0 - test_size))
    if len(trainval) < 2:
        train, val = trainval, trainval.iloc[0:0]
    else:
        y_tv = trainval[label_column]
        strat_tv = None
        if stratify and y_tv.nunique() >= 2 and y_tv.value_counts().min() >= 2:
            strat_tv = y_tv
        i_train, i_val = train_test_split(
            np.arange(len(trainval)),
            test_size=val_ratio,
            random_state=random_state,
            stratify=strat_tv,
        )
        train = trainval.iloc[i_train]
        val = trainval.iloc[i_val]

    return train, val, test


def feature_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_columns is None:
        feature_columns = infer_numeric_feature_columns(df)
    X = df.loc[:, list(feature_columns)].astype(np.float64).fillna(0.0)
    y = df[label_column]
    return X, y


def write_labels_template(features: pd.DataFrame, path: str) -> None:
    """Write a CSV with example_id (+ merge keys) and an empty label column for labeling."""
    cols = ["example_id", "query_name", "candidate_table", "candidate_cols", "label"]
    have = [c for c in cols if c in features.columns]
    tpl = features[have].drop_duplicates(subset=["example_id"]).copy()
    if "label" not in tpl.columns:
        tpl["label"] = np.nan
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tpl.to_csv(path, index=False)


def export_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str,
    fmt: str = "csv",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "csv":
        train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    elif fmt == "parquet":
        try:
            train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
            val.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
            test.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
        except ImportError as e:
            raise ImportError(
                "Parquet export requires pyarrow (pip install pyarrow)."
            ) from e
    else:
        raise ValueError("fmt must be 'csv' or 'parquet'")


def build_training_splits(
    conn,
    workload: List[dict],
    candidates: List[dict],
    queries_dir: str,
    labels_path: Optional[str] = None,
    label_column: str = "label",
    label_columns: Optional[Sequence[str]] = None,
    placeholder_label: Optional[float] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = False,
    schema: str = "public",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (features_all, train, val, test).
    If labels_path is None and placeholder_label is set, fills label_column for smoke tests.
    """
    features = build_feature_dataframe(
        conn, candidates, workload, queries_dir=queries_dir, schema=schema
    )
    if features.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    if labels_path:
        labels = load_labels_csv(labels_path)
        features = merge_features_and_labels(features, labels, label_columns=label_columns)
        if label_column not in features.columns:
            raise ValueError(
                f"After merge, expected label column {label_column!r} is missing"
            )
    elif placeholder_label is not None:
        features = features.copy()
        features[label_column] = float(placeholder_label)
    else:
        # features only; caller may label later
        pass

    if label_column not in features.columns or features[label_column].isna().all():
        return features, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train, val, test = train_val_test_split_dataframe(
        features,
        label_column=label_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
    return features, train, val, test


def _default_queries_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "queries")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training dataset from feature_extractor (+ optional labels CSV)."
    )
    parser.add_argument(
        "--repo-root",
        default=os.path.dirname(_SRC_DIR),
        help="Repository root (contains queries/).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for train/val/test exports. Default: <repo-root>/data/training",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="CSV path from hypopg_labeler (example_id or query/table/cols keys).",
    )
    parser.add_argument(
        "--label-template",
        default=None,
        help="Write a labeling template CSV to this path and exit (no DB split export).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Candidate generator min_frequency.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Target column name for stratified split and matrices.",
    )
    parser.add_argument(
        "--placeholder-label",
        type=float,
        default=None,
        help="If set and --labels omitted, fill this constant label (pipeline smoke test).",
    )
    parser.add_argument(
        "--fmt",
        choices=("csv", "parquet"),
        default="csv",
        help="Output format for splits.",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify splits on label_column when possible.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    queries_dir = _default_queries_dir(repo_root)
    out_dir = args.out_dir or os.path.join(repo_root, "data", "training")

    workload = parse_workload(queries_dir)
    candidates = generate_candidates(workload, min_frequency=args.min_frequency)

    conn = get_connection()
    try:
        if args.label_template:
            features = build_feature_dataframe(
                conn, candidates, workload, queries_dir=queries_dir
            )
            write_labels_template(features, args.label_template)
            print(
                f"Wrote label template ({len(features)} rows) to {args.label_template}"
            )
            return

        features, train, val, test = build_training_splits(
            conn,
            workload,
            candidates,
            queries_dir=queries_dir,
            labels_path=args.labels,
            label_column=args.label_column,
            placeholder_label=args.placeholder_label,
            stratify=args.stratify,
        )
    finally:
        conn.close()

    if features.empty:
        print("No feature rows; check DB, workload, and candidates.")
        return

    if train.empty and args.labels is None and args.placeholder_label is None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "features_unlabeled.csv")
        features.to_csv(path, index=False)
        print(
            f"Wrote {len(features)} unlabeled rows to {path}. "
            "Pass --labels <csv> or --placeholder-label for splits."
        )
        return

    if train.empty:
        print("No labeled rows to split; check labels file and label column.")
        return

    export_splits(train, val, test, out_dir, fmt=args.fmt)
    print(
        f"Wrote train/val/test to {out_dir} ({args.fmt}): "
        f"{len(train)} / {len(val)} / {len(test)} rows."
    )


if __name__ == "__main__":
    main()
