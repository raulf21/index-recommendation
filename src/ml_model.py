"""
ml_model.py
-----------
Train and run an XGBoost regression model for learning-based index ranking.

Input artifacts:
    data/training/train.csv
    data/training/val.csv
    data/training/test.csv

Those files are produced by training_dataset.py and contain one row per
(query, candidate index) pair.  The target column `label` is the signed-log1p
transform of the raw HypoPG optimizer-estimated cost delta.

Important:
    - The model predicts optimizer-estimated benefit, not measured runtime.
    - Metadata/debug/label-generation columns are explicitly excluded from the
      feature matrix to avoid leakage.
    - Recommendation rebuilds fresh candidates/features from the current
      workload rather than reusing training rows.

Pipeline:
    workload_parser -> candidate_generator -> feature_extractor
    -> hypopg_labeler -> training_dataset -> ml_model

Usage:
    python src/ml_model.py --train --no-grid-search
    python src/ml_model.py --train
    python src/ml_model.py --recommend --top-k 10
    python src/ml_model.py --recommend --top-k 10 --budget 5000
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from candidate_generator import generate_candidates
from db_utils import get_connection
from feature_extractor import build_feature_rows
from workload_parser import parse_workload

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)

TRAINING_DIR = os.path.join(_REPO_ROOT, "data", "training")
MODEL_PATH = os.path.join(_REPO_ROOT, "data", "model.json")
DEFAULT_MIN_COST_IMPACT = 50000.0

# These columns identify an example or describe how the label was produced.
# They must not be used as model features.
METADATA_COLUMNS = {
    "example_id",
    "query_name",
    "candidate_table",
    "candidate_cols",
    "candidate_type",
    "label_source",
}

# These columns are direct labels or debug/label-generation information.
# They must never enter the model feature matrix.
LEAKAGE_COLUMNS = {
    "label",
    "label_raw",
    "is_marginal",
    "predicted_log_benefit",
    "predicted_raw_benefit",
}

# Stale field from the old candidate-generator design.  If this appears, the
# patched pipeline is out of sync.
FORBIDDEN_COLUMNS = {"clustered_candidate"}

PARAM_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "n_estimators": [100, 300],
}

FIXED_PARAMS = {
    "early_stopping_rounds": 15,
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def feature_cols_path_for_model(model_path: str) -> str:
    """Save/load feature-column order next to the model file."""
    base, _ = os.path.splitext(model_path)
    return f"{base}_feature_cols.txt"


def inverse_signed_log1p(x: float) -> float:
    """Invert the signed log1p transform used by training_dataset.py."""
    x = float(x)
    if x >= 0:
        return float(np.expm1(x))
    return float(-np.expm1(abs(x)))


def normalize_candidate_cols(value: object) -> str:
    """Canonicalize candidate column strings while preserving column order."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value]
    else:
        parts = [p.strip() for p in str(value).split(",")]
    return ",".join(p for p in parts if p)


def make_example_id(query_name: object, table: object, cols: object) -> str:
    return f"{query_name}|{table}|{normalize_candidate_cols(cols)}"


def add_example_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the shared query-candidate key exists."""
    df = df.copy()
    required = {"query_name", "candidate_table", "candidate_cols"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"cannot build example_id; missing columns: {sorted(missing)}")
    df["candidate_cols"] = df["candidate_cols"].map(normalize_candidate_cols)
    df["example_id"] = df.apply(
        lambda r: make_example_id(r["query_name"], r["candidate_table"], r["candidate_cols"]),
        axis=1,
    )
    return df


def load_splits(training_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train / val / test CSVs produced by training_dataset.py."""
    train = pd.read_csv(os.path.join(training_dir, "train.csv"))
    val = pd.read_csv(os.path.join(training_dir, "val.csv"))
    test = pd.read_csv(os.path.join(training_dir, "test.csv"))
    return add_example_ids(train), add_example_ids(val), add_example_ids(test)


def validate_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, label_col: str) -> None:
    """Validate train/val/test are aligned with the patched dataset contract."""
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if df.empty:
            raise RuntimeError(f"{name} split is empty")
        if label_col not in df.columns:
            raise RuntimeError(f"{name} split missing label column {label_col!r}")
        forbidden = FORBIDDEN_COLUMNS & set(df.columns)
        if forbidden:
            raise RuntimeError(f"{name} split contains stale forbidden columns: {sorted(forbidden)}")
        leaks = ({"label_raw", "is_marginal"} & set(df.columns))
        if leaks:
            raise RuntimeError(f"{name} split contains leakage/debug columns: {sorted(leaks)}")
        if df["example_id"].duplicated().any():
            dups = df.loc[df["example_id"].duplicated(), "example_id"].head(5).tolist()
            raise RuntimeError(f"{name} split has duplicate example_id values, e.g. {dups}")
        if df[label_col].isna().any():
            raise RuntimeError(f"{name} split has null labels")

    train_q = set(train["query_name"].astype(str))
    val_q = set(val["query_name"].astype(str))
    test_q = set(test["query_name"].astype(str))
    if train_q & val_q or train_q & test_q or val_q & test_q:
        raise RuntimeError("query_name leakage across train/val/test splits")


def infer_numeric_feature_columns(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    """Return numeric feature columns, excluding metadata and leakage columns."""
    excluded = set(METADATA_COLUMNS) | set(LEAKAGE_COLUMNS) | {label_col}
    excluded |= FORBIDDEN_COLUMNS

    feature_cols: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError("no numeric feature columns found")
    return feature_cols


def feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X/y matrices with stable feature ordering."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"dataset missing saved feature columns: {missing[:10]}")
    X = df.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.astype(np.float64)
    y = df[label_col].astype(np.float64)
    return X, y


def feature_matrix_inference(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Build inference matrix; missing features are filled with 0.0."""
    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    X = work.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float64)


def check_pipeline_alignment(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, label_col: str) -> List[str]:
    """Print feature/label stats and return the feature column list."""
    validate_splits(train, val, test, label_col)
    feature_cols = infer_numeric_feature_columns(train, label_col=label_col)

    print("\n" + "=" * 60)
    print("ML DATASET ALIGNMENT CHECK")
    print("=" * 60)
    print(f"Train rows / queries: {len(train)} / {train['query_name'].nunique()}")
    print(f"Val rows / queries:   {len(val)} / {val['query_name'].nunique()}")
    print(f"Test rows / queries:  {len(test)} / {test['query_name'].nunique()}")
    print(f"Numeric feature cols: {len(feature_cols)}")
    print(f"Label column:         {label_col}")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        s = df[label_col]
        print(
            f"  {name:5s} label min={s.min():.4f} max={s.max():.4f} "
            f"mean={s.mean():.4f} median={s.median():.4f}"
        )
    print("Sanity check passed: no metadata/leakage columns are used as features.")
    print("=" * 60 + "\n")
    return feature_cols


def grid_search_cv(
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_col: str,
    param_grid: dict = PARAM_GRID,
    fixed_params: dict = FIXED_PARAMS,
) -> Tuple[dict, float, List[str]]:
    """Small explicit grid search using the query-template validation split."""
    feature_cols = infer_numeric_feature_columns(train, label_col=label_col)
    X_train, y_train = feature_matrix(train, feature_cols, label_col)
    X_val, y_val = feature_matrix(val, feature_cols, label_col)

    combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    best_rmse = float("inf")
    best_params: dict = {}

    print(f"\nGrid search: {len(combinations)} combinations")
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        t0 = time.time()
        model = xgb.XGBRegressor(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            n_estimators=params["n_estimators"],
            objective=fixed_params["objective"],
            random_state=fixed_params["random_state"],
            n_jobs=fixed_params["n_jobs"],
            verbosity=fixed_params["verbosity"],
            eval_metric=fixed_params["eval_metric"],
            early_stopping_rounds=fixed_params["early_stopping_rounds"],
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        rmse = math.sqrt(mean_squared_error(y_val, preds))
        flag = " <- best" if rmse < best_rmse else ""
        print(
            f" [{i:2d}/{len(combinations)}] "
            f"lr={params['learning_rate']} depth={params['max_depth']} "
            f"sub={params['subsample']} col={params['colsample_bytree']} "
            f"trees={params['n_estimators']} -> val RMSE {rmse:.4f} "
            f"({time.time() - t0:.1f}s){flag}"
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = dict(params)

    print(f"\nBest params: {best_params}")
    print(f"Best val RMSE: {best_rmse:.4f}")
    return best_params, best_rmse, feature_cols


def train_default(train: pd.DataFrame, val: pd.DataFrame, label_col: str) -> Tuple[xgb.XGBRegressor, List[str]]:
    feature_cols = infer_numeric_feature_columns(train, label_col=label_col)
    X_train, y_train = feature_matrix(train, feature_cols, label_col)
    X_val, y_val = feature_matrix(val, feature_cols, label_col)

    model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=15,
        eval_metric="rmse",
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, feature_cols


def train_with_best_params(
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_col: str,
    best_params: dict,
    feature_cols: Sequence[str],
) -> xgb.XGBRegressor:
    trainval = pd.concat([train, val], ignore_index=True)
    X_all, y_all = feature_matrix(trainval, feature_cols, label_col)
    model = xgb.XGBRegressor(
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        n_estimators=best_params["n_estimators"],
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(X_all, y_all, verbose=False)
    return model


def evaluate(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "label",
    split_name: str = "test",
) -> dict:
    X, y = feature_matrix(df, feature_cols, label_col)
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = math.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds) if len(y) > 1 else float("nan")

    y_raw = np.array([inverse_signed_log1p(v) for v in y], dtype=np.float64)
    pred_raw = np.array([inverse_signed_log1p(v) for v in preds], dtype=np.float64)
    raw_mae = mean_absolute_error(y_raw, pred_raw)

    print(f"\n{split_name.upper()} METRICS")
    print("  Log-space target metrics:")
    print(f"    MAE : {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R^2 : {r2:.4f}")
    print("  Raw optimizer-cost-delta scale:")
    print(f"    MAE : {raw_mae:,.2f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "raw_mae": raw_mae}


def print_feature_importance(model: xgb.XGBRegressor, feature_cols: Sequence[str], top_n: int = 15) -> None:
    importance = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nTOP {top_n} FEATURES BY GAIN")
    for fname, score in ranked:
        try:
            idx = int(fname[1:])
            real_name = feature_cols[idx]
        except (ValueError, IndexError):
            real_name = fname
        print(f"  {score:10.1f}  {real_name}")


def build_recommendation_features(
    repo_root: str,
    min_cost_impact: float = DEFAULT_MIN_COST_IMPACT,
) -> pd.DataFrame:
    """Rebuild fresh feature rows from the current workload/candidate pipeline."""
    queries_dir = os.path.join(repo_root, "queries")
    workload = parse_workload(queries_dir)
    candidates = generate_candidates(workload, min_cost_impact=min_cost_impact)

    conn = get_connection()
    try:
        rows = build_feature_rows(conn, candidates, workload)
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()
    return add_example_ids(pd.DataFrame(rows))


def recommend(
    model: xgb.XGBRegressor,
    feature_cols: Sequence[str],
    features_df: pd.DataFrame,
    top_k: int = 5,
    budget: float | None = None,
) -> pd.DataFrame:
    """
    Predict benefit for every (query, candidate) row and aggregate to candidates.

    The model predicts signed-log benefit per query-candidate row.  To rank
    candidate indexes, predictions are inverted back to raw optimizer-cost-delta
    scale and then summed across source queries.

    If `budget` is provided, we perform a simple greedy selection by predicted
    benefit/maintenance proxy using `write_penalty_proxy` as the cost proxy.
    """
    X = feature_matrix_inference(features_df, feature_cols)
    scored = features_df.copy()
    scored["predicted_log_benefit"] = model.predict(X)
    scored["predicted_raw_benefit"] = scored["predicted_log_benefit"].map(inverse_signed_log1p)

    agg_cols = ["candidate_table", "candidate_cols", "candidate_type"]
    grouped = (
        scored.groupby(agg_cols, as_index=False)
        .agg(
            predicted_raw_benefit=("predicted_raw_benefit", "sum"),
            predicted_log_score_sum=("predicted_log_benefit", "sum"),
            query_count=("query_name", "nunique"),
            candidate_cost_impact=("candidate_cost_impact", "max"),
            write_penalty_proxy=("write_penalty_proxy", "max"),
        )
    )

    # Avoid negative-benefit recommendations unless everything is negative.
    grouped = grouped.sort_values("predicted_raw_benefit", ascending=False).reset_index(drop=True)

    if budget is None:
        ranked = grouped.head(top_k).copy()
        ranked["selection_mode"] = "top_k"
    else:
        work = grouped.copy()
        work["selection_cost"] = work["write_penalty_proxy"].clip(lower=1.0)
        work["benefit_per_cost"] = work["predicted_raw_benefit"] / work["selection_cost"]
        work = work.sort_values(["benefit_per_cost", "predicted_raw_benefit"], ascending=False)

        selected = []
        used = 0.0
        for _, row in work.iterrows():
            if len(selected) >= top_k:
                break
            cost = float(row["selection_cost"])
            if used + cost <= float(budget):
                selected.append(row)
                used += cost
        ranked = pd.DataFrame(selected)
        if ranked.empty:
            ranked = work.head(0).copy()
        ranked["selection_mode"] = f"budgeted_proxy_{budget:g}"

    ranked = ranked.reset_index(drop=True)
    ranked.index += 1
    return ranked


def format_recommendations(ranked: pd.DataFrame) -> None:
    print("\nRECOMMENDED INDEXES")
    print("=" * 78)
    if ranked.empty:
        print("No indexes selected under the requested constraints.")
        print("=" * 78)
        return

    for rank, row in ranked.iterrows():
        print(f"\n#{rank}")
        print(f"  Selection mode:                 {row.get('selection_mode', 'top_k')}")
        print(f"  Predicted total raw benefit:    {row['predicted_raw_benefit']:,.2f}")
        print(f"  Sum of per-query log scores:    {row['predicted_log_score_sum']:.4f}")
        print(f"  Source query count:             {int(row['query_count'])}")
        print(f"  Candidate workload cost impact: {row['candidate_cost_impact']:,.2f}")
        print(f"  Write penalty proxy:            {row['write_penalty_proxy']:,.2f}")
        if "benefit_per_cost" in row.index and not pd.isna(row.get("benefit_per_cost")):
            print(f"  Benefit / proxy cost:           {row['benefit_per_cost']:,.4f}")
        print(f"  SQL: CREATE INDEX ON {row['candidate_table']} ({row['candidate_cols']});")
    print("=" * 78)


def save_feature_columns(path: str, feature_cols: Sequence[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(feature_cols))


def load_feature_columns(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the index recommendation model.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--recommend", action="store_true")
    parser.add_argument("--no-grid-search", action="store_true", help="Use default hyperparameters.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--budget", type=float, default=None, help="Optional greedy budget using write_penalty_proxy units.")
    parser.add_argument("--training-dir", default=TRAINING_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--repo-root", default=_REPO_ROOT)
    parser.add_argument("--min-cost-impact", type=float, default=DEFAULT_MIN_COST_IMPACT)
    args = parser.parse_args()

    cols_path = feature_cols_path_for_model(args.model_path)

    if args.train:
        print("Loading train/val/test splits...")
        train, val, test = load_splits(args.training_dir)
        print(f"  train={len(train)} val={len(val)} test={len(test)}")
        feature_cols = check_pipeline_alignment(train, val, test, args.label_column)

        if args.no_grid_search:
            print("\nTraining with default hyperparameters...")
            model, feature_cols = train_default(train, val, args.label_column)
        else:
            best_params, _, feature_cols = grid_search_cv(train, val, args.label_column)
            print(f"\nTraining final model on train+val ({len(train) + len(val)} rows)...")
            model = train_with_best_params(train, val, args.label_column, best_params, feature_cols)

        evaluate(model, test, feature_cols, args.label_column, split_name="test")
        print_feature_importance(model, feature_cols)

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save_model(args.model_path)
        save_feature_columns(cols_path, feature_cols)
        print(f"\nModel saved to {args.model_path}")
        print(f"Feature columns saved to {cols_path}")

    elif args.recommend:
        if not os.path.exists(args.model_path):
            print(f"No model found at {args.model_path}. Run with --train first.")
            sys.exit(1)
        if not os.path.exists(cols_path):
            print(f"No feature-column file found at {cols_path}. Run with --train first.")
            sys.exit(1)

        model = xgb.XGBRegressor()
        model.load_model(args.model_path)
        feature_cols = load_feature_columns(cols_path)

        print("Building fresh recommendation features from the current workload...")
        features_df = build_recommendation_features(
            repo_root=os.path.abspath(args.repo_root),
            min_cost_impact=args.min_cost_impact,
        )
        if features_df.empty:
            print("No recommendation feature rows were generated.")
            sys.exit(1)

        forbidden = FORBIDDEN_COLUMNS & set(features_df.columns)
        if forbidden:
            raise RuntimeError(f"recommendation features contain stale forbidden columns: {sorted(forbidden)}")

        ranked = recommend(model, feature_cols, features_df, top_k=args.top_k, budget=args.budget)
        format_recommendations(ranked)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

