"""Utilities for Stage 06 Knowledge Unit temporal baselines.

This module provides reusable logic for running baseline models on the frozen
Stage 05F KU temporal prediction benchmark.

Expected Stage 05F input directory:

    data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/

Expected files:

    prediction_feature_table.parquet
    feature_columns.json
    target_columns.json

Main benchmark design:

    example = KU x cutoff_year

    history = 2000..cutoff_year
    future  = cutoff_year+1..cutoff_year+3

Primary task:

    Future Translation Heat Forecasting

    filter:
        eligible_translation_task == true

    heat target:
        target_future_translation_heat_3yr

    auxiliary binary label:
        label_future_translation_emergence_3yr

This module intentionally does not hard-code a single model. It provides:

1. dataset loading helpers;
2. task configuration helpers;
3. safe preprocessing utilities;
4. heuristic baseline scoring;
5. sklearn model constructors;
6. metric computation for regression / classification / ranking;
7. grouped evaluation for validation, test_all, test_stable, and test_by_cutoff.

The CLI wrapper should live in:

    scripts/30_run_knowledge_unit_temporal_baselines.py
"""

from __future__ import annotations

import gc
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a KU temporal prediction task."""

    name: str
    eligibility_column: str
    heat_target_column: str
    count_target_column: str
    emergence_label_column: str
    primary_description: str


TASK_CONFIGS: dict[str, TaskConfig] = {
    "translation": TaskConfig(
        name="translation",
        eligibility_column="eligible_translation_task",
        heat_target_column="target_future_translation_heat_3yr",
        count_target_column="target_future_translation_count_3yr",
        emergence_label_column="label_future_translation_emergence_3yr",
        primary_description=(
            "Overall future translation heat forecasting: patent or trial "
            "evidence in the next 3 years."
        ),
    ),
    "patent": TaskConfig(
        name="patent",
        eligibility_column="eligible_patent_task",
        heat_target_column="target_future_patent_heat_3yr",
        count_target_column="target_future_patent_count_3yr",
        emergence_label_column="label_future_patent_emergence_3yr",
        primary_description=(
            "Future patent translation heat forecasting: patent evidence "
            "in the next 3 years."
        ),
    ),
    "trial": TaskConfig(
        name="trial",
        eligibility_column="eligible_trial_task",
        heat_target_column="target_future_trial_heat_3yr",
        count_target_column="target_future_trial_count_3yr",
        emergence_label_column="label_future_trial_emergence_3yr",
        primary_description=(
            "Future clinical translation heat forecasting: clinical trial "
            "evidence in the next 3 years."
        ),
    ),
}


DEFAULT_EVAL_GROUPS: dict[str, dict[str, Any]] = {
    "validation": {
        "split": "validation",
        "cutoffs": None,
        "description": "All validation cutoffs.",
    },
    "test_all": {
        "split": "test",
        "cutoffs": None,
        "description": "All test cutoffs, including right-edge stress cutoff.",
    },
    "test_stable": {
        "split": "test",
        "cutoffs": [2019, 2020],
        "description": "Stable test subset excluding the right-edge 2021 cutoff.",
    },
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Convert numpy / pandas / pathlib objects into JSON-serializable objects."""

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if value is pd.NA:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    return value


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write JSON with safe conversion."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(dict(data)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame based on file extension."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        frame.to_json(path, orient="records", indent=2, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output suffix: {path}")


def load_feature_spec(prediction_dir: str | Path) -> dict[str, Any]:
    """Load Stage 05F feature column specification."""

    prediction_dir = Path(prediction_dir)
    path = prediction_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature spec: {path}")
    return read_json(path)


def load_target_spec(prediction_dir: str | Path) -> dict[str, Any]:
    """Load Stage 05F target column specification."""

    prediction_dir = Path(prediction_dir)
    path = prediction_dir / "target_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing target spec: {path}")
    return read_json(path)


def load_prediction_feature_table(
    prediction_dir: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load Stage 05F prediction feature table."""

    prediction_dir = Path(prediction_dir)
    path = prediction_dir / "prediction_feature_table.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction feature table: {path}")

    return pd.read_parquet(path, columns=list(columns) if columns else None)


def get_feature_columns(
    feature_spec: Mapping[str, Any],
    *,
    include_categorical: bool = True,
    include_numeric: bool = True,
) -> list[str]:
    """Return safe feature columns from feature_columns.json."""

    columns: list[str] = []

    if include_categorical:
        columns.extend(feature_spec.get("categorical_feature_columns", []))

    if include_numeric:
        columns.extend(feature_spec.get("numeric_feature_columns", []))

    # Keep order and remove duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column not in seen:
            out.append(column)
            seen.add(column)

    return out


def required_columns_for_tasks(
    tasks: Sequence[str],
    feature_columns: Sequence[str],
) -> list[str]:
    """Build a minimal column list for loading the prediction feature table."""

    base_columns = {
        "ku_id",
        "cutoff_year",
        "split",
        *feature_columns,
    }

    for task in tasks:
        config = TASK_CONFIGS[task]
        base_columns.update(
            [
                config.eligibility_column,
                config.heat_target_column,
                config.count_target_column,
                config.emergence_label_column,
            ]
        )

    return sorted(base_columns)


# ---------------------------------------------------------------------------
# Dataset splitting and task selection
# ---------------------------------------------------------------------------


def get_task_config(task: str) -> TaskConfig:
    """Return task config by name."""

    if task not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task {task!r}. Available tasks: {sorted(TASK_CONFIGS)}"
        )
    return TASK_CONFIGS[task]


def select_task_frame(
    frame: pd.DataFrame,
    task: str,
    *,
    copy: bool = False,
) -> pd.DataFrame:
    """Filter a full prediction table to a task-specific eligible frame.

    For wide Stage 06G feature sets, eager deep copies can allocate many GB.
    Prefer loading eligible task frames directly from parquet in script 30.
    """

    config = get_task_config(task)
    if config.eligibility_column not in frame.columns:
        raise ValueError(f"Missing eligibility column: {config.eligibility_column}")

    out = frame.loc[frame[config.eligibility_column].astype(bool)]
    return out.copy(deep=False) if copy else out


def split_task_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return train / validation / test frames.

    This intentionally avoids eager .copy() calls because Stage 06F feature sets
    can contain millions of rows and 100+ columns. Downstream code should copy
    only the columns/chunks it needs.
    """

    if "split" not in frame.columns:
        raise ValueError("Frame missing required column: split")

    return {
        "train": frame.loc[frame["split"] == "train"],
        "validation": frame.loc[frame["split"] == "validation"],
        "test": frame.loc[frame["split"] == "test"],
    }


def make_eval_slices(
    frame: pd.DataFrame,
    *,
    include_test_by_cutoff: bool = True,
    stable_test_cutoffs: Sequence[int] = (2019, 2020),
) -> dict[str, pd.DataFrame]:
    """Create standard evaluation slices.

    This function avoids eager .copy() to reduce memory pressure. The returned
    frames are read-only views/slices for evaluation.
    """

    if "split" not in frame.columns:
        raise ValueError("Frame missing required column: split")
    if "cutoff_year" not in frame.columns:
        raise ValueError("Frame missing required column: cutoff_year")

    slices: dict[str, pd.DataFrame] = {
        "validation": frame.loc[frame["split"] == "validation"],
        "test_all": frame.loc[frame["split"] == "test"],
        "test_stable": frame.loc[
            (frame["split"] == "test")
            & (frame["cutoff_year"].isin(list(stable_test_cutoffs)))
        ],
    }

    if include_test_by_cutoff:
        test_cutoffs = sorted(
            int(v)
            for v in frame.loc[
                frame["split"] == "test",
                "cutoff_year",
            ].dropna().unique()
        )
        for cutoff in test_cutoffs:
            slices[f"test_cutoff_{cutoff}"] = frame.loc[
                (frame["split"] == "test") & (frame["cutoff_year"] == cutoff)
            ]

    return slices


def sample_training_frame(
    frame: pd.DataFrame,
    *,
    max_rows: int | None = None,
    random_state: int = 42,
    stratify_column: str | None = None,
) -> pd.DataFrame:
    """Optionally downsample a training frame.

    This is useful for expensive sklearn baselines on millions of rows.

    If max_rows is None or frame is already small enough, returns frame unchanged.
    """

    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame

    if stratify_column is not None and stratify_column in frame.columns:
        positives = frame[frame[stratify_column].astype(bool)]
        negatives = frame[~frame[stratify_column].astype(bool)]

        positive_fraction = len(positives) / max(len(frame), 1)
        n_pos = min(len(positives), max(1, int(round(max_rows * positive_fraction))))
        n_neg = min(len(negatives), max_rows - n_pos)

        sampled = pd.concat(
            [
                positives.sample(n=n_pos, random_state=random_state)
                if len(positives) > n_pos
                else positives,
                negatives.sample(n=n_neg, random_state=random_state)
                if len(negatives) > n_neg
                else negatives,
            ],
            axis=0,
        )
        return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return frame.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _make_one_hot_encoder(*, sparse_output: bool):
    """Construct OneHotEncoder compatible with multiple sklearn versions."""

    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=sparse_output,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=sparse_output,
        )


def build_preprocessor(
    feature_columns: Sequence[str],
    *,
    categorical_columns: Sequence[str] | None = None,
    numeric_columns: Sequence[str] | None = None,
    dense_output: bool = False,
):
    """Build a sklearn ColumnTransformer for tabular baselines.

    Numeric features:
        median imputation + standard scaling

    Categorical features:
        most-frequent imputation + one-hot encoding

    Args:
        feature_columns:
            Full feature column list.

        categorical_columns:
            Categorical feature columns. If omitted, inferred from common
            Stage 05F columns.

        numeric_columns:
            Numeric feature columns. If omitted, uses all non-categorical
            feature columns.

        dense_output:
            Whether one-hot output should be dense. HistGradientBoosting
            usually needs dense arrays; linear models can use sparse output.
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_columns = list(feature_columns)

    if categorical_columns is None:
        categorical_columns = [
            c for c in ["pair_type", "entity_a_type", "entity_b_type"] if c in feature_columns
        ]
    else:
        categorical_columns = [c for c in categorical_columns if c in feature_columns]

    if numeric_columns is None:
        numeric_columns = [c for c in feature_columns if c not in set(categorical_columns)]
    else:
        numeric_columns = [c for c in numeric_columns if c in feature_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                _make_one_hot_encoder(sparse_output=not dense_output),
            ),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3 if not dense_output else 0.0,
    )


def prepare_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Return feature matrix with columns present and ordered.

    Low-memory behavior:
      - Do not copy the full input frame.
      - Copy only requested feature columns.
      - Add missing feature columns as NaN.
      - Downcast numeric / boolean columns to float32.

    This is important for Stage 06F KU + carrier structural baselines, where
    full train/eval frames can contain millions of rows and 100+ features.
    """

    feature_columns = list(feature_columns)

    present_columns = [column for column in feature_columns if column in frame.columns]
    missing_columns = [column for column in feature_columns if column not in frame.columns]

    # Copy only the feature columns, not the whole frame.
    if present_columns:
        out = frame.loc[:, present_columns].copy()
    else:
        out = pd.DataFrame(index=frame.index)

    for column in missing_columns:
        out[column] = np.nan

    # Restore requested order.
    out = out.loc[:, feature_columns]

    # Downcast numeric columns to float32 to reduce memory pressure.
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_bool_dtype(series):
            out[column] = series.astype("float32")
        elif pd.api.types.is_integer_dtype(series):
            out[column] = series.astype("float32")
        elif pd.api.types.is_float_dtype(series):
            out[column] = series.astype("float32")

    return out


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------


def make_ridge_regression_pipeline(
    feature_columns: Sequence[str],
    *,
    alpha: float = 1.0,
    random_state: int = 42,
):
    """Create Ridge regression pipeline for heat forecasting."""

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(feature_columns, dense_output=False)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                Ridge(
                    alpha=alpha,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_elasticnet_regression_pipeline(
    feature_columns: Sequence[str],
    *,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    random_state: int = 42,
    max_iter: int = 2000,
):
    """Create ElasticNet regression pipeline for heat forecasting."""

    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(feature_columns, dense_output=True)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=random_state,
                    max_iter=max_iter,
                ),
            ),
        ]
    )


def make_logistic_regression_pipeline(
    feature_columns: Sequence[str],
    *,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
):
    """Create Logistic Regression pipeline for emergence classification."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(feature_columns, dense_output=False)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    random_state=random_state,
                    max_iter=max_iter,
                    n_jobs=None,
                ),
            ),
        ]
    )


def make_hist_gradient_boosting_regressor_pipeline(
    feature_columns: Sequence[str],
    *,
    learning_rate: float = 0.05,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.01,
    random_state: int = 42,
):
    """Create sklearn HistGradientBoostingRegressor pipeline.

    Uses dense one-hot preprocessing. This is acceptable because the categorical
    feature set is very small in Stage 05F.
    """

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(feature_columns, dense_output=True)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    random_state=random_state,
                    early_stopping=True,
                ),
            ),
        ]
    )


def make_hist_gradient_boosting_classifier_pipeline(
    feature_columns: Sequence[str],
    *,
    learning_rate: float = 0.05,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.01,
    random_state: int = 42,
):
    """Create sklearn HistGradientBoostingClassifier pipeline."""

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(feature_columns, dense_output=True)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    random_state=random_state,
                    early_stopping=True,
                ),
            ),
        ]
    )


def try_make_lightgbm_regressor(
    *,
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
):
    """Create LightGBM regressor if lightgbm is installed.

    Returns None if unavailable.
    """

    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None

    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
    )


def try_make_lightgbm_classifier(
    *,
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    class_weight: str | None = "balanced",
):
    """Create LightGBM classifier if lightgbm is installed.

    Returns None if unavailable.
    """

    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None

    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Heuristic baseline scoring
# ---------------------------------------------------------------------------


def _safe_numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return numeric series with missing column/value handling."""

    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")

    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("float64")


def heuristic_score(
    frame: pd.DataFrame,
    heuristic: str,
) -> pd.Series:
    """Compute heuristic score.

    These scores are primarily ranking scores. They are not necessarily calibrated
    heat predictions.
    """

    heuristic = heuristic.lower()

    if heuristic in {"recency", "recent"}:
        recency = _safe_numeric_series(frame, "history_recency", default=np.nan)
        finite = recency[np.isfinite(recency)]
        fill_value = float(finite.max() + 1.0) if len(finite) else 9999.0
        recency = recency.fillna(fill_value)
        return -recency

    if heuristic in {"activity_paper", "paper_activity"}:
        return np.log1p(_safe_numeric_series(frame, "history_paper_count"))

    if heuristic in {"activity_total", "total_activity"}:
        return np.log1p(_safe_numeric_series(frame, "history_total_count"))

    if heuristic in {"recent_3yr_paper", "recent_activity_paper"}:
        return np.log1p(_safe_numeric_series(frame, "recent_3yr_paper_count"))

    if heuristic in {"recent_3yr_total", "recent_activity_total"}:
        return np.log1p(_safe_numeric_series(frame, "recent_3yr_total_count"))

    if heuristic in {"recent_5yr_paper"}:
        return np.log1p(_safe_numeric_series(frame, "recent_5yr_paper_count"))

    if heuristic in {"recent_5yr_total"}:
        return np.log1p(_safe_numeric_series(frame, "recent_5yr_total_count"))

    if heuristic in {"growth_paper_3yr", "growth_paper"}:
        return _safe_numeric_series(frame, "growth_paper_3yr_vs_prev3yr")

    if heuristic in {"growth_total_3yr", "growth_total"}:
        return _safe_numeric_series(frame, "growth_total_3yr_vs_prev3yr")

    if heuristic in {"growth_paper_3yr_ratio", "growth_paper_ratio"}:
        return _safe_numeric_series(frame, "growth_paper_3yr_ratio", default=1.0)

    if heuristic in {"growth_total_3yr_ratio", "growth_total_ratio"}:
        return _safe_numeric_series(frame, "growth_total_3yr_ratio", default=1.0)

    if heuristic in {"hybrid_activity_growth"}:
        return (
            np.log1p(_safe_numeric_series(frame, "recent_3yr_total_count"))
            + 0.5 * np.log1p(_safe_numeric_series(frame, "growth_total_3yr_ratio", default=1.0))
        )

    raise KeyError(f"Unknown heuristic: {heuristic}")


def percentile_calibrate_scores(
    train_scores: Sequence[float],
    train_targets: Sequence[float],
    eval_scores: Sequence[float],
    *,
    n_bins: int = 100,
) -> np.ndarray:
    """Calibrate arbitrary heuristic scores to expected heat by score quantiles.

    This is useful if we want heuristic baselines to have a calibrated heat
    prediction for MAE/RMSE, not only a ranking score.

    Procedure:
        1. rank train scores into quantile bins;
        2. compute mean train target per bin;
        3. assign eval rows by train score quantile edges.

    The ranking score remains the raw heuristic score.
    """

    train_scores = np.asarray(train_scores, dtype=float)
    train_targets = np.asarray(train_targets, dtype=float)
    eval_scores = np.asarray(eval_scores, dtype=float)

    valid = np.isfinite(train_scores) & np.isfinite(train_targets)
    if valid.sum() == 0:
        return np.zeros_like(eval_scores, dtype=float)

    train_scores_valid = train_scores[valid]
    train_targets_valid = train_targets[valid]

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(train_scores_valid, quantiles)
    edges = np.unique(edges)

    if len(edges) <= 2:
        return np.full_like(eval_scores, float(np.mean(train_targets_valid)), dtype=float)

    train_bin = np.searchsorted(edges, train_scores_valid, side="right") - 1
    eval_bin = np.searchsorted(edges, eval_scores, side="right") - 1

    max_bin = len(edges) - 2
    train_bin = np.clip(train_bin, 0, max_bin)
    eval_bin = np.clip(eval_bin, 0, max_bin)

    bin_means = np.full(max_bin + 1, float(np.mean(train_targets_valid)), dtype=float)
    for bin_id in range(max_bin + 1):
        mask = train_bin == bin_id
        if mask.any():
            bin_means[bin_id] = float(np.mean(train_targets_valid[mask]))

    return bin_means[eval_bin]


def minmax_to_probability(
    train_scores: Sequence[float],
    train_labels: Sequence[int | bool],
    eval_scores: Sequence[float],
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Map arbitrary scores to rough probabilities using train score min/max.

    This is only a weak probability proxy for heuristic baselines. It is mainly
    useful for AUPRC/AUROC scores where calibration is not required. Brier score
    should be interpreted cautiously for this output.
    """

    train_scores = np.asarray(train_scores, dtype=float)
    eval_scores = np.asarray(eval_scores, dtype=float)

    finite = np.isfinite(train_scores)
    if finite.sum() == 0:
        base_rate = float(np.mean(np.asarray(train_labels, dtype=float)))
        return np.full_like(eval_scores, base_rate, dtype=float)

    lo = float(np.nanmin(train_scores[finite]))
    hi = float(np.nanmax(train_scores[finite]))

    if hi - lo <= eps:
        base_rate = float(np.mean(np.asarray(train_labels, dtype=float)))
        return np.full_like(eval_scores, base_rate, dtype=float)

    scaled = (eval_scores - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _as_float_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)


def _as_binary_array(values: Sequence[Any]) -> np.ndarray:
    series = pd.Series(values)
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).to_numpy()


def safe_pearson(y_true: Sequence[float], y_score: Sequence[float]) -> float | None:
    """Compute Pearson correlation safely."""

    y_true = _as_float_array(y_true)
    y_score = _as_float_array(y_score)
    mask = np.isfinite(y_true) & np.isfinite(y_score)

    if mask.sum() < 2:
        return None

    y_true_valid = y_true[mask]
    y_score_valid = y_score[mask]

    if np.ptp(y_true_valid) == 0 or np.ptp(y_score_valid) == 0:
        return None

    return float(np.corrcoef(y_true_valid, y_score_valid)[0, 1])


def safe_spearman(y_true: Sequence[float], y_score: Sequence[float]) -> float | None:
    """Compute Spearman correlation safely."""

    y_true = _as_float_array(y_true)
    y_score = _as_float_array(y_score)
    mask = np.isfinite(y_true) & np.isfinite(y_score)

    if mask.sum() < 2:
        return None

    y_true_valid = y_true[mask]
    y_score_valid = y_score[mask]

    if np.ptp(y_true_valid) == 0 or np.ptp(y_score_valid) == 0:
        return None

    try:
        from scipy.stats import spearmanr
        from scipy.stats import ConstantInputWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            result = spearmanr(y_true_valid, y_score_valid)

        if result.correlation is None or math.isnan(float(result.correlation)):
            return None
        return float(result.correlation)

    except Exception:
        # Fallback: rank with pandas and compute Pearson on ranks.
        y_true_rank = pd.Series(y_true_valid).rank(method="average").to_numpy(dtype=float)
        y_score_rank = pd.Series(y_score_valid).rank(method="average").to_numpy(dtype=float)
        return safe_pearson(y_true_rank, y_score_rank)


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float] | None,
    *,
    y_score: Sequence[float] | None = None,
) -> dict[str, float | None]:
    """Compute heat regression / correlation metrics.

    MAE/RMSE require y_pred. Spearman/Pearson use y_score if provided,
    otherwise y_pred.
    """

    y_true_arr = _as_float_array(y_true)

    out: dict[str, float | None] = {
        "mae": None,
        "rmse": None,
        "spearman": None,
        "pearson": None,
    }

    if y_pred is not None:
        y_pred_arr = _as_float_array(y_pred)
        mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        if mask.sum() > 0:
            err = y_pred_arr[mask] - y_true_arr[mask]
            out["mae"] = float(np.mean(np.abs(err)))
            out["rmse"] = float(np.sqrt(np.mean(err**2)))

    score = y_score if y_score is not None else y_pred
    if score is not None:
        out["spearman"] = safe_spearman(y_true_arr, score)
        out["pearson"] = safe_pearson(y_true_arr, score)

    return out


def classification_metrics(
    y_true: Sequence[int | bool],
    y_score: Sequence[float] | None,
    *,
    y_prob: Sequence[float] | None = None,
) -> dict[str, float | None]:
    """Compute binary emergence metrics.

    AUROC/AUPRC use y_score if provided. Brier uses y_prob if provided,
    otherwise y_score clipped to [0, 1].
    """

    out: dict[str, float | None] = {
        "auroc": None,
        "auprc": None,
        "brier": None,
    }

    y_true_arr = _as_binary_array(y_true)

    if y_score is not None:
        score_arr = _as_float_array(y_score)
        mask = np.isfinite(score_arr)
        if mask.sum() > 0 and len(np.unique(y_true_arr[mask])) == 2:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score

                out["auroc"] = float(roc_auc_score(y_true_arr[mask], score_arr[mask]))
                out["auprc"] = float(average_precision_score(y_true_arr[mask], score_arr[mask]))
            except Exception:
                out["auroc"] = None
                out["auprc"] = None

    prob_source = y_prob if y_prob is not None else y_score
    if prob_source is not None:
        prob_arr = np.clip(_as_float_array(prob_source), 0.0, 1.0)
        mask = np.isfinite(prob_arr)
        if mask.sum() > 0:
            try:
                from sklearn.metrics import brier_score_loss

                out["brier"] = float(brier_score_loss(y_true_arr[mask], prob_arr[mask]))
            except Exception:
                out["brier"] = None

    return out


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute DCG@K."""

    if k <= 0:
        return 0.0

    rel = np.asarray(relevance, dtype=float)[:k]
    if len(rel) == 0:
        return 0.0

    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(rel * discounts))


def ranking_metrics_at_k(
    y_heat: Sequence[float],
    y_label: Sequence[int | bool],
    y_score: Sequence[float],
    *,
    k: int,
) -> dict[str, float | None]:
    """Compute top-K ranking metrics.

    NDCG uses heat target as graded relevance.
    Precision/Recall/Enrichment use binary emergence label.
    """

    y_heat_arr = _as_float_array(y_heat)
    y_label_arr = _as_binary_array(y_label)
    y_score_arr = _as_float_array(y_score)

    mask = np.isfinite(y_heat_arr) & np.isfinite(y_score_arr)
    if mask.sum() == 0:
        return {
            f"ndcg_at_{k}": None,
            f"precision_at_{k}": None,
            f"recall_at_{k}": None,
            f"enrichment_at_{k}": None,
        }

    y_heat_arr = y_heat_arr[mask]
    y_label_arr = y_label_arr[mask]
    y_score_arr = y_score_arr[mask]

    n = len(y_score_arr)
    k_eff = min(int(k), n)

    if k_eff <= 0:
        return {
            f"ndcg_at_{k}": None,
            f"precision_at_{k}": None,
            f"recall_at_{k}": None,
            f"enrichment_at_{k}": None,
        }

    order = np.argsort(-y_score_arr, kind="mergesort")
    top = order[:k_eff]

    ideal_order = np.argsort(-y_heat_arr, kind="mergesort")
    ideal_top = ideal_order[:k_eff]

    dcg = dcg_at_k(y_heat_arr[top], k_eff)
    ideal_dcg = dcg_at_k(y_heat_arr[ideal_top], k_eff)
    ndcg = float(dcg / ideal_dcg) if ideal_dcg > 0 else None

    positives = int(y_label_arr.sum())
    top_positives = int(y_label_arr[top].sum())

    precision = float(top_positives / k_eff)
    recall = float(top_positives / positives) if positives > 0 else None

    positive_rate = float(positives / n) if n > 0 else 0.0
    enrichment = float(precision / positive_rate) if positive_rate > 0 else None

    return {
        f"ndcg_at_{k}": ndcg,
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        f"enrichment_at_{k}": enrichment,
    }


def ranking_metrics(
    y_heat: Sequence[float],
    y_label: Sequence[int | bool],
    y_score: Sequence[float] | None,
    *,
    ks: Sequence[int] = (100, 1000),
) -> dict[str, float | None]:
    """Compute ranking metrics for multiple K values.

    If the score is constant, ranking is undefined. In that case return None
    for all ranking metrics instead of producing tie-order artifacts.
    """

    out: dict[str, float | None] = {}

    def empty_metrics() -> dict[str, float | None]:
        empty: dict[str, float | None] = {}
        for k in ks:
            empty.update(
                {
                    f"ndcg_at_{k}": None,
                    f"precision_at_{k}": None,
                    f"recall_at_{k}": None,
                    f"enrichment_at_{k}": None,
                }
            )
        return empty

    if y_score is None:
        return empty_metrics()

    score_arr = _as_float_array(y_score)
    finite_score = score_arr[np.isfinite(score_arr)]

    if len(finite_score) < 2 or np.ptp(finite_score) == 0:
        return empty_metrics()

    for k in ks:
        out.update(ranking_metrics_at_k(y_heat, y_label, y_score, k=int(k)))

    return out


def _to_float32_array(values: Sequence[Any]) -> np.ndarray:
    """Convert values to float32 with minimal copying."""

    if isinstance(values, pd.Series):
        if pd.api.types.is_numeric_dtype(values):
            return values.to_numpy(dtype=np.float32, copy=False)
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32, copy=False)

    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)


def _to_binary_int8_array(values: Sequence[Any]) -> np.ndarray:
    """Convert values to int8 binary array with minimal copying."""

    if isinstance(values, pd.Series):
        if values.dtype == bool:
            return values.to_numpy(dtype=np.int8, copy=False)
        if pd.api.types.is_numeric_dtype(values):
            return values.fillna(0).to_numpy(dtype=np.int8, copy=False)
        return pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    arr = np.asarray(values)
    if arr.dtype == bool or np.issubdtype(arr.dtype, np.number):
        return np.nan_to_num(arr, nan=0).astype(np.int8, copy=False)

    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).to_numpy(dtype=np.int8)


def _safe_pearson_low_memory(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float | None:
    """Compute Pearson correlation without np.corrcoef copies."""

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    n = int(mask.sum())
    if n < 2:
        return None

    x = y_true[mask].astype(np.float64, copy=False)
    y = y_score[mask].astype(np.float64, copy=False)

    x_mean = float(x.mean())
    y_mean = float(y.mean())

    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom <= 0.0:
        return None

    return float(np.sum(x_centered * y_centered) / denom)


def _safe_spearman_sampled_low_memory(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    max_rows: int = 1_000_000,
    random_state: int = 42,
) -> float | None:
    """Compute Spearman on a deterministic sample to avoid full-rank OOM.

    Full Spearman requires ranking all rows, which is memory-heavy for multi-
    million-row evaluation slices. This sampled estimate is stable enough for
    representation comparison and avoids catastrophic memory spikes.
    """

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    idx = np.flatnonzero(mask)

    if len(idx) < 2:
        return None

    if len(idx) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=max_rows, replace=False)

    x = y_true[idx]
    y = y_score[idx]

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None

    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float32)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float32)

    return _safe_pearson_low_memory(x_rank, y_rank)


def _trapezoid_auc(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for NumPy trapezoidal integration."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))

    # Manual fallback.
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(y) < 2 or len(x) < 2:
        return 0.0

    dx = x[1:] - x[:-1]
    avg_y = 0.5 * (y[1:] + y[:-1])
    return float(np.sum(dx * avg_y))


def _binned_classification_metrics_low_memory(
    y_true: np.ndarray,
    y_score: np.ndarray | None,
    *,
    y_prob: np.ndarray | None = None,
    n_bins: int = 4096,
) -> dict[str, float | None]:
    """Low-memory AUROC/AUPRC/Brier.

    For large arrays, this avoids sklearn's full sorting by binning scores.
    AUROC/AUPRC are approximate but deterministic.
    """

    out: dict[str, float | None] = {
        "auroc": None,
        "auprc": None,
        "brier": None,
    }

    if y_score is not None:
        score = np.asarray(y_score, dtype=np.float32)
        label = np.asarray(y_true, dtype=np.int8)

        mask = np.isfinite(score)
        if mask.sum() > 0:
            score = score[mask]
            label = label[mask]

            positives = int(label.sum())
            negatives = int(len(label) - positives)

            if positives > 0 and negatives > 0:
                lo = float(np.min(score))
                hi = float(np.max(score))

                if hi > lo:
                    # Bin scores into ascending bins, then evaluate thresholds
                    # from high score to low score.
                    bin_id = np.floor((score - lo) / (hi - lo) * (n_bins - 1)).astype(np.int32)
                    bin_id = np.clip(bin_id, 0, n_bins - 1)

                    pos_hist = np.bincount(
                        bin_id,
                        weights=label.astype(np.float32),
                        minlength=n_bins,
                    ).astype(np.float64)

                    count_hist = np.bincount(
                        bin_id,
                        minlength=n_bins,
                    ).astype(np.float64)

                    neg_hist = count_hist - pos_hist

                    # Descending score order.
                    tp = np.cumsum(pos_hist[::-1])
                    fp = np.cumsum(neg_hist[::-1])

                    tpr = tp / max(float(positives), 1.0)
                    fpr = fp / max(float(negatives), 1.0)

                    # AUROC via trapezoid in FPR/TPR space.
                    out["auroc"] = _trapezoid_auc(tpr, fpr)

                    precision = tp / np.maximum(tp + fp, 1.0)
                    recall = tpr

                    recall_prev = np.concatenate([[0.0], recall[:-1]])
                    delta_recall = np.maximum(recall - recall_prev, 0.0)
                    out["auprc"] = float(np.sum(delta_recall * precision))

    prob_source = y_prob if y_prob is not None else y_score
    if prob_source is not None:
        prob = np.clip(np.asarray(prob_source, dtype=np.float32), 0.0, 1.0)
        label = np.asarray(y_true, dtype=np.float32)
        mask = np.isfinite(prob)
        if mask.sum() > 0:
            err = prob[mask] - label[mask]
            out["brier"] = float(np.mean(err * err))

    return out


def _ranking_metrics_low_memory(
    y_heat: np.ndarray,
    y_label: np.ndarray,
    y_score: np.ndarray | None,
    *,
    ks: Sequence[int] = (100, 1000),
) -> dict[str, float | None]:
    """Compute top-K ranking metrics without full argsort.

    Uses argpartition to extract top-K only.
    """

    out: dict[str, float | None] = {}

    for k in ks:
        out.update(
            {
                f"ndcg_at_{k}": None,
                f"precision_at_{k}": None,
                f"recall_at_{k}": None,
                f"enrichment_at_{k}": None,
            }
        )

    if y_score is None:
        return out

    heat = np.asarray(y_heat, dtype=np.float32)
    label = np.asarray(y_label, dtype=np.int8)
    score = np.asarray(y_score, dtype=np.float32)

    mask = np.isfinite(heat) & np.isfinite(score)
    n_valid = int(mask.sum())

    if n_valid < 2:
        return out

    heat = heat[mask]
    label = label[mask]
    score = score[mask]

    finite_score = score[np.isfinite(score)]
    if len(finite_score) < 2 or np.ptp(finite_score) == 0:
        return out

    positives = int(label.sum())
    positive_rate = float(positives / len(label)) if len(label) else 0.0

    for k in ks:
        k_eff = min(int(k), len(score))
        if k_eff <= 0:
            continue

        # Top-k by score without sorting the full array.
        top_idx_unsorted = np.argpartition(-score, k_eff - 1)[:k_eff]
        top_idx = top_idx_unsorted[np.argsort(-score[top_idx_unsorted], kind="mergesort")]

        # Ideal top-k by heat without sorting the full array.
        ideal_idx_unsorted = np.argpartition(-heat, k_eff - 1)[:k_eff]
        ideal_idx = ideal_idx_unsorted[np.argsort(-heat[ideal_idx_unsorted], kind="mergesort")]

        dcg = dcg_at_k(heat[top_idx], k_eff)
        ideal_dcg = dcg_at_k(heat[ideal_idx], k_eff)

        ndcg = float(dcg / ideal_dcg) if ideal_dcg > 0 else None

        top_positives = int(label[top_idx].sum())
        precision = float(top_positives / k_eff)
        recall = float(top_positives / positives) if positives > 0 else None
        enrichment = float(precision / positive_rate) if positive_rate > 0 else None

        out[f"ndcg_at_{k}"] = ndcg
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"enrichment_at_{k}"] = enrichment

    return out


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    eval_group: str,
    y_pred_heat: Sequence[float] | None = None,
    y_score: Sequence[float] | None = None,
    y_prob: Sequence[float] | None = None,
    ks: Sequence[int] = (100, 1000),
) -> dict[str, Any]:
    """Evaluate predictions on one frame / group using low-memory metrics.

    This version avoids repeated float64 materialization and avoids full sorting
    for top-K metrics. Spearman is computed on a deterministic sample for large
    groups to avoid rank-memory spikes. AUROC/AUPRC are computed using a binned
    approximation to avoid sklearn's full-score sorting on multi-million-row
    slices.
    """

    config = get_task_config(task)

    if len(frame) == 0:
        return {
            "task": task,
            "model": model_name,
            "eval_group": eval_group,
            "split": None,
            "cutoff_year": None,
            "n_examples": 0,
            "n_positive": 0,
            "positive_rate": None,
        }

    y_heat = _to_float32_array(frame[config.heat_target_column])
    y_label = _to_binary_int8_array(frame[config.emergence_label_column])

    score_arr = _to_float32_array(y_score) if y_score is not None else None
    pred_heat_arr = _to_float32_array(y_pred_heat) if y_pred_heat is not None else None
    prob_arr = _to_float32_array(y_prob) if y_prob is not None else None

    n_examples = int(len(frame))
    n_positive = int(y_label.sum())

    out: dict[str, Any] = {
        "task": task,
        "model": model_name,
        "eval_group": eval_group,
        "split": None,
        "cutoff_year": None,
        "cutoff_years": None,
        "n_examples": n_examples,
        "n_positive": n_positive,
        "positive_rate": float(n_positive / n_examples) if n_examples else None,
        "target_heat_mean": float(np.nanmean(y_heat)) if n_examples else None,
        "target_heat_max": float(np.nanmax(y_heat)) if n_examples else None,
        "mae": None,
        "rmse": None,
        "spearman": None,
        "pearson": None,
        "auroc": None,
        "auprc": None,
        "brier": None,
    }

    if "split" in frame.columns:
        split_values = sorted(str(v) for v in frame["split"].dropna().unique())
        out["split"] = ",".join(split_values) if split_values else None

    if "cutoff_year" in frame.columns:
        cutoff_values = sorted(int(v) for v in frame["cutoff_year"].dropna().unique())
        out["cutoff_year"] = cutoff_values[0] if len(cutoff_values) == 1 else None
        out["cutoff_years"] = ",".join(str(v) for v in cutoff_values) if cutoff_values else None

    # Regression metrics.
    if pred_heat_arr is not None:
        mask = np.isfinite(y_heat) & np.isfinite(pred_heat_arr)
        if mask.sum() > 0:
            err = pred_heat_arr[mask].astype(np.float32, copy=False) - y_heat[mask].astype(
                np.float32,
                copy=False,
            )
            out["mae"] = float(np.mean(np.abs(err)))
            out["rmse"] = float(np.sqrt(np.mean(err * err)))

    corr_source = score_arr if score_arr is not None else pred_heat_arr
    if corr_source is not None:
        out["pearson"] = _safe_pearson_low_memory(y_heat, corr_source)
        out["spearman"] = _safe_spearman_sampled_low_memory(
            y_heat,
            corr_source,
            max_rows=1_000_000,
            random_state=42,
        )

    # Classification metrics.
    clf = _binned_classification_metrics_low_memory(
        y_label,
        score_arr,
        y_prob=prob_arr,
        n_bins=4096,
    )
    out.update(clf)

    # Ranking metrics.
    rank = _ranking_metrics_low_memory(
        y_heat,
        y_label,
        corr_source,
        ks=ks,
    )
    out.update(rank)

    return out


def evaluate_prediction_column(
    frame: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    score_column: str,
    heat_prediction_column: str | None = None,
    probability_column: str | None = None,
    ks: Sequence[int] = (100, 1000),
    stable_test_cutoffs: Sequence[int] = (2019, 2020),
) -> pd.DataFrame:
    """Evaluate a frame that already contains prediction columns."""

    slices = make_eval_slices(frame, stable_test_cutoffs=stable_test_cutoffs)

    rows: list[dict[str, Any]] = []
    for group_name, group in slices.items():
        y_pred_heat = (
            group[heat_prediction_column].to_numpy()
            if heat_prediction_column and heat_prediction_column in group.columns
            else None
        )
        y_prob = (
            group[probability_column].to_numpy()
            if probability_column and probability_column in group.columns
            else None
        )
        rows.append(
            evaluate_predictions(
                group,
                task=task,
                model_name=model_name,
                eval_group=group_name,
                y_pred_heat=y_pred_heat,
                y_score=group[score_column].to_numpy() if score_column in group.columns else None,
                y_prob=y_prob,
                ks=ks,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline prediction helpers
# ---------------------------------------------------------------------------


def predict_mean_baseline(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    task: str,
) -> dict[str, np.ndarray]:
    """Predict constant train mean heat and prevalence probability."""

    config = get_task_config(task)

    train_heat = _as_float_array(train_frame[config.heat_target_column])
    train_label = _as_binary_array(train_frame[config.emergence_label_column])

    heat_mean = float(np.nanmean(train_heat)) if len(train_heat) else 0.0
    prevalence = float(np.mean(train_label)) if len(train_label) else 0.0

    n = len(eval_frame)
    return {
        "heat": np.full(n, heat_mean, dtype=float),
        "score": np.full(n, heat_mean, dtype=float),
        "prob": np.full(n, prevalence, dtype=float),
    }


def predict_heuristic_baseline(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    task: str,
    heuristic: str,
    calibrate_heat: bool = True,
) -> dict[str, np.ndarray]:
    """Predict with a heuristic score.

    Returns:
        heat:
            percentile-calibrated heat estimate if calibrate_heat=True;
            otherwise None-like NaN array.

        score:
            raw heuristic score.

        prob:
            rough min-max score probability proxy.
    """

    config = get_task_config(task)

    train_score = heuristic_score(train_frame, heuristic).to_numpy(dtype=float)
    eval_score = heuristic_score(eval_frame, heuristic).to_numpy(dtype=float)

    train_heat = _as_float_array(train_frame[config.heat_target_column])
    train_label = _as_binary_array(train_frame[config.emergence_label_column])

    if calibrate_heat:
        heat = percentile_calibrate_scores(train_score, train_heat, eval_score)
    else:
        heat = np.full(len(eval_frame), np.nan, dtype=float)

    prob = minmax_to_probability(train_score, train_label, eval_score)

    return {
        "heat": heat,
        "score": eval_score,
        "prob": prob,
    }


def iter_frame_chunks(
    frame: pd.DataFrame,
    *,
    chunk_size: int = 10_000,
) -> Iterable[tuple[int, int, pd.DataFrame]]:
    """Yield positional chunks from a frame without copying the full frame."""

    n = len(frame)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        yield start, end, frame.iloc[start:end]


def predict_regression_model_in_chunks(
    model: Any,
    eval_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    chunk_size: int = 10_000,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Predict regression outputs in chunks to avoid materializing huge X_eval."""

    pred = np.empty(len(eval_frame), dtype=np.float32)

    for start, end, chunk in iter_frame_chunks(eval_frame, chunk_size=chunk_size):
        X_chunk = prepare_feature_matrix(chunk, feature_columns)
        chunk_pred = np.asarray(model.predict(X_chunk), dtype=np.float32)

        if clip_nonnegative:
            chunk_pred = np.maximum(chunk_pred, 0.0)

        pred[start:end] = chunk_pred

        del X_chunk, chunk_pred
        gc.collect()

    return pred


def predict_classifier_probability_in_chunks(
    model: Any,
    eval_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    chunk_size: int = 10_000,
) -> np.ndarray:
    """Predict positive-class probability in chunks."""

    y_prob = np.empty(len(eval_frame), dtype=np.float32)

    for start, end, chunk in iter_frame_chunks(eval_frame, chunk_size=chunk_size):
        X_chunk = prepare_feature_matrix(chunk, feature_columns)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_chunk)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                chunk_prob = np.asarray(prob[:, 1], dtype=np.float32)
            else:
                chunk_prob = np.asarray(prob, dtype=np.float32).reshape(-1)

        elif hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(X_chunk), dtype=np.float32)
            chunk_prob = 1.0 / (1.0 + np.exp(-decision))

        else:
            chunk_prob = np.asarray(model.predict(X_chunk), dtype=np.float32)

        y_prob[start:end] = np.clip(chunk_prob, 0.0, 1.0)

        del X_chunk, chunk_prob
        gc.collect()

    return y_prob

def fit_predict_regression_model(
    model: Any,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Fit a regression model and predict eval heat."""

    X_train = prepare_feature_matrix(train_frame, feature_columns)
    y_train = _as_float_array(train_frame[target_column]).astype("float32", copy=False)

    model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    return predict_regression_model_in_chunks(
        model,
        eval_frame,
        feature_columns=feature_columns,
        clip_nonnegative=clip_nonnegative,
    )


def fit_predict_classifier_model(
    model: Any,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    label_column: str,
) -> np.ndarray:
    """Fit a classifier model and predict positive-class probability."""

    X_train = prepare_feature_matrix(train_frame, feature_columns)
    y_train = _as_binary_array(train_frame[label_column])

    model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    return predict_classifier_probability_in_chunks(
        model,
        eval_frame,
        feature_columns=feature_columns,
    )


def build_prediction_frame(
    frame: pd.DataFrame,
    *,
    y_pred_heat: Sequence[float] | None = None,
    y_score: Sequence[float] | None = None,
    y_prob: Sequence[float] | None = None,
    heat_column: str = "pred_heat",
    score_column: str = "pred_score",
    probability_column: str = "pred_probability",
    keep_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach predictions to a compact frame for saving."""

    if keep_columns is None:
        keep_columns = [
            "ku_id",
            "cutoff_year",
            "split",
            "pair_type",
            "entity_a_name",
            "entity_a_type",
            "entity_b_name",
            "entity_b_type",
        ]

    columns = [column for column in keep_columns if column in frame.columns]
    out = frame[columns].copy()

    if y_pred_heat is not None:
        out[heat_column] = np.asarray(y_pred_heat, dtype=float)

    if y_score is not None:
        out[score_column] = np.asarray(y_score, dtype=float)

    if y_prob is not None:
        out[probability_column] = np.asarray(y_prob, dtype=float)

    return out


# ---------------------------------------------------------------------------
# Result table helpers
# ---------------------------------------------------------------------------


def sort_baseline_metrics(
    metrics: pd.DataFrame,
    *,
    primary_eval_group: str = "test_all",
    primary_metric: str = "ndcg_at_1000",
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort baseline metrics using one eval group and metric."""

    if metrics.empty:
        return metrics

    key = metrics[metrics["eval_group"] == primary_eval_group].copy()
    key = key[["task", "model", primary_metric]].rename(
        columns={primary_metric: "_sort_metric"}
    )

    out = metrics.merge(key, on=["task", "model"], how="left")
    out = out.sort_values(
        ["task", "_sort_metric", "model", "eval_group"],
        ascending=[True, ascending, True, True],
    )
    return out.drop(columns=["_sort_metric"])


def make_comparison_table(
    metrics: pd.DataFrame,
    *,
    eval_group: str = "test_all",
    sort_metric: str = "ndcg_at_1000",
    ascending: bool = False,
) -> pd.DataFrame:
    """Create one-row-per task/model comparison table for an eval group."""

    if metrics.empty:
        return metrics

    subset = metrics[metrics["eval_group"] == eval_group].copy()

    preferred_columns = [
        "task",
        "model",
        "eval_group",
        "n_examples",
        "n_positive",
        "positive_rate",
        "mae",
        "rmse",
        "spearman",
        "pearson",
        "auroc",
        "auprc",
        "brier",
        "ndcg_at_100",
        "ndcg_at_1000",
        "precision_at_1000",
        "recall_at_1000",
        "enrichment_at_1000",
    ]
    columns = [column for column in preferred_columns if column in subset.columns]

    subset = subset[columns]

    if sort_metric in subset.columns:
        subset = subset.sort_values(
            ["task", sort_metric, "model"],
            ascending=[True, ascending, True],
        )

    return subset.reset_index(drop=True)


def markdown_table(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    max_rows: int | None = None,
) -> str:
    """Render a small markdown table."""

    if isinstance(rows, pd.DataFrame):
        data = rows.to_dict("records")
        if columns is None:
            columns = list(rows.columns)
    else:
        data = list(rows)
        if columns is None:
            columns = list(data[0].keys()) if data else []

    if max_rows is not None:
        data = data[:max_rows]

    columns = list(columns)

    if not columns:
        return "_No rows._"

    def fmt(value: Any) -> str:
        value = json_safe(value)

        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.6g}"

        text = str(value)
        text = text.replace("|", "\\|").replace("\n", " ")
        return text

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(fmt(row.get(column)) for column in columns) + " |"
        for row in data
    ]

    return "\n".join([header, sep, *body])


def build_baseline_report(
    *,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    title: str = "Knowledge Unit Temporal Baseline Report",
    prediction_dir: str | Path | None = None,
    notes: Sequence[str] | None = None,
) -> str:
    """Build a Markdown report for Stage 06 baseline results."""

    lines: list[str] = [
        f"# {title}",
        "",
    ]

    if prediction_dir is not None:
        lines.extend(
            [
                "## 1. Input prediction dataset",
                "",
                f"`{prediction_dir}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 2. Baseline comparison",
            "",
            markdown_table(comparison, max_rows=100),
            "",
            "## 3. Metrics by evaluation group",
            "",
            markdown_table(metrics, max_rows=300),
            "",
        ]
    )

    if notes:
        lines.extend(["## 4. Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level baseline runner helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Container returned by baseline runner functions."""

    task: str
    model: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame | None = None
    metadata: dict[str, Any] | None = None


def run_mean_baseline(
    task_frame: pd.DataFrame,
    *,
    task: str,
    ks: Sequence[int] = (100, 1000),
    stable_test_cutoffs: Sequence[int] = (2019, 2020),
) -> BaselineResult:
    """Run mean / prevalence baseline."""

    config = get_task_config(task)
    splits = split_task_frame(task_frame)
    train = splits["train"]

    eval_slices = make_eval_slices(
        task_frame,
        stable_test_cutoffs=stable_test_cutoffs,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for group_name, group in eval_slices.items():
        pred = predict_mean_baseline(train, group, task=task)

        metric_rows.append(
            evaluate_predictions(
                group,
                task=task,
                model_name="mean",
                eval_group=group_name,
                y_pred_heat=pred["heat"],
                y_score=pred["score"],
                y_prob=pred["prob"],
                ks=ks,
            )
        )

        pred_frame = build_prediction_frame(
            group,
            y_pred_heat=pred["heat"],
            y_score=pred["score"],
            y_prob=pred["prob"],
        )
        pred_frame["task"] = task
        pred_frame["model"] = "mean"
        pred_frame["eval_group"] = group_name
        pred_frame[config.heat_target_column] = group[config.heat_target_column].to_numpy()
        pred_frame[config.emergence_label_column] = group[
            config.emergence_label_column
        ].to_numpy()
        prediction_frames.append(pred_frame)

    return BaselineResult(
        task=task,
        model="mean",
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else None,
        metadata={"baseline_type": "constant_mean_prevalence"},
    )


def run_heuristic_baseline(
    task_frame: pd.DataFrame,
    *,
    task: str,
    heuristic: str,
    model_name: str | None = None,
    calibrate_heat: bool = True,
    ks: Sequence[int] = (100, 1000),
    stable_test_cutoffs: Sequence[int] = (2019, 2020),
) -> BaselineResult:
    """Run one heuristic baseline."""

    config = get_task_config(task)
    model_name = model_name or heuristic

    splits = split_task_frame(task_frame)
    train = splits["train"]

    eval_slices = make_eval_slices(
        task_frame,
        stable_test_cutoffs=stable_test_cutoffs,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for group_name, group in eval_slices.items():
        pred = predict_heuristic_baseline(
            train,
            group,
            task=task,
            heuristic=heuristic,
            calibrate_heat=calibrate_heat,
        )

        metric_rows.append(
            evaluate_predictions(
                group,
                task=task,
                model_name=model_name,
                eval_group=group_name,
                y_pred_heat=pred["heat"] if calibrate_heat else None,
                y_score=pred["score"],
                y_prob=pred["prob"],
                ks=ks,
            )
        )

        pred_frame = build_prediction_frame(
            group,
            y_pred_heat=pred["heat"] if calibrate_heat else None,
            y_score=pred["score"],
            y_prob=pred["prob"],
        )
        pred_frame["task"] = task
        pred_frame["model"] = model_name
        pred_frame["eval_group"] = group_name
        pred_frame[config.heat_target_column] = group[config.heat_target_column].to_numpy()
        pred_frame[config.emergence_label_column] = group[
            config.emergence_label_column
        ].to_numpy()
        prediction_frames.append(pred_frame)

    return BaselineResult(
        task=task,
        model=model_name,
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else None,
        metadata={"baseline_type": "heuristic", "heuristic": heuristic},
    )


def run_regression_baseline(
    task_frame: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    model_factory: Callable[[], Any],
    feature_columns: Sequence[str],
    max_train_rows: int | None = None,
    random_state: int = 42,
    ks: Sequence[int] = (100, 1000),
    stable_test_cutoffs: Sequence[int] = (2019, 2020),
    return_predictions: bool = False,
    eval_chunk_size: int = 10_000,
) -> BaselineResult:
    """Run a trainable regression baseline for heat forecasting.

    Memory notes:
      - Train frame is sampled before feature-matrix construction.
      - Evaluation predictions are computed in chunks.
      - Per-row prediction frames are disabled by default.
    """

    config = get_task_config(task)
    splits = split_task_frame(task_frame)

    train = sample_training_frame(
        splits["train"],
        max_rows=max_train_rows,
        random_state=random_state,
        stratify_column=config.emergence_label_column,
    )

    model = model_factory()

    # Fit once on train.
    X_train = prepare_feature_matrix(train, feature_columns)
    y_train = _as_float_array(train[config.heat_target_column]).astype("float32", copy=False)

    model.fit(X_train, y_train)

    train_rows_used = int(len(train))

    del X_train, y_train, train
    gc.collect()

    eval_slices = make_eval_slices(
        task_frame,
        stable_test_cutoffs=stable_test_cutoffs,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for group_name, group in eval_slices.items():
        pred_heat = predict_regression_model_in_chunks(
            model,
            group,
            feature_columns=feature_columns,
            chunk_size=eval_chunk_size,
            clip_nonnegative=True,
        )

        metric_rows.append(
            evaluate_predictions(
                group,
                task=task,
                model_name=model_name,
                eval_group=group_name,
                y_pred_heat=pred_heat,
                y_score=pred_heat,
                y_prob=None,
                ks=ks,
            )
        )

        if return_predictions:
            pred_frame = build_prediction_frame(
                group,
                y_pred_heat=pred_heat,
                y_score=pred_heat,
                y_prob=None,
            )
            pred_frame["task"] = task
            pred_frame["model"] = model_name
            pred_frame["eval_group"] = group_name
            pred_frame[config.heat_target_column] = group[
                config.heat_target_column
            ].to_numpy()
            pred_frame[config.emergence_label_column] = group[
                config.emergence_label_column
            ].to_numpy()
            prediction_frames.append(pred_frame)

        del pred_heat
        gc.collect()

    del model
    gc.collect()

    return BaselineResult(
        task=task,
        model=model_name,
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else None,
        metadata={
            "baseline_type": "regression",
            "max_train_rows": max_train_rows,
            "train_rows_used": train_rows_used,
            "feature_count": len(feature_columns),
            "eval_chunk_size": eval_chunk_size,
            "return_predictions": return_predictions,
        },
    )


def run_classification_baseline(
    task_frame: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    model_factory: Callable[[], Any],
    feature_columns: Sequence[str],
    max_train_rows: int | None = None,
    random_state: int = 42,
    ks: Sequence[int] = (100, 1000),
    stable_test_cutoffs: Sequence[int] = (2019, 2020),
    return_predictions: bool = False,
    eval_chunk_size: int = 10_000,
) -> BaselineResult:
    """Run a trainable classification baseline for emergence prediction.

    Classification probability is also used as ranking score. MAE/RMSE for heat
    are intentionally left empty because this model does not predict heat.

    Memory notes:
      - Train frame is sampled before feature-matrix construction.
      - Evaluation predictions are computed in chunks.
      - Per-row prediction frames are disabled by default.
    """

    config = get_task_config(task)
    splits = split_task_frame(task_frame)

    train = sample_training_frame(
        splits["train"],
        max_rows=max_train_rows,
        random_state=random_state,
        stratify_column=config.emergence_label_column,
    )

    model = model_factory()

    X_train = prepare_feature_matrix(train, feature_columns)
    y_train = _as_binary_array(train[config.emergence_label_column])

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Cannot train classifier {model_name}: train labels contain one class."
        )

    model.fit(X_train, y_train)

    train_rows_used = int(len(train))

    del X_train, y_train, train
    gc.collect()

    eval_slices = make_eval_slices(
        task_frame,
        stable_test_cutoffs=stable_test_cutoffs,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for group_name, group in eval_slices.items():
        y_prob = predict_classifier_probability_in_chunks(
            model,
            group,
            feature_columns=feature_columns,
            chunk_size=eval_chunk_size,
        )

        metric_rows.append(
            evaluate_predictions(
                group,
                task=task,
                model_name=model_name,
                eval_group=group_name,
                y_pred_heat=None,
                y_score=y_prob,
                y_prob=y_prob,
                ks=ks,
            )
        )

        if return_predictions:
            pred_frame = build_prediction_frame(
                group,
                y_pred_heat=None,
                y_score=y_prob,
                y_prob=y_prob,
            )
            pred_frame["task"] = task
            pred_frame["model"] = model_name
            pred_frame["eval_group"] = group_name
            pred_frame[config.heat_target_column] = group[
                config.heat_target_column
            ].to_numpy()
            pred_frame[config.emergence_label_column] = group[
                config.emergence_label_column
            ].to_numpy()
            prediction_frames.append(pred_frame)

        del y_prob
        gc.collect()

    del model
    gc.collect()

    return BaselineResult(
        task=task,
        model=model_name,
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else None,
        metadata={
            "baseline_type": "classification",
            "max_train_rows": max_train_rows,
            "train_rows_used": train_rows_used,
            "feature_count": len(feature_columns),
            "eval_chunk_size": eval_chunk_size,
            "return_predictions": return_predictions,
        },
    )


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------


def extract_linear_coefficients(
    pipeline: Any,
    *,
    top_n: int = 100,
) -> pd.DataFrame:
    """Extract approximate coefficient table from a fitted linear pipeline."""

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
    except Exception:
        return pd.DataFrame()

    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(model.coef_.ravel()))])

    coef = np.asarray(model.coef_).ravel()
    n = min(len(feature_names), len(coef))

    out = pd.DataFrame(
        {
            "feature": feature_names[:n],
            "coefficient": coef[:n],
            "abs_coefficient": np.abs(coef[:n]),
        }
    )

    return out.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def extract_tree_feature_importance(
    model: Any,
    *,
    feature_columns: Sequence[str],
    top_n: int = 100,
) -> pd.DataFrame:
    """Extract feature importance when available.

    For sklearn HistGradientBoosting models, native feature_importances_ is not
    available. This function returns empty DataFrame for such models.
    """

    estimator = model

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    values = np.asarray(estimator.feature_importances_, dtype=float)

    # For models without preprocessing, lengths may match raw feature columns.
    if len(values) == len(feature_columns):
        names = list(feature_columns)
    else:
        names = [f"feature_{i}" for i in range(len(values))]

    out = pd.DataFrame(
        {
            "feature": names,
            "importance": values,
        }
    )
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience model registry
# ---------------------------------------------------------------------------


def make_default_model_factory(
    model_name: str,
    feature_columns: Sequence[str],
    *,
    random_state: int = 42,
) -> tuple[str, Callable[[], Any]]:
    """Return baseline type and model factory for common model names.

    Returns:
        (model_kind, factory)

    model_kind:
        regression or classification
    """

    name = model_name.lower()

    if name in {"ridge", "ridge_regression"}:
        return (
            "regression",
            lambda: make_ridge_regression_pipeline(
                feature_columns,
                alpha=1.0,
                random_state=random_state,
            ),
        )

    if name in {"elasticnet", "elastic_net"}:
        return (
            "regression",
            lambda: make_elasticnet_regression_pipeline(
                feature_columns,
                alpha=0.001,
                l1_ratio=0.5,
                random_state=random_state,
            ),
        )

    if name in {"logistic", "logistic_regression", "logistic_balanced", "logistic_regression_balanced"}:
        return (
            "classification",
            lambda: make_logistic_regression_pipeline(
                feature_columns,
                class_weight="balanced",
                random_state=random_state,
            ),
        )

    if name in {"logistic_unweighted", "logistic_regression_unweighted"}:
        return (
            "classification",
            lambda: make_logistic_regression_pipeline(
                feature_columns,
                class_weight=None,
                random_state=random_state,
            ),
        )

    if name in {"histgb", "histgb_regressor", "hgb_regressor"}:
        return (
            "regression",
            lambda: make_hist_gradient_boosting_regressor_pipeline(
                feature_columns,
                random_state=random_state,
            ),
        )

    if name in {"histgb_classifier", "hgb_classifier"}:
        return (
            "classification",
            lambda: make_hist_gradient_boosting_classifier_pipeline(
                feature_columns,
                random_state=random_state,
            ),
        )

    if name in {"lightgbm", "lgbm", "lightgbm_regressor", "lgbm_regressor"}:
        def factory():
            model = try_make_lightgbm_regressor(random_state=random_state)
            if model is None:
                raise ImportError("lightgbm is not installed.")
            return model

        return ("regression", factory)

    if name in {"lightgbm_classifier", "lgbm_classifier"}:
        def factory_cls():
            model = try_make_lightgbm_classifier(random_state=random_state)
            if model is None:
                raise ImportError("lightgbm is not installed.")
            return model

        return ("classification", factory_cls)

    raise KeyError(f"Unknown trainable model name: {model_name}")


HEURISTIC_BASELINES: dict[str, str] = {
    "recency": "recency",
    "activity_paper": "activity_paper",
    "activity_total": "activity_total",
    "recent_3yr_paper": "recent_3yr_paper",
    "recent_3yr_total": "recent_3yr_total",
    "recent_5yr_paper": "recent_5yr_paper",
    "recent_5yr_total": "recent_5yr_total",
    "growth_paper_3yr": "growth_paper_3yr",
    "growth_total_3yr": "growth_total_3yr",
    "growth_paper_3yr_ratio": "growth_paper_3yr_ratio",
    "growth_total_3yr_ratio": "growth_total_3yr_ratio",
    "hybrid_activity_growth": "hybrid_activity_growth",
}


def is_heuristic_model(model_name: str) -> bool:
    return model_name.lower() in HEURISTIC_BASELINES


def is_constant_model(model_name: str) -> bool:
    return model_name.lower() in {"mean", "prevalence", "constant"}


def run_baseline_by_name(
    task_frame: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    feature_columns: Sequence[str],
    max_train_rows: int | None = None,
    random_state: int = 42,
    ks: Sequence[int] = (100, 1000),
    stable_test_cutoffs: Sequence[int] = (2019, 2020),
) -> BaselineResult:
    """Run one baseline by model name."""

    name = model_name.lower()

    if is_constant_model(name):
        return run_mean_baseline(
            task_frame,
            task=task,
            ks=ks,
            stable_test_cutoffs=stable_test_cutoffs,
        )

    if is_heuristic_model(name):
        heuristic = HEURISTIC_BASELINES[name]
        return run_heuristic_baseline(
            task_frame,
            task=task,
            heuristic=heuristic,
            model_name=name,
            calibrate_heat=True,
            ks=ks,
            stable_test_cutoffs=stable_test_cutoffs,
        )

    model_kind, factory = make_default_model_factory(
        name,
        feature_columns,
        random_state=random_state,
    )

    if model_kind == "regression":
        return run_regression_baseline(
            task_frame,
            task=task,
            model_name=name,
            model_factory=factory,
            feature_columns=feature_columns,
            max_train_rows=max_train_rows,
            random_state=random_state,
            ks=ks,
            stable_test_cutoffs=stable_test_cutoffs,
        )

    if model_kind == "classification":
        return run_classification_baseline(
            task_frame,
            task=task,
            model_name=name,
            model_factory=factory,
            feature_columns=feature_columns,
            max_train_rows=max_train_rows,
            random_state=random_state,
            ks=ks,
            stable_test_cutoffs=stable_test_cutoffs,
        )

    raise ValueError(f"Unsupported model kind: {model_kind}")


__all__ = [
    "TaskConfig",
    "TASK_CONFIGS",
    "DEFAULT_EVAL_GROUPS",
    "BaselineResult",
    "HEURISTIC_BASELINES",
    "get_task_config",
    "load_feature_spec",
    "load_target_spec",
    "load_prediction_feature_table",
    "get_feature_columns",
    "required_columns_for_tasks",
    "select_task_frame",
    "split_task_frame",
    "make_eval_slices",
    "sample_training_frame",
    "build_preprocessor",
    "prepare_feature_matrix",
    "make_ridge_regression_pipeline",
    "make_elasticnet_regression_pipeline",
    "make_logistic_regression_pipeline",
    "make_hist_gradient_boosting_regressor_pipeline",
    "make_hist_gradient_boosting_classifier_pipeline",
    "try_make_lightgbm_regressor",
    "try_make_lightgbm_classifier",
    "heuristic_score",
    "percentile_calibrate_scores",
    "minmax_to_probability",
    "regression_metrics",
    "classification_metrics",
    "ranking_metrics",
    "evaluate_predictions",
    "evaluate_prediction_column",
    "predict_mean_baseline",
    "predict_heuristic_baseline",
    "fit_predict_regression_model",
    "fit_predict_classifier_model",
    "build_prediction_frame",
    "sort_baseline_metrics",
    "make_comparison_table",
    "build_baseline_report",
    "run_mean_baseline",
    "run_heuristic_baseline",
    "run_regression_baseline",
    "run_classification_baseline",
    "make_default_model_factory",
    "run_baseline_by_name",
    "write_json",
    "read_json",
    "write_dataframe",
]