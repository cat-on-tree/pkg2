#!/usr/bin/env python
"""Run tree-based baselines for Patent-Paper link prediction.

This script trains feature-based tree models using BioEntity-overlap and degree
features.

Supported models:

- random_forest
- hist_gradient_boosting
- xgboost
- lightgbm

The script reuses feature construction logic from:

    src/pkg2/baselines.py

and evaluation logic from:

    src/pkg2/metrics.py

Recommended first run:

    python scripts/15_baseline_tree_models.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/tree_models_no_product \
      --feature-set no_product \
      --models random_forest hist_gradient_boosting xgboost lightgbm \
      --threads 4 \
      --memory-limit 20GB \
      --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.baselines import (  # noqa: E402
    DEFAULT_SPLITS,
    build_bioentity_overlap_baseline_tables,
    export_table_to_parquet,
    feature_table_name,
    prediction_table_name,
    summarize_bioentity_overlap_features,
    summarize_score_distribution,
)
from pkg2.io import (  # noqa: E402
    connect_duckdb,
    count_rows,
    markdown_table,
    qident,
    write_csv,
    write_text,
)
from pkg2.metrics import (  # noqa: E402
    evaluate_prediction_rows_by_split,
    write_metrics_csv,
    write_metrics_summary_csv,
)


OVERLAP_FEATURE_COLUMNS = [
    "shared_bioentity_count",
    "bioentity_jaccard",
    "weighted_shared_bioentity_min",
    "weighted_bioentity_jaccard",
    "weighted_bioentity_cosine",
]

DEGREE_FEATURE_COLUMNS = [
    "source_bioentity_count",
    "target_bioentity_count",
    "source_context_degree",
    "target_context_degree",
    "source_target_degree_product",
]

NO_PRODUCT_FEATURE_COLUMNS = [
    "shared_bioentity_count",
    "bioentity_jaccard",
    "weighted_shared_bioentity_min",
    "weighted_bioentity_jaccard",
    "weighted_bioentity_cosine",
    "source_bioentity_count",
    "target_bioentity_count",
    "source_context_degree",
    "target_context_degree",
]

ALL_FEATURE_COLUMNS = [
    "shared_bioentity_count",
    "bioentity_jaccard",
    "weighted_shared_bioentity_min",
    "weighted_bioentity_jaccard",
    "weighted_bioentity_cosine",
    "source_bioentity_count",
    "target_bioentity_count",
    "source_context_degree",
    "target_context_degree",
    "source_target_degree_product",
]

FEATURE_SETS = {
    "overlap_only": OVERLAP_FEATURE_COLUMNS,
    "degree_only": DEGREE_FEATURE_COLUMNS,
    "no_product": NO_PRODUCT_FEATURE_COLUMNS,
    "all": ALL_FEATURE_COLUMNS,
}

SUPPORTED_MODELS = [
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
]

IDENTIFIER_COLUMNS = [
    "source_index",
    "target_index",
    "source_type",
    "source_id",
    "target_type",
    "target_id",
    "split",
    "label",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run tree-based baselines for Patent-Paper link prediction."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Input link prediction dataset directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory. Each model will write results into a subdirectory "
            "under this path."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate. Default: train val test",
    )

    parser.add_argument(
        "--feature-set",
        default="no_product",
        choices=sorted(FEATURE_SETS),
        help=(
            "Predefined feature set. Default: no_product. "
            "Ignored if --feature-columns is provided."
        ),
    )

    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help="Explicit feature columns. Overrides --feature-set.",
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=["random_forest", "hist_gradient_boosting"],
        choices=SUPPORTED_MODELS,
        help=(
            "Models to run. Default: random_forest hist_gradient_boosting. "
            "Use all four names to include xgboost and lightgbm."
        ),
    )

    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=[10, 50, 100, 500, 1000],
        help="K values for Precision@K / Recall@K. Default: 10 50 100 500 1000",
    )

    parser.add_argument(
        "--exclude-bioentity-types",
        nargs="*",
        default=None,
        help=(
            "Optional BioEntity types to exclude inside feature construction. "
            "Usually not needed if the dataset has already been filtered."
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees / boosting iterations. Default: 300",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help=(
            "Maximum tree depth for RandomForest and XGBoost. "
            "For LightGBM this is passed as max_depth. Default: 8"
        ),
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=20,
        help="Minimum samples per leaf for RandomForest. Default: 20",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for boosting models. Default: 0.05",
    )

    parser.add_argument(
        "--subsample",
        type=float,
        default=0.9,
        help="Subsample ratio for XGBoost / LightGBM. Default: 0.9",
    )

    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.9,
        help="Column subsample ratio for XGBoost / LightGBM. Default: 0.9",
    )

    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Number of leaves for LightGBM. Default: 31",
    )

    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=31,
        help="Maximum leaf nodes for sklearn HistGradientBoosting. Default: 31",
    )

    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=0.0,
        help="L2 regularization for HistGradientBoosting. Default: 0.0",
    )

    parser.add_argument(
        "--class-weight",
        default=None,
        choices=[None, "balanced"],
        help=(
            "Class weight for RandomForest. For XGBoost/LightGBM the dataset is "
            "balanced by default, so this is usually unnecessary."
        ),
    )

    parser.add_argument(
        "--export-features",
        action="store_true",
        help="Export feature tables into the top-level output directory.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB and model thread count. Default: 1",
    )

    parser.add_argument(
        "--memory-limit",
        default="20GB",
        help="DuckDB memory limit. Default: 20GB",
    )

    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/interim/duckdb_tmp"),
        help="DuckDB temporary directory. Default: data/interim/duckdb_tmp",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    parser.add_argument(
        "--fail-on-missing-optional",
        action="store_true",
        help=(
            "Fail if xgboost or lightgbm is requested but not installed. "
            "By default missing optional models are skipped."
        ),
    )

    return parser.parse_args()


def prepare_result_dir(output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Create an output result directory."""

    output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            existing = list(output_dir.iterdir())
            if existing:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Use --overwrite to replace it."
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_subdir(parent_dir: Path, name: str) -> Path:
    """Create a model-specific output subdirectory."""

    path = parent_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_optional_list(values: list[str] | None) -> list[str]:
    """Normalize optional list arguments."""

    if not values:
        return []

    return [str(value).strip() for value in values if str(value).strip()]


def _format_value(value: Any) -> str:
    """Format values for Markdown tables."""

    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def resolve_feature_columns(
    *,
    feature_set: str,
    feature_columns: list[str] | None,
) -> list[str]:
    """Resolve feature columns from predefined set or explicit columns."""

    if feature_columns:
        selected = [str(column).strip() for column in feature_columns if str(column).strip()]
    else:
        selected = list(FEATURE_SETS[feature_set])

    allowed = set(ALL_FEATURE_COLUMNS)
    unsupported = sorted(set(selected) - allowed)

    if unsupported:
        raise ValueError(
            "Unsupported feature columns: "
            + ", ".join(unsupported)
            + ". Supported columns: "
            + ", ".join(ALL_FEATURE_COLUMNS)
        )

    if not selected:
        raise ValueError("At least one feature column is required.")

    return selected


def fetch_feature_frame(
    connection: Any,
    *,
    split: str,
    feature_columns: list[str],
    feature_table_prefix: str = "bioentity_overlap_features",
) -> Any:
    """Fetch one split feature table as a pandas DataFrame."""

    table_name = feature_table_name(split, prefix=feature_table_prefix)

    selected_columns = IDENTIFIER_COLUMNS + feature_columns
    selected_sql = ",\n            ".join(qident(column) for column in selected_columns)

    return connection.execute(
        f"""
        SELECT
            {selected_sql}
        FROM {qident(table_name)}
        ORDER BY source_id, target_id, label DESC
        """
    ).fetchdf()


def fetch_feature_frames_for_splits(
    connection: Any,
    *,
    splits: list[str],
    feature_columns: list[str],
    feature_table_prefix: str = "bioentity_overlap_features",
) -> dict[str, Any]:
    """Fetch feature DataFrames for multiple splits."""

    return {
        split: fetch_feature_frame(
            connection,
            split=split,
            feature_columns=feature_columns,
            feature_table_prefix=feature_table_prefix,
        )
        for split in splits
    }


def _import_sklearn_common() -> tuple[Any, Any]:
    """Import sklearn common preprocessing components."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for tree model baselines. "
            "Install it with: pip install scikit-learn"
        ) from exc

    return SimpleImputer, Pipeline


def make_model(
    model_name: str,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    num_leaves: int,
    max_leaf_nodes: int,
    l2_regularization: float,
    class_weight: str | None,
    threads: int,
) -> Any:
    """Create one model pipeline."""

    SimpleImputer, Pipeline = _import_sklearn_common()

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=threads,
            random_state=42,
        )

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=42,
        )

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            ) from exc

        estimator = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=threads,
            random_state=42,
        )

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install lightgbm"
            ) from exc

        estimator = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=threads,
            random_state=42,
            verbose=-1,
        )

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    raise ValueError(f"Unsupported model_name: {model_name!r}")


def train_model(
    model: Any,
    train_frame: Any,
    *,
    feature_columns: list[str],
) -> Any:
    """Train a model pipeline."""

    x_train = train_frame[feature_columns]
    y_train = train_frame["label"].astype(int)

    model.fit(x_train, y_train)
    return model


def predict_frame(
    model: Any,
    frame: Any,
    *,
    feature_columns: list[str],
) -> Any:
    """Add prediction scores to a feature DataFrame."""

    prediction_frame = frame[IDENTIFIER_COLUMNS].copy()

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(frame[feature_columns])[:, 1]
    else:
        raw_scores = model.decision_function(frame[feature_columns])
        scores = raw_scores

    prediction_frame["score"] = scores
    return prediction_frame


def create_prediction_table_from_frame(
    connection: Any,
    *,
    prediction_frame: Any,
    split: str,
    table_prefix: str,
) -> str:
    """Register a pandas prediction frame as a DuckDB temp table."""

    table_name = prediction_table_name(split, prefix=table_prefix)
    view_name = f"_{table_name}_frame"

    connection.register(view_name, prediction_frame)

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {qident(table_name)} AS
        SELECT
            source_index,
            target_index,
            source_type,
            source_id,
            target_type,
            target_id,
            split,
            CAST(label AS INTEGER) AS label,
            CAST(score AS DOUBLE) AS score
        FROM {qident(view_name)}
        """
    )

    connection.unregister(view_name)

    return table_name


def create_prediction_tables(
    connection: Any,
    *,
    model: Any,
    frames_by_split: dict[str, Any],
    splits: list[str],
    feature_columns: list[str],
    table_prefix: str,
) -> dict[str, int]:
    """Create prediction tables for all splits."""

    counts = {}

    for split in splits:
        prediction_frame = predict_frame(
            model,
            frames_by_split[split],
            feature_columns=feature_columns,
        )

        table_name = create_prediction_table_from_frame(
            connection,
            prediction_frame=prediction_frame,
            split=split,
            table_prefix=table_prefix,
        )

        counts[table_name] = count_rows(connection, table_name)

    return counts


def fetch_prediction_rows_for_splits(
    connection: Any,
    *,
    splits: list[str],
    prediction_table_prefix: str,
) -> list[dict[str, Any]]:
    """Fetch prediction rows from multiple split prediction tables."""

    rows: list[dict[str, Any]] = []

    for split in splits:
        table_name = prediction_table_name(split, prefix=prediction_table_prefix)

        split_rows = connection.execute(
            f"""
            SELECT
                source_index,
                target_index,
                source_type,
                source_id,
                target_type,
                target_id,
                split,
                label,
                score
            FROM {qident(table_name)}
            ORDER BY split, score DESC, source_id, target_id
            """
        ).fetchall()

        columns = [
            "source_index",
            "target_index",
            "source_type",
            "source_id",
            "target_type",
            "target_id",
            "split",
            "label",
            "score",
        ]

        rows.extend(dict(zip(columns, row)) for row in split_rows)

    return rows


def export_prediction_tables(
    connection: Any,
    *,
    output_dir: Path,
    splits: list[str],
    prediction_table_prefix: str,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Export prediction tables to Parquet."""

    artifacts = []

    for split in splits:
        table_name = prediction_table_name(split, prefix=prediction_table_prefix)
        output_path = output_dir / f"predictions_{split}.parquet"

        export_table_to_parquet(
            connection,
            table_name=table_name,
            output_path=output_path,
            overwrite=overwrite,
        )

        artifacts.append(
            {
                "artifact": table_name,
                "path": str(output_path),
                "description": f"Prediction scores for the {split} split.",
                "row_count": count_rows(connection, table_name),
                "file_size_bytes": output_path.stat().st_size if output_path.exists() else "",
            }
        )

    return artifacts


def export_feature_tables(
    connection: Any,
    *,
    output_dir: Path,
    splits: list[str],
    feature_table_prefix: str = "bioentity_overlap_features",
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Export feature tables to Parquet."""

    artifacts = []

    for split in splits:
        table_name = feature_table_name(split, prefix=feature_table_prefix)
        output_path = output_dir / f"features_{split}.parquet"

        export_table_to_parquet(
            connection,
            table_name=table_name,
            output_path=output_path,
            overwrite=overwrite,
        )

        artifacts.append(
            {
                "artifact": table_name,
                "path": str(output_path),
                "description": f"BioEntity overlap feature table for the {split} split.",
                "row_count": count_rows(connection, table_name),
                "file_size_bytes": output_path.stat().st_size if output_path.exists() else "",
            }
        )

    return artifacts


def extract_feature_importance(
    model: Any,
    *,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Extract feature importance if the estimator exposes it."""

    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        return [
            {
                "feature": feature,
                "importance": "",
                "rank": "",
            }
            for feature in feature_columns
        ]

    importances = list(estimator.feature_importances_)

    rows = [
        {
            "feature": feature,
            "importance": float(importance),
            "rank": 0,
        }
        for feature, importance in zip(feature_columns, importances)
    ]

    rows.sort(key=lambda row: float(row["importance"]), reverse=True)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def write_feature_importance_csv(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write feature importance rows."""

    output_path = output_dir / "feature_importance.csv"

    write_csv(
        output_path,
        rows,
        columns=["rank", "feature", "importance"],
    )

    return output_path


def write_feature_summary_csv(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write feature summary rows."""

    output_path = output_dir / "feature_summary.csv"

    write_csv(
        output_path,
        rows,
        columns=[
            "split",
            "label",
            "example_count",
            "avg_shared_bioentity_count",
            "avg_bioentity_jaccard",
            "avg_weighted_shared_bioentity_min",
            "avg_weighted_bioentity_jaccard",
            "avg_weighted_bioentity_cosine",
            "avg_source_bioentity_count",
            "avg_target_bioentity_count",
            "avg_source_context_degree",
            "avg_target_context_degree",
        ],
    )

    return output_path


def write_score_distribution_csv(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write score distribution rows."""

    output_path = output_dir / "score_distribution.csv"

    write_csv(
        output_path,
        rows,
        columns=[
            "split",
            "label",
            "example_count",
            "score_min",
            "score_max",
            "score_mean",
            "score_q25",
            "score_median",
            "score_q75",
        ],
    )

    return output_path


def artifact_row(
    *,
    artifact: str,
    path: Path,
    description: str,
    row_count: int | str = "",
) -> dict[str, Any]:
    """Create one artifact manifest row."""

    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
    }


def write_manifest(
    output_dir: Path,
    artifact_rows: list[dict[str, Any]],
) -> Path:
    """Write baseline manifest."""

    output_path = output_dir / "baseline_manifest.csv"

    write_csv(
        output_path,
        artifact_rows,
        columns=[
            "artifact",
            "path",
            "description",
            "row_count",
            "file_size_bytes",
        ],
    )

    return output_path


def build_model_report(
    *,
    model_name: str,
    dataset_dir: Path,
    output_dir: Path,
    splits: list[str],
    feature_set: str,
    feature_columns: list[str],
    model_parameters: dict[str, Any],
    table_summary: dict[str, Any],
    prediction_counts: dict[str, int],
    metrics_by_split: dict[str, dict[str, Any]],
    feature_importance_rows: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    score_distribution: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> str:
    """Build Markdown report for one model."""

    selected_metrics = [
        "example_count",
        "positive_count",
        "negative_count",
        "positive_ratio",
        "score_min",
        "score_max",
        "score_mean",
        "auroc",
        "auprc",
        "mrr",
        "precision_at_100",
        "recall_at_100",
        "precision_at_1000",
        "recall_at_1000",
    ]

    metric_rows = []

    for split in splits:
        split_metrics = metrics_by_split.get(split, {})
        row = {"split": split}

        for metric in selected_metrics:
            row[metric] = _format_value(split_metrics.get(metric, ""))

        metric_rows.append(row)

    parameter_rows = [
        {"parameter": "model_name", "value": model_name},
        {"parameter": "dataset_dir", "value": str(dataset_dir)},
        {"parameter": "output_dir", "value": str(output_dir)},
        {"parameter": "splits", "value": ", ".join(splits)},
        {"parameter": "feature_set", "value": feature_set},
        {"parameter": "feature_columns", "value": ", ".join(feature_columns)},
    ]

    for key, value in model_parameters.items():
        parameter_rows.append({"parameter": key, "value": value})

    table_rows = []

    for key, value in table_summary.items():
        table_rows.append({"name": key, "value": _format_value(value)})

    for key, value in prediction_counts.items():
        table_rows.append({"name": key, "value": _format_value(value)})

    formatted_feature_importance = [
        {
            "rank": _format_value(row.get("rank", "")),
            "feature": row.get("feature", ""),
            "importance": _format_value(row.get("importance", "")),
        }
        for row in feature_importance_rows
    ]

    formatted_feature_summary = [
        {key: _format_value(value) for key, value in row.items()}
        for row in feature_summary
    ]

    formatted_score_distribution = [
        {key: _format_value(value) for key, value in row.items()}
        for row in score_distribution
    ]

    formatted_artifacts = [
        {key: _format_value(value) for key, value in row.items()}
        for row in artifact_rows
    ]

    lines = [
        f"# Tree Model Baseline Report: {model_name}",
        "",
        "## 1. Baseline description",
        "",
        "This baseline trains a tree-based classifier on BioEntity-overlap and degree features for Patent-Paper link prediction.",
        "",
        "Input features are constructed from the leakage-controlled context graph. The target Patent-Paper edge table is not used as context evidence.",
        "",
        "## 2. Parameters",
        "",
        markdown_table(parameter_rows, ["parameter", "value"]),
        "",
        "## 3. Constructed table counts",
        "",
        markdown_table(table_rows, ["name", "value"]),
        "",
        "## 4. Metrics",
        "",
        markdown_table(metric_rows, ["split", *selected_metrics]),
        "",
        "## 5. Feature importance",
        "",
        markdown_table(
            formatted_feature_importance,
            ["rank", "feature", "importance"],
        ),
        "",
        "## 6. Feature summary by split and label",
        "",
        markdown_table(
            formatted_feature_summary,
            [
                "split",
                "label",
                "example_count",
                "avg_shared_bioentity_count",
                "avg_bioentity_jaccard",
                "avg_weighted_shared_bioentity_min",
                "avg_weighted_bioentity_jaccard",
                "avg_weighted_bioentity_cosine",
                "avg_source_bioentity_count",
                "avg_target_bioentity_count",
                "avg_source_context_degree",
                "avg_target_context_degree",
            ],
        ),
        "",
        "## 7. Score distribution by split and label",
        "",
        markdown_table(
            formatted_score_distribution,
            [
                "split",
                "label",
                "example_count",
                "score_min",
                "score_max",
                "score_mean",
                "score_q25",
                "score_median",
                "score_q75",
            ],
        ),
        "",
        "## 8. Output artifacts",
        "",
        markdown_table(
            formatted_artifacts,
            [
                "artifact",
                "path",
                "description",
                "row_count",
                "file_size_bytes",
            ],
        ),
        "",
        "## 9. Notes",
        "",
        "- Scores are predicted probabilities for label 1.",
        "- Features are median-imputed before model training.",
        "- Tree feature importance is reported when exposed by the model.",
        "- Larger scores indicate higher predicted likelihood of a Patent-Paper link.",
        "",
    ]

    return "\n".join(lines)


def write_report(output_dir: Path, report_text: str) -> Path:
    """Write Markdown report."""

    output_path = output_dir / "baseline_report.md"
    write_text(output_path, report_text)
    return output_path


def run_one_model(
    *,
    connection: Any,
    model_name: str,
    dataset_dir: Path,
    model_output_dir: Path,
    splits: list[str],
    feature_set: str,
    feature_columns: list[str],
    frames_by_split: dict[str, Any],
    table_summary: dict[str, Any],
    feature_summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train and evaluate one model."""

    print()
    print(f"Running tree model: {model_name}")
    print(f"Output directory:   {model_output_dir}")
    print()

    model = make_model(
        model_name,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        num_leaves=args.num_leaves,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        class_weight=args.class_weight,
        threads=args.threads,
    )

    model = train_model(
        model,
        frames_by_split["train"],
        feature_columns=feature_columns,
    )

    prediction_table_prefix = f"{model_name}_predictions"

    prediction_counts = create_prediction_tables(
        connection,
        model=model,
        frames_by_split=frames_by_split,
        splits=splits,
        feature_columns=feature_columns,
        table_prefix=prediction_table_prefix,
    )

    prediction_rows = fetch_prediction_rows_for_splits(
        connection,
        splits=splits,
        prediction_table_prefix=prediction_table_prefix,
    )

    metrics_by_split = evaluate_prediction_rows_by_split(
        prediction_rows,
        split_column="split",
        label_column="label",
        score_column="score",
        k_values=sorted({int(k) for k in args.k_values if int(k) > 0}),
        threshold=None,
    )

    score_distribution = summarize_score_distribution(
        connection,
        splits=splits,
        prediction_table_prefix=prediction_table_prefix,
    )

    feature_importance_rows = extract_feature_importance(
        model,
        feature_columns=feature_columns,
    )

    artifact_rows = []

    prediction_artifacts = export_prediction_tables(
        connection,
        output_dir=model_output_dir,
        splits=splits,
        prediction_table_prefix=prediction_table_prefix,
        overwrite=True,
    )
    artifact_rows.extend(prediction_artifacts)

    metrics_csv_path = model_output_dir / "metrics.csv"
    write_metrics_csv(
        metrics_csv_path,
        metrics_by_split,
        model_name=f"{model_name}:{feature_set}",
    )
    artifact_rows.append(
        artifact_row(
            artifact="metrics",
            path=metrics_csv_path,
            description="Long-form evaluation metrics.",
        )
    )

    metrics_summary_path = model_output_dir / "metrics_summary.csv"
    write_metrics_summary_csv(metrics_summary_path, metrics_by_split)
    artifact_rows.append(
        artifact_row(
            artifact="metrics_summary",
            path=metrics_summary_path,
            description="Compact evaluation metrics summary.",
        )
    )

    feature_importance_path = write_feature_importance_csv(
        model_output_dir,
        feature_importance_rows,
    )
    artifact_rows.append(
        artifact_row(
            artifact="feature_importance",
            path=feature_importance_path,
            description="Tree model feature importance.",
            row_count=len(feature_importance_rows),
        )
    )

    feature_summary_path = write_feature_summary_csv(model_output_dir, feature_summary)
    artifact_rows.append(
        artifact_row(
            artifact="feature_summary",
            path=feature_summary_path,
            description="BioEntity overlap feature summary by split and label.",
            row_count=len(feature_summary),
        )
    )

    score_distribution_path = write_score_distribution_csv(model_output_dir, score_distribution)
    artifact_rows.append(
        artifact_row(
            artifact="score_distribution",
            path=score_distribution_path,
            description="Prediction score distribution summary by split and label.",
            row_count=len(score_distribution),
        )
    )

    manifest_path = write_manifest(model_output_dir, artifact_rows)
    artifact_rows.append(
        artifact_row(
            artifact="baseline_manifest",
            path=manifest_path,
            description="Manifest of tree model baseline output artifacts.",
            row_count=len(artifact_rows),
        )
    )

    model_parameters = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "num_leaves": args.num_leaves,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "class_weight": args.class_weight or "",
        "threads": args.threads,
        "k_values": ", ".join(str(k) for k in args.k_values),
    }

    report_text = build_model_report(
        model_name=model_name,
        dataset_dir=dataset_dir,
        output_dir=model_output_dir,
        splits=splits,
        feature_set=feature_set,
        feature_columns=feature_columns,
        model_parameters=model_parameters,
        table_summary=table_summary,
        prediction_counts=prediction_counts,
        metrics_by_split=metrics_by_split,
        feature_importance_rows=feature_importance_rows,
        feature_summary=feature_summary,
        score_distribution=score_distribution,
        artifact_rows=artifact_rows,
    )

    report_path = write_report(model_output_dir, report_text)
    artifact_rows.append(
        artifact_row(
            artifact="baseline_report",
            path=report_path,
            description=f"Markdown report for {model_name}.",
        )
    )

    write_manifest(model_output_dir, artifact_rows)

    test_metrics = metrics_by_split.get("test", {})
    print(
        f"{model_name} complete: "
        f"test AUROC={test_metrics.get('auroc'):.6f}, "
        f"test AUPRC={test_metrics.get('auprc'):.6f}"
    )

    return {
        "model": model_name,
        "output_dir": str(model_output_dir),
        "status": "completed",
        "test_auroc": test_metrics.get("auroc", ""),
        "test_auprc": test_metrics.get("auprc", ""),
    }


def build_overall_report(
    *,
    dataset_dir: Path,
    output_dir: Path,
    feature_set: str,
    feature_columns: list[str],
    model_rows: list[dict[str, Any]],
) -> str:
    """Build top-level report for all requested tree models."""

    formatted_rows = [
        {key: _format_value(value) for key, value in row.items()}
        for row in model_rows
    ]

    lines = [
        "# Tree Models Baseline Summary",
        "",
        "## 1. Parameters",
        "",
        markdown_table(
            [
                {"parameter": "dataset_dir", "value": str(dataset_dir)},
                {"parameter": "output_dir", "value": str(output_dir)},
                {"parameter": "feature_set", "value": feature_set},
                {"parameter": "feature_columns", "value": ", ".join(feature_columns)},
            ],
            ["parameter", "value"],
        ),
        "",
        "## 2. Model results",
        "",
        markdown_table(
            formatted_rows,
            ["model", "status", "test_auroc", "test_auprc", "output_dir"],
        ),
        "",
        "## 3. Notes",
        "",
        "- Each model writes its own `metrics_summary.csv`, so `scripts/13_collect_baseline_results.py` can collect them automatically.",
        "- Optional models such as XGBoost and LightGBM are skipped if their packages are unavailable, unless `--fail-on-missing-optional` is used.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    """Run tree-based baselines."""

    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = prepare_result_dir(args.output_dir, overwrite=args.overwrite)

    splits = [str(split).strip() for split in args.splits if str(split).strip()]
    feature_columns = resolve_feature_columns(
        feature_set=args.feature_set,
        feature_columns=args.feature_columns,
    )
    exclude_bioentity_types = _normalize_optional_list(args.exclude_bioentity_types)

    if "train" not in splits:
        raise ValueError("The train split is required to train tree models.")

    connection = connect_duckdb(
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
    )

    table_summary = build_bioentity_overlap_baseline_tables(
        connection,
        dataset_dir=dataset_dir,
        splits=splits,
        score_column="weighted_bioentity_cosine",
        exclude_bioentity_types=exclude_bioentity_types,
        load_views=True,
    )

    frames_by_split = fetch_feature_frames_for_splits(
        connection,
        splits=splits,
        feature_columns=feature_columns,
        feature_table_prefix="bioentity_overlap_features",
    )

    feature_summary = summarize_bioentity_overlap_features(
        connection,
        splits=splits,
        feature_table_prefix="bioentity_overlap_features",
    )

    if args.export_features:
        feature_output_dir = output_dir / "features"
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        export_feature_tables(
            connection,
            output_dir=feature_output_dir,
            splits=splits,
            feature_table_prefix="bioentity_overlap_features",
            overwrite=True,
        )

    model_rows = []

    for model_name in args.models:
        model_output_dir = prepare_subdir(output_dir, model_name)

        try:
            row = run_one_model(
                connection=connection,
                model_name=model_name,
                dataset_dir=dataset_dir,
                model_output_dir=model_output_dir,
                splits=splits,
                feature_set=args.feature_set,
                feature_columns=feature_columns,
                frames_by_split=frames_by_split,
                table_summary=table_summary,
                feature_summary=feature_summary,
                args=args,
            )
            model_rows.append(row)

        except ImportError as exc:
            if args.fail_on_missing_optional:
                raise

            print()
            print(f"Skipping model {model_name}: {exc}")
            print()

            model_rows.append(
                {
                    "model": model_name,
                    "output_dir": str(model_output_dir),
                    "status": f"skipped: {exc}",
                    "test_auroc": "",
                    "test_auprc": "",
                }
            )

        except Exception:
            print()
            print(f"Model failed: {model_name}")
            traceback.print_exc()
            print()

            model_rows.append(
                {
                    "model": model_name,
                    "output_dir": str(model_output_dir),
                    "status": "failed",
                    "test_auroc": "",
                    "test_auprc": "",
                }
            )

    completed = [row for row in model_rows if row.get("status") == "completed"]

    if not completed:
        raise RuntimeError("No tree model completed successfully.")

    summary_csv_path = output_dir / "tree_model_summary.csv"
    write_csv(
        summary_csv_path,
        model_rows,
        columns=["model", "status", "test_auroc", "test_auprc", "output_dir"],
    )

    overall_report_text = build_overall_report(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        feature_set=args.feature_set,
        feature_columns=feature_columns,
        model_rows=model_rows,
    )
    write_text(output_dir / "tree_model_summary_report.md", overall_report_text)

    print()
    print("Tree model baselines complete.")
    print()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Feature set:       {args.feature_set}")
    print()
    print("Completed models:")
    for row in completed:
        print(
            f"  {row['model']}: "
            f"test AUROC={_format_value(row.get('test_auroc'))}, "
            f"test AUPRC={_format_value(row.get('test_auprc'))}"
        )
    print()
    print(f"Summary CSV:    {summary_csv_path}")
    print(f"Summary report: {output_dir / 'tree_model_summary_report.md'}")
    print()


if __name__ == "__main__":
    main()