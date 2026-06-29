#!/usr/bin/env python
"""Run Logistic Regression baseline for Patent-Paper link prediction.

This script trains a feature-based Logistic Regression baseline using
BioEntity-overlap and degree features.

It reuses feature construction logic from:

    src/pkg2/baselines.py

and evaluation logic from:

    src/pkg2/metrics.py

Example for the full dataset:

    python scripts/14_baseline_logistic_regression.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/logistic_regression_bioentity_features \
      --threads 4 \
      --memory-limit 20GB \
      --overwrite

Example for the no-species dataset:

    python scripts/14_baseline_logistic_regression.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/logistic_regression_bioentity_features \
      --threads 4 \
      --memory-limit 20GB \
      --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
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


DEFAULT_FEATURE_COLUMNS = [
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
        description="Run Logistic Regression baseline for Patent-Paper link prediction."
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
        help="Output result directory.",
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate. Default: train val test",
    )

    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Feature columns used by Logistic Regression.",
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
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for LogisticRegression. Default: 1000",
    )

    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength C for LogisticRegression. Default: 1.0",
    )

    parser.add_argument(
        "--penalty",
        default="l2",
        choices=["l2", "none"],
        help="Penalty for LogisticRegression. Default: l2",
    )

    parser.add_argument(
        "--solver",
        default="lbfgs",
        help="Solver for LogisticRegression. Default: lbfgs",
    )

    parser.add_argument(
        "--class-weight",
        default=None,
        choices=[None, "balanced"],
        help="Class weight for LogisticRegression. Default: None",
    )

    parser.add_argument(
        "--export-features",
        action="store_true",
        help="Export Logistic Regression feature tables.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB thread count. Default: 1",
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


def _import_sklearn() -> tuple[Any, Any, Any, Any]:
    """Import sklearn components with a helpful error message."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for Logistic Regression baseline. "
            "Install it with: pip install scikit-learn"
        ) from exc

    return SimpleImputer, LogisticRegression, Pipeline, StandardScaler


def validate_feature_columns(feature_columns: list[str]) -> list[str]:
    """Validate requested feature columns."""

    cleaned = [str(column).strip() for column in feature_columns if str(column).strip()]

    if not cleaned:
        raise ValueError("At least one feature column is required.")

    allowed = set(DEFAULT_FEATURE_COLUMNS)

    unsupported = sorted(set(cleaned) - allowed)
    if unsupported:
        raise ValueError(
            "Unsupported feature columns: "
            + ", ".join(unsupported)
            + ". Supported columns: "
            + ", ".join(DEFAULT_FEATURE_COLUMNS)
        )

    return cleaned


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


def train_logistic_regression_model(
    train_frame: Any,
    *,
    feature_columns: list[str],
    max_iter: int,
    c_value: float,
    penalty: str,
    solver: str,
    class_weight: str | None,
) -> Any:
    """Train sklearn Logistic Regression pipeline."""

    SimpleImputer, LogisticRegression, Pipeline, StandardScaler = _import_sklearn()

    logistic_kwargs = {
        "C": c_value,
        "solver": solver,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "random_state": 42,
    }

    if penalty == "none":
        logistic_kwargs["C"] = float("inf")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(**logistic_kwargs),
            ),
        ]
    )

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
    """Add Logistic Regression scores to a feature DataFrame."""

    prediction_frame = frame[IDENTIFIER_COLUMNS].copy()

    probabilities = model.predict_proba(frame[feature_columns])[:, 1]
    prediction_frame["score"] = probabilities

    return prediction_frame


def create_prediction_table_from_frame(
    connection: Any,
    *,
    prediction_frame: Any,
    split: str,
    table_prefix: str = "logistic_regression_predictions",
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
    table_prefix: str = "logistic_regression_predictions",
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
    prediction_table_prefix: str = "logistic_regression_predictions",
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
    prediction_table_prefix: str = "logistic_regression_predictions",
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
                "description": f"Logistic Regression prediction scores for the {split} split.",
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


def extract_coefficients(
    model: Any,
    *,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Extract Logistic Regression coefficients from fitted sklearn pipeline."""

    logistic_model = model.named_steps["model"]

    coefficients = logistic_model.coef_[0]
    intercept = float(logistic_model.intercept_[0])

    rows = [
        {
            "feature": "__intercept__",
            "coefficient": intercept,
            "abs_coefficient": abs(intercept),
        }
    ]

    for feature, coefficient in zip(feature_columns, coefficients):
        coefficient = float(coefficient)
        rows.append(
            {
                "feature": feature,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
            }
        )

    rows.sort(key=lambda row: float(row["abs_coefficient"]), reverse=True)
    return rows


def write_coefficients_csv(
    output_dir: Path,
    coefficient_rows: list[dict[str, Any]],
) -> Path:
    """Write model coefficients to CSV."""

    output_path = output_dir / "model_coefficients.csv"

    write_csv(
        output_path,
        coefficient_rows,
        columns=["feature", "coefficient", "abs_coefficient"],
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


def build_report(
    *,
    dataset_dir: Path,
    output_dir: Path,
    splits: list[str],
    feature_columns: list[str],
    model_parameters: dict[str, Any],
    table_summary: dict[str, Any],
    prediction_counts: dict[str, int],
    metrics_by_split: dict[str, dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    score_distribution: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> str:
    """Build Markdown report."""

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
        {"parameter": "dataset_dir", "value": str(dataset_dir)},
        {"parameter": "output_dir", "value": str(output_dir)},
        {"parameter": "splits", "value": ", ".join(splits)},
        {"parameter": "feature_columns", "value": ", ".join(feature_columns)},
    ]

    for key, value in model_parameters.items():
        parameter_rows.append({"parameter": key, "value": value})

    table_rows = []

    for key, value in table_summary.items():
        table_rows.append({"name": key, "value": _format_value(value)})

    for key, value in prediction_counts.items():
        table_rows.append({"name": key, "value": _format_value(value)})

    formatted_coefficients = [
        {
            "feature": row["feature"],
            "coefficient": _format_value(row["coefficient"]),
            "abs_coefficient": _format_value(row["abs_coefficient"]),
        }
        for row in coefficient_rows
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
        "# Logistic Regression Baseline Report",
        "",
        "## 1. Baseline description",
        "",
        "This baseline trains a Logistic Regression classifier on BioEntity-overlap and degree features for Patent-Paper link prediction.",
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
        "## 5. Model coefficients",
        "",
        markdown_table(
            formatted_coefficients,
            ["feature", "coefficient", "abs_coefficient"],
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
        "- Features are median-imputed and standardized before Logistic Regression.",
        "- Coefficients are coefficients on standardized features.",
        "- Larger scores indicate higher predicted likelihood of a Patent-Paper link.",
        "",
    ]

    return "\n".join(lines)


def write_report(output_dir: Path, report_text: str) -> Path:
    """Write Markdown report."""

    output_path = output_dir / "baseline_report.md"
    write_text(output_path, report_text)
    return output_path


def print_summary(
    *,
    dataset_dir: Path,
    output_dir: Path,
    metrics_by_split: dict[str, dict[str, Any]],
) -> None:
    """Print compact terminal summary."""

    print()
    print("Logistic Regression baseline complete.")
    print()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print()

    print("Metrics:")

    for split in ["train", "val", "test"]:
        if split not in metrics_by_split:
            continue

        metrics = metrics_by_split[split]

        auroc = metrics.get("auroc")
        auprc = metrics.get("auprc")
        p100 = metrics.get("precision_at_100")
        r100 = metrics.get("recall_at_100")

        print(
            f"  {split}: "
            f"AUROC={auroc:.6f}, "
            f"AUPRC={auprc:.6f}, "
            f"P@100={p100:.6f}, "
            f"R@100={r100:.6f}"
        )

    print()
    print(f"Baseline report: {output_dir / 'baseline_report.md'}")
    print(f"Metrics summary: {output_dir / 'metrics_summary.csv'}")
    print()


def main() -> None:
    """Run Logistic Regression baseline."""

    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = prepare_result_dir(args.output_dir, overwrite=args.overwrite)

    splits = [str(split).strip() for split in args.splits if str(split).strip()]
    k_values = sorted({int(k) for k in args.k_values if int(k) > 0})
    feature_columns = validate_feature_columns(args.feature_columns)
    exclude_bioentity_types = _normalize_optional_list(args.exclude_bioentity_types)

    if "train" not in splits:
        raise ValueError("The train split is required to train Logistic Regression.")

    connection = connect_duckdb(
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
    )

    # Use weighted_bioentity_cosine as the temporary feature-table score column.
    # Logistic Regression will use all selected feature columns, so this does not
    # affect model training except for the intermediate feature table's score field.
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

    model = train_logistic_regression_model(
        frames_by_split["train"],
        feature_columns=feature_columns,
        max_iter=args.max_iter,
        c_value=args.c,
        penalty=args.penalty,
        solver=args.solver,
        class_weight=args.class_weight,
    )

    prediction_counts = create_prediction_tables(
        connection,
        model=model,
        frames_by_split=frames_by_split,
        splits=splits,
        feature_columns=feature_columns,
        table_prefix="logistic_regression_predictions",
    )

    prediction_rows = fetch_prediction_rows_for_splits(
        connection,
        splits=splits,
        prediction_table_prefix="logistic_regression_predictions",
    )

    metrics_by_split = evaluate_prediction_rows_by_split(
        prediction_rows,
        split_column="split",
        label_column="label",
        score_column="score",
        k_values=k_values,
        threshold=None,
    )

    coefficient_rows = extract_coefficients(model, feature_columns=feature_columns)

    feature_summary = summarize_bioentity_overlap_features(
        connection,
        splits=splits,
        feature_table_prefix="bioentity_overlap_features",
    )

    score_distribution = summarize_score_distribution(
        connection,
        splits=splits,
        prediction_table_prefix="logistic_regression_predictions",
    )

    artifact_rows = []

    prediction_artifacts = export_prediction_tables(
        connection,
        output_dir=output_dir,
        splits=splits,
        prediction_table_prefix="logistic_regression_predictions",
        overwrite=True,
    )
    artifact_rows.extend(prediction_artifacts)

    if args.export_features:
        feature_artifacts = export_feature_tables(
            connection,
            output_dir=output_dir,
            splits=splits,
            feature_table_prefix="bioentity_overlap_features",
            overwrite=True,
        )
        artifact_rows.extend(feature_artifacts)

    metrics_csv_path = output_dir / "metrics.csv"
    write_metrics_csv(
        metrics_csv_path,
        metrics_by_split,
        model_name="logistic_regression:bioentity_features",
    )
    artifact_rows.append(
        artifact_row(
            artifact="metrics",
            path=metrics_csv_path,
            description="Long-form evaluation metrics.",
        )
    )

    metrics_summary_path = output_dir / "metrics_summary.csv"
    write_metrics_summary_csv(metrics_summary_path, metrics_by_split)
    artifact_rows.append(
        artifact_row(
            artifact="metrics_summary",
            path=metrics_summary_path,
            description="Compact evaluation metrics summary.",
        )
    )

    coefficients_path = write_coefficients_csv(output_dir, coefficient_rows)
    artifact_rows.append(
        artifact_row(
            artifact="model_coefficients",
            path=coefficients_path,
            description="Logistic Regression coefficients on standardized features.",
            row_count=len(coefficient_rows),
        )
    )

    feature_summary_path = write_feature_summary_csv(output_dir, feature_summary)
    artifact_rows.append(
        artifact_row(
            artifact="feature_summary",
            path=feature_summary_path,
            description="BioEntity overlap feature summary by split and label.",
            row_count=len(feature_summary),
        )
    )

    score_distribution_path = write_score_distribution_csv(output_dir, score_distribution)
    artifact_rows.append(
        artifact_row(
            artifact="score_distribution",
            path=score_distribution_path,
            description="Prediction score distribution summary by split and label.",
            row_count=len(score_distribution),
        )
    )

    manifest_path = write_manifest(output_dir, artifact_rows)
    artifact_rows.append(
        artifact_row(
            artifact="baseline_manifest",
            path=manifest_path,
            description="Manifest of Logistic Regression baseline output artifacts.",
            row_count=len(artifact_rows),
        )
    )

    model_parameters = {
        "model": "LogisticRegression",
        "max_iter": args.max_iter,
        "C": args.c,
        "penalty": args.penalty,
        "solver": args.solver,
        "class_weight": args.class_weight or "",
        "k_values": ", ".join(str(k) for k in k_values),
    }

    report_text = build_report(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        splits=splits,
        feature_columns=feature_columns,
        model_parameters=model_parameters,
        table_summary=table_summary,
        prediction_counts=prediction_counts,
        metrics_by_split=metrics_by_split,
        coefficient_rows=coefficient_rows,
        feature_summary=feature_summary,
        score_distribution=score_distribution,
        artifact_rows=artifact_rows,
    )

    report_path = write_report(output_dir, report_text)
    artifact_rows.append(
        artifact_row(
            artifact="baseline_report",
            path=report_path,
            description="Markdown report for Logistic Regression baseline.",
        )
    )

    write_manifest(output_dir, artifact_rows)

    print_summary(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        metrics_by_split=metrics_by_split,
    )


if __name__ == "__main__":
    main()