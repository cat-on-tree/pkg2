#!/usr/bin/env python
"""Run the BioEntity overlap baseline for Patent-Paper link prediction.

This script evaluates a simple and interpretable baseline:

    score(Patent, Paper) = overlap between Patent BioEntities and Paper BioEntities

The score can be selected from several BioEntity-overlap or degree-based
features, including:

- shared_bioentity_count
- bioentity_jaccard
- weighted_shared_bioentity_min
- weighted_bioentity_jaccard
- weighted_bioentity_cosine
- source_context_degree
- target_context_degree

Example for the full dataset:

    python scripts/12_baseline_bioentity_overlap.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/bioentity_overlap_shared_count \
      --score-column shared_bioentity_count \
      --export-features \
      --overwrite

Example for the no-species dataset:

    python scripts/12_baseline_bioentity_overlap.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/bioentity_overlap_shared_count \
      --score-column shared_bioentity_count \
      --export-features \
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
    BIOENTITY_SCORE_COLUMNS,
    DEFAULT_SPLITS,
    build_bioentity_overlap_baseline_tables,
    export_feature_tables,
    export_prediction_tables,
    fetch_prediction_rows_for_splits,
    summarize_bioentity_overlap_features,
    summarize_score_distribution,
)
from pkg2.io import connect_duckdb, markdown_table, write_csv, write_text  # noqa: E402
from pkg2.metrics import (  # noqa: E402
    evaluate_prediction_rows_by_split,
    write_metrics_csv,
    write_metrics_summary_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run BioEntity overlap baseline for Patent-Paper link prediction."
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
        "--score-column",
        default="shared_bioentity_count",
        choices=sorted(BIOENTITY_SCORE_COLUMNS),
        help="Feature column used as prediction score.",
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate. Default: train val test",
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
            "Optional BioEntity types to exclude inside the baseline. "
            "Usually not needed if the dataset has already been filtered."
        ),
    )

    parser.add_argument(
        "--export-features",
        action="store_true",
        help="Export feature tables in addition to prediction tables.",
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
        help="Overwrite the output directory if it already exists.",
    )

    return parser.parse_args()


def prepare_result_dir(output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Create an output result directory."""

    output_dir = Path(output_dir)

    if output_dir.exists():
        if not overwrite:
            existing = list(output_dir.iterdir())
            if existing:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Use --overwrite to replace it."
                )
        else:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _normalize_optional_list(values: list[str] | None) -> list[str]:
    """Normalize argparse optional list values."""

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


def _write_feature_summary_csv(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    file_name: str = "feature_summary.csv",
) -> Path:
    """Write feature summary rows to CSV."""

    output_path = output_dir / file_name

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


def _write_score_distribution_csv(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    file_name: str = "score_distribution.csv",
) -> Path:
    """Write score distribution rows to CSV."""

    output_path = output_dir / file_name

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


def _write_manifest(
    output_dir: Path,
    artifact_rows: list[dict[str, Any]],
    *,
    file_name: str = "baseline_manifest.csv",
) -> Path:
    """Write baseline artifact manifest."""

    output_path = output_dir / file_name

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


def _build_report(
    *,
    dataset_dir: Path,
    output_dir: Path,
    score_column: str,
    splits: list[str],
    k_values: list[int],
    table_summary: dict[str, Any],
    metrics_by_split: dict[str, dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    score_distribution: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> str:
    """Build a Markdown baseline report."""

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

    table_rows = []

    for key, value in table_summary.items():
        table_rows.append(
            {
                "name": key,
                "value": _format_value(value),
            }
        )

    lines = [
        "# BioEntity Overlap Baseline Report",
        "",
        "## 1. Baseline description",
        "",
        "This baseline scores each Patent-Paper candidate pair using BioEntity overlap features derived from the leakage-controlled context graph.",
        "",
        "The context graph is expected to exclude the target edge table, so direct Patent-Paper target links are not used as context evidence.",
        "",
        "## 2. Parameters",
        "",
        markdown_table(
            [
                {"parameter": "dataset_dir", "value": str(dataset_dir)},
                {"parameter": "output_dir", "value": str(output_dir)},
                {"parameter": "score_column", "value": score_column},
                {"parameter": "splits", "value": ", ".join(splits)},
                {"parameter": "k_values", "value": ", ".join(str(k) for k in k_values)},
            ],
            ["parameter", "value"],
        ),
        "",
        "## 3. Constructed table counts",
        "",
        markdown_table(table_rows, ["name", "value"]),
        "",
        "## 4. Metrics",
        "",
        markdown_table(metric_rows, ["split", *selected_metrics]),
        "",
        "## 5. Feature summary by split and label",
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
        "## 6. Score distribution by split and label",
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
        "## 7. Output artifacts",
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
        "## 8. Notes",
        "",
        "- Larger scores indicate higher predicted likelihood of a Patent-Paper link.",
        "- This baseline is intentionally simple and interpretable.",
        "- It is mainly used to test whether shared BioEntity context carries signal for Patent-Paper link prediction.",
        "",
    ]

    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    *,
    report_text: str,
    file_name: str = "baseline_report.md",
) -> Path:
    """Write Markdown baseline report."""

    output_path = output_dir / file_name
    write_text(output_path, report_text)
    return output_path


def _artifact_row(
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


def _print_summary(
    *,
    dataset_dir: Path,
    output_dir: Path,
    score_column: str,
    metrics_by_split: dict[str, dict[str, Any]],
) -> None:
    """Print a compact summary."""

    print()
    print("BioEntity overlap baseline complete.")
    print()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Score column:      {score_column}")
    print()

    print("Metrics:")
    for split in ["train", "val", "test"]:
        if split not in metrics_by_split:
            continue

        metrics = metrics_by_split[split]
        print(
            f"  {split}: "
            f"AUROC={metrics.get('auroc'):.6f}, "
            f"AUPRC={metrics.get('auprc'):.6f}, "
            f"P@100={metrics.get('precision_at_100'):.6f}, "
            f"R@100={metrics.get('recall_at_100'):.6f}"
        )

    print()
    print(f"Baseline report: {output_dir / 'baseline_report.md'}")
    print(f"Metrics summary: {output_dir / 'metrics_summary.csv'}")
    print()


def main() -> None:
    """Run the BioEntity overlap baseline."""

    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = prepare_result_dir(args.output_dir, overwrite=args.overwrite)

    splits = [str(split).strip() for split in args.splits if str(split).strip()]
    k_values = sorted({int(k) for k in args.k_values if int(k) > 0})
    exclude_bioentity_types = _normalize_optional_list(args.exclude_bioentity_types)

    connection = connect_duckdb(
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
    )

    table_summary = build_bioentity_overlap_baseline_tables(
        connection,
        dataset_dir=dataset_dir,
        splits=splits,
        score_column=args.score_column,
        exclude_bioentity_types=exclude_bioentity_types,
        load_views=True,
    )

    prediction_rows = fetch_prediction_rows_for_splits(
        connection,
        splits=splits,
        prediction_table_prefix="bioentity_overlap_predictions",
    )

    metrics_by_split = evaluate_prediction_rows_by_split(
        prediction_rows,
        split_column="split",
        label_column="label",
        score_column="score",
        k_values=k_values,
        threshold=None,
    )

    feature_summary = summarize_bioentity_overlap_features(
        connection,
        splits=splits,
        feature_table_prefix="bioentity_overlap_features",
    )

    score_distribution = summarize_score_distribution(
        connection,
        splits=splits,
        prediction_table_prefix="bioentity_overlap_predictions",
    )

    artifact_rows = []

    prediction_artifacts = export_prediction_tables(
        connection,
        output_dir=output_dir,
        splits=splits,
        prediction_table_prefix="bioentity_overlap_predictions",
        file_prefix="predictions",
        overwrite=True,
    )
    artifact_rows.extend(prediction_artifacts)

    if args.export_features:
        feature_artifacts = export_feature_tables(
            connection,
            output_dir=output_dir,
            splits=splits,
            feature_table_prefix="bioentity_overlap_features",
            file_prefix="features",
            overwrite=True,
        )
        artifact_rows.extend(feature_artifacts)

    metrics_csv_path = output_dir / "metrics.csv"
    write_metrics_csv(
        metrics_csv_path,
        metrics_by_split,
        model_name=f"bioentity_overlap:{args.score_column}",
    )
    artifact_rows.append(
        _artifact_row(
            artifact="metrics",
            path=metrics_csv_path,
            description="Long-form evaluation metrics.",
        )
    )

    metrics_summary_path = output_dir / "metrics_summary.csv"
    write_metrics_summary_csv(metrics_summary_path, metrics_by_split)
    artifact_rows.append(
        _artifact_row(
            artifact="metrics_summary",
            path=metrics_summary_path,
            description="Compact evaluation metrics summary.",
        )
    )

    feature_summary_path = _write_feature_summary_csv(output_dir, feature_summary)
    artifact_rows.append(
        _artifact_row(
            artifact="feature_summary",
            path=feature_summary_path,
            description="BioEntity overlap feature summary by split and label.",
            row_count=len(feature_summary),
        )
    )

    score_distribution_path = _write_score_distribution_csv(output_dir, score_distribution)
    artifact_rows.append(
        _artifact_row(
            artifact="score_distribution",
            path=score_distribution_path,
            description="Score distribution summary by split and label.",
            row_count=len(score_distribution),
        )
    )

    manifest_path = _write_manifest(output_dir, artifact_rows)
    artifact_rows.append(
        _artifact_row(
            artifact="baseline_manifest",
            path=manifest_path,
            description="Manifest of baseline output artifacts.",
            row_count=len(artifact_rows),
        )
    )

    report_text = _build_report(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        score_column=args.score_column,
        splits=splits,
        k_values=k_values,
        table_summary=table_summary,
        metrics_by_split=metrics_by_split,
        feature_summary=feature_summary,
        score_distribution=score_distribution,
        artifact_rows=artifact_rows,
    )

    report_path = _write_report(output_dir, report_text=report_text)
    artifact_rows.append(
        _artifact_row(
            artifact="baseline_report",
            path=report_path,
            description="Markdown report for the BioEntity overlap baseline.",
        )
    )

    _write_manifest(output_dir, artifact_rows)

    _print_summary(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        score_column=args.score_column,
        metrics_by_split=metrics_by_split,
    )


if __name__ == "__main__":
    main()