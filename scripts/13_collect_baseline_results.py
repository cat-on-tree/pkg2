#!/usr/bin/env python
"""Collect baseline result summaries into comparison tables.

This script scans baseline output directories and collects metrics from each
``metrics_summary.csv`` file.

Expected directory structure
----------------------------

    data/results/link_prediction/patent_paper/
      diabetes_2018_2019_v1_full/
        bioentity_overlap_shared_count/
          metrics_summary.csv
          metrics.csv
          baseline_report.md

      diabetes_2018_2019_v1_no_species/
        bioentity_overlap_shared_count/
          metrics_summary.csv
          metrics.csv
          baseline_report.md

Outputs
-------

By default, this script writes to the results root:

    baseline_comparison.csv
    baseline_comparison_test.csv
    baseline_comparison_report.md

Example
-------

    python scripts/13_collect_baseline_results.py \
      --results-root data/results/link_prediction/patent_paper \
      --sort-split test \
      --sort-metric auprc
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.io import markdown_table, write_csv, write_text  # noqa: E402


DEFAULT_SELECTED_COLUMNS = [
    "dataset",
    "baseline",
    "model",
    "score_column",
    "split",
    "example_count",
    "positive_count",
    "negative_count",
    "auroc",
    "auprc",
    "mrr",
    "precision_at_100",
    "recall_at_100",
    "precision_at_1000",
    "recall_at_1000",
    "metrics_summary_path",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Collect baseline metrics_summary.csv files into comparison tables."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/results/link_prediction/patent_paper"),
        help=(
            "Root directory containing baseline result directories. "
            "Default: data/results/link_prediction/patent_paper"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for comparison files. "
            "Default: same as --results-root"
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default="baseline_comparison",
        help="Output file prefix. Default: baseline_comparison",
    )

    parser.add_argument(
        "--sort-split",
        default="test",
        help="Split used for ranking baselines in the report. Default: test",
    )

    parser.add_argument(
        "--sort-metric",
        default="auprc",
        help="Metric used for ranking baselines in the report. Default: auprc",
    )

    parser.add_argument(
        "--descending",
        action="store_true",
        default=True,
        help="Sort ranking metric in descending order. Default: true",
    )

    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ranking metric in ascending order.",
    )

    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Include train split in the compact Markdown report.",
    )

    return parser.parse_args()


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV file into a list of dictionaries."""

    path = Path(path)

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _safe_float(value: Any) -> float | None:
    """Convert a value to float if possible."""

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _format_value(value: Any) -> str:
    """Format values for Markdown output."""

    if value is None:
        return ""

    text = str(value)

    if text == "":
        return ""

    numeric = _safe_float(value)

    if numeric is None:
        return text

    if abs(numeric) >= 1000 and numeric.is_integer():
        return f"{int(numeric):,}"

    return f"{numeric:.6f}"


def discover_metrics_summary_files(results_root: str | Path) -> list[Path]:
    """Discover metrics_summary.csv files under a results root."""

    results_root = Path(results_root)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    paths = sorted(results_root.glob("**/metrics_summary.csv"))

    return [
        path
        for path in paths
        if path.is_file()
        and not path.name.startswith(".")
    ]


def infer_dataset_and_baseline(
    metrics_summary_path: Path,
    *,
    results_root: Path,
) -> tuple[str, str]:
    """Infer dataset name and baseline name from a metrics_summary.csv path."""

    relative_path = metrics_summary_path.relative_to(results_root)
    parts = relative_path.parts

    if len(parts) < 3:
        raise ValueError(
            "Expected metrics_summary.csv under <dataset>/<baseline>/metrics_summary.csv, "
            f"got: {metrics_summary_path}"
        )

    dataset = parts[0]
    baseline = "/".join(parts[1:-1])

    return dataset, baseline


def extract_model_from_metrics_csv(metrics_csv_path: Path) -> str:
    """Extract model name from long-form metrics.csv if available."""

    if not metrics_csv_path.exists():
        return ""

    rows = read_csv_rows(metrics_csv_path)

    for row in rows:
        model = str(row.get("model", "")).strip()
        if model:
            return model

    return ""


def infer_score_column(
    *,
    baseline: str,
    model: str,
) -> str:
    """Infer score column from model name or baseline directory."""

    if model and ":" in model:
        return model.split(":", 1)[1].strip()

    prefix = "bioentity_overlap_"

    if baseline.startswith(prefix):
        inferred = baseline[len(prefix):]

        alias_map = {
            "shared_count": "shared_bioentity_count",
            "jaccard": "bioentity_jaccard",
            "weighted_jaccard": "weighted_bioentity_jaccard",
            "weighted_cosine": "weighted_bioentity_cosine",
        }

        return alias_map.get(inferred, inferred)

    return ""


def collect_baseline_rows(results_root: str | Path) -> list[dict[str, Any]]:
    """Collect all baseline metric summary rows."""

    results_root = Path(results_root)
    metrics_summary_paths = discover_metrics_summary_files(results_root)

    rows: list[dict[str, Any]] = []

    for metrics_summary_path in metrics_summary_paths:
        dataset, baseline = infer_dataset_and_baseline(
            metrics_summary_path,
            results_root=results_root,
        )

        baseline_dir = metrics_summary_path.parent
        metrics_csv_path = baseline_dir / "metrics.csv"
        report_path = baseline_dir / "baseline_report.md"

        model = extract_model_from_metrics_csv(metrics_csv_path)
        score_column = infer_score_column(baseline=baseline, model=model)

        summary_rows = read_csv_rows(metrics_summary_path)

        for summary_row in summary_rows:
            row = {
                "dataset": dataset,
                "baseline": baseline,
                "model": model,
                "score_column": score_column,
                "metrics_summary_path": str(metrics_summary_path),
                "metrics_csv_path": str(metrics_csv_path) if metrics_csv_path.exists() else "",
                "baseline_report_path": str(report_path) if report_path.exists() else "",
            }

            row.update(summary_row)
            rows.append(row)

    return rows


def sort_rows(
    rows: list[dict[str, Any]],
    *,
    sort_split: str,
    sort_metric: str,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Sort rows by split, metric, and identifiers."""

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        split_priority = 0 if str(row.get("split", "")) == sort_split else 1
        metric_value = _safe_float(row.get(sort_metric))

        if metric_value is None:
            metric_value = float("-inf") if descending else float("inf")

        ordered_metric = -metric_value if descending else metric_value

        return (
            split_priority,
            ordered_metric,
            str(row.get("dataset", "")),
            str(row.get("baseline", "")),
            str(row.get("score_column", "")),
            str(row.get("split", "")),
        )

    return sorted(rows, key=sort_key)


def write_comparison_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write full comparison CSV."""

    path = Path(path)

    all_columns = list(DEFAULT_SELECTED_COLUMNS)

    extra_columns = []

    for row in rows:
        for key in row.keys():
            if key not in all_columns and key not in extra_columns:
                extra_columns.append(key)

    columns = all_columns + extra_columns

    write_csv(path, rows, columns=columns)
    return path


def filter_split_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Filter rows for one split."""

    return [
        row
        for row in rows
        if str(row.get("split", "")) == split
    ]


def build_report(
    *,
    results_root: Path,
    output_dir: Path,
    comparison_csv_path: Path,
    split_csv_path: Path,
    rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    sort_split: str,
    sort_metric: str,
    include_train: bool = False,
) -> str:
    """Build a Markdown comparison report."""

    split_order = ["val", "test"]
    if include_train:
        split_order = ["train", "val", "test"]

    compact_rows = []

    for row in ranked_rows:
        if str(row.get("split", "")) not in split_order:
            continue

        compact_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "baseline": row.get("baseline", ""),
                "score_column": row.get("score_column", ""),
                "split": row.get("split", ""),
                "auroc": _format_value(row.get("auroc", "")),
                "auprc": _format_value(row.get("auprc", "")),
                "mrr": _format_value(row.get("mrr", "")),
                "precision_at_100": _format_value(row.get("precision_at_100", "")),
                "recall_at_100": _format_value(row.get("recall_at_100", "")),
                "precision_at_1000": _format_value(row.get("precision_at_1000", "")),
                "recall_at_1000": _format_value(row.get("recall_at_1000", "")),
            }
        )

    best_rows = [
        row
        for row in ranked_rows
        if str(row.get("split", "")) == sort_split
    ][:20]

    formatted_best_rows = [
        {
            "rank": index,
            "dataset": row.get("dataset", ""),
            "baseline": row.get("baseline", ""),
            "score_column": row.get("score_column", ""),
            "split": row.get("split", ""),
            sort_metric: _format_value(row.get(sort_metric, "")),
            "auroc": _format_value(row.get("auroc", "")),
            "auprc": _format_value(row.get("auprc", "")),
            "precision_at_100": _format_value(row.get("precision_at_100", "")),
        }
        for index, row in enumerate(best_rows, start=1)
    ]

    dataset_count = len({str(row.get("dataset", "")) for row in rows})
    baseline_count = len(
        {
            (
                str(row.get("dataset", "")),
                str(row.get("baseline", "")),
                str(row.get("score_column", "")),
            )
            for row in rows
        }
    )

    lines = [
        "# Baseline Comparison Report",
        "",
        "## 1. Summary",
        "",
        markdown_table(
            [
                {"name": "results_root", "value": str(results_root)},
                {"name": "output_dir", "value": str(output_dir)},
                {"name": "dataset_count", "value": dataset_count},
                {"name": "baseline_run_count", "value": baseline_count},
                {"name": "metric_row_count", "value": len(rows)},
                {"name": "ranking_split", "value": sort_split},
                {"name": "ranking_metric", "value": sort_metric},
                {"name": "comparison_csv", "value": str(comparison_csv_path)},
                {"name": "split_csv", "value": str(split_csv_path)},
            ],
            ["name", "value"],
        ),
        "",
        "## 2. Top baseline runs",
        "",
        markdown_table(
            formatted_best_rows,
            [
                "rank",
                "dataset",
                "baseline",
                "score_column",
                "split",
                sort_metric,
                "auroc",
                "auprc",
                "precision_at_100",
            ],
        ),
        "",
        "## 3. Compact comparison",
        "",
        markdown_table(
            compact_rows,
            [
                "dataset",
                "baseline",
                "score_column",
                "split",
                "auroc",
                "auprc",
                "mrr",
                "precision_at_100",
                "recall_at_100",
                "precision_at_1000",
                "recall_at_1000",
            ],
        ),
        "",
        "## 4. Notes",
        "",
        "- Rows are collected from each baseline run's `metrics_summary.csv` file.",
        "- Ranking is based on the selected split and metric.",
        "- AUROC and AUPRC are the primary ranking metrics for balanced link prediction evaluation.",
        "- Precision@K is useful for top-ranked candidate inspection.",
        "",
    ]

    return "\n".join(lines)


def print_summary(
    *,
    rows: list[dict[str, Any]],
    ranked_split_rows: list[dict[str, Any]],
    comparison_csv_path: Path,
    split_csv_path: Path,
    report_path: Path,
    sort_split: str,
    sort_metric: str,
) -> None:
    """Print compact collection summary."""

    print()
    print("Baseline result collection complete.")
    print()
    print(f"Metric rows collected: {len(rows)}")
    print(f"Ranking split:         {sort_split}")
    print(f"Ranking metric:        {sort_metric}")
    print()
    print(f"Comparison CSV:        {comparison_csv_path}")
    print(f"{sort_split} CSV:              {split_csv_path}")
    print(f"Report:                {report_path}")
    print()

    if ranked_split_rows:
        print("Top runs:")
        for index, row in enumerate(ranked_split_rows[:10], start=1):
            metric_value = _format_value(row.get(sort_metric, ""))
            auroc = _format_value(row.get("auroc", ""))
            auprc = _format_value(row.get("auprc", ""))

            print(
                f"  {index}. "
                f"{row.get('dataset')} / {row.get('baseline')} "
                f"[{row.get('score_column')}] "
                f"{sort_metric}={metric_value}, "
                f"AUROC={auroc}, AUPRC={auprc}"
            )

    print()


def main() -> None:
    """Collect baseline results."""

    args = parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir is not None else results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    descending = not args.ascending

    rows = collect_baseline_rows(results_root)

    if not rows:
        raise RuntimeError(f"No metrics_summary.csv files found under: {results_root}")

    ranked_rows = sort_rows(
        rows,
        sort_split=args.sort_split,
        sort_metric=args.sort_metric,
        descending=descending,
    )

    comparison_csv_path = output_dir / f"{args.output_prefix}.csv"
    write_comparison_csv(comparison_csv_path, ranked_rows)

    split_rows = filter_split_rows(ranked_rows, split=args.sort_split)

    split_csv_path = output_dir / f"{args.output_prefix}_{args.sort_split}.csv"
    write_comparison_csv(split_csv_path, split_rows)

    report_path = output_dir / f"{args.output_prefix}_report.md"
    report_text = build_report(
        results_root=results_root,
        output_dir=output_dir,
        comparison_csv_path=comparison_csv_path,
        split_csv_path=split_csv_path,
        rows=rows,
        ranked_rows=ranked_rows,
        sort_split=args.sort_split,
        sort_metric=args.sort_metric,
        include_train=args.include_train,
    )
    write_text(report_path, report_text)

    print_summary(
        rows=rows,
        ranked_split_rows=split_rows,
        comparison_csv_path=comparison_csv_path,
        split_csv_path=split_csv_path,
        report_path=report_path,
        sort_split=args.sort_split,
        sort_metric=args.sort_metric,
    )


if __name__ == "__main__":
    main()