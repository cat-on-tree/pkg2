#!/usr/bin/env python
"""Run Stage 06 Knowledge Unit temporal baselines.

This script runs baseline models on the frozen Stage 05F KU temporal prediction
benchmark.

Input directory:

    data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/

Expected files:

    prediction_feature_table.parquet
    feature_columns.json
    target_columns.json

Default first-round task:

    Future Translation Heat Forecasting

    filter:
        eligible_translation_task == true

    heat target:
        target_future_translation_heat_3yr

    auxiliary binary label:
        label_future_translation_emergence_3yr

Default evaluation groups:

    validation:
        cutoff 2017-2018

    test_all:
        cutoff 2019-2021

    test_stable:
        cutoff 2019-2020

    test_by_cutoff:
        cutoff 2019
        cutoff 2020
        cutoff 2021

Examples:

    # Recommended first run: translation task, fast core baselines
    python scripts/30_run_knowledge_unit_temporal_baselines.py \\
      --tasks translation \\
      --models mean recency activity_paper activity_total recent_3yr_total growth_total_3yr_ratio ridge logistic histgb \\
      --max-train-rows 500000 \\
      --overwrite

    # Run all three tasks with core baselines
    python scripts/30_run_knowledge_unit_temporal_baselines.py \\
      --tasks translation patent trial \\
      --models mean recency activity_paper activity_total recent_3yr_total growth_total_3yr_ratio ridge logistic histgb \\
      --max-train-rows 500000 \\
      --overwrite

    # Heuristic-only full-data sanity run
    python scripts/30_run_knowledge_unit_temporal_baselines.py \\
      --tasks translation patent trial \\
      --models mean recency activity_paper activity_total recent_3yr_paper recent_3yr_total growth_total_3yr_ratio \\
      --overwrite

    # Stage 06E representation ablation:
    # raw carrier-count-like temporal features only
    python scripts/30_run_knowledge_unit_temporal_baselines.py \\
      --tasks translation patent trial \\
      --models ridge logistic logistic_unweighted histgb \\
      --feature-subset raw_counts_only \\
      --max-train-rows 0 \\
      --overwrite

    # Stage 06E representation ablation:
    # numeric temporal features only, no KU schema categorical features
    python scripts/30_run_knowledge_unit_temporal_baselines.py \\
      --tasks translation patent trial \\
      --models ridge logistic logistic_unweighted histgb \\
      --feature-subset numeric_only \\
      --max-train-rows 0 \\
      --overwrite

    # Stage 06E representation ablation:
    # KU schema categorical features only
    python scripts/30_run_knowledge_unit_temporal_baselines.py \\
      --tasks translation patent trial \\
      --models ridge logistic logistic_unweighted histgb \\
      --feature-subset ku_schema_only \\
      --max-train-rows 0 \\
      --overwrite

Notes:

    Trainable sklearn baselines can be expensive on millions of rows.
    Use --max-train-rows to subsample only the training split for fitting.
    Evaluation still runs on the full validation/test task frames.

    Stage 06E representation ablation is controlled by --feature-subset:

        all / ku_full:
            Use the original feature_columns.json selection. This is the
            full KU representation.

        numeric_only:
            Use numeric features only. This removes KU schema categorical
            features such as pair_type and entity types.

        raw_counts_only:
            Use cutoff-safe temporal count / recent / growth / recency
            numeric features only. This acts as a raw carrier-count-like
            evidence representation.

        ku_schema_only:
            Use categorical KU schema features only, such as pair_type and
            entity_a_type / entity_b_type.
"""

from __future__ import annotations

import gc
import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2 import knowledge_unit_baselines as kub  # noqa: E402


DEFAULT_PREDICTION_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1"
)

DEFAULT_TASKS = ["translation"]

DEFAULT_MODELS = [
    "mean",
    "recency",
    "activity_paper",
    "activity_total",
    "recent_3yr_total",
    "growth_total_3yr_ratio",
    "ridge",
    "logistic",
    "histgb",
]

DEFAULT_KS = [100, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 06 Knowledge Unit temporal baselines."
    )

    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help="Input Stage 05F temporal prediction dataset directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for Stage 06 baseline results.",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=sorted(kub.TASK_CONFIGS),
        help="Tasks to run.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Baseline model names. Supported examples: mean, recency, "
            "activity_paper, activity_total, recent_3yr_total, "
            "growth_total_3yr_ratio, ridge, elasticnet, logistic, "
            "logistic_unweighted, histgb, histgb_classifier, lightgbm, "
            "lightgbm_classifier."
        ),
    )

    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=DEFAULT_KS,
        help="K values for ranking metrics.",
    )

    parser.add_argument(
        "--stable-test-cutoffs",
        nargs="+",
        type=int,
        default=[2019, 2020],
        help="Cutoffs used for the stable test subset.",
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help=(
            "Maximum training rows for trainable models. "
            "Use 0 or negative value to train on all eligible train rows. "
            "Default: 0, use all training rows."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--include-categorical",
        action="store_true",
        default=True,
        help="Include categorical feature columns from feature_columns.json.",
    )

    parser.add_argument(
        "--no-categorical",
        action="store_true",
        help="Disable categorical feature columns.",
    )

    parser.add_argument(
        "--include-numeric",
        action="store_true",
        default=True,
        help="Include numeric feature columns from feature_columns.json.",
    )

    parser.add_argument(
        "--feature-subset",
        type=str,
        default="all",
        choices=[
            "all",
            "ku_full",
            "numeric_only",
            "raw_counts_only",
            "ku_schema_only",
        ],
        help=(
            "Feature subset for Stage 06E representation ablation. "
            "'all'/'ku_full' uses the original feature_columns.json selection; "
            "'numeric_only' uses numeric features only; "
            "'raw_counts_only' uses temporal count/recency/growth numeric features only; "
            "'ku_schema_only' uses categorical KU schema features only."
        ),
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "Save per-task/per-model prediction parquet files. "
            "This can use substantial disk space because evaluation groups overlap."
        ),
    )

    parser.add_argument(
        "--save-task-frames",
        action="store_true",
        help="Save task-filtered input frames used by baselines. Usually not needed.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any baseline fails.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress logging.",
    )

    return parser.parse_args()


def log(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, flush=True)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "task_frames").mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_importance").mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if not args.prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {args.prediction_dir}")

    feature_spec_path = args.prediction_dir / "feature_columns.json"
    target_spec_path = args.prediction_dir / "target_columns.json"
    feature_table_path = args.prediction_dir / "prediction_feature_table.parquet"

    missing = [
        path
        for path in [feature_spec_path, target_spec_path, feature_table_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required Stage 05F files:\n"
            + "\n".join(str(path) for path in missing)
        )

    if any(k <= 0 for k in args.ks):
        raise ValueError("--ks must contain positive integers.")

    if not args.tasks:
        raise ValueError("--tasks cannot be empty.")

    if not args.models:
        raise ValueError("--models cannot be empty.")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.prediction_dir = args.prediction_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    args.tasks = [str(task).lower() for task in args.tasks]
    args.models = [str(model).lower() for model in args.models]
    args.feature_subset = str(args.feature_subset).lower()

    if args.no_categorical:
        args.include_categorical = False

    args.max_train_rows = None if args.max_train_rows <= 0 else int(args.max_train_rows)

    return args


def _feature_spec_numeric_columns(feature_spec: dict[str, Any]) -> list[str]:
    """Return numeric feature columns from feature_columns.json."""

    for key in [
        "numeric_feature_columns",
        "numeric_columns",
        "numerical_feature_columns",
    ]:
        value = feature_spec.get(key)
        if value:
            return [str(col) for col in value]

    return []


def _feature_spec_categorical_columns(feature_spec: dict[str, Any]) -> list[str]:
    """Return categorical feature columns from feature_columns.json."""

    for key in [
        "categorical_feature_columns",
        "categorical_columns",
        "category_feature_columns",
    ]:
        value = feature_spec.get(key)
        if value:
            return [str(col) for col in value]

    return []


def _ordered_unique(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for column in columns:
        if column not in seen:
            out.append(column)
            seen.add(column)

    return out


def _is_raw_count_like_feature(column: str) -> bool:
    """Heuristic for raw carrier-count-like temporal features.

    This subset intentionally excludes KU schema categorical columns and keeps
    only cutoff-safe temporal evidence summaries such as history counts,
    recent activity, growth ratios, and recency.

    It is intended for Stage 06E representation ablation:

        raw_counts_only vs ku_full

    Interpretation:

        raw_counts_only:
            raw carrier-evidence-like temporal summaries

        ku_full:
            full KU abstraction representation
    """

    name = column.lower()

    include_prefixes = (
        "history_",
        "recent_",
        "growth_",
        "prior_",
        "lag_",
    )

    include_tokens = (
        "count",
        "recency",
        "ratio",
        "growth",
        "age",
        "span",
        "first_year",
        "last_year",
    )

    exclude_tokens = (
        "target_",
        "label_",
        "eligible_",
        "split",
        "cutoff",
        "future_",
        "entity_a_id",
        "entity_b_id",
        "entity_a_name",
        "entity_b_name",
        "entity_a_type",
        "entity_b_type",
        "pair_type",
    )

    if any(token in name for token in exclude_tokens):
        return False

    if name.startswith(include_prefixes):
        return True

    if any(token in name for token in include_tokens):
        return True

    return False


def _is_ku_schema_categorical_feature(column: str) -> bool:
    """Return whether a categorical feature describes the KU schema."""

    name = column.lower()

    return name in {
        "pair_type",
        "entity_a_type",
        "entity_b_type",
    } or name.endswith("_type")


def select_feature_columns_for_subset(
    feature_spec: dict[str, Any],
    *,
    feature_subset: str,
    include_categorical: bool,
    include_numeric: bool,
) -> list[str]:
    """Select feature columns for Stage 06E representation ablation."""

    feature_subset = str(feature_subset).lower()

    numeric_columns = _feature_spec_numeric_columns(feature_spec)
    categorical_columns = _feature_spec_categorical_columns(feature_spec)

    if feature_subset in {"all", "ku_full"}:
        return kub.get_feature_columns(
            feature_spec,
            include_categorical=include_categorical,
            include_numeric=include_numeric,
        )

    if feature_subset == "numeric_only":
        return numeric_columns if include_numeric else []

    if feature_subset == "raw_counts_only":
        if not include_numeric:
            return []
        return [
            column
            for column in numeric_columns
            if _is_raw_count_like_feature(column)
        ]

    if feature_subset == "ku_schema_only":
        if not include_categorical:
            return []
        return [
            column
            for column in categorical_columns
            if _is_ku_schema_categorical_feature(column)
        ]

    raise ValueError(f"Unsupported feature subset: {feature_subset}")


def load_stage05f_specs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Load feature/target specs and select feature columns.

    Important:
        This function intentionally does NOT load prediction_feature_table.parquet.
        Wide Stage 06G feature sets can exceed memory if the full table is
        loaded before task filtering. Task frames are loaded later, one task at
        a time, with parquet filters.
    """

    log(args, "Loading feature and target specs ...")
    feature_spec = kub.load_feature_spec(args.prediction_dir)
    target_spec = kub.load_target_spec(args.prediction_dir)

    feature_columns = select_feature_columns_for_subset(
        feature_spec,
        feature_subset=args.feature_subset,
        include_categorical=args.include_categorical,
        include_numeric=args.include_numeric,
    )

    feature_columns = _ordered_unique(feature_columns)

    if not feature_columns:
        raise ValueError(
            f"No feature columns selected for --feature-subset={args.feature_subset}. "
            "Check feature_columns.json or choose a different feature subset."
        )

    log(args, f"Feature subset: {args.feature_subset}")
    log(args, f"Selected feature columns: {len(feature_columns):,}")

    return feature_spec, target_spec, feature_columns


def model_is_trainable(model_name: str) -> bool:
    return not kub.is_constant_model(model_name) and not kub.is_heuristic_model(model_name)


def run_one_baseline(
    *,
    args: argparse.Namespace,
    task_frame: pd.DataFrame,
    task: str,
    model_name: str,
    feature_columns: list[str],
) -> kub.BaselineResult:
    task_config = kub.get_task_config(task)

    log(args, "-" * 80)
    log(args, f"Task: {task}")
    log(args, f"Model: {model_name}")
    log(args, f"Feature subset: {args.feature_subset}")
    log(args, f"Eligibility: {task_config.eligibility_column}")
    log(args, f"Heat target: {task_config.heat_target_column}")
    log(args, f"Emergence label: {task_config.emergence_label_column}")
    log(args, f"Eligible task rows: {len(task_frame):,}")

    start = time.time()

    result = kub.run_baseline_by_name(
        task_frame,
        task=task,
        model_name=model_name,
        feature_columns=feature_columns,
        max_train_rows=args.max_train_rows if model_is_trainable(model_name) else None,
        random_state=args.random_state,
        ks=args.ks,
        stable_test_cutoffs=args.stable_test_cutoffs,
    )

    elapsed = time.time() - start
    log(args, f"Finished {task}/{model_name} in {elapsed:.1f}s")

    metadata = dict(result.metadata or {})
    metadata.update(
        {
            "task": task,
            "model": model_name,
            "runtime_seconds": elapsed,
            "eligible_task_rows": int(len(task_frame)),
            "feature_subset": args.feature_subset,
            "feature_count": int(len(feature_columns)),
            "feature_columns": list(feature_columns),
            "max_train_rows": args.max_train_rows if model_is_trainable(model_name) else None,
        }
    )

    return kub.BaselineResult(
        task=result.task,
        model=result.model,
        metrics=result.metrics,
        predictions=result.predictions,
        metadata=metadata,
    )


def save_baseline_result(
    *,
    args: argparse.Namespace,
    result: kub.BaselineResult,
) -> None:
    task = result.task
    model = result.model

    metrics_path = args.output_dir / "metrics" / f"{task}_{model}_metrics.csv"
    kub.write_dataframe(result.metrics, metrics_path)

    config_path = args.output_dir / "configs" / f"{task}_{model}.json"
    kub.write_json(config_path, result.metadata or {})

    if args.save_predictions and result.predictions is not None:
        prediction_path = args.output_dir / "predictions" / f"{task}_{model}_predictions.parquet"
        kub.write_dataframe(result.predictions, prediction_path)


def save_error(
    *,
    args: argparse.Namespace,
    task: str,
    model_name: str,
    error: BaseException,
) -> dict[str, Any]:
    row = {
        "task": task,
        "model": model_name,
        "feature_subset": args.feature_subset,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }

    error_path = args.output_dir / "configs" / f"{task}_{model_name}_error.json"
    kub.write_json(error_path, row)
    return row


def collect_artifacts(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "artifact": path.name,
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_dir)),
                    "file_size_bytes": path.stat().st_size,
                }
            )

    return pd.DataFrame(rows)


def build_run_summary(
    *,
    args: argparse.Namespace,
    feature_columns: list[str],
    metrics: pd.DataFrame,
    errors: list[dict[str, Any]],
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "script": "scripts/30_run_knowledge_unit_temporal_baselines.py",
        "command": " ".join(sys.argv),
        "prediction_dir": str(args.prediction_dir),
        "output_dir": str(args.output_dir),
        "tasks": list(args.tasks),
        "models": list(args.models),
        "ks": list(args.ks),
        "stable_test_cutoffs": list(args.stable_test_cutoffs),
        "max_train_rows": args.max_train_rows,
        "random_state": args.random_state,
        "feature_subset": args.feature_subset,
        "include_categorical": args.include_categorical,
        "include_numeric": args.include_numeric,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "metric_rows": int(len(metrics)),
        "error_count": int(len(errors)),
        "errors": errors,
        "runtime_seconds": float(runtime_seconds),
        "created_at_unix": time.time(),
    }


def write_final_outputs(
    *,
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    errors: list[dict[str, Any]],
    feature_columns: list[str],
    runtime_seconds: float,
) -> None:
    output_dir = args.output_dir

    expected_metric_columns = [
        "task",
        "model",
        "eval_group",
        "split",
        "cutoff_year",
        "cutoff_years",
        "n_examples",
        "n_positive",
        "positive_rate",
        "target_heat_mean",
        "target_heat_max",
        "mae",
        "rmse",
        "spearman",
        "pearson",
        "auroc",
        "auprc",
        "brier",
        "ndcg_at_100",
        "precision_at_100",
        "recall_at_100",
        "enrichment_at_100",
        "ndcg_at_1000",
        "precision_at_1000",
        "recall_at_1000",
        "enrichment_at_1000",
    ]

    if metrics is None or metrics.empty:
        metrics = pd.DataFrame(columns=expected_metric_columns)

    baseline_metrics_path = output_dir / "baseline_metrics.csv"
    kub.write_dataframe(metrics, baseline_metrics_path)

    if metrics.empty or "eval_group" not in metrics.columns:
        sorted_metrics = metrics.copy()
        comparison_test_all = pd.DataFrame(columns=expected_metric_columns)
        comparison_validation = pd.DataFrame(columns=expected_metric_columns)
        comparison_test_stable = pd.DataFrame(columns=expected_metric_columns)
        by_cutoff = pd.DataFrame(columns=expected_metric_columns)
    else:
        sorted_metrics = kub.sort_baseline_metrics(
            metrics,
            primary_eval_group="test_all",
            primary_metric="ndcg_at_1000",
            ascending=False,
        )

        comparison_test_all = kub.make_comparison_table(
            metrics,
            eval_group="test_all",
            sort_metric="ndcg_at_1000",
            ascending=False,
        )
        comparison_validation = kub.make_comparison_table(
            metrics,
            eval_group="validation",
            sort_metric="ndcg_at_1000",
            ascending=False,
        )
        comparison_test_stable = kub.make_comparison_table(
            metrics,
            eval_group="test_stable",
            sort_metric="ndcg_at_1000",
            ascending=False,
        )

        by_cutoff = metrics[
            metrics["eval_group"].astype(str).str.startswith("test_cutoff_")
        ].copy()

    kub.write_dataframe(sorted_metrics, output_dir / "baseline_metrics_sorted.csv")

    kub.write_dataframe(
        comparison_test_all,
        output_dir / "baseline_comparison_test_all.csv",
    )
    kub.write_dataframe(
        comparison_validation,
        output_dir / "baseline_comparison_validation.csv",
    )
    kub.write_dataframe(
        comparison_test_stable,
        output_dir / "baseline_comparison_test_stable.csv",
    )

    # Backward-compatible default comparison file: use test_all.
    kub.write_dataframe(
        comparison_test_all,
        output_dir / "baseline_comparison.csv",
    )

    kub.write_dataframe(by_cutoff, output_dir / "baseline_metrics_by_cutoff.csv")

    if errors:
        error_frame = pd.DataFrame(errors)
        kub.write_dataframe(error_frame, output_dir / "baseline_errors.csv")
    else:
        error_frame = pd.DataFrame(
            columns=[
                "task",
                "model",
                "feature_subset",
                "error_type",
                "error_message",
                "traceback",
            ]
        )
        kub.write_dataframe(error_frame, output_dir / "baseline_errors.csv")

    run_summary = build_run_summary(
        args=args,
        feature_columns=feature_columns,
        metrics=metrics,
        errors=errors,
        runtime_seconds=runtime_seconds,
    )
    kub.write_json(output_dir / "baseline_run_summary.json", run_summary)

    notes = [
        "All baselines use the frozen Stage 05F prediction dataset.",
        f"Feature subset: {args.feature_subset}.",
        "The primary comparison file uses test_all cutoffs 2019-2021.",
        "test_stable excludes cutoff 2021 and uses cutoffs 2019-2020.",
        "cutoff 2021 should be interpreted as a right-edge stress test.",
        "Trainable baselines may use --max-train-rows subsampling for fitting; evaluation uses full validation/test task frames.",
    ]

    if args.feature_subset == "raw_counts_only":
        notes.append(
            "Stage 06E representation ablation: raw_counts_only keeps only "
            "cutoff-safe temporal count / recent / growth / recency numeric features."
        )
    elif args.feature_subset == "numeric_only":
        notes.append(
            "Stage 06E representation ablation: numeric_only removes KU schema categorical features."
        )
    elif args.feature_subset == "ku_schema_only":
        notes.append(
            "Stage 06E representation ablation: ku_schema_only uses only categorical KU schema features."
        )
    elif args.feature_subset in {"all", "ku_full"}:
        notes.append(
            "Stage 06E representation ablation: all/ku_full is the full KU representation."
        )

    report = kub.build_baseline_report(
        metrics=metrics,
        comparison=comparison_test_all,
        title="Stage 06 Knowledge Unit Temporal Baseline Report",
        prediction_dir=args.prediction_dir,
        notes=notes,
    )

    # Add extra sections for validation and stable test comparisons.
    report += "\n\n## Additional comparison: validation\n\n"
    report += kub.markdown_table(comparison_validation, max_rows=100)

    report += "\n\n## Additional comparison: test_stable\n\n"
    report += kub.markdown_table(comparison_test_stable, max_rows=100)

    if not by_cutoff.empty:
        report += "\n\n## Test metrics by cutoff\n\n"
        report += kub.markdown_table(by_cutoff, max_rows=300)

    if errors:
        report += "\n\n## Errors\n\n"
        report += kub.markdown_table(error_frame, max_rows=100)

    kub.write_dataframe(
        collect_artifacts(output_dir),
        output_dir / "baseline_manifest.csv",
    )

    (output_dir / "baseline_comparison_report.md").write_text(
        report,
        encoding="utf-8",
    )


def print_final_summary(
    *,
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    errors: list[dict[str, Any]],
    runtime_seconds: float,
) -> None:
    print()
    print("Stage 06 KU temporal baselines complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature subset:   {args.feature_subset}")
    print(f"Metric rows:      {len(metrics):,}")
    print(f"Errors:           {len(errors):,}")
    print(f"Runtime seconds:  {runtime_seconds:.1f}")
    print()
    print("Main outputs:")
    print(f"  {args.output_dir / 'baseline_metrics.csv'}")
    print(f"  {args.output_dir / 'baseline_comparison.csv'}")
    print(f"  {args.output_dir / 'baseline_comparison_test_all.csv'}")
    print(f"  {args.output_dir / 'baseline_comparison_test_stable.csv'}")
    print(f"  {args.output_dir / 'baseline_metrics_by_cutoff.csv'}")
    print(f"  {args.output_dir / 'baseline_comparison_report.md'}")
    print()


def downcast_loaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Downcast loaded prediction table columns to reduce memory pressure."""

    for col in frame.select_dtypes(include=["float64"]).columns:
        frame[col] = frame[col].astype("float32")

    for col in frame.select_dtypes(include=["int64"]).columns:
        series = frame[col]

        if series.isna().any():
            continue

        col_min = series.min()
        col_max = series.max()

        if col_min >= -2_147_483_648 and col_max <= 2_147_483_647:
            frame[col] = series.astype("int32")

    return frame


def task_required_columns(
    *,
    task: str,
    feature_columns: list[str],
) -> list[str]:
    """Build minimal columns needed for one task."""

    columns = kub.required_columns_for_tasks([task], feature_columns)

    optional_context_columns = [
        "pair_type",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "history_paper_count",
        "history_total_count",
        "recent_3yr_total_count",
        "history_recency",
    ]

    for column in optional_context_columns:
        if column not in columns:
            columns.append(column)

    return _ordered_unique(columns)


def load_task_prediction_frame(
    *,
    args: argparse.Namespace,
    task: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Load one eligible task frame directly from parquet.

    This avoids:
        full 12M-row table load
        then huge eligible-row copy

    This is essential for wide Stage 06G feature sets such as:
        KU + carrier structural + diffusion proxy
    """

    config = kub.get_task_config(task)
    table_path = args.prediction_dir / "prediction_feature_table.parquet"

    columns = task_required_columns(
        task=task,
        feature_columns=feature_columns,
    )

    log(args, f"Loading task frame for {task} with {len(columns):,} columns ...")

    try:
        frame = pd.read_parquet(
            table_path,
            columns=columns,
            filters=[(config.eligibility_column, "=", True)],
        )
    except Exception as filter_error:
        log(
            args,
            (
                "Parquet filter load failed; falling back to full selected-column "
                f"load then filtering. Reason: {type(filter_error).__name__}: {filter_error}"
            ),
        )
        frame = pd.read_parquet(
            table_path,
            columns=columns,
        )
        frame = frame.loc[frame[config.eligibility_column].astype(bool)]

    frame = downcast_loaded_frame(frame)

    log(args, f"Loaded eligible task rows for {task}: {len(frame):,}")
    return frame


def main() -> None:
    args = parse_args()
    args = normalize_args(args)
    validate_args(args)

    start = time.time()

    prepare_output_dir(args.output_dir, args.overwrite)

    kub.write_json(
        args.output_dir / "configs" / "run_args.json",
        vars(args),
    )

    feature_spec, target_spec, feature_columns = load_stage05f_specs(args)

    kub.write_json(
        args.output_dir / "configs" / "feature_spec.json",
        feature_spec,
    )
    kub.write_json(
        args.output_dir / "configs" / "target_spec.json",
        target_spec,
    )
    kub.write_json(
        args.output_dir / "configs" / "selected_feature_columns.json",
        {
            "feature_subset": args.feature_subset,
            "include_categorical": args.include_categorical,
            "include_numeric": args.include_numeric,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
        },
    )

    all_metrics: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []

    total_runs = len(args.tasks) * len(args.models)
    run_index = 0

    for task in args.tasks:
        task_frame: pd.DataFrame | None = None

        try:
            task_frame = load_task_prediction_frame(
                args=args,
                task=task,
                feature_columns=feature_columns,
            )

            if args.save_task_frames:
                task_frame_path = args.output_dir / "task_frames" / f"{task}.parquet"
                kub.write_dataframe(task_frame, task_frame_path)

            for model_name in args.models:
                run_index += 1
                log(args, "")
                log(args, f"Run {run_index}/{total_runs}: {task}/{model_name}")

                try:
                    result = run_one_baseline(
                        args=args,
                        task_frame=task_frame,
                        task=task,
                        model_name=model_name,
                        feature_columns=feature_columns,
                    )
                    save_baseline_result(args=args, result=result)
                    all_metrics.append(result.metrics)

                    del result
                    gc.collect()

                except BaseException as error:
                    log(
                        args,
                        f"ERROR in {task}/{model_name}: "
                        f"{type(error).__name__}: {error}",
                    )
                    error_row = save_error(
                        args=args,
                        task=task,
                        model_name=model_name,
                        error=error,
                    )
                    errors.append(error_row)

                    gc.collect()

                    if args.fail_fast:
                        raise

        finally:
            if task_frame is not None:
                del task_frame
            gc.collect()

    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics = pd.DataFrame()

    runtime_seconds = time.time() - start

    write_final_outputs(
        args=args,
        metrics=metrics,
        errors=errors,
        feature_columns=feature_columns,
        runtime_seconds=runtime_seconds,
    )

    print_final_summary(
        args=args,
        metrics=metrics,
        errors=errors,
        runtime_seconds=runtime_seconds,
    )


if __name__ == "__main__":
    main()