#!/usr/bin/env python
"""Run DynamicRDS hyperparameter sensitivity experiments.

This script is intentionally a thin orchestration layer. It reuses the existing
training entry point:

    scripts/35_run_temporal_knowledge_field_gnn.py

and then aggregates each run's existing artifacts:

    run_summary.json
    test_metrics_by_cutoff.csv

The default plan is one-factor-at-a-time sensitivity around the current Stage 07
best DynamicRDS v2 configuration:

    model=dynamic_rds
    gnn_input_dir=data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_proxy_latest_split
    hidden_dim=128
    num_layers=2
    num_neighbors=15 10
    dynamic_state_dim=8
    seed=42

Examples
--------
Dry-run the default plan:

    python scripts/36_run_dynamic_rds_hyperparameter_sensitivity.py --dry-run

Run only state_dim and seed sweeps:

    python scripts/36_run_dynamic_rds_hyperparameter_sensitivity.py \
      --sweeps state_dim seed \
      --device cuda --amp --num-workers 4

Run quick debug jobs:

    python scripts/36_run_dynamic_rds_hyperparameter_sensitivity.py \
      --sweeps quick \
      --epochs 3 --patience 2 --max-batches-per-cutoff 10 --max-eval-batches 10 \
      --dry-run

Summarize existing runs without launching training:

    python scripts/36_run_dynamic_rds_hyperparameter_sensitivity.py --summarize-only

Notes
-----
This script only sweeps hyperparameters exposed by script 35 / TrainingConfig.
Mechanistic parameters such as explicit diffusion_scale or decay_scale are not
currently CLI-exposed in the model. To analyze those, first add trainable or
fixed scale options to DynamicRDSRegressor and script 35, then add them to this
orchestration plan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GNN_INPUT_DIR = Path(
    "data/datasets/knowledge_field/"
    "diabetes_2000_2024_v1_gnn_local_source_proxy_latest_split"
)

DEFAULT_OUTPUT_ROOT = Path(
    "data/results/knowledge_field/hyperparameter_sensitivity/dynamic_rds_v2"
)

DEFAULT_TRAIN_SCRIPT = Path("scripts/35_run_temporal_knowledge_field_gnn.py")


@dataclass(frozen=True)
class RunSpec:
    """One training run in the sensitivity plan."""

    sweep: str
    value: str
    run_name: str
    args: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch and summarize DynamicRDS v2 hyperparameter sensitivity runs "
            "by reusing scripts/35_run_temporal_knowledge_field_gnn.py."
        )
    )

    parser.add_argument(
        "--gnn-input-dir",
        type=Path,
        default=DEFAULT_GNN_INPUT_DIR,
        help=f"GNN input directory. Default: {DEFAULT_GNN_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for sensitivity runs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=DEFAULT_TRAIN_SCRIPT,
        help=f"Training script to reuse. Default: {DEFAULT_TRAIN_SCRIPT}",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for subprocess training runs. Default: current interpreter.",
    )

    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=["state_dim", "seed", "num_layers", "num_neighbors"],
        choices=[
            "quick",
            "state_dim",
            "seed",
            "num_layers",
            "num_neighbors",
            "dropout",
            "lr",
            "weight_decay",
            "all",
        ],
        help=(
            "Sweeps to run. Default: state_dim seed num_layers num_neighbors. "
            "Use quick for a tiny smoke-test plan, or all for all exposed sweeps."
        ),
    )

    # Baseline formal configuration.
    parser.add_argument(
        "--task",
        default="translation",
        choices=["translation", "patent", "trial"],
    )
    parser.add_argument(
        "--model",
        default="dynamic_rds",
        choices=["dynamic_rds"],
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--base-state-dim", type=int, default=8)
    parser.add_argument("--base-num-layers", type=int, default=2)
    parser.add_argument("--base-num-neighbors", nargs="+", type=int, default=[15, 10])
    parser.add_argument("--base-dropout", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--base-weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-seed", type=int, default=42)

    # Sweep values.
    parser.add_argument("--state-dims", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 42, 2024])
    parser.add_argument("--num-layers-values", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument(
        "--num-neighbors-values",
        nargs="+",
        default=["10,5", "15,10", "25,15", "50,25"],
        help=(
            "Neighbor fanout settings, each as comma-separated ints. "
            "Default: 10,5 15,10 25,15 50,25"
        ),
    )
    parser.add_argument("--dropout-values", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--lr-values", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument(
        "--weight-decay-values",
        nargs="+",
        type=float,
        default=[0.0, 1e-5, 1e-4, 1e-3],
    )

    # Training settings forwarded to script 35.
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--loss", default="smoothl1", choices=["smoothl1", "mse", "huber"])
    parser.add_argument("--monitor-metric", default="auprc")
    parser.add_argument("--topk", nargs="+", type=int, default=[100, 1000, 5000])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # Debug controls forwarded to script 35.
    parser.add_argument("--max-train-cutoffs", type=int, default=None)
    parser.add_argument("--max-batches-per-cutoff", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    # Orchestration controls.
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize existing run outputs.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run when its output directory already contains run_summary.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs if one subprocess fails.",
    )

    return parser.parse_args()


def expand_sweeps(raw_sweeps: list[str]) -> list[str]:
    sweeps = list(raw_sweeps)
    if "all" in sweeps:
        return [
            "state_dim",
            "seed",
            "num_layers",
            "num_neighbors",
            "dropout",
            "lr",
            "weight_decay",
        ]
    if "quick" in sweeps:
        return ["quick"]
    return sweeps


def parse_neighbors(text: str) -> list[int]:
    values = [int(v.strip()) for v in str(text).split(",") if v.strip()]
    if not values:
        raise ValueError(f"Invalid neighbor fanout setting: {text!r}")
    return values


def fmt_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def base_run_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dynamic_state_dim": args.base_state_dim,
        "num_layers": args.base_num_layers,
        "num_neighbors": list(args.base_num_neighbors),
        "dropout": args.base_dropout,
        "lr": args.base_lr,
        "weight_decay": args.base_weight_decay,
        "seed": args.base_seed,
    }


def make_plan(args: argparse.Namespace) -> list[RunSpec]:
    sweeps = expand_sweeps(args.sweeps)
    base = base_run_args(args)
    runs: list[RunSpec] = []

    if sweeps == ["quick"]:
        quick_specs = [
            ("quick_state4", {"dynamic_state_dim": 4}),
            ("quick_state8", {"dynamic_state_dim": 8}),
            ("quick_seed1", {"seed": 1}),
        ]
        for name, overrides in quick_specs:
            run_args = {**base, **overrides}
            runs.append(RunSpec("quick", name, name, run_args))
        return runs

    if "state_dim" in sweeps:
        for value in args.state_dims:
            run_args = {**base, "dynamic_state_dim": int(value)}
            runs.append(
                RunSpec(
                    sweep="state_dim",
                    value=str(value),
                    run_name=f"state_dim_{value}",
                    args=run_args,
                )
            )

    if "seed" in sweeps:
        for value in args.seeds:
            run_args = {**base, "seed": int(value)}
            runs.append(
                RunSpec(
                    sweep="seed",
                    value=str(value),
                    run_name=f"seed_{value}",
                    args=run_args,
                )
            )

    if "num_layers" in sweeps:
        for value in args.num_layers_values:
            value = int(value)
            base_fanout = list(args.base_num_neighbors)

            # Keep NeighborLoader fanout length consistent with num_layers.
            if value <= len(base_fanout):
                fanout = base_fanout[:value]
            else:
                fanout = base_fanout + [base_fanout[-1]] * (value - len(base_fanout))

            run_args = {**base, "num_layers": value, "num_neighbors": fanout}
            runs.append(
                RunSpec(
                    sweep="num_layers",
                    value=str(value),
                    run_name=f"num_layers_{value}",
                    args=run_args,
                )
            )

    if "num_neighbors" in sweeps:
        for text in args.num_neighbors_values:
            fanout = parse_neighbors(text)
            run_args = {**base, "num_layers": len(fanout), "num_neighbors": fanout}
            runs.append(
                RunSpec(
                    sweep="num_neighbors",
                    value="-".join(str(v) for v in fanout),
                    run_name="num_neighbors_" + "_".join(str(v) for v in fanout),
                    args=run_args,
                )
            )

    if "dropout" in sweeps:
        for value in args.dropout_values:
            run_args = {**base, "dropout": float(value)}
            runs.append(
                RunSpec(
                    sweep="dropout",
                    value=str(value),
                    run_name=f"dropout_{fmt_float(value)}",
                    args=run_args,
                )
            )

    if "lr" in sweeps:
        for value in args.lr_values:
            run_args = {**base, "lr": float(value)}
            runs.append(
                RunSpec(
                    sweep="lr",
                    value=str(value),
                    run_name=f"lr_{fmt_float(value)}",
                    args=run_args,
                )
            )

    if "weight_decay" in sweeps:
        for value in args.weight_decay_values:
            run_args = {**base, "weight_decay": float(value)}
            runs.append(
                RunSpec(
                    sweep="weight_decay",
                    value=str(value),
                    run_name=f"weight_decay_{fmt_float(value)}",
                    args=run_args,
                )
            )

    # Preserve order while removing exact duplicate output names.
    deduped: list[RunSpec] = []
    seen: set[tuple[str, str]] = set()
    for run in runs:
        key = (run.sweep, run.run_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(run)

    return deduped


def append_flag(cmd: list[str], name: str, value: Any) -> None:
    cmd.append(f"--{name.replace('_', '-')}")
    if isinstance(value, (list, tuple)):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def build_command(args: argparse.Namespace, run: RunSpec, output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(args.train_script),
        "--gnn-input-dir",
        str(args.gnn_input_dir),
        "--output-dir",
        str(output_dir),
        "--task",
        args.task,
        "--model",
        args.model,
        "--hidden-dim",
        str(args.hidden_dim),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--optimizer",
        args.optimizer,
        "--loss",
        args.loss,
        "--monitor-metric",
        args.monitor_metric,
        "--topk",
        *[str(k) for k in args.topk],
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]

    for key in [
        "num_layers",
        "num_neighbors",
        "dropout",
        "lr",
        "weight_decay",
        "seed",
        "dynamic_state_dim",
    ]:
        append_flag(cmd, key, run.args[key])

    if args.amp:
        cmd.append("--amp")

    if args.tensorboard:
        cmd.append("--tensorboard")
        cmd.extend(["--tensorboard-run-name", f"hparam_{run.sweep}_{run.run_name}"])

    if args.write_predictions:
        cmd.append("--write-predictions")

    if args.overwrite:
        cmd.append("--overwrite")

    if args.max_train_cutoffs is not None:
        cmd.extend(["--max-train-cutoffs", str(args.max_train_cutoffs)])

    if args.max_batches_per_cutoff is not None:
        cmd.extend(["--max-batches-per-cutoff", str(args.max_batches_per_cutoff)])

    if args.max_eval_batches is not None:
        cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])

    return cmd


def shell_join(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def write_plan(output_root: Path, plan: list[RunSpec]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_rows = [
        {
            "sweep": run.sweep,
            "value": run.value,
            "run_name": run.run_name,
            "args": run.args,
        }
        for run in plan
    ]

    (output_root / "sensitivity_plan.json").write_text(
        json.dumps(json_rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    csv_rows = []
    for run in plan:
        row = {
            "sweep": run.sweep,
            "value": run.value,
            "run_name": run.run_name,
            **run.args,
        }
        row["num_neighbors"] = " ".join(str(v) for v in run.args["num_neighbors"])
        csv_rows.append(row)

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(output_root / "sensitivity_plan.csv", index=False)


def collect_run_metrics(output_root: Path, plan: list[RunSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run in plan:
        run_dir = output_root / run.sweep / run.run_name
        summary_path = run_dir / "run_summary.json"
        test_path = run_dir / "test_metrics_by_cutoff.csv"
        config_path = run_dir / "run_config.json"

        row: dict[str, Any] = {
            "sweep": run.sweep,
            "value": run.value,
            "run_name": run.run_name,
            "output_dir": str(run_dir),
            **run.args,
            "status": "missing",
        }
        row["num_neighbors"] = " ".join(str(v) for v in run.args["num_neighbors"])

        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row.update(
                    {
                        "status": summary.get("status", "ok"),
                        "best_epoch": summary.get("best_epoch"),
                        "best_monitor_metric": summary.get("best_monitor_metric"),
                        "best_monitor_value": summary.get("best_monitor_value"),
                        "runtime_seconds": summary.get("runtime_seconds"),
                        "trainable_parameters": summary.get("trainable_parameters"),
                    }
                )

                test_metrics = summary.get("test_metrics") or {}
                for key, value in test_metrics.items():
                    row[f"test_{key}"] = value

            except Exception as exc:
                row["status"] = "summary_read_error"
                row["error"] = str(exc)

        if test_path.exists():
            try:
                test_df = pd.read_csv(test_path)
                if not test_df.empty:
                    test_row = test_df.iloc[0].to_dict()
                    for key, value in test_row.items():
                        row[f"cutoff_{key}"] = value
            except Exception as exc:
                row["test_csv_error"] = str(exc)

        if config_path.exists():
            row["has_run_config"] = True

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def write_summary_tables(output_root: Path, metrics_df: pd.DataFrame) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(output_root / "sensitivity_metrics_summary.csv", index=False)

    numeric_cols = [
        "test_auprc",
        "test_auroc",
        "test_spearman",
        "test_precision_at_100",
        "test_precision_at_1000",
        "test_precision_at_5000",
        "test_ndcg_at_1000",
        "test_enrichment_at_1000",
        "best_monitor_value",
        "runtime_seconds",
    ]
    available_numeric = [col for col in numeric_cols if col in metrics_df.columns]

    if available_numeric and "sweep" in metrics_df.columns:
        completed = metrics_df[metrics_df["status"].ne("missing")]
        if not completed.empty:
            grouped = (
                completed.groupby("sweep", dropna=False)[available_numeric]
                .agg(["count", "mean", "std", "max"])
                .reset_index()
            )
            grouped.to_csv(output_root / "sensitivity_metrics_by_sweep.csv", index=False)

    report_lines = [
        "# DynamicRDS hyperparameter sensitivity summary",
        "",
        f"Total planned runs: {len(metrics_df)}",
        f"Completed run summaries: {int(metrics_df['status'].ne('missing').sum())}",
        "",
    ]

    key_cols = [
        "sweep",
        "value",
        "run_name",
        "status",
        "best_epoch",
        "test_auprc",
        "test_auroc",
        "test_spearman",
        "test_precision_at_1000",
        "test_ndcg_at_1000",
        "test_enrichment_at_1000",
    ]
    key_cols = [col for col in key_cols if col in metrics_df.columns]

    if key_cols:
        report_lines.append("## Key metrics")
        report_lines.append("")
        try:
            report_lines.append(metrics_df[key_cols].to_markdown(index=False))
        except Exception:
            report_lines.append(metrics_df[key_cols].to_csv(index=False))
        report_lines.append("")

    if "test_precision_at_1000" in metrics_df.columns:
        report_lines.append("## Best run by P@1000")
        report_lines.append("")
        valid = metrics_df.dropna(subset=["test_precision_at_1000"])
        if not valid.empty:
            best = valid.sort_values("test_precision_at_1000", ascending=False).iloc[0]
            report_lines.append(
                f"Best P@1000: **{best['test_precision_at_1000']:.6f}** "
                f"({best['sweep']}={best['value']}, run={best['run_name']})."
            )
        else:
            report_lines.append("No completed P@1000 values found yet.")
        report_lines.append("")

    (output_root / "sensitivity_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def resolve_under_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()

    args.gnn_input_dir = resolve_under_repo(args.gnn_input_dir)
    args.output_root = resolve_under_repo(args.output_root)
    args.train_script = resolve_under_repo(args.train_script)

    plan = make_plan(args)
    write_plan(args.output_root, plan)

    print(f"Planned runs: {len(plan)}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print("", flush=True)

    if not args.summarize_only:
        for idx, run in enumerate(plan, start=1):
            output_dir = args.output_root / run.sweep / run.run_name
            summary_path = output_dir / "run_summary.json"

            if args.skip_existing and summary_path.exists():
                print(
                    f"[{idx}/{len(plan)}] SKIP existing {run.sweep}/{run.run_name}",
                    flush=True,
                )
                continue

            cmd = build_command(args, run, output_dir)

            print(f"[{idx}/{len(plan)}] {run.sweep}/{run.run_name}", flush=True)
            print(shell_join(cmd), flush=True)
            print("", flush=True)

            if args.dry_run:
                continue

            try:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            except subprocess.CalledProcessError as exc:
                print(
                    f"Run failed: {run.sweep}/{run.run_name} exit={exc.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.continue_on_error:
                    raise

    metrics_df = collect_run_metrics(args.output_root, plan)
    write_summary_tables(args.output_root, metrics_df)

    print("Sensitivity summary written:", flush=True)
    print(f"  {args.output_root / 'sensitivity_plan.csv'}", flush=True)
    print(f"  {args.output_root / 'sensitivity_metrics_summary.csv'}", flush=True)
    print(f"  {args.output_root / 'sensitivity_report.md'}", flush=True)


if __name__ == "__main__":
    main()