#!/usr/bin/env python
"""Export predictions from an existing Temporal Knowledge Field run.

This script is a post-hoc inference utility for Stage 07 models. It reloads an
existing run checkpoint and writes per-node predictions without retraining.

Typical use
-----------

    python scripts/36_export_knowledge_field_predictions.py \
      --run-dir data/results/knowledge_field/temporal_gnn/local_source_proxy_dynamic_rds_v2_state8_seed42 \
      --split test \
      --cutoff-year 2021 \
      --device cuda \
      --amp \
      --overwrite

Expected outputs
----------------

    <run-dir>/predictions/predictions_translation_test_cutoff_2021.parquet
    <run-dir>/exported_test_metrics_by_cutoff.csv
    <run-dir>/exported_predictions_summary.json

The exported prediction frame contains:

    node_idx, split, cutoff_year, y_heat, y_label, score

These files can be used by downstream stratified evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import torch

# Make src/ importable when running this script directly from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pkg2.knowledge_field.data import KnowledgeFieldDataset, load_normalization_stats
from pkg2.knowledge_field.metrics import aggregate_metric_rows
from pkg2.knowledge_field.models import (
    build_model,
    count_parameters,
    model_summary_dict,
)
from pkg2.knowledge_field.trainer import (
    GRAPH_MODELS,
    KnowledgeFieldTrainer,
    TrainingConfig,
)


EDGE_WEIGHT_MODELS = {
    "weighted_diffusion",
    "reaction_diffusion_source",
    "rdgnn_style",
    "grand_style",
    "dynamic_rds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-node predictions from an existing Knowledge Field run."
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing model run directory containing best_model.pt and run_config.json.",
    )
    parser.add_argument(
        "--gnn-input-dir",
        type=Path,
        default=None,
        help=(
            "Optional GNN input directory. If omitted, use the path recorded in "
            "best_model.pt or run_config.json."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/best_model.pt.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Split to export predictions for.",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=None,
        help="Optional cutoff year. If omitted, export all cutoffs in the split.",
    )
    parser.add_argument(
        "--task",
        choices=["translation", "patent", "trial"],
        default=None,
        help="Optional task override. Defaults to task recorded in the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cuda or cpu. Defaults to recorded config.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Force-enable CUDA AMP for inference.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Force-disable AMP for inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch size override.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional NeighborLoader worker count override.",
    )
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=None,
        help="Optional top-K values for exported metrics.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug: limit evaluation batches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing prediction files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def training_config_from_checkpoint(
    ckpt: dict[str, Any],
    *,
    run_config_path: Path | None = None,
) -> TrainingConfig:
    """Reconstruct TrainingConfig from checkpoint/run_config metadata."""

    raw: dict[str, Any] = {}

    if isinstance(ckpt.get("config"), dict):
        raw.update(ckpt["config"])

    # run_config.json may contain the same fields plus derived metadata.
    # Use it as a fallback only, so checkpoint remains the source of truth.
    if run_config_path is not None and run_config_path.exists():
        try:
            run_raw = read_json(run_config_path)
            for key, value in run_raw.items():
                raw.setdefault(key, value)
        except Exception:
            pass

    valid_fields = {f.name for f in fields(TrainingConfig)}
    filtered = {key: value for key, value in raw.items() if key in valid_fields}

    return TrainingConfig(**filtered)


def resolve_gnn_input_dir(
    *,
    args: argparse.Namespace,
    ckpt: dict[str, Any],
    run_config_path: Path,
) -> Path:
    if args.gnn_input_dir is not None:
        return args.gnn_input_dir.resolve()

    dataset_summary = ckpt.get("dataset_summary")
    if isinstance(dataset_summary, dict) and dataset_summary.get("gnn_input_dir"):
        return Path(dataset_summary["gnn_input_dir"]).resolve()

    if run_config_path.exists():
        run_config = read_json(run_config_path)
        dataset = run_config.get("dataset")
        if isinstance(dataset, dict) and dataset.get("gnn_input_dir"):
            return Path(dataset["gnn_input_dir"]).resolve()

    raise ValueError(
        "Could not infer --gnn-input-dir from checkpoint/run_config. "
        "Please pass --gnn-input-dir explicitly."
    )


def prediction_output_path(
    run_dir: Path,
    *,
    task: str,
    split: str,
    cutoff_year: int,
) -> Path:
    return (
        run_dir
        / "predictions"
        / f"predictions_{task}_{split}_cutoff_{int(cutoff_year)}.parquet"
    )


def apply_config_overrides(
    config: TrainingConfig,
    *,
    args: argparse.Namespace,
) -> TrainingConfig:
    """Apply CLI overrides to a reconstructed TrainingConfig."""

    if args.task is not None:
        config.task = args.task

    if args.device is not None:
        config.device = args.device

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.num_workers is not None:
        config.num_workers = args.num_workers

    if args.topk is not None:
        config.topk = args.topk

    if args.max_eval_batches is not None:
        config.max_eval_batches = args.max_eval_batches

    if args.amp:
        config.amp = True

    if args.no_amp:
        config.amp = False

    config.write_predictions = True
    config.progress = not args.no_progress
    config.command = " ".join(sys.argv)
    config.extra_run_metadata = {
        **dict(config.extra_run_metadata or {}),
        "export_script": "scripts/36_export_knowledge_field_predictions.py",
    }

    config.validate()
    return config


def prepare_trainer(
    *,
    args: argparse.Namespace,
    config: TrainingConfig,
    gnn_input_dir: Path,
) -> KnowledgeFieldTrainer:
    """Build trainer/model and load checkpoint without retraining."""

    dataset = KnowledgeFieldDataset(
        gnn_input_dir,
        task=config.task,
        log1p_target=config.log1p_target,
    )

    trainer = KnowledgeFieldTrainer(
        dataset=dataset,
        output_dir=args.run_dir,
        config=config,
    )

    # Resolve device from trainer internals.
    # The normal trainer.setup() also estimates normalization, but here we must
    # reuse the exact train-only normalization from the original run.
    norm_npz = args.run_dir / "normalization_stats.npz"
    norm_json = args.run_dir / "normalization_stats.json"

    if norm_npz.exists():
        trainer.normalization = load_normalization_stats(norm_npz)
    elif norm_json.exists():
        trainer.normalization = load_normalization_stats(norm_json)
    else:
        raise FileNotFoundError(
            "Could not find normalization stats in run dir. Expected one of: "
            f"{norm_npz} or {norm_json}"
        )

    if config.model_name in GRAPH_MODELS:
        trainer.edge_index = dataset.load_edge_index()

        if config.model_name in EDGE_WEIGHT_MODELS:
            trainer.edge_weight = dataset.load_edge_weight()
        else:
            trainer.edge_weight = None

    feature_groups = trainer._build_feature_groups()

    trainer.model = build_model(
        config.model_name,
        input_dim=dataset.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        feature_groups=feature_groups,
        project_input=config.project_input,
        add_self_loop=config.add_self_loop,
        use_source_branch=config.use_source_branch,
        use_proxy_branch=config.use_proxy_branch,
        dynamic_state_dim=config.dynamic_state_dim,
    ).to(trainer.device)

    trainer.load_checkpoint(args.checkpoint)
    trainer.model.eval()

    if config.progress:
        print("Loaded model for prediction export", flush=True)
        print(f"  run_dir:       {args.run_dir}", flush=True)
        print(f"  checkpoint:    {args.checkpoint}", flush=True)
        print(f"  gnn_input_dir: {gnn_input_dir}", flush=True)
        print(f"  task:          {dataset.task}", flush=True)
        print(f"  model:         {config.model_name}", flush=True)
        print(f"  device:        {trainer.device}", flush=True)
        print(f"  parameters:    {count_parameters(trainer.model):,}", flush=True)
        print("", flush=True)

    return trainer


def export_predictions(args: argparse.Namespace) -> None:
    args.run_dir = args.run_dir.resolve()

    if args.checkpoint is None:
        args.checkpoint = args.run_dir / "best_model.pt"
    else:
        args.checkpoint = args.checkpoint.resolve()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    run_config_path = args.run_dir / "run_config.json"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = training_config_from_checkpoint(ckpt, run_config_path=run_config_path)

    gnn_input_dir = resolve_gnn_input_dir(
        args=args,
        ckpt=ckpt,
        run_config_path=run_config_path,
    )

    config = apply_config_overrides(config, args=args)
    task_for_path = config.task

    # Probe snapshots and fail early if output already exists.
    dataset_probe = KnowledgeFieldDataset(
        gnn_input_dir,
        task=config.task,
        log1p_target=config.log1p_target,
    )

    if args.cutoff_year is not None:
        snapshots = [dataset_probe.snapshot(args.split, args.cutoff_year)]
    else:
        snapshots = list(dataset_probe.iter_snapshots(args.split))

    for snapshot in snapshots:
        out_path = prediction_output_path(
            args.run_dir,
            task=task_for_path,
            split=args.split,
            cutoff_year=snapshot.cutoff_year,
        )

        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Prediction file already exists: {out_path}. "
                "Use --overwrite to replace it."
            )

    del dataset_probe

    trainer = prepare_trainer(
        args=args,
        config=config,
        gnn_input_dir=gnn_input_dir,
    )

    rows: list[dict[str, Any]] = []

    for snapshot in snapshots:
        if config.progress:
            print(
                f"Exporting predictions: split={snapshot.split} "
                f"cutoff={snapshot.cutoff_year}",
                flush=True,
            )

        row, pred_df = trainer.evaluate_snapshot(
            snapshot,
            epoch=trainer.best_epoch if trainer.best_epoch > 0 else None,
        )

        rows.append(row)

        if pred_df is None:
            raise RuntimeError("Prediction export returned no prediction frame.")

        trainer.save_predictions(
            pred_df,
            split=snapshot.split,
            cutoff_year=snapshot.cutoff_year,
        )

        if config.progress:
            out_path = prediction_output_path(
                args.run_dir,
                task=trainer.dataset.task,
                split=snapshot.split,
                cutoff_year=snapshot.cutoff_year,
            )
            print(f"  wrote: {out_path}", flush=True)

    metrics_df = pd.DataFrame(rows)
    metrics_path = args.run_dir / f"exported_{args.split}_metrics_by_cutoff.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "status": "ok",
        "script": "scripts/36_export_knowledge_field_predictions.py",
        "run_dir": str(args.run_dir),
        "checkpoint": str(args.checkpoint),
        "gnn_input_dir": str(gnn_input_dir),
        "task": trainer.dataset.task,
        "split": args.split,
        "cutoff_year": args.cutoff_year,
        "cutoffs_exported": [int(s.cutoff_year) for s in snapshots],
        "model_name": config.model_name,
        "dynamic_state_dim": config.dynamic_state_dim,
        "topk": config.topk,
        "metrics_summary": aggregate_metric_rows(rows, prefix=args.split),
        "model_summary": (
            model_summary_dict(trainer.model)
            if trainer.model is not None
            else None
        ),
    }

    summary_path = args.run_dir / "exported_predictions_summary.json"
    write_json(summary_path, summary)

    if config.progress:
        print("", flush=True)
        print("Prediction export complete.", flush=True)
        print(f"  metrics: {metrics_path}", flush=True)
        print(f"  summary: {summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    export_predictions(args)


if __name__ == "__main__":
    main()