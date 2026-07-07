#!/usr/bin/env python
"""Run Temporal Knowledge Field neural training.

This is the thin CLI entry point for Knowledge Field experiments.

The core implementation lives in:

    src/pkg2/knowledge_field/data.py
    src/pkg2/knowledge_field/metrics.py
    src/pkg2/knowledge_field/models.py
    src/pkg2/knowledge_field/trainer.py

Example: local laptop debug
---------------------------

    python scripts/35_run_temporal_knowledge_field_gnn.py \\
      --gnn-input-dir data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_latest_split \\
      --output-dir data/results/knowledge_field/temporal_gnn/debug_local_source_graphsage \\
      --task translation \\
      --model graphsage \\
      --hidden-dim 64 \\
      --num-layers 2 \\
      --num-neighbors 10 5 \\
      --batch-size 4096 \\
      --epochs 3 \\
      --patience 2 \\
      --max-batches-per-cutoff 20 \\
      --lr 1e-3 \\
      --weight-decay 1e-4 \\
      --dropout 0.1 \\
      --device cuda \\
      --amp \\
      --num-workers 0 \\
      --tensorboard \\
      --overwrite

Example: A100 / Slurm formal run
--------------------------------

    python scripts/35_run_temporal_knowledge_field_gnn.py \\
      --gnn-input-dir data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_latest_split \\
      --output-dir data/results/knowledge_field/temporal_gnn/local_source_graphsage_v0_seed42 \\
      --task translation \\
      --model graphsage \\
      --hidden-dim 128 \\
      --num-layers 2 \\
      --num-neighbors 15 10 \\
      --batch-size 16384 \\
      --epochs 50 \\
      --patience 8 \\
      --lr 1e-3 \\
      --weight-decay 1e-4 \\
      --dropout 0.1 \\
      --device cuda \\
      --amp \\
      --num-workers 4 \\
      --tensorboard \\
      --overwrite

TensorBoard
-----------

By default, when --tensorboard is enabled, logs are written to:

    tensorboard_logs/<timestamp>_<run_name>/

from the project root.

Open with:

    tensorboard --logdir tensorboard_logs
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make src/ importable when running this script directly from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pkg2.knowledge_field.trainer import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Temporal Knowledge Field GNN models."
    )

    parser.add_argument(
        "--gnn-input-dir",
        type=Path,
        required=True,
        help="GNN input directory produced by script 34.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for checkpoints, CSV metrics, config, predictions.",
    )

    parser.add_argument(
        "--task",
        choices=["translation", "patent", "trial"],
        default="translation",
    )

    parser.add_argument(
        "--model",
        choices=[
            "mlp",
            "graphsage",
            "weighted_diffusion",
            "reaction_diffusion_source",
        ],
        default="graphsage",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--num-neighbors",
        nargs="+",
        type=int,
        default=[15, 10],
        help=(
            "Neighbor sampling fanouts for graph models. "
            "Length should match --num-layers."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam", "sgd"],
        default="adamw",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--loss",
        choices=["smoothl1", "mse", "huber"],
        default="smoothl1",
    )

    parser.add_argument(
        "--no-log1p-target",
        action="store_true",
        help="Disable log1p transform for heat target.",
    )

    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="auprc",
        help=(
            "Metric used for early stopping. Examples: auprc, auroc, "
            "spearman, ndcg_at_1000, precision_at_1000."
        ),
    )

    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[100, 1000, 5000],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "NeighborLoader workers. Use 0 on Windows; 4-8 is usually OK "
            "on Linux/A100."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Debug controls.
    parser.add_argument(
        "--max-train-cutoffs",
        type=int,
        default=None,
        help="Debug: train on only the first N train cutoffs.",
    )

    parser.add_argument(
        "--max-batches-per-cutoff",
        type=int,
        default=None,
        help="Debug: limit train batches per cutoff.",
    )

    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug: limit validation/test batches.",
    )

    # Model options.
    parser.add_argument(
        "--project-input",
        action="store_true",
        help="Use an input projection before GraphSAGE layers.",
    )

    parser.add_argument(
        "--no-add-self-loop",
        action="store_true",
        help="Disable self loops in weighted diffusion layers.",
    )

    parser.add_argument(
        "--no-source-branch",
        action="store_true",
        help="Disable source branch in reaction_diffusion_source model.",
    )

    parser.add_argument(
        "--no-proxy-branch",
        action="store_true",
        help="Disable proxy branch in reaction_diffusion_source model.",
    )

    parser.add_argument(
        "--no-infer-feature-groups",
        action="store_true",
        help="Disable heuristic feature grouping for RDS model.",
    )

    # TensorBoard / logging.
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard logs.",
    )

    parser.add_argument(
        "--tensorboard-root",
        type=Path,
        default=Path("tensorboard_logs"),
        help=(
            "TensorBoard root directory. Default: tensorboard_logs/ "
            "under the project root/current working directory."
        ),
    )

    parser.add_argument(
        "--tensorboard-run-name",
        type=str,
        default=None,
        help=(
            "Optional TensorBoard run directory name. If omitted, a "
            "timestamped name is generated."
        ),
    )

    parser.add_argument(
        "--tensorboard-histograms",
        action="store_true",
        help="Log parameter/gradient histograms.",
    )

    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write test predictions for eligible nodes.",
    )

    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save one checkpoint per epoch.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def make_timestamped_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name = args.gnn_input_dir.name
    output_name = args.output_dir.name

    # Keep names readable but not too long.
    return (
        f"{timestamp}"
        f"__{args.task}"
        f"__{args.model}"
        f"__{dataset_name}"
        f"__{output_name}"
        f"__seed{args.seed}"
    )


def build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.tensorboard_run_name is None:
        tensorboard_run_name = make_timestamped_run_name(args)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_run_name = f"{timestamp}__{args.tensorboard_run_name}"

    command = " ".join(sys.argv)

    return TrainingConfig(
        task=args.task,
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        patience=args.patience,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        loss=args.loss,
        log1p_target=not args.no_log1p_target,
        monitor_metric=args.monitor_metric,
        monitor_split="validation",
        topk=args.topk,
        device=args.device,
        amp=args.amp,
        seed=args.seed,
        max_train_cutoffs=args.max_train_cutoffs,
        max_batches_per_cutoff=args.max_batches_per_cutoff,
        max_eval_batches=args.max_eval_batches,
        project_input=args.project_input,
        add_self_loop=not args.no_add_self_loop,
        use_source_branch=not args.no_source_branch,
        use_proxy_branch=not args.no_proxy_branch,
        infer_feature_groups=not args.no_infer_feature_groups,
        tensorboard=args.tensorboard,
        tensorboard_histograms=args.tensorboard_histograms,
        tensorboard_root=str(args.tensorboard_root),
        tensorboard_run_name=tensorboard_run_name,
        write_predictions=args.write_predictions,
        save_every_epoch=args.save_every_epoch,
        progress=not args.no_progress,
        command=command,
        extra_run_metadata={
            "repo_root": str(REPO_ROOT),
            "script": "scripts/35_run_temporal_knowledge_field_gnn.py",
            "tensorboard_log_dir": str(
                args.tensorboard_root.resolve() / tensorboard_run_name
            )
            if args.tensorboard
            else None,
        },
    )


def main() -> None:
    args = parse_args()

    args.gnn_input_dir = args.gnn_input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.tensorboard_root = args.tensorboard_root.resolve()

    config = build_config(args)

    if config.progress:
        print("Running Temporal Knowledge Field training", flush=True)
        print(f"  gnn_input_dir: {args.gnn_input_dir}", flush=True)
        print(f"  output_dir:    {args.output_dir}", flush=True)
        print(f"  task:          {config.task}", flush=True)
        print(f"  model:         {config.model_name}", flush=True)
        if config.tensorboard:
            print(
                "  tensorboard:   "
                f"{Path(config.tensorboard_root).resolve() / config.tensorboard_run_name}",
                flush=True,
            )
        print("", flush=True)

    run_training(
        gnn_input_dir=args.gnn_input_dir,
        output_dir=args.output_dir,
        config=config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()