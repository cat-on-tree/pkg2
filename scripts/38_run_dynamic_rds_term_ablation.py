#!/usr/bin/env python
"""Run DynamicRDS term ablation experiments without modifying src/.

This script keeps src/pkg2 unchanged by defining a local ablated DynamicRDS
subclass and monkey-patching the trainer's build_model function at runtime.

Ablation formula:

    Full:
      R + S + P + D - Lambda

    no_reaction:
      S + P + D - Lambda

    no_source:
      R + P + D - Lambda

    no_proxy:
      R + S + D - Lambda

    no_diffusion:
      R + S + P - Lambda

    no_decay:
      R + S + P + D

Typical use
-----------

    python scripts/38_run_dynamic_rds_term_ablation.py \
      --gnn-input-dir data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_proxy_latest_split \
      --output-dir data/results/knowledge_field/temporal_gnn/local_source_proxy_dynamic_rds_v2_state8_no_diffusion_seed42 \
      --ablation no_diffusion \
      --task translation \
      --dynamic-state-dim 8 \
      --hidden-dim 128 \
      --num-layers 2 \
      --num-neighbors 15 10 \
      --batch-size 16384 \
      --epochs 50 \
      --patience 8 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --dropout 0.1 \
      --device cuda \
      --amp \
      --num-workers 4 \
      --tensorboard \
      --write-predictions \
      --overwrite

Notes
-----

- This script intentionally does not add new model names to src/.
- For reproducibility, ablation metadata is written to extra_run_metadata.
- If you need post-hoc prediction export for these ablation checkpoints, either
  rerun this script with --write-predictions or create an ablation-aware export
  script. The existing script 36 loads the original DynamicRDS forward behavior.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Make src/ importable when running this script directly from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pkg2.knowledge_field import trainer as trainer_mod
from pkg2.knowledge_field.models import DynamicRDSRegressor
from pkg2.knowledge_field.trainer import TrainingConfig


ABLATION_CHOICES = (
    "full",
    "no_reaction",
    "no_source",
    "no_proxy",
    "no_diffusion",
    "no_decay",
)


class AblatedDynamicRDSRegressor(DynamicRDSRegressor):
    """DynamicRDS with runtime term switches for one-term ablation.

    This subclass keeps the same module structure as DynamicRDSRegressor but
    zeroes out selected terms in forward().
    """

    def __init__(
        self,
        *args: Any,
        use_reaction_term: bool = True,
        use_diffusion_term: bool = True,
        use_decay_term: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.use_reaction_term = bool(use_reaction_term)
        self.use_diffusion_term = bool(use_diffusion_term)
        self.use_decay_term = bool(use_decay_term)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_x, source_x, proxy_x = self.split_features(x)

        # Explicit vector knowledge state u_i(t).
        u = self.state_norm(self.state_encoder(local_x))
        reaction_input = torch.cat([u, local_x], dim=1)

        # R_i(t): local reaction.
        if self.use_reaction_term:
            reaction = self.reaction(reaction_input)
        else:
            reaction = torch.zeros_like(u)

        # S_i(t): gated source injection.
        if (
            self.source is not None
            and self.source_gate is not None
            and source_x.size(1) > 0
        ):
            source_gate = torch.sigmoid(self.source_gate(source_x))
            source = source_gate * self.source(source_x)
        else:
            source = torch.zeros_like(u)

        # P_i(t): gated proxy / prior field term.
        if (
            self.proxy is not None
            and self.proxy_gate is not None
            and proxy_x.size(1) > 0
        ):
            proxy_gate = torch.sigmoid(self.proxy_gate(proxy_x))
            proxy = proxy_gate * self.proxy(proxy_x)
        else:
            proxy = torch.zeros_like(u)

        # D_i(t): true graph diffusion flux.
        if self.use_diffusion_term:
            diffusion = self.flux_diffusion(
                u,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        else:
            diffusion = torch.zeros_like(u)

        # Lambda_i(t): nonnegative vector decay / dissipation.
        if self.use_decay_term:
            decay_rate = F.softplus(self.decay_rate(reaction_input))
            decay = decay_rate * u
        else:
            decay = torch.zeros_like(u)

        raw_du = reaction + source + proxy + diffusion - decay

        # Stabilize one-step dynamics.
        du_scale = F.softplus(self.log_du_scale)
        du = torch.tanh(raw_du) * du_scale

        dt = F.softplus(self.log_dt)
        u_next = u + dt * du

        h_context = self.context(x)

        readout_x = torch.cat(
            [
                u,
                du,
                u_next,
                h_context,
            ],
            dim=1,
        )

        return self.readout(readout_x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DynamicRDS one-term ablation models."
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
        help="Output directory for this ablation run.",
    )

    parser.add_argument(
        "--ablation",
        choices=ABLATION_CHOICES,
        required=True,
        help="Which DynamicRDS term to remove.",
    )

    parser.add_argument(
        "--task",
        choices=["translation", "patent", "trial"],
        default="translation",
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
        "--dynamic-state-dim",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-neighbors",
        nargs="+",
        type=int,
        default=[15, 10],
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
    )

    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="auprc",
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
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
    )

    parser.add_argument(
        "--max-batches-per-cutoff",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
    )

    # Existing model options.
    parser.add_argument(
        "--project-input",
        action="store_true",
        help="Kept for compatibility; not used by DynamicRDS.",
    )

    parser.add_argument(
        "--no-add-self-loop",
        action="store_true",
        help="Kept for compatibility; not used by DynamicRDS.",
    )

    parser.add_argument(
        "--no-infer-feature-groups",
        action="store_true",
        help="Disable heuristic local/source/proxy feature grouping.",
    )

    # TensorBoard / logging.
    parser.add_argument(
        "--tensorboard",
        action="store_true",
    )

    parser.add_argument(
        "--tensorboard-root",
        type=Path,
        default=Path("tensorboard_logs"),
    )

    parser.add_argument(
        "--tensorboard-run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tensorboard-histograms",
        action="store_true",
    )

    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help=(
            "Recommended for ablation runs. Script 36 is not ablation-aware by default."
        ),
    )

    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
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

    return (
        f"{timestamp}"
        f"__{args.task}"
        f"__dynamic_rds"
        f"__{args.ablation}"
        f"__{dataset_name}"
        f"__{output_name}"
        f"__seed{args.seed}"
    )


def ablation_flags(ablation: str) -> dict[str, bool]:
    return {
        "use_reaction_term": ablation != "no_reaction",
        "use_source_branch": ablation != "no_source",
        "use_proxy_branch": ablation != "no_proxy",
        "use_diffusion_term": ablation != "no_diffusion",
        "use_decay_term": ablation != "no_decay",
    }


def build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.tensorboard_run_name is None:
        tensorboard_run_name = make_timestamped_run_name(args)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_run_name = f"{timestamp}__{args.tensorboard_run_name}"

    flags = ablation_flags(args.ablation)
    command = " ".join(sys.argv)

    return TrainingConfig(
        task=args.task,
        model_name="dynamic_rds",
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
        use_source_branch=flags["use_source_branch"],
        use_proxy_branch=flags["use_proxy_branch"],
        infer_feature_groups=not args.no_infer_feature_groups,
        dynamic_state_dim=args.dynamic_state_dim,
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
            "script": "scripts/38_run_dynamic_rds_term_ablation.py",
            "ablation": args.ablation,
            "dynamic_rds_formula_full": "R + S + P + D - Lambda",
            "use_reaction_term": flags["use_reaction_term"],
            "use_source_branch": flags["use_source_branch"],
            "use_proxy_branch": flags["use_proxy_branch"],
            "use_diffusion_term": flags["use_diffusion_term"],
            "use_decay_term": flags["use_decay_term"],
            "tensorboard_log_dir": str(
                args.tensorboard_root.resolve() / tensorboard_run_name
            )
            if args.tensorboard
            else None,
        },
    )


def make_ablation_build_model(
    *,
    ablation: str,
):
    """Return a build_model replacement closing over ablation flags."""

    original_build_model = trainer_mod.build_model
    flags = ablation_flags(ablation)

    def patched_build_model(
        model_name: str,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        feature_groups=None,
        **kwargs: Any,
    ):
        if model_name != "dynamic_rds":
            return original_build_model(
                model_name,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                feature_groups=feature_groups,
                **kwargs,
            )

        return AblatedDynamicRDSRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            feature_groups=feature_groups,
            use_source_branch=bool(kwargs.get("use_source_branch", True)),
            use_proxy_branch=bool(kwargs.get("use_proxy_branch", True)),
            state_dim=int(kwargs.get("dynamic_state_dim", 16)),
            use_reaction_term=flags["use_reaction_term"],
            use_diffusion_term=flags["use_diffusion_term"],
            use_decay_term=flags["use_decay_term"],
        )

    return patched_build_model


def main() -> None:
    args = parse_args()

    args.gnn_input_dir = args.gnn_input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.tensorboard_root = args.tensorboard_root.resolve()

    config = build_config(args)

    if config.progress:
        print("Running DynamicRDS term ablation training", flush=True)
        print(f"  gnn_input_dir: {args.gnn_input_dir}", flush=True)
        print(f"  output_dir:    {args.output_dir}", flush=True)
        print(f"  task:          {config.task}", flush=True)
        print(f"  ablation:      {args.ablation}", flush=True)
        print(f"  state_dim:     {config.dynamic_state_dim}", flush=True)
        print("", flush=True)

    # Monkey-patch only inside this script process.
    trainer_mod.build_model = make_ablation_build_model(ablation=args.ablation)

    trainer_mod.run_training(
        gnn_input_dir=args.gnn_input_dir,
        output_dir=args.output_dir,
        config=config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()