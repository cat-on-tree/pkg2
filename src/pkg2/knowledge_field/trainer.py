"""Training utilities for Temporal Knowledge Field models.

This module provides the shared trainer for:

- MLP no-graph baseline
- GraphSAGE baseline
- Weighted diffusion GNN
- Reaction-Diffusion-Source GNN

It is designed to work with datasets produced by:

    scripts/34_build_knowledge_field_gnn_inputs.py

and data utilities in:

    src/pkg2/knowledge_field/data.py

Responsibilities
----------------

1. Build model / optimizer / loss.
2. Estimate train-only normalization.
3. Load each temporal snapshot.
4. Train with either:
   - direct eligible-node batches for MLP
   - PyG NeighborLoader for graph models
5. Evaluate validation/test metrics.
6. Save checkpoints, predictions, CSV metrics, TensorBoard logs.

Optimizer location
------------------

For now, optimizer helpers are defined here because they are tightly coupled to
the trainer. If optimizer/scheduler logic becomes more complex, move them later
to:

    src/pkg2/knowledge_field/optim.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pkg2.knowledge_field.data import (
    KnowledgeFieldDataset,
    NormalizationStats,
    SnapshotBatch,
    SnapshotInfo,
    make_mlp_batches,
    make_neighbor_loader,
    make_pyg_data,
    save_normalization_stats,
)
from pkg2.knowledge_field.metrics import (
    aggregate_metric_rows,
    compute_split_cutoff_metrics,
    get_monitor_value,
)
from pkg2.knowledge_field.models import (
    FeatureGroups,
    build_model,
    count_parameters,
    infer_feature_groups_from_names,
    model_summary_dict,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


GRAPH_MODELS = {
    "graphsage",
    "weighted_diffusion",
    "reaction_diffusion_source",
    "rdgnn_style",
    "grand_style",
    "dynamic_rds"
}


@dataclass
class TrainingConfig:
    """Configuration for one Knowledge Field training run."""

    task: str = "translation"
    model_name: str = "graphsage"

    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    # Neighbor sampling only applies to graph models.
    num_neighbors: list[int] = field(default_factory=lambda: [15, 10])
    batch_size: int = 8192
    num_workers: int = 0

    epochs: int = 50
    patience: int = 8

    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9

    loss: str = "smoothl1"
    log1p_target: bool = True

    monitor_metric: str = "auprc"
    monitor_split: str = "validation"
    topk: list[int] = field(default_factory=lambda: [100, 1000, 5000])

    device: str = "cuda"
    amp: bool = False

    seed: int = 42

    # Debug controls.
    max_train_cutoffs: int | None = None
    max_batches_per_cutoff: int | None = None
    max_eval_batches: int | None = None

    # Normalization.
    normalization_chunk_size: int = 200_000

    # Model options.
    project_input: bool = False
    add_self_loop: bool = True
    use_source_branch: bool = True
    use_proxy_branch: bool = True
    infer_feature_groups: bool = True
    dynamic_state_dim: int = 16

    # Logging / outputs.
    tensorboard: bool = False
    tensorboard_histograms: bool = False
    tensorboard_root: str = "tensorboard_logs"
    tensorboard_run_name: str | None = None
    write_predictions: bool = False
    save_every_epoch: bool = False
    progress: bool = True

    # Provenance.
    command: str | None = None
    extra_run_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if self.dynamic_state_dim <= 0:
            raise ValueError("dynamic_state_dim must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if self.patience <= 0:
            raise ValueError("patience must be positive.")

        self.topk = sorted(set(int(k) for k in self.topk if int(k) > 0))
        if not self.topk:
            raise ValueError("topk must contain at least one positive integer.")

        if self.model_name in GRAPH_MODELS:
            if len(self.num_neighbors) != self.num_layers:
                raise ValueError(
                    "For graph models, len(num_neighbors) should match "
                    f"num_layers. Got num_neighbors={self.num_neighbors}, "
                    f"num_layers={self.num_layers}."
                )

        if self.optimizer not in {"adamw", "adam", "sgd"}:
            raise ValueError(
                "optimizer must be one of {'adamw', 'adam', 'sgd'}."
            )

        if self.loss not in {"smoothl1", "mse", "huber"}:
            raise ValueError("loss must be one of {'smoothl1', 'mse', 'huber'}.")


@dataclass
class EpochResult:
    epoch: int
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    history_row: dict[str, Any]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_text: str) -> torch.device:
    if device_text == "cpu":
        return torch.device("cpu")

    if device_text.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_text)

    return torch.device("cpu")


def build_optimizer(
    parameters,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build optimizer.

    Currently defined in trainer.py. Later it can be moved to optim.py if
    optimizer/scheduler logic becomes more complex.
    """

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_loss(loss_name: str) -> nn.Module:
    if loss_name in {"smoothl1", "huber"}:
        return nn.SmoothL1Loss()

    if loss_name == "mse":
        return nn.MSELoss()

    raise ValueError(f"Unsupported loss: {loss_name}")


def make_grad_scaler(config: TrainingConfig, device: torch.device):
    if not (config.amp and device.type == "cuda"):
        return None

    # Compatibility across PyTorch versions.
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def amp_context(config: TrainingConfig, device: torch.device):
    if config.amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


class TensorBoardRunLogger:
    """Thin wrapper around SummaryWriter."""

    def __init__(
        self,
        output_dir: Path,
        *,
        enabled: bool,
        log_dir: Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.writer = None
        self.log_dir = None

        if not self.enabled:
            return

        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard requested, but torch.utils.tensorboard.SummaryWriter "
                "is unavailable. Install tensorboard or disable TensorBoard logging."
            )

        if log_dir is None:
            log_dir = output_dir / "tensorboard"

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.writer is None:
            return

        if value is None:
            return

        try:
            value_f = float(value)
        except Exception:
            return

        def add_scalar(self, tag: str, value: Any, step: int) -> None:
            if self.writer is None:
                return

            if value is None:
                return

            try:
                value_f = float(value)
            except Exception:
                return

            if not math.isfinite(value_f):
                return

            self.writer.add_scalar(tag, value_f, step)

        self.writer.add_scalar(tag, value_f, step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.writer is None:
            return
        self.writer.add_text(tag, text, step)

    def add_histograms(
        self,
        model: nn.Module,
        *,
        epoch: int,
        include_grads: bool,
    ) -> None:
        if self.writer is None:
            return

        for name, param in model.named_parameters():
            self.writer.add_histogram(
                f"parameters/{name}",
                param.detach().cpu(),
                epoch,
            )

            if include_grads and param.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{name}",
                    param.grad.detach().cpu(),
                    epoch,
                )

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class KnowledgeFieldTrainer:
    """Trainer for Temporal Knowledge Field models."""

    def __init__(
        self,
        *,
        dataset: KnowledgeFieldDataset,
        output_dir: str | Path,
        config: TrainingConfig,
    ) -> None:
        config.validate()

        self.dataset = dataset
        self.output_dir = Path(output_dir).resolve()
        self.config = config

        self.device = resolve_device(config.device)

        self.normalization: NormalizationStats | None = None
        self.edge_index: torch.Tensor | None = None
        self.edge_weight: torch.Tensor | None = None

        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion: nn.Module | None = None
        self.scaler: Any = None

        self.best_epoch = -1
        self.best_monitor_value = -math.inf
        self.no_improve = 0

        self.history_rows: list[dict[str, Any]] = []
        self.validation_metric_rows: list[dict[str, Any]] = []
        self.test_metric_rows: list[dict[str, Any]] = []

        tb_log_dir = None
        if config.tensorboard:
            run_name = config.tensorboard_run_name
            if run_name is None:
                run_name = self.output_dir.name

            tb_log_dir = Path(config.tensorboard_root).resolve() / run_name

        self.tb = TensorBoardRunLogger(
            self.output_dir,
            enabled=config.tensorboard,
            log_dir=tb_log_dir,
        )

    def log(self, message: str) -> None:
        if self.config.progress:
            print(message, flush=True)

    def _cleanup_memory(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "best_model.pt"

    def prepare_output_dir(self, *, overwrite: bool = False) -> None:
        if self.output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {self.output_dir}. "
                    "Use overwrite=True or --overwrite."
                )
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        set_seed(self.config.seed)

        self.log("Setting up Knowledge Field trainer ...")
        self.log(f"  dataset:    {self.dataset.gnn_input_dir}")
        self.log(f"  output_dir: {self.output_dir}")
        self.log(f"  task:       {self.dataset.task}")
        self.log(f"  model:      {self.config.model_name}")
        self.log(f"  device:     {self.device}")
        self.log("")

        self.normalization = self.dataset.estimate_normalization(
            max_train_cutoffs=self.config.max_train_cutoffs,
            chunk_size=self.config.normalization_chunk_size,
        )

        save_normalization_stats(
            self.output_dir,
            self.normalization,
            self.dataset.numeric_features,
        )

        if self.config.model_name in GRAPH_MODELS:
            self.log("Loading graph tensors ...")
            self.edge_index = self.dataset.load_edge_index()

            if self.config.model_name in {
                "weighted_diffusion",
                "reaction_diffusion_source",
                "rdgnn_style",
                "grand_style",
                "dynamic_rds"
            }:
                self.edge_weight = self.dataset.load_edge_weight()
            else:
                self.edge_weight = None

            self.log(f"  edge_index: {tuple(self.edge_index.shape)}")
            if self.edge_weight is not None:
                self.log(f"  edge_weight: {tuple(self.edge_weight.shape)}")
            self.log("")

        feature_groups = self._build_feature_groups()

        self.model = build_model(
            self.config.model_name,
            input_dim=self.dataset.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            feature_groups=feature_groups,
            project_input=self.config.project_input,
            add_self_loop=self.config.add_self_loop,
            use_source_branch=self.config.use_source_branch,
            use_proxy_branch=self.config.use_proxy_branch,
            dynamic_state_dim=self.config.dynamic_state_dim,
        ).to(self.device)

        self.optimizer = build_optimizer(
            self.model.parameters(),
            optimizer_name=self.config.optimizer,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum,
        )

        self.criterion = build_loss(self.config.loss)
        self.scaler = make_grad_scaler(self.config, self.device)

        self._write_run_config(feature_groups)
        self._log_tensorboard_config(feature_groups)

        self.log("Model:")
        self.log(f"  class: {self.model.__class__.__name__}")
        self.log(f"  trainable parameters: {count_parameters(self.model):,}")
        self.log("")

    def _build_feature_groups(self) -> FeatureGroups | None:
        if self.config.model_name not in {
            "reaction_diffusion_source",
            "dynamic_rds"
        }:
            return None

        if not self.config.infer_feature_groups:
            return None

        return infer_feature_groups_from_names(
            numeric_features=self.dataset.numeric_features,
            categorical_one_hot_dim=self.dataset.categorical_one_hot_dim,
        )

    def _write_run_config(self, feature_groups: FeatureGroups | None) -> None:
        config_dict = asdict(self.config)

        config_dict.update(
            {
                "dataset": self.dataset.summary_dict(),
                "device_resolved": str(self.device),
                "input_dim": self.dataset.input_dim,
                "numeric_dim": self.dataset.numeric_dim,
                "categorical_one_hot_dim": self.dataset.categorical_one_hot_dim,
                "model_summary": (
                    model_summary_dict(self.model)
                    if self.model is not None
                    else None
                ),
                "feature_groups": (
                    {
                        "local_indices": feature_groups.local_indices,
                        "source_indices": feature_groups.source_indices,
                        "proxy_indices": feature_groups.proxy_indices,
                        "categorical_indices": feature_groups.categorical_indices,
                    }
                    if feature_groups is not None
                    else None
                ),
            }
        )

        write_json(self.output_dir / "run_config.json", config_dict)

    def _log_tensorboard_config(self, feature_groups: FeatureGroups | None) -> None:
        command = self.config.command or " ".join(sys.argv)

        self.tb.add_text(
            "run/command",
            f"```bash\n{command}\n```",
            step=0,
        )

        config_payload = {
            "config": asdict(self.config),
            "dataset": self.dataset.summary_dict(),
            "device": str(self.device),
            "model_parameter_count": (
                count_parameters(self.model)
                if self.model is not None
                else None
            ),
            "feature_groups": (
                {
                    "local_indices": feature_groups.local_indices,
                    "source_indices": feature_groups.source_indices,
                    "proxy_indices": feature_groups.proxy_indices,
                    "categorical_indices": feature_groups.categorical_indices,
                }
                if feature_groups is not None
                else None
            ),
        }

        self.tb.add_text(
            "run/config_json",
            "```json\n"
            + json.dumps(config_payload, ensure_ascii=False, indent=2, default=str)
            + "\n```",
            step=0,
        )

    def train(self) -> dict[str, Any]:
        if self.model is None:
            self.setup()

        assert self.model is not None
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.normalization is not None

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_result = self.train_one_epoch(epoch)

            self.history_rows.append(epoch_result.history_row)
            self.validation_metric_rows.extend(epoch_result.validation_rows)

            self._write_history_files()

            monitor_value = float(epoch_result.history_row["monitor_value"])

            if monitor_value > self.best_monitor_value:
                self.best_monitor_value = monitor_value
                self.best_epoch = epoch
                self.no_improve = 0
                self.save_checkpoint(self.best_checkpoint_path, epoch, monitor_value)
                self.log(f"  new best checkpoint saved at epoch={epoch}")
            else:
                self.no_improve += 1
                self.log(f"  no improvement: {self.no_improve}/{self.config.patience}")

            if self.config.save_every_epoch:
                self.save_checkpoint(
                    self.output_dir / f"checkpoint_epoch_{epoch}.pt",
                    epoch,
                    monitor_value,
                )

            if self.config.tensorboard_histograms:
                self.tb.add_histograms(
                    self.model,
                    epoch=epoch,
                    include_grads=True,
                )

            self.tb.flush()
            self.log("")

            if self.no_improve >= self.config.patience:
                self.log("Early stopping triggered.")
                break

        if self.best_checkpoint_path.exists():
            self.load_checkpoint(self.best_checkpoint_path)

        test_summary = self.evaluate_test()

        run_summary = {
            "status": "ok",
            "dataset": self.dataset.summary_dict(),
            "output_dir": str(self.output_dir),
            "task": self.dataset.task,
            "model_name": self.config.model_name,
            "best_epoch": self.best_epoch,
            "best_monitor_metric": self.config.monitor_metric,
            "best_monitor_split": self.config.monitor_split,
            "best_monitor_value": self.best_monitor_value,
            "test_metrics": test_summary,
            "runtime_seconds": time.time() - start_time,
            "device": str(self.device),
            "trainable_parameters": (
                count_parameters(self.model)
                if self.model is not None
                else None
            ),
        }

        write_json(self.output_dir / "run_summary.json", run_summary)

        self.tb.add_text(
            "run/summary_json",
            "```json\n"
            + json.dumps(run_summary, ensure_ascii=False, indent=2, default=str)
            + "\n```",
            step=max(self.best_epoch, 0),
        )

        for key, value in test_summary.items():
            self.tb.add_scalar(
                f"test/{key}",
                value,
                step=max(self.best_epoch, 0),
            )

        self.tb.close()

        self.log("Training complete.")
        self.log(f"  best_epoch: {self.best_epoch}")
        self.log(f"  best {self.config.monitor_metric}: {self.best_monitor_value:.6f}")
        self.log(f"  output_dir: {self.output_dir}")

        return run_summary

    def train_one_epoch(self, epoch: int) -> EpochResult:
        epoch_start = time.time()

        self.log(f"Epoch {epoch}/{self.config.epochs}")

        train_rows: list[dict[str, Any]] = []

        for snapshot in self.dataset.iter_snapshots(
            "train",
            max_items=self.config.max_train_cutoffs,
        ):
            self.log(f"  train cutoff={snapshot.cutoff_year}")
            row = self.train_snapshot(snapshot, epoch=epoch)
            train_rows.append(row)

            self.log(
                "    "
                f"loss={row['loss']:.6f} "
                f"examples={int(row['examples']):,} "
                f"batches={int(row['batches']):,}"
            )

            step = (epoch - 1) * 1000 + snapshot.cutoff_year
            self.tb.add_scalar(
                f"train_cutoff_{snapshot.cutoff_year}/loss",
                row["loss"],
                step,
            )
            self.tb.add_scalar(
                f"train_cutoff_{snapshot.cutoff_year}/examples",
                row["examples"],
                step,
            )
            self.tb.add_scalar(
                f"train_cutoff_{snapshot.cutoff_year}/batches",
                row["batches"],
                step,
            )

        validation_rows = self.evaluate_split("validation", epoch=epoch)

        train_loss = (
            float(np.mean([row["loss"] for row in train_rows]))
            if train_rows
            else None
        )
        train_examples = (
            float(np.sum([row["examples"] for row in train_rows]))
            if train_rows
            else 0.0
        )
        train_batches = (
            float(np.sum([row["batches"] for row in train_rows]))
            if train_rows
            else 0.0
        )

        val_agg = aggregate_metric_rows(
            validation_rows,
            prefix="validation",
        )

        monitor_value = get_monitor_value(
            val_agg,
            split=self.config.monitor_split,
            monitor_metric=self.config.monitor_metric,
        )

        epoch_seconds = time.time() - epoch_start

        history_row: dict[str, Any] = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "train_loss": train_loss,
            "train_examples": train_examples,
            "train_batches": train_batches,
            "monitor_metric": self.config.monitor_metric,
            "monitor_split": self.config.monitor_split,
            "monitor_value": monitor_value,
            **val_agg,
        }

        self.tb.add_scalar("epoch/train_loss", train_loss, epoch)
        self.tb.add_scalar("epoch/train_examples", train_examples, epoch)
        self.tb.add_scalar("epoch/train_batches", train_batches, epoch)
        self.tb.add_scalar("epoch/seconds", epoch_seconds, epoch)
        self.tb.add_scalar("epoch/monitor_value", monitor_value, epoch)

        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0].get("lr", None)
            self.tb.add_scalar("epoch/lr", lr, epoch)

        for key, value in val_agg.items():
            self.tb.add_scalar(f"epoch/{key}", value, epoch)

        self.log(
            "  epoch summary: "
            f"train_loss={train_loss:.6f} "
            f"{self.config.monitor_metric}={monitor_value:.6f} "
            f"seconds={epoch_seconds:.1f}"
        )

        return EpochResult(
            epoch=epoch,
            train_rows=train_rows,
            validation_rows=validation_rows,
            history_row=history_row,
        )

    def train_snapshot(
        self,
        snapshot: SnapshotInfo,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.normalization is not None

        batch = self.dataset.load_snapshot_batch(
            snapshot,
            normalization=self.normalization,
        )

        if self.config.model_name == "mlp":
            stats = self._train_mlp_snapshot(batch)
        elif self.config.model_name in GRAPH_MODELS:
            stats = self._train_graph_snapshot(batch)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

        row = {
            "epoch": epoch,
            "split": snapshot.split,
            "cutoff_year": snapshot.cutoff_year,
            **stats,
        }

        del batch
        self._cleanup_memory()

        return row

    def _train_mlp_snapshot(
        self,
        batch: SnapshotBatch,
    ) -> dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.criterion is not None

        self.model.train()

        total_loss = 0.0
        total_examples = 0
        batch_count = 0

        for idx in make_mlp_batches(
            batch.eligible,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
        ):
            if (
                self.config.max_batches_per_cutoff is not None
                and batch_count >= self.config.max_batches_per_cutoff
            ):
                break

            xb = batch.x[idx].to(self.device, non_blocking=True)
            yb = batch.y_train[idx].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with amp_context(self.config, self.device):
                pred = self.model(xb)
                loss = self.criterion(pred, yb)

            self._backward_step(loss)

            n = int(idx.numel())
            total_loss += float(loss.detach().cpu()) * n
            total_examples += n
            batch_count += 1

        return {
            "loss": total_loss / max(total_examples, 1),
            "examples": float(total_examples),
            "batches": float(batch_count),
        }

    def _train_graph_snapshot(
        self,
        batch: SnapshotBatch,
    ) -> dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.edge_index is not None

        self.model.train()

        pyg_data = make_pyg_data(
            batch,
            self.edge_index,
            edge_weight=self.edge_weight,
        )

        loader = make_neighbor_loader(
            pyg_data,
            batch.eligible,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        total_loss = 0.0
        total_examples = 0
        batch_count = 0

        for pyg_batch in loader:
            if (
                self.config.max_batches_per_cutoff is not None
                and batch_count >= self.config.max_batches_per_cutoff
            ):
                break

            pyg_batch = pyg_batch.to(self.device, non_blocking=True)
            batch_size = int(pyg_batch.batch_size)

            self.optimizer.zero_grad(set_to_none=True)

            with amp_context(self.config, self.device):
                pred_all = self._model_forward_pyg_batch(pyg_batch)
                pred = pred_all[:batch_size]
                yb = pyg_batch.y[:batch_size]
                loss = self.criterion(pred, yb)

            self._backward_step(loss)

            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            batch_count += 1

        return {
            "loss": total_loss / max(total_examples, 1),
            "examples": float(total_examples),
            "batches": float(batch_count),
        }

    def _backward_step(self, loss: torch.Tensor) -> None:
        assert self.optimizer is not None

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def _model_forward_pyg_batch(self, pyg_batch: Any) -> torch.Tensor:
        assert self.model is not None

        edge_weight = getattr(pyg_batch, "edge_weight", None)

        return self.model(
            pyg_batch.x,
            pyg_batch.edge_index,
            edge_weight=edge_weight,
        )

    def evaluate_split(
        self,
        split: str,
        *,
        epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for snapshot in self.dataset.iter_snapshots(split):
            self.log(f"  {split} cutoff={snapshot.cutoff_year}")
            row, pred_df = self.evaluate_snapshot(snapshot, epoch=epoch)

            rows.append(row)

            preview = []
            for name in [
                "auprc",
                "auroc",
                "spearman",
                "precision_at_1000",
                "ndcg_at_1000",
            ]:
                value = row.get(name)
                if value is not None and not pd.isna(value):
                    preview.append(f"{name}={float(value):.5f}")

            if preview:
                self.log("    " + " ".join(preview))

            if self.config.write_predictions and pred_df is not None:
                self.save_predictions(
                    pred_df,
                    split=split,
                    cutoff_year=snapshot.cutoff_year,
                )

            if epoch is not None:
                for key, value in row.items():
                    if key in {"split", "cutoff_year", "epoch"}:
                        continue
                    self.tb.add_scalar(
                        f"{split}_cutoff_{snapshot.cutoff_year}/{key}",
                        value,
                        epoch,
                    )

        return rows

    @torch.no_grad()
    def evaluate_snapshot(
        self,
        snapshot: SnapshotInfo,
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame | None]:
        assert self.model is not None
        assert self.normalization is not None

        batch = self.dataset.load_snapshot_batch(
            snapshot,
            normalization=self.normalization,
        )

        if self.config.model_name == "mlp":
            score, node_idx = self._predict_mlp_snapshot(batch)
        else:
            score, node_idx = self._predict_graph_snapshot(batch)

        y_heat = batch.y_heat[node_idx]
        y_label = batch.y_label[node_idx]

        result = compute_split_cutoff_metrics(
            split=snapshot.split,
            cutoff_year=snapshot.cutoff_year,
            y_heat=y_heat,
            y_label=y_label,
            score=score,
            topk=self.config.topk,
        )

        row = result.to_row(epoch=epoch)

        pred_df = None
        if self.config.write_predictions:
            pred_df = pd.DataFrame(
                {
                    "node_idx": node_idx.astype(np.int64),
                    "split": snapshot.split,
                    "cutoff_year": snapshot.cutoff_year,
                    "y_heat": y_heat.astype(np.float32),
                    "y_label": y_label.astype(np.int8),
                    "score": score.astype(np.float32),
                }
            )

        del batch
        self._cleanup_memory()

        return row, pred_df

    @torch.no_grad()
    def _predict_mlp_snapshot(
        self,
        batch: SnapshotBatch,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None

        self.model.eval()

        scores: list[np.ndarray] = []
        node_ids: list[np.ndarray] = []

        batch_count = 0

        for idx in make_mlp_batches(
            batch.eligible,
            batch_size=self.config.batch_size,
            shuffle=False,
            seed=self.config.seed,
        ):
            if (
                self.config.max_eval_batches is not None
                and batch_count >= self.config.max_eval_batches
            ):
                break

            xb = batch.x[idx].to(self.device, non_blocking=True)

            with amp_context(self.config, self.device):
                pred = self.model(xb)

            scores.append(pred.detach().float().cpu().numpy())
            node_ids.append(idx.numpy().astype(np.int64))

            batch_count += 1

        if not scores:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return (
            np.concatenate(scores).astype(np.float32),
            np.concatenate(node_ids).astype(np.int64),
        )

    @torch.no_grad()
    def _predict_graph_snapshot(
        self,
        batch: SnapshotBatch,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        assert self.edge_index is not None

        self.model.eval()

        pyg_data = make_pyg_data(
            batch,
            self.edge_index,
            edge_weight=self.edge_weight,
        )

        loader = make_neighbor_loader(
            pyg_data,
            batch.eligible,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        scores: list[np.ndarray] = []
        node_ids: list[np.ndarray] = []

        batch_count = 0

        for pyg_batch in loader:
            if (
                self.config.max_eval_batches is not None
                and batch_count >= self.config.max_eval_batches
            ):
                break

            pyg_batch = pyg_batch.to(self.device, non_blocking=True)
            batch_size = int(pyg_batch.batch_size)

            with amp_context(self.config, self.device):
                pred_all = self._model_forward_pyg_batch(pyg_batch)
                pred = pred_all[:batch_size]

            input_node_idx = pyg_batch.n_id[:batch_size].detach().cpu().numpy()

            scores.append(pred.detach().float().cpu().numpy())
            node_ids.append(input_node_idx.astype(np.int64))

            batch_count += 1

        if not scores:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return (
            np.concatenate(scores).astype(np.float32),
            np.concatenate(node_ids).astype(np.int64),
        )

    def evaluate_test(self) -> dict[str, float | None]:
        self.log("Evaluating test split ...")

        rows = self.evaluate_split("test", epoch=self.best_epoch if self.best_epoch > 0 else None)
        self.test_metric_rows = rows

        pd.DataFrame(rows).to_csv(
            self.output_dir / "test_metrics_by_cutoff.csv",
            index=False,
        )

        test_summary = aggregate_metric_rows(
            rows,
            prefix="test",
        )

        for key, value in test_summary.items():
            self.tb.add_scalar(
                f"test/{key}",
                value,
                step=max(self.best_epoch, 0),
            )

        return test_summary

    def _write_history_files(self) -> None:
        pd.DataFrame(self.history_rows).to_csv(
            self.output_dir / "training_history.csv",
            index=False,
        )

        pd.DataFrame(self.validation_metric_rows).to_csv(
            self.output_dir / "validation_metrics_by_cutoff.csv",
            index=False,
        )

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        monitor_value: float,
    ) -> None:
        assert self.model is not None
        assert self.optimizer is not None

        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": int(epoch),
                "monitor_metric": self.config.monitor_metric,
                "monitor_split": self.config.monitor_split,
                "monitor_value": float(monitor_value),
                "config": asdict(self.config),
                "dataset_summary": self.dataset.summary_dict(),
                "model_summary": model_summary_dict(self.model),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        assert self.model is not None

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if self.optimizer is not None and "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                # Optimizer state loading is convenient but not required for
                # final evaluation.
                pass

        return ckpt

    def save_predictions(
        self,
        pred_df: pd.DataFrame,
        *,
        split: str,
        cutoff_year: int,
    ) -> None:
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        out_base = pred_dir / (
            f"predictions_{self.dataset.task}_{split}_cutoff_{cutoff_year}"
        )

        try:
            pred_df.to_parquet(out_base.with_suffix(".parquet"), index=False)
        except Exception:
            pred_df.to_csv(out_base.with_suffix(".csv"), index=False)


def run_training(
    *,
    gnn_input_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convenience entry point for scripts."""

    dataset = KnowledgeFieldDataset(
        gnn_input_dir,
        task=config.task,
        log1p_target=config.log1p_target,
    )

    trainer = KnowledgeFieldTrainer(
        dataset=dataset,
        output_dir=output_dir,
        config=config,
    )

    trainer.prepare_output_dir(overwrite=overwrite)
    trainer.setup()
    return trainer.train()