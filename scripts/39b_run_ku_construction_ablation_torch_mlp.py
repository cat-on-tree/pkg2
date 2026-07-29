#!/usr/bin/env python
"""Run KU construction / representation-level ablation with PyTorch MLP.

This script complements scripts/39_run_ku_construction_ablation.py.

It trains GPU-accelerated PyTorch MLP baselines for:

    1. Entity-only pre-KU
    2. Raw pair evidence
    3. Full KU features

and optionally appends existing KU graph / DynamicRDS run metrics.

Main output:

    ku_construction_ablation_main_table.csv

Typical use
-----------

    python scripts/39b_run_ku_construction_ablation_torch_mlp.py \
      --feature-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy_latest_split \
      --output-dir data/results/knowledge_field/ku_construction_ablation/stage07i_ku_layer_ablation_torch_mlp \
      --task translation \
      --test-cutoff-year 2021 \
      --hidden-dims 512 256 128 \
      --dropout 0.15 \
      --batch-size 65536 \
      --epochs 80 \
      --patience 10 \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --device cuda \
      --amp \
      --num-workers 4 \
      --grand-run-dir data/results/knowledge_field/temporal_gnn/local_source_proxy_grand_style_v0_seed42 \
      --dynamic-rds-run-dir data/results/knowledge_field/temporal_gnn/local_source_proxy_dynamic_rds_v2_state8_seed42 \
      --extra-run "KU graph|RDGNN-style|Yes|Yes|Yes|data/results/knowledge_field/temporal_gnn/local_source_proxy_rdgnn_style_v0_seed42" \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Make src/ importable when running this script directly from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pkg2.knowledge_field.metrics import compute_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KU construction ablation with PyTorch MLP."
    )

    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help=(
            "Prediction feature dataset directory containing "
            "prediction_feature_table.parquet and feature_columns.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--task",
        choices=["translation", "patent", "trial"],
        default="translation",
    )

    parser.add_argument(
        "--test-cutoff-year",
        type=int,
        default=2021,
    )

    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[512, 256, 128],
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
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
        "--loss",
        choices=["smoothl1", "mse", "huber"],
        default="smoothl1",
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
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[100, 1000, 5000],
    )

    parser.add_argument(
        "--validation-sample-fraction",
        type=float,
        default=0.10,
        help=(
            "If no validation split exists, sample this fraction from train rows "
            "for validation."
        ),
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional debug/sample limit after reading full table.",
    )

    parser.add_argument(
        "--max-validation-rows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-categorical-cardinality",
        type=int,
        default=50,
        help=(
            "Only keep categorical columns whose train-set cardinality is <= this. "
            "This prevents accidental one-hot explosion."
        ),
    )

    parser.add_argument(
        "--raw-include-pair-type",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--raw-no-pair-type",
        action="store_true",
        help="Disable pair_type in Raw pair evidence.",
    )

    # Existing Stage 07 external runs.
    parser.add_argument(
        "--graphsage-run-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--grand-run-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dynamic-rds-run-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--extra-run",
        nargs="*",
        default=[],
        help=(
            "Optional external rows in the form "
            "Representation|Model|UsesPairHistory|UsesSchema|UsesGraph|run_dir"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def log(args: argparse.Namespace, msg: str) -> None:
    if not args.no_progress:
        print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output suffix: {path}")


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def task_columns(task: str) -> dict[str, str]:
    return {
        "eligible": f"eligible_{task}_task",
        "heat": f"target_future_{task}_heat_3yr",
        "label": f"label_future_{task}_emergence_3yr",
    }


def is_forbidden_feature(col: str) -> bool:
    lower = col.lower()

    if lower in {
        "split",
        "is_train",
        "is_validation",
        "is_val",
        "is_test",
        "train_mask",
        "validation_mask",
        "val_mask",
        "test_mask",
    }:
        return True

    return lower.startswith(
        (
            "target_future_",
            "label_future_",
            "eligible_",
        )
    )


def existing(cols: list[str], frame: pd.DataFrame) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col in frame.columns and col not in seen and not is_forbidden_feature(col):
            out.append(col)
            seen.add(col)
    return out


# ---------------------------------------------------------------------------
# Feature loading / inference
# ---------------------------------------------------------------------------


def load_feature_table(feature_dir: Path) -> pd.DataFrame:
    path = feature_dir / "prediction_feature_table.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction_feature_table.parquet: {path}")
    return pd.read_parquet(path)


def load_feature_spec(feature_dir: Path) -> dict[str, Any]:
    path = feature_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_columns.json: {path}")
    return read_json(path)


def list_from_spec(spec: dict[str, Any], keys: list[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = spec.get(key)
        if isinstance(value, list):
            out.extend(str(x) for x in value)
    return out


def infer_feature_columns(
    frame: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[list[str], list[str]]:
    numeric = list_from_spec(
        spec,
        [
            "numeric_features",
            "numeric_feature_columns",
            "numeric_columns",
            "numerical_features",
            "numerical_columns",
        ],
    )

    categorical = list_from_spec(
        spec,
        [
            "categorical_features",
            "categorical_feature_columns",
            "categorical_columns",
            "category_features",
        ],
    )

    explicit_all = list_from_spec(
        spec,
        [
            "feature_columns",
            "features",
            "all_features",
            "all_feature_columns",
            "model_features",
        ],
    )

    if not numeric and not categorical and explicit_all:
        for col in explicit_all:
            if col not in frame.columns or is_forbidden_feature(col):
                continue
            if pd.api.types.is_numeric_dtype(frame[col]):
                numeric.append(col)
            else:
                categorical.append(col)

    def clean(cols: list[str]) -> list[str]:
        seen = set()
        out = []
        for col in cols:
            if col not in frame.columns:
                continue
            if col in seen:
                continue
            if is_forbidden_feature(col):
                continue
            seen.add(col)
            out.append(col)
        return out

    numeric = clean(numeric)
    categorical = clean(categorical)

    cat_set = set(categorical)
    numeric = [c for c in numeric if c not in cat_set]

    if not numeric and not categorical:
        raise ValueError("Could not infer feature columns from feature_columns.json.")

    return numeric, categorical


# ---------------------------------------------------------------------------
# Representation feature selection
# ---------------------------------------------------------------------------


def select_entity_only_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str]]:
    """Pre-KU entity-only features.

    Avoids pair/KU history columns and tries to use marginal entity_a/entity_b
    features if available.
    """

    numeric: list[str] = []

    entity_markers = (
        "entity_a_",
        "entity_b_",
        "entity1_",
        "entity2_",
        "source_entity_",
        "target_entity_",
        "a_entity_",
        "b_entity_",
    )

    exclude_tokens = (
        "name",
        "id",
        "ku",
        "pair",
        "target",
        "label",
        "eligible",
    )

    for col in numeric_features:
        lower = col.lower()

        if not any(marker in lower for marker in entity_markers):
            continue

        if any(tok in lower for tok in exclude_tokens):
            continue

        numeric.append(col)

    categorical_candidates = [
        "entity_a_type",
        "entity_b_type",
        "entity1_type",
        "entity2_type",
        "source_entity_type",
        "target_entity_type",
    ]

    categorical: list[str] = []
    for col in categorical_candidates:
        if col in frame.columns:
            categorical.append(col)

    return existing(numeric, frame), existing(categorical, frame)


def select_raw_pair_evidence_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    include_pair_type: bool,
) -> tuple[list[str], list[str]]:
    """Minimal pair-level evidence features.

    Uses simple paper trajectory features, avoiding source/proxy/graph features.
    """

    numeric: list[str] = []

    include_tokens = (
        "history_paper",
        "recent_3yr_paper",
        "recent_paper",
        "paper_count",
        "paper_age",
        "paper_recency",
        "first_paper",
        "last_paper",
        "paper_growth",
        "growth_paper",
    )

    exclude_tokens = (
        "target",
        "label",
        "eligible",
        "carrier",
        "source",
        "structural",
        "exposure",
        "cgs_",
        "src_",
        "kud_",
        "proxy",
        "diffusion",
        "neighbor",
        "neighbour",
        "nbr_",
        "patent",
        "trial",
    )

    for col in numeric_features:
        lower = col.lower()

        if any(tok in lower for tok in exclude_tokens):
            continue

        if any(tok in lower for tok in include_tokens):
            numeric.append(col)

    categorical: list[str] = []
    if include_pair_type and "pair_type" in frame.columns:
        categorical.append("pair_type")

    return existing(numeric, frame), existing(categorical, frame)


def select_full_ku_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str]]:
    return existing(numeric_features, frame), existing(categorical_features, frame)


def filter_low_cardinality_categoricals(
    train_df: pd.DataFrame,
    categorical_cols: list[str],
    *,
    max_cardinality: int,
) -> list[str]:
    kept: list[str] = []

    for col in categorical_cols:
        if col not in train_df.columns:
            continue

        nunique = train_df[col].astype("string").nunique(dropna=True)

        if nunique <= max_cardinality:
            kept.append(col)

    return kept


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_to_float32(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    columns: list[str],
    *,
    fit: bool,
) -> np.ndarray:
    x_df = frame[columns].copy()

    if fit:
        x = preprocessor.fit_transform(x_df)
    else:
        x = preprocessor.transform(x_df)

    if hasattr(x, "toarray"):
        x = x.toarray()

    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Torch model / training
# ---------------------------------------------------------------------------


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = int(input_dim)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = int(hidden_dim)

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loss(name: str) -> nn.Module:
    if name == "smoothl1":
        return nn.SmoothL1Loss()
    if name == "huber":
        return nn.HuberLoss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {name}")


def make_loader(
    x: np.ndarray,
    y: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))

    if y is None:
        dataset = TensorDataset(x_tensor)
    else:
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
        dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> np.ndarray:
    model.eval()

    loader = make_loader(
        x,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    preds: list[np.ndarray] = []

    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            out = model(xb)

        preds.append(out.detach().float().cpu().numpy())

    return np.concatenate(preds, axis=0)


def train_torch_mlp(
    *,
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> tuple[nn.Module, dict[str, Any]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = TabularMLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = make_loss(args.loss)

    train_loader = make_loader(
        x_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    eval_batch_size = args.eval_batch_size or args.batch_size
    val_loader = make_loader(
        x_val,
        y_val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0

    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_losses: list[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    pred = model(xb)
                    loss = criterion(pred, yb)

                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "best_validation_loss": best_val_loss,
                "improved": improved,
                "seconds_elapsed": time.time() - start,
            }
        )

        log(
            args,
            (
                f"    epoch {epoch:03d}: "
                f"train_loss={train_loss:.6g} "
                f"val_loss={val_loss:.6g} "
                f"best={best_val_loss:.6g} "
                f"patience={patience_count}/{args.patience}"
            ),
        )

        if patience_count >= args.patience:
            log(args, "    early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_run": len(history),
        "history": history,
        "runtime_seconds": time.time() - start,
        "device": str(device),
    }

    return model, summary


# ---------------------------------------------------------------------------
# Metrics / external runs
# ---------------------------------------------------------------------------


def standardize_metric_names(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": metrics.get("n"),
        "positive_count": metrics.get("positive_count"),
        "positive_rate": metrics.get("positive_rate"),
        "auprc": metrics.get("auprc"),
        "auroc": metrics.get("auroc"),
        "spearman": metrics.get("spearman"),
        "precision_at_100": metrics.get("precision_at_100"),
        "precision_at_1000": metrics.get("precision_at_1000"),
        "precision_at_5000": metrics.get("precision_at_5000"),
        "recall_at_1000": metrics.get("recall_at_1000"),
        "ndcg_at_1000": metrics.get("ndcg_at_1000"),
        "enrichment_at_1000": metrics.get("enrichment_at_1000"),
    }


def standardize_external_metric_names(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": row.get("n"),
        "positive_count": row.get("positive_count"),
        "positive_rate": row.get("positive_rate"),
        "auprc": row.get("auprc"),
        "auroc": row.get("auroc"),
        "spearman": row.get("spearman"),
        "precision_at_100": row.get("precision_at_100"),
        "precision_at_1000": row.get("precision_at_1000"),
        "precision_at_5000": row.get("precision_at_5000"),
        "recall_at_1000": row.get("recall_at_1000"),
        "ndcg_at_1000": row.get("ndcg_at_1000"),
        "enrichment_at_1000": row.get("enrichment_at_1000"),
    }


def read_external_run_metrics(
    *,
    run_dir: Path,
    representation: str,
    model: str,
    uses_pair_history: str,
    uses_schema: str,
    uses_graph: str,
    test_cutoff_year: int,
) -> dict[str, Any]:
    metrics_path = run_dir / "test_metrics_by_cutoff.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing test_metrics_by_cutoff.csv: {metrics_path}")

    df = pd.read_csv(metrics_path)

    row0: dict[str, Any]

    if "cutoff_year" in df.columns:
        sub = df[pd.to_numeric(df["cutoff_year"], errors="coerce") == test_cutoff_year]
        row0 = sub.iloc[0].to_dict() if not sub.empty else df.iloc[0].to_dict()
    elif "test_cutoff" in df.columns:
        sub = df[pd.to_numeric(df["test_cutoff"], errors="coerce") == test_cutoff_year]
        row0 = sub.iloc[0].to_dict() if not sub.empty else df.iloc[0].to_dict()
    else:
        row0 = df.iloc[0].to_dict()

    run_summary_path = run_dir / "run_summary.json"
    run_summary = read_json(run_summary_path) if run_summary_path.exists() else {}

    out = {
        "representation": representation,
        "model": model,
        "uses_ku_pair_history": uses_pair_history,
        "uses_ku_schema": uses_schema,
        "uses_ku_graph": uses_graph,
        "source": "external_run",
        "run_dir": str(run_dir),
        "best_epoch": run_summary.get("best_epoch"),
        "best_val_metric": run_summary.get("best_monitor_metric"),
        "best_val_value": run_summary.get("best_monitor_value"),
    }

    out.update(standardize_external_metric_names(row0))
    return out


def parse_extra_run(text: str) -> tuple[str, str, str, str, str, Path]:
    parts = text.split("|")
    if len(parts) != 6:
        raise ValueError(
            "--extra-run must have 6 pipe-separated fields: "
            "Representation|Model|UsesPairHistory|UsesSchema|UsesGraph|run_dir"
        )

    return (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        parts[4],
        Path(parts[5]),
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def make_splits(
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = task_columns(args.task)

    eligible = frame[cols["eligible"]].fillna(False).astype(bool)

    split = frame["split"].astype(str)
    cutoff = pd.to_numeric(frame["cutoff_year"], errors="coerce")

    train_mask = eligible & (split == "train")
    val_mask = eligible & split.isin(["validation", "val"])
    test_mask = eligible & (split == "test") & (cutoff == args.test_cutoff_year)

    train_df = frame.loc[train_mask].copy()
    val_df = frame.loc[val_mask].copy()
    test_df = frame.loc[test_mask].copy()

    if val_df.empty:
        frac = args.validation_sample_fraction
        if not (0 < frac < 1):
            raise ValueError("--validation-sample-fraction must be between 0 and 1.")

        val_df = train_df.sample(frac=frac, random_state=args.seed)
        train_df = train_df.drop(index=val_df.index).copy()

    if args.max_train_rows is not None and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(n=args.max_train_rows, random_state=args.seed)

    if args.max_validation_rows is not None and len(val_df) > args.max_validation_rows:
        val_df = val_df.sample(n=args.max_validation_rows, random_state=args.seed)

    if args.max_test_rows is not None and len(test_df) > args.max_test_rows:
        test_df = test_df.sample(n=args.max_test_rows, random_state=args.seed)

    return train_df, val_df, test_df


def train_and_evaluate_representation(
    *,
    args: argparse.Namespace,
    representation: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    uses_pair_history: str,
    uses_schema: str,
    uses_graph: str,
) -> dict[str, Any]:
    task_cols = task_columns(args.task)
    heat_col = task_cols["heat"]
    label_col = task_cols["label"]

    categorical_cols = filter_low_cardinality_categoricals(
        train_df,
        categorical_cols,
        max_cardinality=args.max_categorical_cardinality,
    )

    columns = numeric_cols + categorical_cols

    if not columns:
        return {
            "representation": representation,
            "model": "torch_mlp",
            "uses_ku_pair_history": uses_pair_history,
            "uses_ku_schema": uses_schema,
            "uses_ku_graph": uses_graph,
            "source": "skipped_no_features",
        }

    feature_meta = {
        "representation": representation,
        "model": "torch_mlp",
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_count_before_encoding": len(columns),
    }
    write_json(
        args.output_dir
        / "feature_sets"
        / f"{slugify(representation)}_torch_mlp_features.json",
        feature_meta,
    )

    log(
        args,
        (
            f"Training {representation} / torch_mlp: "
            f"numeric={len(numeric_cols)} categorical={len(categorical_cols)}"
        ),
    )

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)

    log(args, "  preprocessing train ...")
    x_train = transform_to_float32(
        preprocessor,
        train_df,
        columns,
        fit=True,
    )

    log(args, "  preprocessing validation ...")
    x_val = transform_to_float32(
        preprocessor,
        val_df,
        columns,
        fit=False,
    )

    log(args, "  preprocessing test ...")
    x_test = transform_to_float32(
        preprocessor,
        test_df,
        columns,
        fit=False,
    )

    y_train = (
        pd.to_numeric(train_df[heat_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    y_val = (
        pd.to_numeric(val_df[heat_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    log(
        args,
        (
            f"  encoded dims: train={x_train.shape} "
            f"val={x_val.shape} test={x_test.shape}"
        ),
    )

    model, train_summary = train_torch_mlp(
        args=args,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        input_dim=x_train.shape[1],
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eval_batch_size = args.eval_batch_size or args.batch_size

    score = predict(
        model,
        x_test,
        device=device,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )

    y_heat = (
        pd.to_numeric(test_df[heat_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    y_label = (
        pd.to_numeric(test_df[label_col], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    metrics = compute_metrics(
        y_heat=y_heat,
        y_label=y_label,
        score=score,
        topk=args.topk,
    )

    pred_out = test_df[
        [
            col
            for col in [
                "ku_id",
                "knowledge_unit_id",
                "cutoff_year",
                "split",
                "pair_type",
                "entity_a_type",
                "entity_b_type",
                label_col,
                heat_col,
            ]
            if col in test_df.columns
        ]
    ].copy()
    pred_out["score"] = score

    pred_path = (
        args.output_dir
        / "predictions"
        / f"{slugify(representation)}_torch_mlp_test_cutoff_{args.test_cutoff_year}.parquet"
    )
    write_dataframe(pred_out, pred_path)

    checkpoint_path = (
        args.output_dir
        / "checkpoints"
        / f"{slugify(representation)}_torch_mlp.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "representation": representation,
            "feature_meta": feature_meta,
            "train_summary": train_summary,
        },
        checkpoint_path,
    )

    history_path = (
        args.output_dir
        / "training_history"
        / f"{slugify(representation)}_torch_mlp_training_history.csv"
    )
    write_dataframe(pd.DataFrame(train_summary["history"]), history_path)

    row = {
        "representation": representation,
        "model": "torch_mlp",
        "uses_ku_pair_history": uses_pair_history,
        "uses_ku_schema": uses_schema,
        "uses_ku_graph": uses_graph,
        "source": "trained_torch_mlp",
        "feature_count_before_encoding": len(columns),
        "encoded_feature_count": x_train.shape[1],
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "best_epoch": train_summary["best_epoch"],
        "best_val_metric": "validation_loss",
        "best_val_value": train_summary["best_validation_loss"],
        "runtime_seconds": train_summary["runtime_seconds"],
        "prediction_path": str(pred_path),
        "checkpoint_path": str(checkpoint_path),
        **standardize_metric_names(metrics),
    }

    return row


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"

    cols = [c for c in columns if c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            if math.isnan(x):
                return ""
            return f"{x:.6g}"
        return str(x)

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    return "\n".join(lines)


def write_report(args: argparse.Namespace, rows: pd.DataFrame) -> None:
    report: list[str] = []

    report.append("# KU Construction Ablation Report: PyTorch MLP\n")

    report.append("## Configuration\n")
    report.append("```json")
    report.append(
        json.dumps(
            {
                "feature_dir": str(args.feature_dir),
                "task": args.task,
                "test_cutoff_year": args.test_cutoff_year,
                "hidden_dims": args.hidden_dims,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "device": args.device,
                "amp": args.amp,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    report.append("```\n")

    report.append("## Main comparison\n")

    cols = [
        "representation",
        "model",
        "uses_ku_pair_history",
        "uses_ku_schema",
        "uses_ku_graph",
        "auprc",
        "auroc",
        "spearman",
        "precision_at_1000",
        "ndcg_at_1000",
        "enrichment_at_1000",
    ]
    report.append(markdown_table(rows, cols))
    report.append("")

    report.append("## Interpretation guide\n")
    report.append(
        """
- `Entity-only pre-KU` tests whether marginal entity-level information is sufficient.
- `Raw pair evidence` tests whether minimal pair-level paper trajectory is sufficient.
- `Full KU features` tests whether complete KU-level features improve over minimal pair evidence.
- External `KU graph` and `KU field dynamics` rows are read from existing Stage 07 runs.
"""
    )

    path = args.output_dir / "ku_construction_ablation_report.md"
    path.write_text("\n".join(report), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.feature_dir = args.feature_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.raw_no_pair_type:
        args.raw_include_pair_type = False

    prepare_output_dir(args.output_dir, args.overwrite)
    set_seed(args.seed)

    log(args, "Running KU construction ablation with PyTorch MLP ...")
    log(args, f"  feature_dir: {args.feature_dir}")
    log(args, f"  output_dir:  {args.output_dir}")
    log(args, f"  task:        {args.task}")
    log(args, f"  device:      {args.device}")
    log(args, "")

    frame = load_feature_table(args.feature_dir)
    spec = load_feature_spec(args.feature_dir)
    numeric_features, categorical_features = infer_feature_columns(frame, spec)

    task_cols = task_columns(args.task)
    required = [
        "split",
        "cutoff_year",
        task_cols["eligible"],
        task_cols["heat"],
        task_cols["label"],
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    train_df, val_df, test_df = make_splits(frame, args=args)

    log(
        args,
        (
            f"Rows: train={len(train_df):,} "
            f"validation={len(val_df):,} "
            f"test={len(test_df):,}"
        ),
    )
    log(
        args,
        (
            f"Feature columns: numeric={len(numeric_features)} "
            f"categorical={len(categorical_features)}"
        ),
    )

    feature_sets: dict[str, dict[str, Any]] = {}

    ent_num, ent_cat = select_entity_only_features(
        frame,
        numeric_features,
        categorical_features,
    )
    feature_sets["Entity-only pre-KU"] = {
        "numeric": ent_num,
        "categorical": ent_cat,
        "uses_pair_history": "No",
        "uses_schema": "No",
        "uses_graph": "No",
    }

    raw_num, raw_cat = select_raw_pair_evidence_features(
        frame,
        numeric_features,
        categorical_features,
        include_pair_type=args.raw_include_pair_type,
    )
    feature_sets["Raw pair evidence"] = {
        "numeric": raw_num,
        "categorical": raw_cat,
        "uses_pair_history": "Minimal",
        "uses_schema": "Partial" if "pair_type" in raw_cat else "No",
        "uses_graph": "No",
    }

    full_num, full_cat = select_full_ku_features(
        frame,
        numeric_features,
        categorical_features,
    )
    feature_sets["Full KU features"] = {
        "numeric": full_num,
        "categorical": full_cat,
        "uses_pair_history": "Yes",
        "uses_schema": "Yes",
        "uses_graph": "No",
    }

    feature_info = {
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "feature_sets": {
            name: {
                "numeric_count": len(cfg["numeric"]),
                "categorical_count": len(cfg["categorical"]),
                "numeric_preview": cfg["numeric"][:30],
                "categorical_preview": cfg["categorical"][:30],
            }
            for name, cfg in feature_sets.items()
        },
    }
    write_json(args.output_dir / "feature_selection_summary.json", feature_info)

    rows: list[dict[str, Any]] = []

    for representation, cfg in feature_sets.items():
        row = train_and_evaluate_representation(
            args=args,
            representation=representation,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            numeric_cols=cfg["numeric"],
            categorical_cols=cfg["categorical"],
            uses_pair_history=cfg["uses_pair_history"],
            uses_schema=cfg["uses_schema"],
            uses_graph=cfg["uses_graph"],
        )
        rows.append(row)

    # External rows.
    if args.graphsage_run_dir is not None:
        rows.append(
            read_external_run_metrics(
                run_dir=args.graphsage_run_dir.resolve(),
                representation="KU graph",
                model="GraphSAGE",
                uses_pair_history="Yes",
                uses_schema="Yes",
                uses_graph="Yes",
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    if args.grand_run_dir is not None:
        rows.append(
            read_external_run_metrics(
                run_dir=args.grand_run_dir.resolve(),
                representation="KU graph",
                model="GRAND-style",
                uses_pair_history="Yes",
                uses_schema="Yes",
                uses_graph="Yes",
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    if args.dynamic_rds_run_dir is not None:
        rows.append(
            read_external_run_metrics(
                run_dir=args.dynamic_rds_run_dir.resolve(),
                representation="KU field dynamics",
                model="DynamicRDS v2 state8",
                uses_pair_history="Yes",
                uses_schema="Yes",
                uses_graph="Yes",
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    for extra in args.extra_run:
        rep, model, uses_pair, uses_schema, uses_graph, run_dir = parse_extra_run(extra)
        rows.append(
            read_external_run_metrics(
                run_dir=run_dir.resolve(),
                representation=rep,
                model=model,
                uses_pair_history=uses_pair,
                uses_schema=uses_schema,
                uses_graph=uses_graph,
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    result = pd.DataFrame(rows)

    rep_order = {
        "Entity-only pre-KU": 0,
        "Raw pair evidence": 1,
        "Full KU features": 2,
        "KU graph": 3,
        "KU field dynamics": 4,
    }
    model_order = {
        "torch_mlp": 0,
        "GraphSAGE": 1,
        "GRAND-style": 2,
        "RDGNN-style": 3,
        "DynamicRDS v2 state8": 4,
    }

    result["_rep_order"] = result["representation"].map(rep_order).fillna(99)
    result["_model_order"] = result["model"].map(model_order).fillna(99)
    result = (
        result.sort_values(["_rep_order", "_model_order", "model"])
        .drop(columns=["_rep_order", "_model_order"])
        .reset_index(drop=True)
    )

    metrics_path = args.output_dir / "ku_construction_ablation_metrics.csv"
    main_path = args.output_dir / "ku_construction_ablation_main_table.csv"

    write_dataframe(result, metrics_path)

    main_cols = [
        "representation",
        "model",
        "uses_ku_pair_history",
        "uses_ku_schema",
        "uses_ku_graph",
        "auprc",
        "auroc",
        "spearman",
        "precision_at_1000",
        "ndcg_at_1000",
        "enrichment_at_1000",
    ]
    write_dataframe(result[[c for c in main_cols if c in result.columns]], main_path)

    write_json(
        args.output_dir / "run_config.json",
        {
            "script": "scripts/39b_run_ku_construction_ablation_torch_mlp.py",
            "feature_dir": str(args.feature_dir),
            "output_dir": str(args.output_dir),
            "task": args.task,
            "test_cutoff_year": args.test_cutoff_year,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "device": args.device,
            "amp": args.amp,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "topk": args.topk,
            "max_categorical_cardinality": args.max_categorical_cardinality,
            "external_runs": {
                "graphsage_run_dir": str(args.graphsage_run_dir) if args.graphsage_run_dir else None,
                "grand_run_dir": str(args.grand_run_dir) if args.grand_run_dir else None,
                "dynamic_rds_run_dir": str(args.dynamic_rds_run_dir) if args.dynamic_rds_run_dir else None,
                "extra_run": args.extra_run,
            },
        },
    )

    write_report(args, result)

    log(args, "")
    log(args, "PyTorch MLP KU construction ablation complete.")
    log(args, f"  metrics: {metrics_path}")
    log(args, f"  main:    {main_path}")
    log(args, f"  report:  {args.output_dir / 'ku_construction_ablation_report.md'}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()