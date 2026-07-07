"""Data utilities for Temporal Knowledge Field GNN experiments.

This module loads the algorithm-ready GNN inputs produced by:

    scripts/34_build_knowledge_field_gnn_inputs.py

It intentionally avoids using absolute paths stored inside split_manifest.json,
because the dataset may be transferred from a Windows workstation to a Linux
A100/Slurm server.

Main responsibilities
---------------------

1. Discover train / validation / test snapshot directories.
2. Load feature schema and categorical vocabularies.
3. Estimate train-only numeric normalization statistics.
4. Materialize per-cutoff node feature tensors.
5. Load task labels and eligibility masks.
6. Load KU-KU graph tensors.
7. Build PyG Data / NeighborLoader objects.

Expected GNN input layout
-------------------------

    gnn_input_dir/
      graph/
        edge_index.npy
        edge_weight.npy
        ku_node_index.parquet
        ku_ku_edges.parquet
        graph_summary.json

      snapshots/
        train/
          cutoff_2005/
            x_numeric.npy
            x_categorical.npy
            y_translation_heat.npy
            y_translation_label.npy
            mask_translation_eligible.npy
            ...
        validation/
          cutoff_2019/
          cutoff_2020/
        test/
          cutoff_2021/

      feature_schema.json
      categorical_vocab.json
      gnn_input_summary.json

Notes
-----

- Numeric normalization statistics must be estimated from train snapshots only.
- Categorical features are one-hot encoded in this v0 data layer. This keeps
  the first GraphSAGE baseline simple and stable across local/A100 runs.
- The first GraphSAGE baseline may ignore edge_weight. Future weighted /
  physical diffusion layers should use edge_weight explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
except Exception:  # pragma: no cover
    Data = None
    NeighborLoader = None


TASKS = ("translation", "patent", "trial")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SnapshotInfo:
    """Metadata for one temporal snapshot."""

    split: str
    cutoff_year: int
    path: Path


@dataclass(frozen=True)
class FeatureSchema:
    """Feature schema loaded from feature_schema.json."""

    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def numeric_dim(self) -> int:
        return len(self.numeric_features)

    @property
    def categorical_feature_count(self) -> int:
        return len(self.categorical_features)


@dataclass(frozen=True)
class NormalizationStats:
    """Train-only normalization statistics for numeric features."""

    mean: np.ndarray
    std: np.ndarray
    source: str = "train_snapshots_only"

    def to_json_dict(self, numeric_features: list[str]) -> dict[str, Any]:
        return {
            "source": self.source,
            "numeric_feature_count": int(len(numeric_features)),
            "numeric_features": list(numeric_features),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


@dataclass
class SnapshotBatch:
    """Loaded arrays/tensors for one snapshot.

    Attributes
    ----------
    x:
        Float tensor of shape [num_nodes, input_dim].
    y_train:
        Float tensor used for training loss. This may be log1p-transformed.
    y_heat:
        Raw heat target as numpy array.
    y_label:
        Binary emergence label as numpy array.
    eligible:
        Boolean eligibility mask as numpy array.
    """

    snapshot: SnapshotInfo
    x: torch.Tensor
    y_train: torch.Tensor
    y_heat: np.ndarray
    y_label: np.ndarray
    eligible: np.ndarray


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def task_file_names(task: str) -> dict[str, str]:
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}. Expected one of {TASKS}.")

    return {
        "heat": f"y_{task}_heat.npy",
        "label": f"y_{task}_label.npy",
        "eligible": f"mask_{task}_eligible.npy",
    }


def load_feature_schema(gnn_input_dir: Path) -> FeatureSchema:
    schema_path = Path(gnn_input_dir) / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing feature schema: {schema_path}")

    raw = read_json(schema_path)

    numeric = raw.get("numeric_features", [])
    categorical = raw.get("categorical_features", [])

    if not isinstance(numeric, list):
        raise ValueError("feature_schema.json field `numeric_features` must be a list.")

    if not isinstance(categorical, list):
        raise ValueError(
            "feature_schema.json field `categorical_features` must be a list."
        )

    return FeatureSchema(
        numeric_features=[str(x) for x in numeric],
        categorical_features=[str(x) for x in categorical],
    )


def load_categorical_vocab(
    gnn_input_dir: Path,
) -> dict[str, dict[str, int]]:
    vocab_path = Path(gnn_input_dir) / "categorical_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing categorical vocab: {vocab_path}")

    raw = read_json(vocab_path)

    out: dict[str, dict[str, int]] = {}
    for key, mapping in raw.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"Categorical vocab for {key} must be a dict.")
        out[str(key)] = {str(k): int(v) for k, v in mapping.items()}

    return out


def categorical_one_hot_dim(
    categorical_features: list[str],
    categorical_vocab: dict[str, dict[str, int]],
) -> int:
    total = 0
    for col in categorical_features:
        if col not in categorical_vocab:
            raise ValueError(f"Missing categorical vocabulary for feature: {col}")
        total += len(categorical_vocab[col])
    return int(total)


def discover_snapshots(gnn_input_dir: Path) -> dict[str, list[SnapshotInfo]]:
    """Discover split/cutoff snapshot folders from the local filesystem."""

    gnn_input_dir = Path(gnn_input_dir)
    snapshots_root = gnn_input_dir / "snapshots"

    if not snapshots_root.exists():
        raise FileNotFoundError(f"Missing snapshots directory: {snapshots_root}")

    result: dict[str, list[SnapshotInfo]] = {
        split: [] for split in SPLITS
    }

    for split in SPLITS:
        split_dir = snapshots_root / split
        if not split_dir.exists():
            continue

        for child in sorted(split_dir.glob("cutoff_*")):
            if not child.is_dir():
                continue

            try:
                cutoff_year = int(child.name.replace("cutoff_", ""))
            except ValueError:
                continue

            result[split].append(
                SnapshotInfo(
                    split=split,
                    cutoff_year=cutoff_year,
                    path=child,
                )
            )

        result[split] = sorted(result[split], key=lambda x: x.cutoff_year)

    if not result["train"]:
        raise ValueError(f"No train snapshots found under {snapshots_root}")

    if not result["validation"]:
        raise ValueError(f"No validation snapshots found under {snapshots_root}")

    if not result["test"]:
        raise ValueError(f"No test snapshots found under {snapshots_root}")

    return result


def validate_snapshot_files(
    snapshots: dict[str, list[SnapshotInfo]],
    task: str,
) -> None:
    names = task_file_names(task)

    missing: list[Path] = []

    for split_items in snapshots.values():
        for item in split_items:
            required = [
                item.path / "x_numeric.npy",
                item.path / "x_categorical.npy",
                item.path / names["heat"],
                item.path / names["label"],
                item.path / names["eligible"],
            ]

            missing.extend([p for p in required if not p.exists()])

    if missing:
        preview = "\n".join(str(p) for p in missing[:30])
        suffix = "\n..." if len(missing) > 30 else ""
        raise FileNotFoundError(
            "Missing required snapshot files:\n"
            f"{preview}{suffix}"
        )


def iter_row_chunks(
    array: np.ndarray,
    chunk_size: int,
) -> Iterable[np.ndarray]:
    """Yield row chunks from a 2D numpy or memmap array."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    n = int(array.shape[0])
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield array[start:end]


def estimate_train_numeric_stats(
    train_snapshots: list[SnapshotInfo],
    numeric_feature_count: int,
    *,
    max_train_cutoffs: int | None = None,
    chunk_size: int = 200_000,
) -> NormalizationStats:
    """Estimate mean/std from train snapshots only.

    This uses chunked memmap reads to avoid loading all train snapshots into RAM
    at once.
    """

    selected = train_snapshots
    if max_train_cutoffs is not None:
        selected = selected[:max_train_cutoffs]

    if not selected:
        raise ValueError("No train snapshots provided for normalization stats.")

    total_count = 0
    total_sum = np.zeros(numeric_feature_count, dtype=np.float64)
    total_sq_sum = np.zeros(numeric_feature_count, dtype=np.float64)

    for item in selected:
        x_path = item.path / "x_numeric.npy"
        x = np.load(x_path, mmap_mode="r")

        if x.ndim != 2:
            raise ValueError(f"Expected 2D x_numeric array: {x_path}")

        if x.shape[1] != numeric_feature_count:
            raise ValueError(
                f"Numeric feature count mismatch in {x_path}: "
                f"{x.shape[1]} != {numeric_feature_count}"
            )

        for chunk in iter_row_chunks(x, chunk_size=chunk_size):
            block = np.asarray(chunk, dtype=np.float64)
            total_count += int(block.shape[0])
            total_sum += block.sum(axis=0)
            total_sq_sum += np.square(block).sum(axis=0)

    if total_count <= 0:
        raise ValueError("No rows observed while estimating normalization stats.")

    mean = total_sum / total_count
    var = total_sq_sum / total_count - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    # Avoid exploding near-constant columns.
    std = np.where(std < 1e-6, 1.0, std)

    return NormalizationStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def load_normalization_stats(path: Path) -> NormalizationStats:
    """Load stats from .npz or .json."""

    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path)
        return NormalizationStats(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )

    if path.suffix == ".json":
        raw = read_json(path)
        return NormalizationStats(
            mean=np.asarray(raw["mean"], dtype=np.float32),
            std=np.asarray(raw["std"], dtype=np.float32),
            source=str(raw.get("source", "unknown")),
        )

    raise ValueError(f"Unsupported normalization stats format: {path}")


def save_normalization_stats(
    output_dir: Path,
    stats: NormalizationStats,
    numeric_features: list[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "normalization_stats.npz",
        mean=stats.mean,
        std=stats.std,
    )

    write_json(
        output_dir / "normalization_stats.json",
        stats.to_json_dict(numeric_features),
    )


def build_snapshot_features(
    snapshot_dir: Path,
    *,
    normalization: NormalizationStats,
    categorical_features: list[str],
    categorical_vocab: dict[str, dict[str, int]],
) -> torch.Tensor:
    """Build dense float node feature matrix for one snapshot.

    Memory-optimized version.

    Numeric features are normalized with train-only stats. Categorical features
    are one-hot encoded directly into the final output matrix to avoid
    x_num + x_cat + concatenate peak memory.
    """

    snapshot_dir = Path(snapshot_dir)

    x_numeric_path = snapshot_dir / "x_numeric.npy"
    x_categorical_path = snapshot_dir / "x_categorical.npy"

    if not x_numeric_path.exists():
        raise FileNotFoundError(f"Missing numeric feature file: {x_numeric_path}")

    if not x_categorical_path.exists():
        raise FileNotFoundError(f"Missing categorical feature file: {x_categorical_path}")

    x_num_raw = np.load(x_numeric_path, mmap_mode="r")

    if x_num_raw.ndim != 2:
        raise ValueError(f"Expected 2D x_numeric array: {x_numeric_path}")

    num_nodes = int(x_num_raw.shape[0])
    numeric_dim = int(x_num_raw.shape[1])

    if numeric_dim != normalization.mean.shape[0]:
        raise ValueError(
            f"Numeric feature mismatch in {snapshot_dir}: "
            f"{numeric_dim} != {normalization.mean.shape[0]}"
        )

    cat_dim = categorical_one_hot_dim(categorical_features, categorical_vocab)
    total_dim = numeric_dim + cat_dim

    # Allocate final output once.
    x = np.empty((num_nodes, total_dim), dtype=np.float32)

    # Copy numeric features directly into final matrix, then normalize in-place.
    x_numeric_view = x[:, :numeric_dim]
    np.copyto(x_numeric_view, x_num_raw, casting="unsafe")
    x_numeric_view -= normalization.mean[None, :]
    x_numeric_view /= normalization.std[None, :]

    if cat_dim == 0:
        return torch.from_numpy(x)

    x_cat_raw = np.load(x_categorical_path, mmap_mode="r")

    if x_cat_raw.ndim != 2:
        raise ValueError(f"Expected 2D x_categorical array: {x_categorical_path}")

    if x_cat_raw.shape[1] != len(categorical_features):
        raise ValueError(
            f"Categorical feature mismatch in {snapshot_dir}: "
            f"{x_cat_raw.shape[1]} != {len(categorical_features)}"
        )

    # Initialize categorical one-hot block directly inside final matrix.
    x[:, numeric_dim:] = 0.0

    offset = numeric_dim
    rows = np.arange(num_nodes)

    for j, col in enumerate(categorical_features):
        vocab = categorical_vocab[col]
        vocab_size = len(vocab)

        values = np.asarray(x_cat_raw[:, j], dtype=np.int64)
        values = np.clip(values, 0, vocab_size - 1)

        x[rows, offset + values] = 1.0
        offset += vocab_size

    return torch.from_numpy(x)


def load_task_arrays(
    snapshot_dir: Path,
    task: str,
    *,
    log1p_target: bool = True,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Load task targets and mask for one snapshot.

    Returns
    -------
    y_train:
        Torch float tensor used by the training loss. This is log1p transformed
        when log1p_target=True.
    y_heat:
        Raw heat target numpy array.
    y_label:
        Binary emergence label numpy array.
    eligible:
        Boolean eligibility mask numpy array.
    """

    names = task_file_names(task)
    snapshot_dir = Path(snapshot_dir)

    heat_path = snapshot_dir / names["heat"]
    label_path = snapshot_dir / names["label"]
    eligible_path = snapshot_dir / names["eligible"]

    for path in [heat_path, label_path, eligible_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing task array: {path}")

    y_heat = np.array(
        np.load(heat_path, mmap_mode="r"),
        dtype=np.float32,
        copy=True,
    )

    if log1p_target:
        y_train_np = np.log1p(np.maximum(y_heat, 0.0)).astype(np.float32)
    else:
        y_train_np = y_heat.astype(np.float32)

    y_label = np.array(
        np.load(label_path, mmap_mode="r"),
        dtype=np.int8,
        copy=True,
    )

    eligible = np.array(
        np.load(eligible_path, mmap_mode="r"),
        dtype=bool,
        copy=True,
    )

    if not (len(y_heat) == len(y_label) == len(eligible)):
        raise ValueError(
            f"Task array length mismatch in {snapshot_dir}: "
            f"heat={len(y_heat)}, label={len(y_label)}, eligible={len(eligible)}"
        )

    return torch.from_numpy(y_train_np), y_heat, y_label, eligible


def load_edge_index(
    gnn_input_dir: Path,
    *,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    path = Path(gnn_input_dir) / "graph" / "edge_index.npy"

    if not path.exists():
        raise FileNotFoundError(f"Missing edge_index.npy: {path}")

    edge_index_np = np.array(
        np.load(path, mmap_mode="r"),
        dtype=np.int64,
        copy=True,
    )

    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index shape [2, num_edges], got {edge_index_np.shape}"
        )

    return torch.as_tensor(edge_index_np, dtype=dtype)


def load_edge_weight(
    gnn_input_dir: Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    path = Path(gnn_input_dir) / "graph" / "edge_weight.npy"

    if not path.exists():
        raise FileNotFoundError(f"Missing edge_weight.npy: {path}")

    edge_weight_np = np.array(
        np.load(path, mmap_mode="r"),
        dtype=np.float32,
        copy=True,
    )

    if edge_weight_np.ndim != 1:
        raise ValueError(
            f"Expected edge_weight shape [num_edges], got {edge_weight_np.shape}"
        )

    return torch.as_tensor(edge_weight_np, dtype=dtype)


def load_node_index_frame(gnn_input_dir: Path) -> pd.DataFrame:
    """Load KU node index table.

    Requires pandas parquet support through pyarrow or fastparquet.
    """

    path = Path(gnn_input_dir) / "graph" / "ku_node_index.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing ku_node_index.parquet: {path}")

    return pd.read_parquet(path)


class KnowledgeFieldDataset:
    """Dataset manager for Temporal Knowledge Field GNN inputs."""

    def __init__(
        self,
        gnn_input_dir: str | Path,
        *,
        task: str = "translation",
        log1p_target: bool = True,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}. Expected one of {TASKS}.")

        self.gnn_input_dir = Path(gnn_input_dir).resolve()
        self.task = task
        self.log1p_target = bool(log1p_target)

        self.feature_schema = load_feature_schema(self.gnn_input_dir)
        self.categorical_vocab = load_categorical_vocab(self.gnn_input_dir)
        self.snapshots = discover_snapshots(self.gnn_input_dir)

        validate_snapshot_files(self.snapshots, task=self.task)

    @property
    def numeric_features(self) -> list[str]:
        return self.feature_schema.numeric_features

    @property
    def categorical_features(self) -> list[str]:
        return self.feature_schema.categorical_features

    @property
    def numeric_dim(self) -> int:
        return self.feature_schema.numeric_dim

    @property
    def categorical_one_hot_dim(self) -> int:
        return categorical_one_hot_dim(
            self.categorical_features,
            self.categorical_vocab,
        )

    @property
    def input_dim(self) -> int:
        return int(self.numeric_dim + self.categorical_one_hot_dim)

    def cutoffs(self, split: str) -> list[int]:
        self._validate_split(split)
        return [item.cutoff_year for item in self.snapshots[split]]

    def snapshot(self, split: str, cutoff_year: int) -> SnapshotInfo:
        self._validate_split(split)

        for item in self.snapshots[split]:
            if item.cutoff_year == cutoff_year:
                return item

        raise KeyError(f"No snapshot found for split={split}, cutoff_year={cutoff_year}")

    def iter_snapshots(
        self,
        split: str,
        *,
        max_items: int | None = None,
    ) -> Iterable[SnapshotInfo]:
        self._validate_split(split)

        items = self.snapshots[split]
        if max_items is not None:
            items = items[:max_items]

        yield from items

    def estimate_normalization(
        self,
        *,
        max_train_cutoffs: int | None = None,
        chunk_size: int = 200_000,
    ) -> NormalizationStats:
        return estimate_train_numeric_stats(
            self.snapshots["train"],
            numeric_feature_count=self.numeric_dim,
            max_train_cutoffs=max_train_cutoffs,
            chunk_size=chunk_size,
        )

    def load_snapshot_batch(
        self,
        snapshot: SnapshotInfo,
        normalization: NormalizationStats,
    ) -> SnapshotBatch:
        x = build_snapshot_features(
            snapshot.path,
            normalization=normalization,
            categorical_features=self.categorical_features,
            categorical_vocab=self.categorical_vocab,
        )

        y_train, y_heat, y_label, eligible = load_task_arrays(
            snapshot.path,
            self.task,
            log1p_target=self.log1p_target,
        )

        if x.shape[0] != len(y_train):
            raise ValueError(
                f"Feature/target node count mismatch in {snapshot.path}: "
                f"x={x.shape[0]}, y={len(y_train)}"
            )

        return SnapshotBatch(
            snapshot=snapshot,
            x=x,
            y_train=y_train,
            y_heat=y_heat,
            y_label=y_label,
            eligible=eligible,
        )

    def load_edge_index(self) -> torch.Tensor:
        return load_edge_index(self.gnn_input_dir)

    def load_edge_weight(self) -> torch.Tensor:
        return load_edge_weight(self.gnn_input_dir)

    def load_node_index_frame(self) -> pd.DataFrame:
        return load_node_index_frame(self.gnn_input_dir)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "gnn_input_dir": str(self.gnn_input_dir),
            "task": self.task,
            "log1p_target": self.log1p_target,
            "numeric_dim": self.numeric_dim,
            "categorical_feature_count": len(self.categorical_features),
            "categorical_one_hot_dim": self.categorical_one_hot_dim,
            "input_dim": self.input_dim,
            "splits": {
                split: self.cutoffs(split)
                for split in SPLITS
            },
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
        }

    @staticmethod
    def _validate_split(split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}. Expected one of {SPLITS}.")


def make_pyg_data(
    snapshot_batch: SnapshotBatch,
    edge_index: torch.Tensor,
    *,
    edge_weight: torch.Tensor | None = None,
) -> Any:
    """Create a PyG Data object for one snapshot."""

    if Data is None:
        raise ImportError(
            "torch_geometric.data.Data is unavailable. "
            "Please install PyTorch Geometric."
        )

    kwargs: dict[str, Any] = {
        "x": snapshot_batch.x,
        "edge_index": edge_index,
        "y": snapshot_batch.y_train,
    }

    if edge_weight is not None:
        kwargs["edge_weight"] = edge_weight

    return Data(**kwargs)


def eligible_node_indices(
    eligible: np.ndarray,
) -> torch.Tensor:
    return torch.from_numpy(np.flatnonzero(eligible).astype(np.int64))


def make_neighbor_loader(
    data: Any,
    eligible: np.ndarray,
    *,
    num_neighbors: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> Any:
    """Create a PyG NeighborLoader over eligible target nodes."""

    if NeighborLoader is None:
        raise ImportError(
            "torch_geometric.loader.NeighborLoader is unavailable. "
            "Please install PyTorch Geometric and its sampling dependencies."
        )

    input_nodes = eligible_node_indices(eligible)

    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=input_nodes,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def make_mlp_batches(
    eligible: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> Iterable[torch.Tensor]:
    """Yield node-index batches for no-graph MLP training/evaluation."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    idx = np.flatnonzero(eligible).astype(np.int64)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        yield torch.from_numpy(idx[start : start + batch_size])


def snapshot_debug_summary(
    snapshot_batch: SnapshotBatch,
) -> dict[str, Any]:
    eligible_count = int(snapshot_batch.eligible.sum())

    if eligible_count > 0:
        positive_count = int(snapshot_batch.y_label[snapshot_batch.eligible].sum())
        positive_rate = positive_count / eligible_count
    else:
        positive_count = 0
        positive_rate = None

    return {
        "split": snapshot_batch.snapshot.split,
        "cutoff_year": snapshot_batch.snapshot.cutoff_year,
        "path": str(snapshot_batch.snapshot.path),
        "node_count": int(snapshot_batch.x.shape[0]),
        "input_dim": int(snapshot_batch.x.shape[1]),
        "eligible_count": eligible_count,
        "positive_count": positive_count,
        "positive_rate": positive_rate,
        "y_heat_mean_eligible": (
            float(snapshot_batch.y_heat[snapshot_batch.eligible].mean())
            if eligible_count > 0
            else None
        ),
    }