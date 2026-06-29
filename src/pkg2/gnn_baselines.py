"""Supervised GNN baselines for carrier-level link prediction.

This module provides lightweight GraphSAGE and R-GCN baselines for
Patent-Paper link prediction on leakage-controlled carrier context graphs.

Design assumptions
------------------
1. Message passing uses context graph edges only.
   Target Patent-Paper edges should not be included in ``edge_index``.

2. Supervision comes from the fixed labeled Patent-Paper splits:
   labeled_edges_train / labeled_edges_val / labeled_edges_test.

3. Nodes are initialized with trainable embeddings and optional node-type
   embeddings. This keeps the baseline focused on graph structure rather than
   external text or handcrafted node features.

4. The default edge decoder is a concat MLP:
       score(u, v) = MLP([h_u, h_v])

The script layer should handle:
- Loading graph tables.
- Building context triples / edge_index / edge_type.
- Writing artifacts and reports.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import RGCNConv, SAGEConv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for gnn_baselines.py. "
        "Install PyTorch Geometric or skip GraphSAGE/R-GCN baselines."
    ) from exc


DEFAULT_SOURCE_INDEX_COLUMNS = [
    "graph_source_idx",
    "source_graph_node_idx",
    "source_node_idx",
    "source_idx",
]

DEFAULT_TARGET_INDEX_COLUMNS = [
    "graph_target_idx",
    "target_graph_node_idx",
    "target_node_idx",
    "target_idx",
]


@dataclass
class EdgeSplitTensors:
    """Tensor representation of one labeled edge split."""

    split: str
    edge_index: torch.Tensor
    labels: torch.Tensor

    @property
    def example_count(self) -> int:
        return int(self.labels.numel())

    @property
    def positive_count(self) -> int:
        return int(self.labels.sum().item())

    @property
    def negative_count(self) -> int:
        return int(self.example_count - self.positive_count)

    def to(self, device: torch.device) -> "EdgeSplitTensors":
        """Move tensors to device."""

        return EdgeSplitTensors(
            split=self.split,
            edge_index=self.edge_index.to(device),
            labels=self.labels.to(device),
        )


@dataclass
class GNNTrainingResult:
    """Result object returned by supervised GNN training."""

    model: nn.Module
    best_state_dict: dict[str, torch.Tensor]
    training_history: pd.DataFrame
    metadata: dict[str, Any]


def set_torch_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_torch_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve device string to a torch.device."""

    if isinstance(device, torch.device):
        return device

    device = str(device).strip().lower()

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device)


def encode_node_types(
    node_types: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode node type strings into integer IDs.

    Parameters
    ----------
    node_types:
        Sequence of node type names aligned to global node indices.

    Returns
    -------
    node_type_ids:
        Integer array with shape ``[num_nodes]``.
    node_type_to_id:
        Mapping from node type string to integer ID.
    """

    unique_types = sorted({str(node_type) for node_type in node_types})
    node_type_to_id = {node_type: idx for idx, node_type in enumerate(unique_types)}

    node_type_ids = np.array(
        [node_type_to_id[str(node_type)] for node_type in node_types],
        dtype=np.int64,
    )

    return node_type_ids, node_type_to_id


def find_first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    column_role: str,
) -> str:
    """Find the first candidate column present in a DataFrame."""

    for column in candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        f"Could not find {column_role} column. "
        f"Looked for {list(candidates)} in columns {list(frame.columns)}."
    )


def edge_frame_to_tensors(
    frame: pd.DataFrame,
    *,
    split: str,
    label_column: str = "label",
    source_column: str | None = None,
    target_column: str | None = None,
) -> EdgeSplitTensors:
    """Convert a labeled edge DataFrame into tensors.

    The function expects global graph node indices for source and target nodes.
    If ``source_column`` or ``target_column`` is not supplied, common column
    names are detected automatically.
    """

    if frame.empty:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        labels = torch.empty((0,), dtype=torch.float32)
        return EdgeSplitTensors(split=split, edge_index=edge_index, labels=labels)

    if label_column not in frame.columns:
        raise ValueError(
            f"Label column {label_column!r} not found in split {split!r}. "
            f"Available columns: {list(frame.columns)}"
        )

    source_column = source_column or find_first_existing_column(
        frame,
        DEFAULT_SOURCE_INDEX_COLUMNS,
        column_role="source index",
    )
    target_column = target_column or find_first_existing_column(
        frame,
        DEFAULT_TARGET_INDEX_COLUMNS,
        column_role="target index",
    )

    source = frame[source_column].to_numpy(dtype=np.int64, copy=True)
    target = frame[target_column].to_numpy(dtype=np.int64, copy=True)
    labels = frame[label_column].to_numpy(dtype=np.float32, copy=True)

    edge_index = torch.as_tensor(
        np.stack([source, target], axis=0),
        dtype=torch.long,
    )
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)

    return EdgeSplitTensors(
        split=split,
        edge_index=edge_index,
        labels=label_tensor,
    )


def labeled_edges_by_split_to_tensors(
    labeled_edges_by_split: Mapping[str, pd.DataFrame],
    *,
    label_column: str = "label",
    source_column: str | None = None,
    target_column: str | None = None,
) -> dict[str, EdgeSplitTensors]:
    """Convert labeled edge split DataFrames into tensors."""

    return {
        split: edge_frame_to_tensors(
            frame,
            split=split,
            label_column=label_column,
            source_column=source_column,
            target_column=target_column,
        )
        for split, frame in labeled_edges_by_split.items()
    }


def triples_to_edge_tensors(
    triples: np.ndarray | torch.Tensor,
    *,
    make_writable_copy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert KG triples into PyG edge_index and edge_type tensors.

    Parameters
    ----------
    triples:
        Array with shape ``[num_triples, 3]`` and columns
        ``head_idx, relation_idx, tail_idx``.

    Returns
    -------
    edge_index:
        LongTensor with shape ``[2, num_triples]``.
    edge_type:
        LongTensor with shape ``[num_triples]``.
    """

    if isinstance(triples, torch.Tensor):
        triples_array = triples.detach().cpu().numpy()
    else:
        triples_array = triples

    triples_array = np.array(
        triples_array,
        dtype=np.int64,
        copy=make_writable_copy,
    )

    if triples_array.ndim != 2 or triples_array.shape[1] != 3:
        raise ValueError(
            f"triples must have shape [num_triples, 3], got {triples_array.shape}"
        )

    if triples_array.shape[0] == 0:
        raise ValueError("triples is empty; cannot construct GNN context graph.")

    heads = triples_array[:, 0]
    relations = triples_array[:, 1]
    tails = triples_array[:, 2]

    edge_index = torch.as_tensor(
        np.stack([heads, tails], axis=0),
        dtype=torch.long,
    )
    edge_type = torch.as_tensor(relations, dtype=torch.long)

    return edge_index, edge_type


class TrainableNodeFeatures(nn.Module):
    """Trainable node embeddings with optional node-type embeddings."""

    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dim: int,
        node_type_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        num_node_types: int | None = None,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()

        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)

        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)

        if node_type_ids is None:
            self.register_buffer("node_type_ids", None)
            self.node_type_embeddings = None
        else:
            node_type_tensor = torch.as_tensor(node_type_ids, dtype=torch.long)

            if node_type_tensor.numel() != num_nodes:
                raise ValueError(
                    "node_type_ids must have length num_nodes. "
                    f"Got {node_type_tensor.numel()} vs {num_nodes}."
                )

            inferred_num_node_types = int(node_type_tensor.max().item()) + 1
            num_node_types = int(num_node_types or inferred_num_node_types)

            if num_node_types < inferred_num_node_types:
                raise ValueError(
                    "num_node_types is smaller than max node type id. "
                    f"num_node_types={num_node_types}, "
                    f"required={inferred_num_node_types}"
                )

            self.register_buffer("node_type_ids", node_type_tensor)
            self.node_type_embeddings = nn.Embedding(num_node_types, hidden_dim)

        self.embedding_std = float(embedding_std)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings."""

        nn.init.normal_(self.node_embeddings.weight, mean=0.0, std=self.embedding_std)

        if self.node_type_embeddings is not None:
            nn.init.normal_(
                self.node_type_embeddings.weight,
                mean=0.0,
                std=self.embedding_std,
            )

    def forward(self) -> torch.Tensor:
        """Return node feature matrix with shape ``[num_nodes, hidden_dim]``."""

        x = self.node_embeddings.weight

        if self.node_type_embeddings is not None and self.node_type_ids is not None:
            x = x + self.node_type_embeddings(self.node_type_ids)

        return x


class GraphSAGEEncoder(nn.Module):
    """Full-batch GraphSAGE encoder for homogeneous context graphs."""

    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        node_type_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        num_node_types: int | None = None,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.model_name = "graphsage"
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.node_features = TrainableNodeFeatures(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            node_type_ids=node_type_ids,
            num_node_types=num_node_types,
            embedding_std=embedding_std,
        )

        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset trainable parameters."""

        self.node_features.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute node embeddings.

        ``edge_type`` is ignored for GraphSAGE.
        """

        del edge_type

        x = self.node_features()

        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if layer_idx < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class RGCNEncoder(nn.Module):
    """Full-batch R-GCN encoder for relation-aware context graphs."""

    def __init__(
        self,
        *,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int | None = 8,
        dropout: float = 0.2,
        node_type_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        num_node_types: int | None = None,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        if num_relations <= 0:
            raise ValueError(f"num_relations must be positive, got {num_relations}")

        self.model_name = "rgcn"
        self.num_nodes = int(num_nodes)
        self.num_relations = int(num_relations)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_bases = num_bases
        self.dropout = float(dropout)

        self.node_features = TrainableNodeFeatures(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            node_type_ids=node_type_ids,
            num_node_types=num_node_types,
            embedding_std=embedding_std,
        )

        self.convs = nn.ModuleList(
            [
                RGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases,
                )
                for _ in range(num_layers)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset trainable parameters."""

        self.node_features.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute node embeddings."""

        if edge_type is None:
            raise ValueError("edge_type is required for RGCNEncoder.")

        x = self.node_features()

        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)

            if layer_idx < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class LinkMLPDecoder(nn.Module):
    """Concat MLP edge decoder.

    For an edge ``(u, v)``, the decoder computes:

        logit = MLP([h_u, h_v])

    The output is a raw logit suitable for ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        hidden_dim = int(hidden_dim or embedding_dim)

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize decoder weights."""

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score edges.

        Parameters
        ----------
        node_embeddings:
            Tensor with shape ``[num_nodes, embedding_dim]``.
        edge_index:
            Tensor with shape ``[2, num_edges]``.

        Returns
        -------
        logits:
            Tensor with shape ``[num_edges]``.
        """

        source = edge_index[0]
        target = edge_index[1]

        pair_features = torch.cat(
            [node_embeddings[source], node_embeddings[target]],
            dim=-1,
        )

        return self.mlp(pair_features).squeeze(-1)


class GNNLinkPredictor(nn.Module):
    """Encoder + decoder wrapper for supervised link prediction."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: LinkMLPDecoder,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    @property
    def model_name(self) -> str:
        return getattr(self.encoder, "model_name", self.encoder.__class__.__name__)

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode graph nodes."""

        return self.encoder(edge_index=edge_index, edge_type=edge_type)

    def decode(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode labeled edges."""

        return self.decoder(node_embeddings, edge_index)

    def forward(
        self,
        *,
        graph_edge_index: torch.Tensor,
        graph_edge_type: torch.Tensor | None,
        prediction_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for edge logits."""

        node_embeddings = self.encode(
            edge_index=graph_edge_index,
            edge_type=graph_edge_type,
        )

        return self.decode(node_embeddings, prediction_edge_index)


def build_gnn_link_predictor(
    *,
    model_name: str,
    num_nodes: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    node_type_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    num_node_types: int | None = None,
    num_relations: int | None = None,
    num_bases: int | None = 8,
    decoder_hidden_dim: int | None = None,
    embedding_std: float = 0.02,
) -> GNNLinkPredictor:
    """Factory for GraphSAGE / R-GCN link predictors."""

    normalized_model_name = model_name.strip().lower()

    if normalized_model_name in {"graphsage", "sage"}:
        encoder = GraphSAGEEncoder(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            node_type_ids=node_type_ids,
            num_node_types=num_node_types,
            embedding_std=embedding_std,
        )

    elif normalized_model_name in {"rgcn", "r-gcn"}:
        if num_relations is None:
            raise ValueError("num_relations is required for R-GCN.")

        encoder = RGCNEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=dropout,
            node_type_ids=node_type_ids,
            num_node_types=num_node_types,
            embedding_std=embedding_std,
        )

    else:
        raise ValueError(
            f"Unsupported GNN model {model_name!r}. "
            "Supported models: graphsage, rgcn."
        )

    decoder = LinkMLPDecoder(
        embedding_dim=hidden_dim,
        hidden_dim=decoder_hidden_dim or hidden_dim,
        dropout=dropout,
    )

    return GNNLinkPredictor(encoder=encoder, decoder=decoder)


def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute AUROC and AUPRC safely."""

    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)

    metrics: dict[str, float | int | None] = {
        "example_count": int(labels.shape[0]),
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.shape[0] - labels.sum()),
        "positive_ratio": float(labels.mean()) if labels.shape[0] else None,
        "auroc": None,
        "auprc": None,
    }

    if labels.shape[0] == 0:
        return metrics

    if len(np.unique(labels)) >= 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))

    metrics["auprc"] = float(average_precision_score(labels, scores))

    return metrics


@torch.no_grad()
def score_edge_split(
    model: GNNLinkPredictor,
    *,
    graph_edge_index: torch.Tensor,
    graph_edge_type: torch.Tensor | None,
    edge_split: EdgeSplitTensors,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score one split and return source, target, labels, probabilities."""

    model.eval()

    graph_edge_index = graph_edge_index.to(device)
    graph_edge_type = graph_edge_type.to(device) if graph_edge_type is not None else None
    edge_split = edge_split.to(device)

    node_embeddings = model.encode(
        edge_index=graph_edge_index,
        edge_type=graph_edge_type,
    )

    logits = model.decode(node_embeddings, edge_split.edge_index)
    scores = torch.sigmoid(logits)

    source = edge_split.edge_index[0].detach().cpu().numpy()
    target = edge_split.edge_index[1].detach().cpu().numpy()
    labels = edge_split.labels.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()

    return source, target, labels, scores_np


@torch.no_grad()
def predict_labeled_edges_by_split(
    model: GNNLinkPredictor,
    *,
    graph_edge_index: torch.Tensor,
    graph_edge_type: torch.Tensor | None,
    edge_splits: Mapping[str, EdgeSplitTensors],
    device: str | torch.device = "auto",
    score_column: str = "score",
) -> dict[str, pd.DataFrame]:
    """Predict labeled edges for all splits.

    Returns one DataFrame per split with columns:
        split, graph_source_idx, graph_target_idx, label, score
    """

    torch_device = resolve_torch_device(device)

    predictions: dict[str, pd.DataFrame] = {}

    for split, edge_split in edge_splits.items():
        source, target, labels, scores = score_edge_split(
            model,
            graph_edge_index=graph_edge_index,
            graph_edge_type=graph_edge_type,
            edge_split=edge_split,
            device=torch_device,
        )

        predictions[split] = pd.DataFrame(
            {
                "split": split,
                "graph_source_idx": source.astype(np.int64),
                "graph_target_idx": target.astype(np.int64),
                "label": labels.astype(np.int64),
                score_column: scores.astype(np.float64),
            }
        )

    return predictions


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Deep-copy a module state dict to CPU tensors."""

    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def train_supervised_gnn_link_predictor(
    *,
    model: GNNLinkPredictor,
    graph_edge_index: torch.Tensor,
    graph_edge_type: torch.Tensor | None,
    edge_splits: Mapping[str, EdgeSplitTensors],
    train_split: str = "train",
    val_split: str = "val",
    epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str | torch.device = "auto",
    seed: int = 42,
    deterministic: bool = False,
    early_stopping_patience: int | None = 10,
    min_delta: float = 0.0,
    log_every: int = 1,
) -> GNNTrainingResult:
    """Train a supervised GNN link predictor.

    This uses full-batch message passing over the context graph and supervised
    BCE loss over the fixed training labeled Patent-Paper edges.
    """

    if train_split not in edge_splits:
        raise ValueError(f"Missing train split {train_split!r} in edge_splits.")

    if val_split not in edge_splits:
        raise ValueError(f"Missing validation split {val_split!r} in edge_splits.")

    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    set_torch_seed(seed, deterministic=deterministic)

    torch_device = resolve_torch_device(device)

    model = model.to(torch_device)
    graph_edge_index = graph_edge_index.to(torch_device)
    graph_edge_type = graph_edge_type.to(torch_device) if graph_edge_type is not None else None

    train_edges = edge_splits[train_split].to(torch_device)
    val_edges = edge_splits[val_split].to(torch_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    history_rows: list[dict[str, Any]] = []

    best_metric = -math.inf
    best_epoch = 0
    best_state_dict = clone_state_dict(model)
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        node_embeddings = model.encode(
            edge_index=graph_edge_index,
            edge_type=graph_edge_type,
        )

        train_logits = model.decode(node_embeddings, train_edges.edge_index)
        train_loss = loss_fn(train_logits, train_edges.labels)

        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()

            node_embeddings = model.encode(
                edge_index=graph_edge_index,
                edge_type=graph_edge_type,
            )

            train_scores = torch.sigmoid(
                model.decode(node_embeddings, train_edges.edge_index)
            )
            val_scores = torch.sigmoid(
                model.decode(node_embeddings, val_edges.edge_index)
            )

            train_metrics = binary_classification_metrics(
                train_edges.labels.detach().cpu().numpy(),
                train_scores.detach().cpu().numpy(),
            )
            val_metrics = binary_classification_metrics(
                val_edges.labels.detach().cpu().numpy(),
                val_scores.detach().cpu().numpy(),
            )

        val_auprc = val_metrics.get("auprc")
        monitored_metric = float(val_auprc) if val_auprc is not None else -math.inf

        improved = monitored_metric > best_metric + float(min_delta)

        if improved:
            best_metric = monitored_metric
            best_epoch = epoch
            best_state_dict = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach().cpu().item()),
            "train_auroc": train_metrics.get("auroc"),
            "train_auprc": train_metrics.get("auprc"),
            "val_auroc": val_metrics.get("auroc"),
            "val_auprc": val_metrics.get("auprc"),
            "best_val_auprc": best_metric if math.isfinite(best_metric) else None,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_seconds": epoch_seconds,
        }
        history_rows.append(row)

        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(
                f"Epoch {epoch:03d}/{epochs}: "
                f"loss={row['train_loss']:.6f}, "
                f"train_auprc={row['train_auprc']:.6f}, "
                f"val_auprc={row['val_auprc']:.6f}, "
                f"best_val_auprc={row['best_val_auprc']:.6f}, "
                f"seconds={epoch_seconds:.2f}"
            )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}, best_val_auprc={best_metric:.6f}"
            )
            break

    model.load_state_dict(copy.deepcopy(best_state_dict))

    total_seconds = time.time() - start_time
    training_history = pd.DataFrame(history_rows)

    metadata = {
        "model_name": model.model_name,
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(training_history)),
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_metric) if math.isfinite(best_metric) else None,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "early_stopping_patience": early_stopping_patience,
        "min_delta": float(min_delta),
        "device": str(torch_device),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "total_seconds": float(total_seconds),
        "train_examples": int(train_edges.example_count),
        "train_positive": int(train_edges.positive_count),
        "train_negative": int(train_edges.negative_count),
        "val_examples": int(val_edges.example_count),
        "val_positive": int(val_edges.positive_count),
        "val_negative": int(val_edges.negative_count),
    }

    return GNNTrainingResult(
        model=model,
        best_state_dict=best_state_dict,
        training_history=training_history,
        metadata=metadata,
    )


def summarize_context_graph_tensors(
    *,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor | None = None,
    num_nodes: int | None = None,
) -> dict[str, Any]:
    """Summarize context graph tensors."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    edge_count = int(edge_index.shape[1])

    if num_nodes is None:
        if edge_count == 0:
            num_nodes = 0
        else:
            num_nodes = int(edge_index.max().item()) + 1

    summary: dict[str, Any] = {
        "num_nodes": int(num_nodes),
        "num_context_edges": edge_count,
        "has_edge_type": edge_type is not None,
    }

    if edge_type is not None:
        edge_type_cpu = edge_type.detach().cpu().numpy()
        unique_edge_types, counts = np.unique(edge_type_cpu, return_counts=True)

        summary["num_relations"] = int(len(unique_edge_types))

        for relation_idx, count in zip(unique_edge_types, counts, strict=False):
            summary[f"relation_edges:{int(relation_idx)}"] = int(count)

    return summary


def summarize_edge_splits(
    edge_splits: Mapping[str, EdgeSplitTensors],
) -> dict[str, Any]:
    """Summarize labeled edge split tensors."""

    summary: dict[str, Any] = {}

    for split, tensors in edge_splits.items():
        summary[f"labeled_edges:{split}"] = tensors.example_count
        summary[f"positive_edges:{split}"] = tensors.positive_count
        summary[f"negative_edges:{split}"] = tensors.negative_count

    return summary


def parameter_count(module: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def summarize_model(module: nn.Module) -> dict[str, Any]:
    """Return compact model summary."""

    summary = {
        "model_class": module.__class__.__name__,
        **parameter_count(module),
    }

    if isinstance(module, GNNLinkPredictor):
        summary["model_name"] = module.model_name
        summary["encoder_class"] = module.encoder.__class__.__name__
        summary["decoder_class"] = module.decoder.__class__.__name__

        for attr in [
            "num_nodes",
            "num_relations",
            "hidden_dim",
            "num_layers",
            "num_bases",
            "dropout",
        ]:
            if hasattr(module.encoder, attr):
                summary[f"encoder_{attr}"] = getattr(module.encoder, attr)

    return summary