"""Knowledge graph embedding utilities for carrier graph baselines.

This module trains relation-aware KG embedding models on context graph triples.

Important distinction:
    - This module's negative sampling is for KG embedding training.
    - It is NOT the Patent-Paper downstream negative sampling used to build
      labeled_edges_train/val/test.

Typical pipeline:

    context_edges / heterogeneous edge_index_dict
        -> build KG triples
        -> train DistMult / TransE
        -> export entity embeddings
        -> evaluate Patent-Paper labeled pairs with embedding_baselines.py

Leakage rule:
    KG triples should be built from leakage-controlled context_edges only.
    The target Patent-Paper edge table must not be included in KG training triples.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


KGModelName = Literal["distmult", "transe"]
KGLossName = Literal["bce"]


@dataclass(frozen=True)
class RelationInfo:
    """Metadata for one KG relation."""

    relation_idx: int
    relation_name: str
    source_type: str
    target_type: str
    edge_count: int


@dataclass
class KGTrainingResult:
    """Return object for KG embedding training."""

    model_name: str
    entity_embeddings: np.ndarray
    relation_embeddings: np.ndarray
    training_history: pd.DataFrame
    metadata: dict[str, Any]
    model: nn.Module | None = None


def set_random_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    """Resolve device string."""

    device = str(device).strip()

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def edge_type_to_relation_name(edge_type: tuple[str, str, str]) -> str:
    """Return a stable relation name for a heterogeneous edge type."""

    source_type, relation, target_type = edge_type
    return f"{source_type}|{relation}|{target_type}"


def relation_infos_to_frame(relation_infos: list[RelationInfo]) -> pd.DataFrame:
    """Convert relation metadata to a DataFrame."""

    return pd.DataFrame([asdict(info) for info in relation_infos])


def _validate_node_index(node_index: Any) -> pd.DataFrame:
    """Validate and return node_index.nodes DataFrame."""

    if not hasattr(node_index, "nodes"):
        raise TypeError(
            "node_index must have a `.nodes` DataFrame attribute. "
            "Expected object from pkg2.graph_data.build_node_index(...)."
        )

    nodes = node_index.nodes

    if not isinstance(nodes, pd.DataFrame):
        raise TypeError("node_index.nodes must be a pandas DataFrame.")

    required_columns = {"node_type", "graph_node_idx", "local_node_idx"}
    missing = required_columns.difference(nodes.columns)

    if missing:
        raise ValueError(
            "node_index.nodes is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return nodes


def build_node_type_to_entity_ids(node_index: Any) -> dict[str, np.ndarray]:
    """Build mapping from node type to global graph node indices.

    Returned arrays contain global entity ids used by KG embedding models.
    """

    nodes = _validate_node_index(node_index)

    result: dict[str, np.ndarray] = {}

    for node_type, group in nodes.groupby(nodes["node_type"].astype(str), sort=True):
        entity_ids = (
            group["graph_node_idx"]
            .astype(np.int64)
            .sort_values()
            .to_numpy(dtype=np.int64)
        )
        result[str(node_type)] = entity_ids

    return result


def build_local_to_global_index(node_index: Any) -> dict[str, np.ndarray]:
    """Build local-id to global-id arrays for each node type.

    PyG heterogeneous edge_index_dict usually stores local ids per node type.
    This function converts those local ids back to the global node index used by
    downstream embedding_baselines.py.
    """

    nodes = _validate_node_index(node_index)

    result: dict[str, np.ndarray] = {}

    for node_type, group in nodes.groupby(nodes["node_type"].astype(str), sort=True):
        group = group.copy()
        group["local_node_idx"] = group["local_node_idx"].astype(np.int64)
        group["graph_node_idx"] = group["graph_node_idx"].astype(np.int64)

        max_local_idx = int(group["local_node_idx"].max())
        local_to_global = np.full(max_local_idx + 1, -1, dtype=np.int64)

        local_to_global[group["local_node_idx"].to_numpy(dtype=np.int64)] = group[
            "graph_node_idx"
        ].to_numpy(dtype=np.int64)

        if np.any(local_to_global < 0):
            missing_count = int(np.sum(local_to_global < 0))
            raise ValueError(
                f"Node type {node_type!r} has {missing_count} missing local ids "
                "in local-to-global mapping."
            )

        result[str(node_type)] = local_to_global

    return result


def build_kg_triples_from_edge_index_dict(
    edge_index_dict: dict[tuple[str, str, str], np.ndarray],
    *,
    node_index: Any | None = None,
    edge_indices_are_global: bool = False,
    deduplicate: bool = True,
) -> tuple[np.ndarray, list[RelationInfo], pd.DataFrame]:
    """Build KG triples from a heterogeneous edge_index_dict.

    Parameters
    ----------
    edge_index_dict:
        Mapping from (source_type, relation, target_type) to edge_index arrays
        with shape [2, num_edges].

    node_index:
        NodeIndex object from pkg2.graph_data. Required when edge indices are
        local per node type.

    edge_indices_are_global:
        If False, edge_index rows are interpreted as local ids per node type and
        converted to global graph_node_idx using node_index.
        If True, edge_index rows are already global graph_node_idx.

    deduplicate:
        Whether to remove duplicate (head, relation, tail) triples.

    Returns
    -------
    triples:
        int64 numpy array with shape [num_triples, 3], columns:
        head_entity_idx, relation_idx, tail_entity_idx.

    relation_infos:
        List of RelationInfo metadata objects.

    relation_frame:
        DataFrame version of relation_infos.
    """

    if not edge_index_dict:
        raise ValueError("edge_index_dict is empty; cannot build KG triples.")

    local_to_global_by_type: dict[str, np.ndarray] | None = None

    if not edge_indices_are_global:
        if node_index is None:
            raise ValueError(
                "node_index is required when edge_indices_are_global=False."
            )
        local_to_global_by_type = build_local_to_global_index(node_index)

    triple_chunks: list[np.ndarray] = []
    relation_infos: list[RelationInfo] = []

    for relation_idx, edge_type in enumerate(sorted(edge_index_dict)):
        source_type, relation, target_type = edge_type
        edge_index = np.asarray(edge_index_dict[edge_type])

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index for {edge_type!r} must have shape [2, num_edges], "
                f"got {edge_index.shape}."
            )

        source_local_or_global = edge_index[0].astype(np.int64, copy=False)
        target_local_or_global = edge_index[1].astype(np.int64, copy=False)

        if edge_indices_are_global:
            head = source_local_or_global
            tail = target_local_or_global
        else:
            assert local_to_global_by_type is not None

            if source_type not in local_to_global_by_type:
                raise KeyError(f"Unknown source node type: {source_type!r}")
            if target_type not in local_to_global_by_type:
                raise KeyError(f"Unknown target node type: {target_type!r}")

            source_map = local_to_global_by_type[source_type]
            target_map = local_to_global_by_type[target_type]

            if source_local_or_global.size:
                max_source = int(source_local_or_global.max())
                if max_source >= len(source_map):
                    raise ValueError(
                        f"Source local index out of range for {source_type!r}: "
                        f"max={max_source}, map_size={len(source_map)}."
                    )

            if target_local_or_global.size:
                max_target = int(target_local_or_global.max())
                if max_target >= len(target_map):
                    raise ValueError(
                        f"Target local index out of range for {target_type!r}: "
                        f"max={max_target}, map_size={len(target_map)}."
                    )

            head = source_map[source_local_or_global]
            tail = target_map[target_local_or_global]

        rel = np.full(head.shape[0], relation_idx, dtype=np.int64)

        triples = np.stack([head, rel, tail], axis=1).astype(np.int64, copy=False)
        triple_chunks.append(triples)

        relation_infos.append(
            RelationInfo(
                relation_idx=relation_idx,
                relation_name=edge_type_to_relation_name(edge_type),
                source_type=str(source_type),
                target_type=str(target_type),
                edge_count=int(triples.shape[0]),
            )
        )

    if not triple_chunks:
        raise ValueError("No triples were constructed from edge_index_dict.")

    all_triples = np.concatenate(triple_chunks, axis=0).astype(np.int64, copy=False)

    if deduplicate:
        before = int(all_triples.shape[0])
        frame = pd.DataFrame(all_triples, columns=["head", "relation", "tail"])
        frame = frame.drop_duplicates(ignore_index=True)
        all_triples = frame[["head", "relation", "tail"]].to_numpy(dtype=np.int64)
        after = int(all_triples.shape[0])

        if after != before:
            relation_counts = (
                frame.groupby("relation", sort=True)
                .size()
                .rename("edge_count")
                .reset_index()
            )
            count_map = {
                int(row.relation): int(row.edge_count)
                for row in relation_counts.itertuples(index=False)
            }
            relation_infos = [
                RelationInfo(
                    relation_idx=info.relation_idx,
                    relation_name=info.relation_name,
                    source_type=info.source_type,
                    target_type=info.target_type,
                    edge_count=count_map.get(info.relation_idx, 0),
                )
                for info in relation_infos
            ]

    relation_frame = relation_infos_to_frame(relation_infos)

    return all_triples, relation_infos, relation_frame


def summarize_triples(
    triples: np.ndarray,
    relation_infos: list[RelationInfo],
) -> dict[str, Any]:
    """Summarize KG triples."""

    triples = np.asarray(triples)

    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"triples must have shape [num_triples, 3], got {triples.shape}")

    relation_counts = (
        pd.Series(triples[:, 1])
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary: dict[str, Any] = {
        "num_triples": int(triples.shape[0]),
        "num_relations": int(len(relation_infos)),
        "num_unique_heads": int(pd.Series(triples[:, 0]).nunique()),
        "num_unique_tails": int(pd.Series(triples[:, 2]).nunique()),
        "num_unique_entities_in_triples": int(
            pd.Series(np.concatenate([triples[:, 0], triples[:, 2]])).nunique()
        ),
    }

    for info in relation_infos:
        summary[f"relation_triples:{info.relation_name}"] = int(
            relation_counts.get(info.relation_idx, 0)
        )

    return summary


class DistMultModel(nn.Module):
    """DistMult knowledge graph embedding model.

    score(h, r, t) = <e_h, r, e_t>

    Larger score means the triple is predicted to be more likely.
    """

    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)

        self.entity_embedding = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize model parameters."""

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        """Score triples.

        Parameters
        ----------
        triples:
            LongTensor with shape [batch_size, 3].
        """

        head = self.entity_embedding(triples[:, 0])
        relation = self.relation_embedding(triples[:, 1])
        tail = self.entity_embedding(triples[:, 2])

        return torch.sum(head * relation * tail, dim=-1)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Forward alias for score_triples."""

        return self.score_triples(triples)


class TransEModel(nn.Module):
    """TransE knowledge graph embedding model.

    score(h, r, t) = - || e_h + r - e_t ||_p

    Larger score means the triple is predicted to be more likely.
    """

    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        p_norm: int = 1,
    ) -> None:
        super().__init__()

        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.p_norm = int(p_norm)

        self.entity_embedding = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize model parameters."""

        bound = 6.0 / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.uniform_(self.relation_embedding.weight, -bound, bound)

        with torch.no_grad():
            self.entity_embedding.weight.data = F.normalize(
                self.entity_embedding.weight.data,
                p=2,
                dim=-1,
            )

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        """Score triples.

        Parameters
        ----------
        triples:
            LongTensor with shape [batch_size, 3].
        """

        head = self.entity_embedding(triples[:, 0])
        relation = self.relation_embedding(triples[:, 1])
        tail = self.entity_embedding(triples[:, 2])

        return -torch.linalg.vector_norm(
            head + relation - tail,
            ord=self.p_norm,
            dim=-1,
        )

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Forward alias for score_triples."""

        return self.score_triples(triples)

    def normalize_entity_embeddings_(self) -> None:
        """Normalize entity embeddings in-place.

        This is commonly used for TransE-style models.
        """

        with torch.no_grad():
            self.entity_embedding.weight.data = F.normalize(
                self.entity_embedding.weight.data,
                p=2,
                dim=-1,
            )


def create_kg_model(
    *,
    model_name: str,
    num_entities: int,
    num_relations: int,
    embedding_dim: int,
    transe_p_norm: int = 1,
) -> nn.Module:
    """Create a KG embedding model."""

    normalized_name = str(model_name).strip().lower()

    if normalized_name == "distmult":
        return DistMultModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
        )

    if normalized_name == "transe":
        return TransEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            p_norm=transe_p_norm,
        )

    raise ValueError(
        f"Unsupported KG model: {model_name!r}. "
        "Supported models: distmult, transe."
    )


class TypeConstrainedNegativeSampler:
    """Type-constrained negative sampler for KG embedding training.

    For each positive triple (h, r, t), the sampler corrupts either h or t.

    If type_constrained=True:
        - corrupted h is sampled from the source node type of relation r;
        - corrupted t is sampled from the target node type of relation r.

    This avoids trivially invalid negatives such as:

        BioEntity --patent_mentions_bioentity--> Project

    for a relation that should be:

        Patent --patent_mentions_bioentity--> BioEntity
    """

    def __init__(
        self,
        *,
        relation_infos: list[RelationInfo],
        node_type_to_entity_ids: dict[str, Iterable[int] | np.ndarray],
        num_entities: int,
        device: torch.device,
        type_constrained: bool = True,
    ) -> None:
        self.relation_infos = list(relation_infos)
        self.num_entities = int(num_entities)
        self.device = device
        self.type_constrained = bool(type_constrained)

        self.relation_to_source_type = {
            int(info.relation_idx): str(info.source_type)
            for info in self.relation_infos
        }
        self.relation_to_target_type = {
            int(info.relation_idx): str(info.target_type)
            for info in self.relation_infos
        }

        self.node_type_to_entity_ids: dict[str, torch.Tensor] = {}

        for node_type, entity_ids in node_type_to_entity_ids.items():
            tensor = torch.as_tensor(
                np.asarray(list(entity_ids), dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )

            if tensor.numel() == 0:
                raise ValueError(f"Node type {node_type!r} has no entity ids.")

            self.node_type_to_entity_ids[str(node_type)] = tensor

        self.all_entity_ids = torch.arange(
            self.num_entities,
            dtype=torch.long,
            device=self.device,
        )

    def _candidate_ids_for(
        self,
        *,
        relation_idx: int,
        corrupt_head: bool,
    ) -> torch.Tensor:
        """Return candidate replacement entity ids."""

        if not self.type_constrained:
            return self.all_entity_ids

        if corrupt_head:
            node_type = self.relation_to_source_type[int(relation_idx)]
        else:
            node_type = self.relation_to_target_type[int(relation_idx)]

        if node_type not in self.node_type_to_entity_ids:
            raise KeyError(
                f"Missing entity ids for node type {node_type!r} "
                f"needed by relation {relation_idx}."
            )

        return self.node_type_to_entity_ids[node_type]

    def sample(
        self,
        positive_triples: torch.Tensor,
        *,
        num_negative_samples: int,
    ) -> torch.Tensor:
        """Sample negative triples for a positive batch.

        Parameters
        ----------
        positive_triples:
            LongTensor with shape [batch_size, 3].

        num_negative_samples:
            Number of negatives per positive triple.

        Returns
        -------
        negative_triples:
            LongTensor with shape [batch_size * num_negative_samples, 3].
        """

        if positive_triples.ndim != 2 or positive_triples.shape[1] != 3:
            raise ValueError(
                "positive_triples must have shape [batch_size, 3], "
                f"got {tuple(positive_triples.shape)}."
            )

        k = int(num_negative_samples)

        if k <= 0:
            raise ValueError("num_negative_samples must be positive.")

        negative_triples = positive_triples.repeat_interleave(k, dim=0).clone()

        total = int(negative_triples.shape[0])
        corrupt_head_mask = torch.rand(total, device=self.device) < 0.5

        relation_ids = negative_triples[:, 1]
        unique_relation_ids = torch.unique(relation_ids).detach().cpu().tolist()

        for relation_idx in unique_relation_ids:
            relation_idx = int(relation_idx)
            relation_mask = relation_ids == relation_idx

            head_mask = relation_mask & corrupt_head_mask
            tail_mask = relation_mask & (~corrupt_head_mask)

            if bool(head_mask.any()):
                candidates = self._candidate_ids_for(
                    relation_idx=relation_idx,
                    corrupt_head=True,
                )
                sampled_positions = torch.randint(
                    low=0,
                    high=int(candidates.numel()),
                    size=(int(head_mask.sum().item()),),
                    device=self.device,
                )
                negative_triples[head_mask, 0] = candidates[sampled_positions]

            if bool(tail_mask.any()):
                candidates = self._candidate_ids_for(
                    relation_idx=relation_idx,
                    corrupt_head=False,
                )
                sampled_positions = torch.randint(
                    low=0,
                    high=int(candidates.numel()),
                    size=(int(tail_mask.sum().item()),),
                    device=self.device,
                )
                negative_triples[tail_mask, 2] = candidates[sampled_positions]

        return negative_triples


def compute_bce_kg_loss(
    *,
    model: nn.Module,
    positive_triples: torch.Tensor,
    negative_triples: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute binary cross-entropy KG loss for positive and negative triples."""

    positive_scores = model(positive_triples)
    negative_scores = model(negative_triples)

    scores = torch.cat([positive_scores, negative_scores], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(positive_scores),
            torch.zeros_like(negative_scores),
        ],
        dim=0,
    )

    loss = F.binary_cross_entropy_with_logits(scores, labels)

    with torch.no_grad():
        diagnostics = {
            "positive_score_mean": float(positive_scores.detach().mean().cpu()),
            "negative_score_mean": float(negative_scores.detach().mean().cpu()),
            "positive_score_std": float(positive_scores.detach().std().cpu())
            if positive_scores.numel() > 1
            else 0.0,
            "negative_score_std": float(negative_scores.detach().std().cpu())
            if negative_scores.numel() > 1
            else 0.0,
        }

    return loss, diagnostics


def export_entity_embeddings(model: nn.Module) -> np.ndarray:
    """Export entity embeddings from a KG model."""

    if not hasattr(model, "entity_embedding"):
        raise AttributeError("model does not have entity_embedding.")

    with torch.no_grad():
        return (
            model.entity_embedding.weight.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )


def export_relation_embeddings(model: nn.Module) -> np.ndarray:
    """Export relation embeddings from a KG model."""

    if not hasattr(model, "relation_embedding"):
        raise AttributeError("model does not have relation_embedding.")

    with torch.no_grad():
        return (
            model.relation_embedding.weight.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )


def train_kg_embedding_model(
    *,
    triples: np.ndarray,
    relation_infos: list[RelationInfo],
    node_type_to_entity_ids: dict[str, Iterable[int] | np.ndarray],
    num_entities: int,
    model_name: str,
    embedding_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 4096,
    num_negative_samples: int = 5,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    loss_name: str = "bce",
    type_constrained_negative_sampling: bool = True,
    transe_p_norm: int = 1,
    seed: int = 42,
    device: str = "auto",
    deterministic: bool = False,
    keep_model: bool = False,
    log_every: int = 1,
) -> KGTrainingResult:
    """Train a KG embedding model.

    Parameters
    ----------
    triples:
        int64 numpy array with shape [num_triples, 3]. Columns are:
        head_entity_idx, relation_idx, tail_entity_idx.

    relation_infos:
        Metadata for relations.

    node_type_to_entity_ids:
        Mapping from node type to global entity ids. Used for type-constrained
        negative sampling.

    num_entities:
        Total number of global entities / graph nodes.

    model_name:
        "distmult" or "transe".

    embedding_dim:
        Entity and relation embedding dimension.

    epochs:
        Number of training epochs.

    batch_size:
        Positive triples per batch.

    num_negative_samples:
        Number of corrupted negative triples per positive triple.

    learning_rate:
        Adam learning rate.

    weight_decay:
        Adam weight decay.

    loss_name:
        Currently only "bce" is supported.

    type_constrained_negative_sampling:
        Whether to corrupt heads/tails using relation source/target node types.

    transe_p_norm:
        Norm used by TransE. Usually 1 or 2.

    seed:
        Random seed.

    device:
        "auto", "cpu", "cuda", "cuda:0", etc.

    deterministic:
        Whether to request deterministic CuDNN behavior.

    keep_model:
        Whether to return the PyTorch model inside KGTrainingResult.
        For large models, keep this False unless needed.

    log_every:
        Print progress every N epochs. Set <=0 to disable printing.
    """

    normalized_model_name = str(model_name).strip().lower()
    normalized_loss_name = str(loss_name).strip().lower()

    if normalized_loss_name != "bce":
        raise ValueError(
            f"Unsupported KG loss: {loss_name!r}. Currently supported: bce."
        )

    set_random_seed(seed, deterministic=deterministic)

    torch_device = resolve_device(device)

    triples = np.asarray(triples, dtype=np.int64).copy()

    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"triples must have shape [num_triples, 3], got {triples.shape}")

    if triples.shape[0] == 0:
        raise ValueError("triples is empty; cannot train KG embedding model.")

    num_entities = int(num_entities)
    num_relations = int(len(relation_infos))

    if num_relations <= 0:
        raise ValueError("relation_infos is empty; cannot train KG embedding model.")

    if int(triples[:, 0].max()) >= num_entities or int(triples[:, 2].max()) >= num_entities:
        raise ValueError(
            "Triple entity id exceeds num_entities. "
            f"max_head={int(triples[:, 0].max())}, "
            f"max_tail={int(triples[:, 2].max())}, "
            f"num_entities={num_entities}."
        )

    if int(triples[:, 1].max()) >= num_relations:
        raise ValueError(
            "Triple relation id exceeds number of relations. "
            f"max_relation={int(triples[:, 1].max())}, "
            f"num_relations={num_relations}."
        )

    model = create_kg_model(
        model_name=normalized_model_name,
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        transe_p_norm=transe_p_norm,
    ).to(torch_device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    positive_triples_tensor = torch.as_tensor(
        triples,
        dtype=torch.long,
        device=torch_device,
    )

    sampler = TypeConstrainedNegativeSampler(
        relation_infos=relation_infos,
        node_type_to_entity_ids=node_type_to_entity_ids,
        num_entities=num_entities,
        device=torch_device,
        type_constrained=type_constrained_negative_sampling,
    )

    history_rows: list[dict[str, Any]] = []

    num_triples = int(positive_triples_tensor.shape[0])
    batch_size = int(batch_size)
    epochs = int(epochs)
    num_negative_samples = int(num_negative_samples)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if num_negative_samples <= 0:
        raise ValueError("num_negative_samples must be positive.")

    started_at = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        epoch_started_at = time.time()

        permutation = torch.randperm(num_triples, device=torch_device)

        total_loss_weighted = 0.0
        total_examples = 0
        total_batches = 0

        positive_score_means: list[float] = []
        negative_score_means: list[float] = []

        for start in range(0, num_triples, batch_size):
            end = min(start + batch_size, num_triples)
            batch_indices = permutation[start:end]
            positive_batch = positive_triples_tensor[batch_indices]

            negative_batch = sampler.sample(
                positive_batch,
                num_negative_samples=num_negative_samples,
            )

            optimizer.zero_grad()

            loss, diagnostics = compute_bce_kg_loss(
                model=model,
                positive_triples=positive_batch,
                negative_triples=negative_batch,
            )

            loss.backward()
            optimizer.step()

            if normalized_model_name == "transe" and hasattr(
                model, "normalize_entity_embeddings_"
            ):
                model.normalize_entity_embeddings_()

            example_count = int(positive_batch.shape[0]) * (1 + num_negative_samples)
            total_loss_weighted += float(loss.detach().cpu()) * example_count
            total_examples += example_count
            total_batches += 1

            positive_score_means.append(diagnostics["positive_score_mean"])
            negative_score_means.append(diagnostics["negative_score_mean"])

        epoch_seconds = time.time() - epoch_started_at
        avg_loss = total_loss_weighted / max(total_examples, 1)

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "positive_score_mean": float(np.mean(positive_score_means))
            if positive_score_means
            else np.nan,
            "negative_score_mean": float(np.mean(negative_score_means))
            if negative_score_means
            else np.nan,
            "total_batches": total_batches,
            "positive_triples": num_triples,
            "training_examples_including_negatives": total_examples,
            "epoch_seconds": epoch_seconds,
        }
        history_rows.append(row)

        if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(
                f"Epoch {epoch:03d}/{epochs}: "
                f"loss={avg_loss:.6f}, "
                f"pos_score={row['positive_score_mean']:.6f}, "
                f"neg_score={row['negative_score_mean']:.6f}, "
                f"batches={total_batches}, "
                f"seconds={epoch_seconds:.2f}"
            )

    model.eval()

    entity_embeddings = export_entity_embeddings(model)
    relation_embeddings = export_relation_embeddings(model)

    training_history = pd.DataFrame(history_rows)

    metadata = {
        "model_name": normalized_model_name,
        "loss_name": normalized_loss_name,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_triples": num_triples,
        "embedding_dim": int(embedding_dim),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_negative_samples": num_negative_samples,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "type_constrained_negative_sampling": bool(
            type_constrained_negative_sampling
        ),
        "transe_p_norm": int(transe_p_norm),
        "seed": int(seed),
        "device": str(torch_device),
        "deterministic": bool(deterministic),
        "training_seconds": time.time() - started_at,
        "torch_version": str(torch.__version__),
        "final_loss": float(training_history["loss"].iloc[-1])
        if not training_history.empty
        else None,
        "final_positive_score_mean": float(
            training_history["positive_score_mean"].iloc[-1]
        )
        if not training_history.empty
        else None,
        "final_negative_score_mean": float(
            training_history["negative_score_mean"].iloc[-1]
        )
        if not training_history.empty
        else None,
        "note": (
            "KG negative sampling is used only for training context-triple "
            "embeddings. It is distinct from downstream Patent-Paper negative "
            "samples in labeled_edges_train/val/test."
        ),
    }

    return KGTrainingResult(
        model_name=normalized_model_name,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        training_history=training_history,
        metadata=metadata,
        model=model if keep_model else None,
    )


def score_kg_triples(
    *,
    model: nn.Module,
    triples: np.ndarray,
    device: str = "auto",
    batch_size: int = 65536,
) -> np.ndarray:
    """Score triples with a trained KG model.

    This helper is mainly for diagnostics. Downstream Patent-Paper evaluation
    should usually use embedding_baselines.py on exported entity embeddings.
    """

    torch_device = resolve_device(device)

    triples = np.asarray(triples, dtype=np.int64)

    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"triples must have shape [num_triples, 3], got {triples.shape}")

    model = model.to(torch_device)
    model.eval()

    scores: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, int(triples.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(triples.shape[0]))
            batch = torch.as_tensor(
                triples[start:end],
                dtype=torch.long,
                device=torch_device,
            )
            batch_scores = model(batch).detach().cpu().numpy().astype(np.float32)
            scores.append(batch_scores)

    if not scores:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(scores, axis=0)


def build_kg_run_metadata(
    *,
    dataset_dir: str,
    output_dir: str,
    model_name: str,
    triples_summary: dict[str, Any],
    relation_infos: list[RelationInfo],
    training_metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build serializable metadata for a KG embedding run."""

    metadata: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "model_name": str(model_name),
        "triples_summary": dict(triples_summary),
        "relations": [asdict(info) for info in relation_infos],
        "training_metadata": dict(training_metadata),
        "leakage_control": (
            "KG embeddings should be trained only on context_edges. "
            "Target Patent-Paper edges must be excluded from KG training triples."
        ),
    }

    if extra:
        metadata.update(extra)

    return metadata