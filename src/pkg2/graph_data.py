"""Utilities for loading graph representation baseline data.

This module provides reusable data-loading and indexing utilities for graph
representation baselines such as Node2Vec, MetaPath2Vec, KG embeddings, and GNNs.

It intentionally does not depend on PyTorch Geometric or DGL. Instead, it builds
plain pandas / numpy structures that downstream scripts can convert into:

- homogeneous PyG edge_index for Node2Vec / GraphSAGE
- heterogeneous edge_index_dict for MetaPath2Vec / HeteroData
- KG triples for TransE / DistMult / ComplEx / RotatE
- labeled Patent-Paper pairs for link prediction evaluation

Expected dataset directory structure
------------------------------------

The loader is intentionally permissive. It searches common file names such as:

    nodes.parquet
    context_edges.parquet
    labeled_edges_train.parquet
    labeled_edges_val.parquet
    labeled_edges_test.parquet

or a combined:

    labeled_edges.parquet

with a `split` column.

Minimum required logical columns
--------------------------------

Nodes table:

    node_type
    node_id

Context edges table, preferred form:

    source_type
    source_id
    target_type
    target_id

Optional context edge columns:

    edge_type
    edge_table
    weight

Labeled edge tables, preferred form:

    source_type
    source_id
    target_type
    target_id
    split
    label

The functions also try to handle variants such as source_index / target_index
when the nodes table contains a compatible node_index column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_SPLITS = ("train", "val", "test")

NODE_TYPE_COLUMN = "node_type"
NODE_ID_COLUMN = "node_id"

SOURCE_TYPE_COLUMN = "source_type"
SOURCE_ID_COLUMN = "source_id"
TARGET_TYPE_COLUMN = "target_type"
TARGET_ID_COLUMN = "target_id"

SOURCE_INDEX_COLUMN = "source_index"
TARGET_INDEX_COLUMN = "target_index"

EDGE_TYPE_COLUMN = "edge_type"
EDGE_TABLE_COLUMN = "edge_table"
RELATION_COLUMN = "relation"

SPLIT_COLUMN = "split"
LABEL_COLUMN = "label"

GRAPH_NODE_INDEX_COLUMN = "graph_node_idx"
NODE_KEY_COLUMN = "node_key"
LOCAL_NODE_INDEX_COLUMN = "local_node_idx"

SOURCE_GRAPH_INDEX_COLUMN = "source_graph_node_idx"
TARGET_GRAPH_INDEX_COLUMN = "target_graph_node_idx"
SOURCE_LOCAL_INDEX_COLUMN = "source_local_node_idx"
TARGET_LOCAL_INDEX_COLUMN = "target_local_node_idx"
RELATION_INDEX_COLUMN = "relation_idx"


@dataclass(frozen=True)
class DatasetFiles:
    """Discovered input files for a link prediction graph dataset."""

    dataset_dir: Path
    nodes_path: Path
    context_edges_path: Path
    labeled_edges_paths: dict[str, Path]
    combined_labeled_edges_path: Path | None = None


@dataclass
class GraphTables:
    """Loaded graph tables."""

    nodes: pd.DataFrame
    context_edges: pd.DataFrame
    labeled_edges_by_split: dict[str, pd.DataFrame]


@dataclass
class NodeIndex:
    """Global and type-local node indices."""

    nodes: pd.DataFrame
    node_to_idx: dict[tuple[str, str], int]
    idx_to_node: dict[int, tuple[str, str]]
    node_to_local_idx: dict[tuple[str, str], int]
    local_idx_to_node: dict[tuple[str, int], tuple[str, str]]
    num_nodes_by_type: dict[str, int]


@dataclass
class RelationIndex:
    """Relation-name index mapping."""

    relation_to_idx: dict[str, int]
    idx_to_relation: dict[int, str]


@dataclass
class HomogeneousGraph:
    """Homogeneous graph representation for Node2Vec-like baselines."""

    edge_index: np.ndarray
    edge_frame: pd.DataFrame
    num_nodes: int


@dataclass
class HeterogeneousGraph:
    """Heterogeneous graph representation for MetaPath2Vec / HeteroData."""

    edge_index_dict: dict[tuple[str, str, str], np.ndarray]
    num_nodes_dict: dict[str, int]
    edge_frame: pd.DataFrame


@dataclass
class KGTriples:
    """Knowledge graph triple representation."""

    triples: np.ndarray
    triple_frame: pd.DataFrame
    relation_index: RelationIndex
    num_entities: int
    num_relations: int


def _as_path(path: str | Path) -> Path:
    """Normalize a path-like value."""

    return Path(path)


def _first_existing(base_dir: Path, candidates: Iterable[str]) -> Path | None:
    """Return the first existing candidate path under a base directory."""

    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path

    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read a table from parquet, csv, tsv, json, jsonl, or a parquet directory."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Table path does not exist: {path}")

    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            return pd.concat(
                [pd.read_parquet(file) for file in parquet_files],
                ignore_index=True,
            )

        csv_files = sorted(path.glob("*.csv"))
        if csv_files:
            return pd.concat(
                [pd.read_csv(file) for file in csv_files],
                ignore_index=True,
            )

        raise FileNotFoundError(
            f"Directory contains no supported table files: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported table file suffix for: {path}")


def discover_dataset_files(
    dataset_dir: str | Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    nodes_path: str | Path | None = None,
    context_edges_path: str | Path | None = None,
    labeled_edges_paths: dict[str, str | Path] | None = None,
    combined_labeled_edges_path: str | Path | None = None,
) -> DatasetFiles:
    """Discover graph dataset files.

    Explicit paths override discovery.
    """

    dataset_dir = _as_path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    discovered_nodes_path = (
        Path(nodes_path)
        if nodes_path is not None
        else _first_existing(
            dataset_dir,
            [
                "nodes.parquet",
                "nodes.csv",
                "graph_nodes.parquet",
                "graph_nodes.csv",
                "filtered_nodes.parquet",
                "filtered_nodes.csv",
            ],
        )
    )

    if discovered_nodes_path is None:
        raise FileNotFoundError(
            "Could not find nodes table. Expected one of: "
            "nodes.parquet, nodes.csv, graph_nodes.parquet, graph_nodes.csv, "
            "filtered_nodes.parquet, filtered_nodes.csv"
        )

    discovered_context_edges_path = (
        Path(context_edges_path)
        if context_edges_path is not None
        else _first_existing(
            dataset_dir,
            [
                "context_edges.parquet",
                "context_edges.csv",
                "edges_context.parquet",
                "edges_context.csv",
                "graph_context_edges.parquet",
                "graph_context_edges.csv",
                "filtered_context_edges.parquet",
                "filtered_context_edges.csv",
            ],
        )
    )

    if discovered_context_edges_path is None:
        raise FileNotFoundError(
            "Could not find context edge table. Expected one of: "
            "context_edges.parquet, context_edges.csv, edges_context.parquet, "
            "edges_context.csv, graph_context_edges.parquet, graph_context_edges.csv, "
            "filtered_context_edges.parquet, filtered_context_edges.csv"
        )

    discovered_combined_labeled_edges_path = (
        Path(combined_labeled_edges_path)
        if combined_labeled_edges_path is not None
        else _first_existing(
            dataset_dir,
            [
                "labeled_edges.parquet",
                "labeled_edges.csv",
                "labels.parquet",
                "labels.csv",
            ],
        )
    )

    discovered_labeled_edges_paths: dict[str, Path] = {}

    explicit_labeled_edges_paths = labeled_edges_paths or {}

    for split in splits:
        split = str(split)

        if split in explicit_labeled_edges_paths:
            discovered_labeled_edges_paths[split] = Path(
                explicit_labeled_edges_paths[split]
            )
            continue

        split_path = _first_existing(
            dataset_dir,
            [
                f"labeled_edges_{split}.parquet",
                f"labeled_edges_{split}.csv",
                f"{split}_labeled_edges.parquet",
                f"{split}_labeled_edges.csv",
                f"edges_{split}.parquet",
                f"edges_{split}.csv",
                f"{split}.parquet",
                f"{split}.csv",
            ],
        )

        if split_path is not None:
            discovered_labeled_edges_paths[split] = split_path

    if not discovered_labeled_edges_paths and discovered_combined_labeled_edges_path is None:
        raise FileNotFoundError(
            "Could not find labeled edge files. Expected per-split files such as "
            "labeled_edges_train.parquet / labeled_edges_val.parquet / "
            "labeled_edges_test.parquet, or a combined labeled_edges.parquet."
        )

    return DatasetFiles(
        dataset_dir=dataset_dir,
        nodes_path=discovered_nodes_path,
        context_edges_path=discovered_context_edges_path,
        labeled_edges_paths=discovered_labeled_edges_paths,
        combined_labeled_edges_path=discovered_combined_labeled_edges_path,
    )


def load_graph_tables(
    dataset_dir: str | Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    nodes_path: str | Path | None = None,
    context_edges_path: str | Path | None = None,
    labeled_edges_paths: dict[str, str | Path] | None = None,
    combined_labeled_edges_path: str | Path | None = None,
) -> GraphTables:
    """Load graph tables from a link prediction dataset directory."""

    split_list = [str(split) for split in splits]

    files = discover_dataset_files(
        dataset_dir,
        splits=split_list,
        nodes_path=nodes_path,
        context_edges_path=context_edges_path,
        labeled_edges_paths=labeled_edges_paths,
        combined_labeled_edges_path=combined_labeled_edges_path,
    )

    nodes = normalize_nodes(_read_table(files.nodes_path))
    context_edges = _read_table(files.context_edges_path)

    labeled_edges_by_split: dict[str, pd.DataFrame] = {}

    if files.labeled_edges_paths:
        for split, path in files.labeled_edges_paths.items():
            frame = _read_table(path)
            if SPLIT_COLUMN not in frame.columns:
                frame = frame.copy()
                frame[SPLIT_COLUMN] = split
            labeled_edges_by_split[split] = frame

    if files.combined_labeled_edges_path is not None:
        combined = _read_table(files.combined_labeled_edges_path)

        if SPLIT_COLUMN not in combined.columns:
            raise ValueError(
                f"Combined labeled edge table must contain a {SPLIT_COLUMN!r} column: "
                f"{files.combined_labeled_edges_path}"
            )

        for split in split_list:
            if split not in labeled_edges_by_split:
                split_frame = combined[combined[SPLIT_COLUMN].astype(str) == split].copy()
                if not split_frame.empty:
                    labeled_edges_by_split[split] = split_frame

    missing_splits = [
        split
        for split in split_list
        if split not in labeled_edges_by_split
    ]

    if missing_splits:
        raise FileNotFoundError(
            "Missing labeled edge data for splits: "
            + ", ".join(missing_splits)
        )

    return GraphTables(
        nodes=nodes,
        context_edges=context_edges,
        labeled_edges_by_split={
            split: normalize_labeled_edges(frame, nodes=nodes)
            for split, frame in labeled_edges_by_split.items()
        },
    )


def normalize_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Normalize a nodes table to contain node_type and node_id string columns."""

    nodes = nodes.copy()

    rename_map = {}

    if NODE_TYPE_COLUMN not in nodes.columns:
        for candidate in ["type", "label", "node_label", "entity_type"]:
            if candidate in nodes.columns:
                rename_map[candidate] = NODE_TYPE_COLUMN
                break

    if NODE_ID_COLUMN not in nodes.columns:
        for candidate in ["id", "identifier", "node_identifier", "entity_id"]:
            if candidate in nodes.columns:
                rename_map[candidate] = NODE_ID_COLUMN
                break

    if rename_map:
        nodes = nodes.rename(columns=rename_map)

    required = {NODE_TYPE_COLUMN, NODE_ID_COLUMN}
    missing = sorted(required - set(nodes.columns))

    if missing:
        raise ValueError(
            "Nodes table is missing required columns: "
            + ", ".join(missing)
            + ". Required logical columns: node_type, node_id."
        )

    nodes[NODE_TYPE_COLUMN] = nodes[NODE_TYPE_COLUMN].astype(str)
    nodes[NODE_ID_COLUMN] = nodes[NODE_ID_COLUMN].astype(str)

    if NODE_KEY_COLUMN not in nodes.columns:
        nodes[NODE_KEY_COLUMN] = make_node_keys(
            nodes[NODE_TYPE_COLUMN],
            nodes[NODE_ID_COLUMN],
        )

    return nodes


def make_node_key(node_type: Any, node_id: Any) -> str:
    """Create a stable string key for a node."""

    return f"{str(node_type)}::{str(node_id)}"


def make_node_keys(node_types: Iterable[Any], node_ids: Iterable[Any]) -> list[str]:
    """Create stable node keys for vectorized node type / ID values."""

    return [
        make_node_key(node_type, node_id)
        for node_type, node_id in zip(node_types, node_ids)
    ]


def build_node_index(nodes: pd.DataFrame) -> NodeIndex:
    """Build global and type-local node indices.

    The global graph node index is used for homogeneous graph methods.

    The type-local node index is used for heterogeneous graph methods such as
    PyG MetaPath2Vec / HeteroData.
    """

    nodes = normalize_nodes(nodes)

    index_nodes = nodes.drop_duplicates(
        subset=[NODE_TYPE_COLUMN, NODE_ID_COLUMN],
        keep="first",
    ).copy()

    index_nodes = index_nodes.sort_values(
        [NODE_TYPE_COLUMN, NODE_ID_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)

    index_nodes[GRAPH_NODE_INDEX_COLUMN] = np.arange(len(index_nodes), dtype=np.int64)

    node_to_idx: dict[tuple[str, str], int] = {}
    idx_to_node: dict[int, tuple[str, str]] = {}

    for row in index_nodes.itertuples(index=False):
        node_type = str(getattr(row, NODE_TYPE_COLUMN))
        node_id = str(getattr(row, NODE_ID_COLUMN))
        graph_idx = int(getattr(row, GRAPH_NODE_INDEX_COLUMN))

        key = (node_type, node_id)
        node_to_idx[key] = graph_idx
        idx_to_node[graph_idx] = key

    local_indices = []

    node_to_local_idx: dict[tuple[str, str], int] = {}
    local_idx_to_node: dict[tuple[str, int], tuple[str, str]] = {}
    num_nodes_by_type: dict[str, int] = {}

    for node_type, group in index_nodes.groupby(NODE_TYPE_COLUMN, sort=True):
        for local_idx, original_idx in enumerate(group.index):
            node_id = str(index_nodes.at[original_idx, NODE_ID_COLUMN])
            key = (str(node_type), node_id)

            node_to_local_idx[key] = int(local_idx)
            local_idx_to_node[(str(node_type), int(local_idx))] = key
            local_indices.append((original_idx, int(local_idx)))

        num_nodes_by_type[str(node_type)] = int(len(group))

    local_index_series = pd.Series(
        {
            original_idx: local_idx
            for original_idx, local_idx in local_indices
        },
        dtype="int64",
    )

    index_nodes[LOCAL_NODE_INDEX_COLUMN] = index_nodes.index.map(local_index_series)

    return NodeIndex(
        nodes=index_nodes,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        node_to_local_idx=node_to_local_idx,
        local_idx_to_node=local_idx_to_node,
        num_nodes_by_type=num_nodes_by_type,
    )


def _build_original_index_lookup(nodes: pd.DataFrame) -> dict[Any, tuple[str, str]]:
    """Build lookup from original node index column to node type / ID when available."""

    candidate_columns = [
        "node_index",
        "index",
        "original_node_index",
        "source_index",
    ]

    index_column = None

    for candidate in candidate_columns:
        if candidate in nodes.columns:
            index_column = candidate
            break

    if index_column is None:
        return {}

    lookup = {}

    for row in nodes.itertuples(index=False):
        original_index = getattr(row, index_column)
        node_type = str(getattr(row, NODE_TYPE_COLUMN))
        node_id = str(getattr(row, NODE_ID_COLUMN))
        lookup[original_index] = (node_type, node_id)

    return lookup


def _ensure_edge_endpoint_columns(
    edges: pd.DataFrame,
    *,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure an edge table has source/target type and ID columns."""

    edges = edges.copy()

    rename_map = {}

    source_type_candidates = [
        SOURCE_TYPE_COLUMN,
        "src_type",
        "head_type",
        "from_type",
        "source_node_type",
    ]
    source_id_candidates = [
        SOURCE_ID_COLUMN,
        "src_id",
        "head_id",
        "from_id",
        "source_node_id",
    ]
    target_type_candidates = [
        TARGET_TYPE_COLUMN,
        "dst_type",
        "tail_type",
        "to_type",
        "target_node_type",
    ]
    target_id_candidates = [
        TARGET_ID_COLUMN,
        "dst_id",
        "tail_id",
        "to_id",
        "target_node_id",
    ]

    def find_existing(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in edges.columns:
                return candidate
        return None

    source_type_col = find_existing(source_type_candidates)
    source_id_col = find_existing(source_id_candidates)
    target_type_col = find_existing(target_type_candidates)
    target_id_col = find_existing(target_id_candidates)

    if source_type_col and source_type_col != SOURCE_TYPE_COLUMN:
        rename_map[source_type_col] = SOURCE_TYPE_COLUMN
    if source_id_col and source_id_col != SOURCE_ID_COLUMN:
        rename_map[source_id_col] = SOURCE_ID_COLUMN
    if target_type_col and target_type_col != TARGET_TYPE_COLUMN:
        rename_map[target_type_col] = TARGET_TYPE_COLUMN
    if target_id_col and target_id_col != TARGET_ID_COLUMN:
        rename_map[target_id_col] = TARGET_ID_COLUMN

    if rename_map:
        edges = edges.rename(columns=rename_map)

    required = {
        SOURCE_TYPE_COLUMN,
        SOURCE_ID_COLUMN,
        TARGET_TYPE_COLUMN,
        TARGET_ID_COLUMN,
    }

    if required.issubset(edges.columns):
        edges[SOURCE_TYPE_COLUMN] = edges[SOURCE_TYPE_COLUMN].astype(str)
        edges[SOURCE_ID_COLUMN] = edges[SOURCE_ID_COLUMN].astype(str)
        edges[TARGET_TYPE_COLUMN] = edges[TARGET_TYPE_COLUMN].astype(str)
        edges[TARGET_ID_COLUMN] = edges[TARGET_ID_COLUMN].astype(str)
        return edges

    original_lookup = _build_original_index_lookup(nodes)

    if (
        original_lookup
        and SOURCE_INDEX_COLUMN in edges.columns
        and TARGET_INDEX_COLUMN in edges.columns
    ):
        source_values = edges[SOURCE_INDEX_COLUMN].map(original_lookup)
        target_values = edges[TARGET_INDEX_COLUMN].map(original_lookup)

        missing_source = source_values.isna().sum()
        missing_target = target_values.isna().sum()

        if missing_source or missing_target:
            raise ValueError(
                "Could not map all source_index / target_index values to nodes. "
                f"Missing source mappings: {missing_source}; "
                f"missing target mappings: {missing_target}."
            )

        edges[SOURCE_TYPE_COLUMN] = [value[0] for value in source_values]
        edges[SOURCE_ID_COLUMN] = [value[1] for value in source_values]
        edges[TARGET_TYPE_COLUMN] = [value[0] for value in target_values]
        edges[TARGET_ID_COLUMN] = [value[1] for value in target_values]

        return edges

    missing = sorted(required - set(edges.columns))

    raise ValueError(
        "Edge table is missing endpoint columns: "
        + ", ".join(missing)
        + ". Required logical columns are source_type, source_id, "
        "target_type, target_id. Alternatively, provide source_index / "
        "target_index with a nodes table containing node_index."
    )


def normalize_context_edges(
    context_edges: pd.DataFrame,
    *,
    nodes: pd.DataFrame,
    default_relation: str = "context",
) -> pd.DataFrame:
    """Normalize context edge table endpoint and relation columns."""

    context_edges = _ensure_edge_endpoint_columns(context_edges, nodes=nodes)

    if RELATION_COLUMN not in context_edges.columns:
        if EDGE_TYPE_COLUMN in context_edges.columns:
            context_edges[RELATION_COLUMN] = context_edges[EDGE_TYPE_COLUMN].astype(str)
        elif EDGE_TABLE_COLUMN in context_edges.columns:
            context_edges[RELATION_COLUMN] = context_edges[EDGE_TABLE_COLUMN].astype(str)
        else:
            context_edges[RELATION_COLUMN] = default_relation

    context_edges[RELATION_COLUMN] = context_edges[RELATION_COLUMN].astype(str)

    return context_edges


def normalize_labeled_edges(
    labeled_edges: pd.DataFrame,
    *,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize labeled link prediction edge table."""

    labeled_edges = _ensure_edge_endpoint_columns(labeled_edges, nodes=nodes)

    if LABEL_COLUMN not in labeled_edges.columns:
        raise ValueError(f"Labeled edge table must contain a {LABEL_COLUMN!r} column.")

    if SPLIT_COLUMN not in labeled_edges.columns:
        raise ValueError(f"Labeled edge table must contain a {SPLIT_COLUMN!r} column.")

    labeled_edges[SPLIT_COLUMN] = labeled_edges[SPLIT_COLUMN].astype(str)
    labeled_edges[LABEL_COLUMN] = labeled_edges[LABEL_COLUMN].astype(int)

    return labeled_edges


def add_graph_indices_to_edges(
    edges: pd.DataFrame,
    *,
    node_index: NodeIndex,
    source_type_column: str = SOURCE_TYPE_COLUMN,
    source_id_column: str = SOURCE_ID_COLUMN,
    target_type_column: str = TARGET_TYPE_COLUMN,
    target_id_column: str = TARGET_ID_COLUMN,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Add global and type-local graph indices to an edge table."""

    edges = edges.copy()

    source_keys = list(
        zip(
            edges[source_type_column].astype(str),
            edges[source_id_column].astype(str),
        )
    )
    target_keys = list(
        zip(
            edges[target_type_column].astype(str),
            edges[target_id_column].astype(str),
        )
    )

    source_global = [node_index.node_to_idx.get(key) for key in source_keys]
    target_global = [node_index.node_to_idx.get(key) for key in target_keys]
    source_local = [node_index.node_to_local_idx.get(key) for key in source_keys]
    target_local = [node_index.node_to_local_idx.get(key) for key in target_keys]

    edges[SOURCE_GRAPH_INDEX_COLUMN] = source_global
    edges[TARGET_GRAPH_INDEX_COLUMN] = target_global
    edges[SOURCE_LOCAL_INDEX_COLUMN] = source_local
    edges[TARGET_LOCAL_INDEX_COLUMN] = target_local

    missing_mask = (
        edges[SOURCE_GRAPH_INDEX_COLUMN].isna()
        | edges[TARGET_GRAPH_INDEX_COLUMN].isna()
        | edges[SOURCE_LOCAL_INDEX_COLUMN].isna()
        | edges[TARGET_LOCAL_INDEX_COLUMN].isna()
    )

    missing_count = int(missing_mask.sum())

    if missing_count and not drop_missing:
        raise ValueError(
            f"Could not map {missing_count} edges to graph node indices. "
            "Set drop_missing=True to drop them."
        )

    if missing_count and drop_missing:
        edges = edges.loc[~missing_mask].copy()

    index_columns = [
        SOURCE_GRAPH_INDEX_COLUMN,
        TARGET_GRAPH_INDEX_COLUMN,
        SOURCE_LOCAL_INDEX_COLUMN,
        TARGET_LOCAL_INDEX_COLUMN,
    ]

    for column in index_columns:
        edges[column] = edges[column].astype(np.int64)

    return edges


def build_relation_index(edges: pd.DataFrame) -> RelationIndex:
    """Build a relation index from normalized edge relations."""

    if RELATION_COLUMN not in edges.columns:
        raise ValueError(
            f"Edge table must contain {RELATION_COLUMN!r}. "
            "Call normalize_context_edges first."
        )

    relation_names = sorted(edges[RELATION_COLUMN].astype(str).unique())

    relation_to_idx = {
        relation: index
        for index, relation in enumerate(relation_names)
    }
    idx_to_relation = {
        index: relation
        for relation, index in relation_to_idx.items()
    }

    return RelationIndex(
        relation_to_idx=relation_to_idx,
        idx_to_relation=idx_to_relation,
    )


def add_relation_indices_to_edges(
    edges: pd.DataFrame,
    *,
    relation_index: RelationIndex,
) -> pd.DataFrame:
    """Add relation_idx column to a normalized edge table."""

    edges = edges.copy()

    if RELATION_COLUMN not in edges.columns:
        raise ValueError(f"Edge table must contain {RELATION_COLUMN!r}.")

    edges[RELATION_INDEX_COLUMN] = edges[RELATION_COLUMN].astype(str).map(
        relation_index.relation_to_idx
    )

    missing = int(edges[RELATION_INDEX_COLUMN].isna().sum())
    if missing:
        raise ValueError(f"Could not map {missing} relations to relation indices.")

    edges[RELATION_INDEX_COLUMN] = edges[RELATION_INDEX_COLUMN].astype(np.int64)

    return edges


def build_homogeneous_graph(
    context_edges: pd.DataFrame,
    *,
    nodes: pd.DataFrame,
    node_index: NodeIndex | None = None,
    add_reverse_edges: bool = True,
    drop_missing: bool = False,
) -> HomogeneousGraph:
    """Build homogeneous graph edge_index from context edges.

    Returns edge_index with shape [2, num_edges].
    """

    if node_index is None:
        node_index = build_node_index(nodes)

    normalized_edges = normalize_context_edges(context_edges, nodes=nodes)

    indexed_edges = add_graph_indices_to_edges(
        normalized_edges,
        node_index=node_index,
        drop_missing=drop_missing,
    )

    source = indexed_edges[SOURCE_GRAPH_INDEX_COLUMN].to_numpy(dtype=np.int64)
    target = indexed_edges[TARGET_GRAPH_INDEX_COLUMN].to_numpy(dtype=np.int64)

    if add_reverse_edges:
        edge_index = np.vstack(
            [
                np.concatenate([source, target]),
                np.concatenate([target, source]),
            ]
        )
    else:
        edge_index = np.vstack([source, target])

    return HomogeneousGraph(
        edge_index=edge_index,
        edge_frame=indexed_edges,
        num_nodes=len(node_index.nodes),
    )


def build_heterogeneous_graph(
    context_edges: pd.DataFrame,
    *,
    nodes: pd.DataFrame,
    node_index: NodeIndex | None = None,
    add_reverse_edges: bool = True,
    reverse_relation_suffix: str = "__rev",
    drop_missing: bool = False,
) -> HeterogeneousGraph:
    """Build heterogeneous edge_index_dict using type-local node indices.

    The returned edge_index_dict keys have the PyG-style form:

        (source_type, relation, target_type)

    Values are numpy arrays with shape [2, num_edges], using local node indices
    within source_type and target_type.
    """

    if node_index is None:
        node_index = build_node_index(nodes)

    normalized_edges = normalize_context_edges(context_edges, nodes=nodes)

    indexed_edges = add_graph_indices_to_edges(
        normalized_edges,
        node_index=node_index,
        drop_missing=drop_missing,
    )

    edge_index_dict: dict[tuple[str, str, str], np.ndarray] = {}

    group_columns = [
        SOURCE_TYPE_COLUMN,
        RELATION_COLUMN,
        TARGET_TYPE_COLUMN,
    ]

    for key_values, group in indexed_edges.groupby(group_columns, sort=True):
        source_type, relation, target_type = [str(value) for value in key_values]

        source = group[SOURCE_LOCAL_INDEX_COLUMN].to_numpy(dtype=np.int64)
        target = group[TARGET_LOCAL_INDEX_COLUMN].to_numpy(dtype=np.int64)

        edge_index_dict[(source_type, relation, target_type)] = np.vstack(
            [source, target]
        )

        if add_reverse_edges:
            reverse_key = (
                target_type,
                f"{relation}{reverse_relation_suffix}",
                source_type,
            )

            edge_index_dict[reverse_key] = np.vstack([target, source])

    return HeterogeneousGraph(
        edge_index_dict=edge_index_dict,
        num_nodes_dict=dict(node_index.num_nodes_by_type),
        edge_frame=indexed_edges,
    )


def build_kg_triples(
    context_edges: pd.DataFrame,
    *,
    nodes: pd.DataFrame,
    node_index: NodeIndex | None = None,
    relation_index: RelationIndex | None = None,
    add_reverse_edges: bool = True,
    reverse_relation_suffix: str = "__rev",
    drop_missing: bool = False,
) -> KGTriples:
    """Build KG triples from context edges.

    Triples use global entity indices:

        head_idx, relation_idx, tail_idx
    """

    if node_index is None:
        node_index = build_node_index(nodes)

    normalized_edges = normalize_context_edges(context_edges, nodes=nodes)

    if add_reverse_edges:
        reverse_edges = normalized_edges.copy()
        reverse_edges[SOURCE_TYPE_COLUMN] = normalized_edges[TARGET_TYPE_COLUMN].values
        reverse_edges[SOURCE_ID_COLUMN] = normalized_edges[TARGET_ID_COLUMN].values
        reverse_edges[TARGET_TYPE_COLUMN] = normalized_edges[SOURCE_TYPE_COLUMN].values
        reverse_edges[TARGET_ID_COLUMN] = normalized_edges[SOURCE_ID_COLUMN].values
        reverse_edges[RELATION_COLUMN] = (
            normalized_edges[RELATION_COLUMN].astype(str) + reverse_relation_suffix
        )

        normalized_edges = pd.concat(
            [normalized_edges, reverse_edges],
            ignore_index=True,
        )

    if relation_index is None:
        relation_index = build_relation_index(normalized_edges)

    indexed_edges = add_graph_indices_to_edges(
        normalized_edges,
        node_index=node_index,
        drop_missing=drop_missing,
    )

    indexed_edges = add_relation_indices_to_edges(
        indexed_edges,
        relation_index=relation_index,
    )

    triples = indexed_edges[
        [
            SOURCE_GRAPH_INDEX_COLUMN,
            RELATION_INDEX_COLUMN,
            TARGET_GRAPH_INDEX_COLUMN,
        ]
    ].to_numpy(dtype=np.int64)

    return KGTriples(
        triples=triples,
        triple_frame=indexed_edges,
        relation_index=relation_index,
        num_entities=len(node_index.nodes),
        num_relations=len(relation_index.relation_to_idx),
    )


def add_graph_indices_to_labeled_edges(
    labeled_edges: pd.DataFrame,
    *,
    node_index: NodeIndex,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Add graph indices to labeled link prediction pairs."""

    return add_graph_indices_to_edges(
        labeled_edges,
        node_index=node_index,
        drop_missing=drop_missing,
    )


def add_graph_indices_to_labeled_edges_by_split(
    labeled_edges_by_split: dict[str, pd.DataFrame],
    *,
    node_index: NodeIndex,
    drop_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    """Add graph indices to labeled edge tables for all splits."""

    return {
        split: add_graph_indices_to_labeled_edges(
            frame,
            node_index=node_index,
            drop_missing=drop_missing,
        )
        for split, frame in labeled_edges_by_split.items()
    }


def load_indexed_graph_dataset(
    dataset_dir: str | Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    add_reverse_homogeneous_edges: bool = True,
    add_reverse_heterogeneous_edges: bool = True,
    add_reverse_kg_edges: bool = True,
    drop_missing_edges: bool = False,
) -> dict[str, Any]:
    """Load a full indexed graph dataset bundle.

    This is a convenience wrapper for downstream baseline scripts.

    Returns a dictionary containing:

        tables
        node_index
        homogeneous_graph
        heterogeneous_graph
        kg_triples
        labeled_edges_by_split
    """

    tables = load_graph_tables(dataset_dir, splits=splits)

    node_index = build_node_index(tables.nodes)

    homogeneous_graph = build_homogeneous_graph(
        tables.context_edges,
        nodes=tables.nodes,
        node_index=node_index,
        add_reverse_edges=add_reverse_homogeneous_edges,
        drop_missing=drop_missing_edges,
    )

    heterogeneous_graph = build_heterogeneous_graph(
        tables.context_edges,
        nodes=tables.nodes,
        node_index=node_index,
        add_reverse_edges=add_reverse_heterogeneous_edges,
        drop_missing=drop_missing_edges,
    )

    kg_triples = build_kg_triples(
        tables.context_edges,
        nodes=tables.nodes,
        node_index=node_index,
        add_reverse_edges=add_reverse_kg_edges,
        drop_missing=drop_missing_edges,
    )

    labeled_edges_by_split = add_graph_indices_to_labeled_edges_by_split(
        tables.labeled_edges_by_split,
        node_index=node_index,
        drop_missing=drop_missing_edges,
    )

    return {
        "tables": tables,
        "node_index": node_index,
        "homogeneous_graph": homogeneous_graph,
        "heterogeneous_graph": heterogeneous_graph,
        "kg_triples": kg_triples,
        "labeled_edges_by_split": labeled_edges_by_split,
    }


def node_index_to_frame(node_index: NodeIndex) -> pd.DataFrame:
    """Return node index as a DataFrame suitable for export."""

    return node_index.nodes.copy()


def relation_index_to_frame(relation_index: RelationIndex) -> pd.DataFrame:
    """Return relation index as a DataFrame suitable for export."""

    rows = [
        {
            RELATION_INDEX_COLUMN: index,
            RELATION_COLUMN: relation,
        }
        for index, relation in relation_index.idx_to_relation.items()
    ]

    return pd.DataFrame(rows).sort_values(RELATION_INDEX_COLUMN).reset_index(drop=True)


def summarize_indexed_graph_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Summarize an indexed graph dataset bundle."""

    node_index: NodeIndex = dataset["node_index"]
    homogeneous_graph: HomogeneousGraph = dataset["homogeneous_graph"]
    heterogeneous_graph: HeterogeneousGraph = dataset["heterogeneous_graph"]
    kg_triples: KGTriples = dataset["kg_triples"]
    labeled_edges_by_split: dict[str, pd.DataFrame] = dataset["labeled_edges_by_split"]

    summary: dict[str, Any] = {
        "num_nodes": len(node_index.nodes),
        "num_node_types": len(node_index.num_nodes_by_type),
        "num_homogeneous_edges": int(homogeneous_graph.edge_index.shape[1]),
        "num_heterogeneous_edge_types": len(heterogeneous_graph.edge_index_dict),
        "num_kg_triples": int(kg_triples.triples.shape[0]),
        "num_relations": kg_triples.num_relations,
    }

    for node_type, count in sorted(node_index.num_nodes_by_type.items()):
        summary[f"num_nodes:{node_type}"] = int(count)

    for split, frame in labeled_edges_by_split.items():
        summary[f"labeled_edges:{split}"] = int(len(frame))
        summary[f"positive_edges:{split}"] = int(frame[LABEL_COLUMN].sum())
        summary[f"negative_edges:{split}"] = int(len(frame) - frame[LABEL_COLUMN].sum())

    return summary