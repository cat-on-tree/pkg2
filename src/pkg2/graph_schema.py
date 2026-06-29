"""Shared graph schema definitions for the PKG2 carrier graph pipeline.

This module defines the stable file names, node types, edge tables, and default
task configuration used by the algorithm-stage scripts.

Conceptual positioning
----------------------
The graph currently produced by scripts 06-10 is a Knowledge Carrier Graph
prototype. Its nodes include papers, patents, clinical trials, projects, and
BioEntities. Paper / patent / trial nodes are treated as knowledge carriers,
not as knowledge units themselves.

The initial downstream task, Patent-Paper link prediction, is a carrier-level
proxy task. It is used to validate graph signal and establish reusable dataset
construction infrastructure before moving to knowledge-unit-level modeling.

Long-term direction
-------------------
Later stages may add knowledge-unit-level schemas, for example entity pairs or
entity-relation triples:

    z = (entity_a, entity_b)
    z = (head_entity, relation, tail_entity)

Those schemas should be added in separate modules such as
``knowledge_units.py`` rather than mixed into this carrier graph schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class NodeSchema:
    """Schema metadata for one node table."""

    node_type: str
    table_name: str
    file_name: str
    id_column: str = "node_id"
    label_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeSchema:
    """Schema metadata for one edge table."""

    edge_table: str
    file_name: str
    source_type: str
    target_type: str
    source_column: str = "source_id"
    target_column: str = "target_id"
    edge_type_column: str = "edge_type"
    weight_column: str | None = None
    default_weight: float = 1.0
    is_bioentity_edge: bool = False


NODE_SCHEMAS: Final[dict[str, NodeSchema]] = {
    "Patent": NodeSchema(
        node_type="Patent",
        table_name="nodes_patent",
        file_name="nodes_patent.parquet",
        label_columns=("Title", "Abstract"),
    ),
    "Paper": NodeSchema(
        node_type="Paper",
        table_name="nodes_paper",
        file_name="nodes_paper.parquet",
        label_columns=("ArticleTitle",),
    ),
    "ClinicalTrial": NodeSchema(
        node_type="ClinicalTrial",
        table_name="nodes_clinicaltrial",
        file_name="nodes_clinicaltrial.parquet",
        label_columns=("brief_title", "official_title", "conditions"),
    ),
    "Project": NodeSchema(
        node_type="Project",
        table_name="nodes_project",
        file_name="nodes_project.parquet",
        label_columns=("PROJECT_TITLE", "Abstract"),
    ),
    "BioEntity": NodeSchema(
        node_type="BioEntity",
        table_name="nodes_bioentity",
        file_name="nodes_bioentity.parquet",
        label_columns=("Mention",),
    ),
}


EDGE_SCHEMAS: Final[dict[str, EdgeSchema]] = {
    "edges_patent_paper": EdgeSchema(
        edge_table="edges_patent_paper",
        file_name="edges_patent_paper.parquet",
        source_type="Patent",
        target_type="Paper",
    ),
    "edges_patent_bioentity": EdgeSchema(
        edge_table="edges_patent_bioentity",
        file_name="edges_patent_bioentity.parquet",
        source_type="Patent",
        target_type="BioEntity",
        weight_column="mention_count",
        is_bioentity_edge=True,
    ),
    "edges_patent_project": EdgeSchema(
        edge_table="edges_patent_project",
        file_name="edges_patent_project.parquet",
        source_type="Patent",
        target_type="Project",
    ),
    "edges_paper_trial": EdgeSchema(
        edge_table="edges_paper_trial",
        file_name="edges_paper_trial.parquet",
        source_type="Paper",
        target_type="ClinicalTrial",
    ),
    "edges_paper_project": EdgeSchema(
        edge_table="edges_paper_project",
        file_name="edges_paper_project.parquet",
        source_type="Paper",
        target_type="Project",
    ),
    "edges_paper_bioentity": EdgeSchema(
        edge_table="edges_paper_bioentity",
        file_name="edges_paper_bioentity.parquet",
        source_type="Paper",
        target_type="BioEntity",
        weight_column="mention_count",
        is_bioentity_edge=True,
    ),
    "edges_trial_project": EdgeSchema(
        edge_table="edges_trial_project",
        file_name="edges_trial_project.parquet",
        source_type="ClinicalTrial",
        target_type="Project",
    ),
    "edges_trial_bioentity": EdgeSchema(
        edge_table="edges_trial_bioentity",
        file_name="edges_trial_bioentity.parquet",
        source_type="ClinicalTrial",
        target_type="BioEntity",
        weight_column="mention_count",
        is_bioentity_edge=True,
    ),
}


NODE_TYPES: Final[tuple[str, ...]] = tuple(NODE_SCHEMAS.keys())

EDGE_TABLES: Final[tuple[str, ...]] = tuple(EDGE_SCHEMAS.keys())

BIOENTITY_EDGE_TABLES: Final[tuple[str, ...]] = tuple(
    edge_table
    for edge_table, schema in EDGE_SCHEMAS.items()
    if schema.is_bioentity_edge
)

CARRIER_NODE_TYPES: Final[tuple[str, ...]] = (
    "Patent",
    "Paper",
    "ClinicalTrial",
    "Project",
)

DOCUMENT_CARRIER_NODE_TYPES: Final[tuple[str, ...]] = (
    "Patent",
    "Paper",
    "ClinicalTrial",
)

CONTEXT_EDGE_TABLES: Final[tuple[str, ...]] = tuple(
    edge_table for edge_table in EDGE_TABLES if edge_table != "edges_patent_paper"
)

DEFAULT_TARGET_EDGE_TABLE: Final[str] = "edges_patent_paper"

DEFAULT_TARGET_SOURCE_TYPE: Final[str] = EDGE_SCHEMAS[DEFAULT_TARGET_EDGE_TABLE].source_type

DEFAULT_TARGET_TARGET_TYPE: Final[str] = EDGE_SCHEMAS[DEFAULT_TARGET_EDGE_TABLE].target_type

DEFAULT_TARGET_EDGE_TYPE: Final[str] = "patent_links_paper"


# BioEntity types frequently used for sensitivity analyses.
#
# In the current diabetes carrier graph, species hubs such as patients, mice,
# rat, and mouse dominate degree statistics. They are useful as biomedical
# context but should often be filtered or downweighted in sensitivity analyses.
DEFAULT_EXCLUDABLE_BIOENTITY_TYPES: Final[tuple[str, ...]] = (
    "species",
)

DEFAULT_EXCLUDABLE_BIOENTITY_IDS: Final[tuple[str, ...]] = (
    "NCBITaxon9606",   # human / patients / persons
    "NCBITaxon10095",  # mice
    "NCBITaxon10116",  # rat
    "NCBITaxon10090",  # mouse
)


# Model-ready dataset output file names.
DATASET_OUTPUT_FILES: Final[dict[str, str]] = {
    "nodes": "nodes.parquet",
    "node_mapping": "node_mapping.parquet",
    "context_edges": "context_edges.parquet",
    "positive_edges_train": "positive_edges_train.parquet",
    "positive_edges_val": "positive_edges_val.parquet",
    "positive_edges_test": "positive_edges_test.parquet",
    "negative_edges_train": "negative_edges_train.parquet",
    "negative_edges_val": "negative_edges_val.parquet",
    "negative_edges_test": "negative_edges_test.parquet",
    "labeled_edges_train": "labeled_edges_train.parquet",
    "labeled_edges_val": "labeled_edges_val.parquet",
    "labeled_edges_test": "labeled_edges_test.parquet",
    "dataset_manifest": "dataset_manifest.csv",
    "dataset_report": "dataset_report.md",
}


def get_node_schema(node_type: str) -> NodeSchema:
    """Return schema metadata for a node type."""

    try:
        return NODE_SCHEMAS[node_type]
    except KeyError as exc:
        valid = ", ".join(NODE_TYPES)
        raise KeyError(f"Unknown node type: {node_type!r}. Valid node types: {valid}") from exc


def get_edge_schema(edge_table: str) -> EdgeSchema:
    """Return schema metadata for an edge table."""

    try:
        return EDGE_SCHEMAS[edge_table]
    except KeyError as exc:
        valid = ", ".join(EDGE_TABLES)
        raise KeyError(f"Unknown edge table: {edge_table!r}. Valid edge tables: {valid}") from exc


def get_required_graph_files() -> tuple[str, ...]:
    """Return all required node and edge file names for a carrier graph directory."""

    node_files = [schema.file_name for schema in NODE_SCHEMAS.values()]
    edge_files = [schema.file_name for schema in EDGE_SCHEMAS.values()]
    return tuple(node_files + edge_files)


def get_context_edge_tables(target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE) -> tuple[str, ...]:
    """Return edge tables to use as context edges for a target link prediction task.

    For leakage-safe Patent-Paper link prediction, the default behavior is to
    exclude all target-edge-table rows from the context graph.
    """

    get_edge_schema(target_edge_table)
    return tuple(edge_table for edge_table in EDGE_TABLES if edge_table != target_edge_table)


def get_target_node_types(target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE) -> tuple[str, str]:
    """Return source and target node types for a target edge table."""

    schema = get_edge_schema(target_edge_table)
    return schema.source_type, schema.target_type