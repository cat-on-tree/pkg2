"""Link prediction dataset construction utilities.

This module builds carrier-level link prediction datasets from the PKG2
Knowledge Carrier Graph prototype.

The current primary task is Patent-Paper link prediction. This is a
carrier-level proxy task used to validate graph signal and establish reusable
dataset construction infrastructure. It is not the final knowledge-unit-level
translation prediction task.

Main outputs:
- nodes.parquet
- node_mapping.parquet
- context_edges.parquet
- positive_edges_train.parquet
- positive_edges_val.parquet
- positive_edges_test.parquet
- negative_edges_train.parquet
- negative_edges_val.parquet
- negative_edges_test.parquet
- labeled_edges_train.parquet
- labeled_edges_val.parquet
- labeled_edges_test.parquet
- dataset_manifest.csv
- dataset_report.md

The context graph excludes the target edge table by default to avoid target
edge leakage. For the default task, all edges_patent_paper rows are excluded
from context_edges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .graph_filtering import (
    create_filtered_graph,
    summarize_filtered_edge_tables,
    summarize_filtered_node_types,
    summarize_filtering,
)
from .graph_loading import load_carrier_graph
from .graph_schema import (
    DATASET_OUTPUT_FILES,
    DEFAULT_TARGET_EDGE_TABLE,
    get_context_edge_tables,
    get_edge_schema,
)
from .io import (
    count_rows,
    prepare_output_dir,
    qident,
    scan_sql,
    sql_literal,
)
from .negative_sampling import (
    sample_negative_edges_for_splits,
    summarize_negative_splits,
)
from .reporting import (
    build_artifact_rows,
    write_dataset_report,
    write_manifest,
)
from .splitting import (
    create_target_edge_splits,
    summarize_positive_splits,
)


def export_table_to_parquet(
    connection: Any,
    *,
    table_name: str,
    output_path: str | Path,
    overwrite: bool = True,
    compression: str = "ZSTD",
) -> Path:
    """Export a DuckDB table to a Parquet file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    if output_path.exists():
        output_path.unlink()

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM {qident(table_name)}
        )
        TO {sql_literal(output_path)}
        (
            FORMAT PARQUET,
            COMPRESSION {compression}
        )
        """
    )

    return output_path


def create_node_mapping(
    connection: Any,
    *,
    node_table: str = "filtered_node_all",
    output_table: str = "node_mapping",
    replace: bool = True,
) -> int:
    """Create a stable integer node mapping table.

    The resulting table maps each typed node to a zero-based node_index.

    Columns:
    - node_index
    - node_type
    - node_id
    - label
    - bioentity_type
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY node_type, node_id) - 1 AS node_index,
            node_type,
            node_id,
            label,
            bioentity_type
        FROM {qident(node_table)}
        """
    )

    return count_rows(connection, output_table)


def create_nodes_table(
    connection: Any,
    *,
    node_mapping_table: str = "node_mapping",
    output_table: str = "nodes",
    replace: bool = True,
) -> int:
    """Create the exported model node table.

    At this stage, the exported nodes table is intentionally the same as the
    node mapping table. Keeping both names makes downstream scripts clearer:
    node_mapping.parquet is the mapping artifact, while nodes.parquet is the
    model-facing node table.
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            node_index,
            node_type,
            node_id,
            label,
            bioentity_type
        FROM {qident(node_mapping_table)}
        """
    )

    return count_rows(connection, output_table)


def create_context_edges(
    connection: Any,
    *,
    edge_table: str = "filtered_edge_all",
    node_mapping_table: str = "node_mapping",
    output_table: str = "context_edges",
    target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE,
    replace: bool = True,
) -> int:
    """Create model context edges with integer node indices.

    The target edge table is excluded to avoid leakage.

    For the default Patent-Paper link prediction task, this means all
    edges_patent_paper rows are removed from the context graph.
    """

    context_edge_tables = get_context_edge_tables(target_edge_table)
    context_edge_list_sql = "(" + ", ".join(sql_literal(value) for value in context_edge_tables) + ")"

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            sm.node_index AS source_index,
            tm.node_index AS target_index,
            e.source_type,
            e.source_id,
            e.target_type,
            e.target_id,
            e.edge_table,
            e.edge_type,
            e.edge_weight,
            e.mention_count
        FROM {qident(edge_table)} e
        INNER JOIN {qident(node_mapping_table)} sm
            ON e.source_type = sm.node_type
           AND e.source_id = sm.node_id
        INNER JOIN {qident(node_mapping_table)} tm
            ON e.target_type = tm.node_type
           AND e.target_id = tm.node_id
        WHERE e.edge_table IN {context_edge_list_sql}
        """
    )

    return count_rows(connection, output_table)


def create_labeled_edge_table(
    connection: Any,
    *,
    positive_table: str,
    negative_table: str,
    node_mapping_table: str = "node_mapping",
    output_table: str,
    replace: bool = True,
) -> int:
    """Create a labeled edge table for one split.

    The output contains both positive and negative examples with integer node
    indices and original node IDs.
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            sm.node_index AS source_index,
            tm.node_index AS target_index,
            e.source_type,
            e.source_id,
            e.target_type,
            e.target_id,
            e.edge_table,
            e.edge_type,
            e.split,
            e.label
        FROM (
            SELECT
                source_type,
                source_id,
                target_type,
                target_id,
                edge_table,
                edge_type,
                split,
                label
            FROM {qident(positive_table)}

            UNION ALL

            SELECT
                source_type,
                source_id,
                target_type,
                target_id,
                edge_table,
                edge_type,
                split,
                label
            FROM {qident(negative_table)}
        ) e
        INNER JOIN {qident(node_mapping_table)} sm
            ON e.source_type = sm.node_type
           AND e.source_id = sm.node_id
        INNER JOIN {qident(node_mapping_table)} tm
            ON e.target_type = tm.node_type
           AND e.target_id = tm.node_id
        """
    )

    return count_rows(connection, output_table)


def create_labeled_edge_tables(
    connection: Any,
    *,
    node_mapping_table: str = "node_mapping",
    positive_train_table: str = "positive_edges_train",
    positive_val_table: str = "positive_edges_val",
    positive_test_table: str = "positive_edges_test",
    negative_train_table: str = "negative_edges_train",
    negative_val_table: str = "negative_edges_val",
    negative_test_table: str = "negative_edges_test",
    labeled_train_table: str = "labeled_edges_train",
    labeled_val_table: str = "labeled_edges_val",
    labeled_test_table: str = "labeled_edges_test",
) -> dict[str, int]:
    """Create labeled train / validation / test edge tables."""

    counts = {}

    counts[labeled_train_table] = create_labeled_edge_table(
        connection,
        positive_table=positive_train_table,
        negative_table=negative_train_table,
        node_mapping_table=node_mapping_table,
        output_table=labeled_train_table,
    )

    counts[labeled_val_table] = create_labeled_edge_table(
        connection,
        positive_table=positive_val_table,
        negative_table=negative_val_table,
        node_mapping_table=node_mapping_table,
        output_table=labeled_val_table,
    )

    counts[labeled_test_table] = create_labeled_edge_table(
        connection,
        positive_table=positive_test_table,
        negative_table=negative_test_table,
        node_mapping_table=node_mapping_table,
        output_table=labeled_test_table,
    )

    return counts


def export_link_prediction_tables(
    connection: Any,
    *,
    output_dir: str | Path,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Export standard link prediction dataset tables to Parquet files."""

    output_dir = Path(output_dir)

    table_to_file = {
        "nodes": DATASET_OUTPUT_FILES["nodes"],
        "node_mapping": DATASET_OUTPUT_FILES["node_mapping"],
        "context_edges": DATASET_OUTPUT_FILES["context_edges"],
        "positive_edges_train": DATASET_OUTPUT_FILES["positive_edges_train"],
        "positive_edges_val": DATASET_OUTPUT_FILES["positive_edges_val"],
        "positive_edges_test": DATASET_OUTPUT_FILES["positive_edges_test"],
        "negative_edges_train": DATASET_OUTPUT_FILES["negative_edges_train"],
        "negative_edges_val": DATASET_OUTPUT_FILES["negative_edges_val"],
        "negative_edges_test": DATASET_OUTPUT_FILES["negative_edges_test"],
        "labeled_edges_train": DATASET_OUTPUT_FILES["labeled_edges_train"],
        "labeled_edges_val": DATASET_OUTPUT_FILES["labeled_edges_val"],
        "labeled_edges_test": DATASET_OUTPUT_FILES["labeled_edges_test"],
    }

    descriptions = {
        "nodes": "Model-facing node table with integer node indices.",
        "node_mapping": "Mapping from typed graph nodes to integer node indices.",
        "context_edges": "Leakage-controlled context graph edges excluding the target edge table.",
        "positive_edges_train": "Positive target edges for the training split.",
        "positive_edges_val": "Positive target edges for the validation split.",
        "positive_edges_test": "Positive target edges for the test split.",
        "negative_edges_train": "Negative target edges for the training split.",
        "negative_edges_val": "Negative target edges for the validation split.",
        "negative_edges_test": "Negative target edges for the test split.",
        "labeled_edges_train": "Combined positive and negative labeled training examples.",
        "labeled_edges_val": "Combined positive and negative labeled validation examples.",
        "labeled_edges_test": "Combined positive and negative labeled test examples.",
    }

    artifacts = []

    for table_name, file_name in table_to_file.items():
        output_path = output_dir / file_name

        export_table_to_parquet(
            connection,
            table_name=table_name,
            output_path=output_path,
            overwrite=overwrite,
        )

        artifacts.append(
            {
                "artifact": table_name,
                "file_name": file_name,
                "description": descriptions.get(table_name, ""),
                "row_count": count_rows(connection, table_name),
            }
        )

    return artifacts


def _build_parameter_summary(
    *,
    graph_dir: str | Path,
    output_dir: str | Path,
    target_edge_table: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    negative_ratio: float,
    seed: int,
    filter_isolated_nodes: bool,
    exclude_bioentity_types: Iterable[str] | None,
    exclude_bioentity_ids: Iterable[str] | None,
) -> dict[str, Any]:
    """Build a parameter dictionary for reporting."""

    schema = get_edge_schema(target_edge_table)

    return {
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "task": "Patent-Paper link prediction",
        "task_level": "carrier-level proxy task",
        "target_edge_table": target_edge_table,
        "source_node_type": schema.source_type,
        "target_node_type": schema.target_type,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "negative_ratio": negative_ratio,
        "seed": seed,
        "filter_isolated_nodes": filter_isolated_nodes,
        "exclude_bioentity_types": ", ".join(exclude_bioentity_types or []),
        "exclude_bioentity_ids": ", ".join(exclude_bioentity_ids or []),
        "context_edge_policy": "exclude target edge table from context graph",
    }


def _write_dataset_artifacts_and_reports(
    connection: Any,
    *,
    output_dir: str | Path,
    parameters: dict[str, Any],
    graph_counts: dict[str, Any],
    filtering_summary: dict[str, Any],
    node_type_filtering: list[dict[str, Any]],
    edge_table_filtering: list[dict[str, Any]],
    positive_split_summary: list[dict[str, Any]],
    negative_split_summary: list[dict[str, Any]],
    negative_sampling_summary: dict[str, Any],
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Export Parquet artifacts and write manifest/report files."""

    output_dir = Path(output_dir)

    parquet_artifacts = export_link_prediction_tables(
        connection,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    artifact_rows = build_artifact_rows(output_dir, parquet_artifacts)

    manifest_path = write_manifest(
        output_dir,
        artifact_rows,
        file_name=DATASET_OUTPUT_FILES["dataset_manifest"],
    )

    artifact_rows_with_manifest = artifact_rows + [
        {
            "artifact": "dataset_manifest",
            "path": str(manifest_path),
            "description": "CSV manifest of dataset output artifacts.",
            "row_count": len(artifact_rows),
            "file_size_bytes": manifest_path.stat().st_size if manifest_path.exists() else "",
        }
    ]

    report_path = write_dataset_report(
        output_dir,
        parameters=parameters,
        graph_counts=graph_counts,
        filtering_summary=filtering_summary,
        node_type_filtering=node_type_filtering,
        edge_table_filtering=edge_table_filtering,
        positive_split_summary=positive_split_summary,
        negative_split_summary=negative_split_summary,
        negative_sampling_summary=negative_sampling_summary,
        artifact_rows=artifact_rows_with_manifest,
        file_name=DATASET_OUTPUT_FILES["dataset_report"],
    )

    final_artifact_rows = artifact_rows_with_manifest + [
        {
            "artifact": "dataset_report",
            "path": str(report_path),
            "description": "Markdown report describing dataset construction.",
            "row_count": "",
            "file_size_bytes": report_path.stat().st_size if report_path.exists() else "",
        }
    ]

    write_manifest(
        output_dir,
        final_artifact_rows,
        file_name=DATASET_OUTPUT_FILES["dataset_manifest"],
    )

    return final_artifact_rows


def build_patent_paper_link_prediction_dataset(
    connection: Any,
    *,
    graph_dir: str | Path,
    output_dir: str | Path,
    target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    negative_ratio: float = 1.0,
    seed: int = 42,
    filter_isolated_nodes: bool = True,
    exclude_bioentity_types: Iterable[str] | None = None,
    exclude_bioentity_ids: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a Patent-Paper link prediction dataset.

    This is the main entry point used by scripts/11_build_link_prediction_dataset.py.

    Parameters
    ----------
    connection:
        DuckDB connection.
    graph_dir:
        Input graph directory produced by graph construction scripts.
    output_dir:
        Output dataset directory.
    target_edge_table:
        Target edge table for link prediction. Defaults to edges_patent_paper.
    train_ratio:
        Training positive split ratio.
    val_ratio:
        Validation positive split ratio.
    test_ratio:
        Test positive split ratio.
    negative_ratio:
        Number of negative examples per positive example.
    seed:
        Random seed used for positive splitting and negative sampling.
    filter_isolated_nodes:
        Whether to remove isolated nodes before dataset construction.
    exclude_bioentity_types:
        Optional BioEntity types to remove, for example ["species"].
    exclude_bioentity_ids:
        Optional specific BioEntity node IDs to remove.
    overwrite:
        Whether to overwrite an existing non-empty output directory.

    Returns
    -------
    dict[str, Any]
        Summary dictionary containing counts and output paths.
    """

    graph_dir = Path(graph_dir)
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)

    exclude_bioentity_types = [
        str(value).strip()
        for value in (exclude_bioentity_types or [])
        if str(value).strip()
    ]

    exclude_bioentity_ids = [
        str(value).strip()
        for value in (exclude_bioentity_ids or [])
        if str(value).strip()
    ]

    graph_counts = load_carrier_graph(
        connection,
        graph_dir,
        node_table="node_all",
        edge_table="edge_all",
        degree_table="node_degree",
        require_files=True,
        use_mention_count_weight=True,
        distinct_basic_edges=False,
        create_degree_table=True,
    )

    filtered_counts = create_filtered_graph(
        connection,
        source_node_table="node_all",
        source_edge_table="edge_all",
        source_degree_table="node_degree",
        filtered_node_table="filtered_node_all",
        filtered_edge_table="filtered_edge_all",
        filtered_degree_table="filtered_node_degree",
        filter_isolated_nodes=filter_isolated_nodes,
        exclude_bioentity_types=exclude_bioentity_types,
        exclude_bioentity_ids=exclude_bioentity_ids,
    )

    filtering_summary = summarize_filtering(
        connection,
        source_node_table="node_all",
        source_edge_table="edge_all",
        filtered_node_table="filtered_node_all",
        filtered_edge_table="filtered_edge_all",
    )

    node_type_filtering = summarize_filtered_node_types(
        connection,
        source_node_table="node_all",
        filtered_node_table="filtered_node_all",
    )

    edge_table_filtering = summarize_filtered_edge_tables(
        connection,
        source_edge_table="edge_all",
        filtered_edge_table="filtered_edge_all",
    )

    split_counts = create_target_edge_splits(
        connection,
        source_edge_table="filtered_edge_all",
        target_edge_table=target_edge_table,
        target_positive_table="target_positive_edges",
        randomized_table="target_positive_edges_randomized",
        train_table="positive_edges_train",
        val_table="positive_edges_val",
        test_table="positive_edges_test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    negative_sampling_summary = sample_negative_edges_for_splits(
        connection,
        node_table="filtered_node_all",
        target_positive_table="target_positive_edges",
        train_positive_table="positive_edges_train",
        val_positive_table="positive_edges_val",
        test_positive_table="positive_edges_test",
        train_negative_table="negative_edges_train",
        val_negative_table="negative_edges_val",
        test_negative_table="negative_edges_test",
        all_positive_pair_table="all_positive_pairs",
        candidate_table="negative_candidate_pairs",
        randomized_candidate_table="negative_candidate_pairs_randomized",
        target_edge_table=target_edge_table,
        negative_ratio=negative_ratio,
        seed=seed + 1,
    )

    node_mapping_count = create_node_mapping(
        connection,
        node_table="filtered_node_all",
        output_table="node_mapping",
    )

    nodes_count = create_nodes_table(
        connection,
        node_mapping_table="node_mapping",
        output_table="nodes",
    )

    context_edge_count = create_context_edges(
        connection,
        edge_table="filtered_edge_all",
        node_mapping_table="node_mapping",
        output_table="context_edges",
        target_edge_table=target_edge_table,
    )

    labeled_counts = create_labeled_edge_tables(
        connection,
        node_mapping_table="node_mapping",
        positive_train_table="positive_edges_train",
        positive_val_table="positive_edges_val",
        positive_test_table="positive_edges_test",
        negative_train_table="negative_edges_train",
        negative_val_table="negative_edges_val",
        negative_test_table="negative_edges_test",
        labeled_train_table="labeled_edges_train",
        labeled_val_table="labeled_edges_val",
        labeled_test_table="labeled_edges_test",
    )

    positive_split_summary = summarize_positive_splits(
        connection,
        train_table="positive_edges_train",
        val_table="positive_edges_val",
        test_table="positive_edges_test",
    )

    negative_split_summary = summarize_negative_splits(
        connection,
        train_table="negative_edges_train",
        val_table="negative_edges_val",
        test_table="negative_edges_test",
    )

    parameters = _build_parameter_summary(
        graph_dir=graph_dir,
        output_dir=output_dir,
        target_edge_table=target_edge_table,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        negative_ratio=negative_ratio,
        seed=seed,
        filter_isolated_nodes=filter_isolated_nodes,
        exclude_bioentity_types=exclude_bioentity_types,
        exclude_bioentity_ids=exclude_bioentity_ids,
    )

    report_graph_counts = {
        "loaded_node_count": graph_counts.get("node_all", ""),
        "loaded_edge_count": graph_counts.get("edge_all", ""),
        "loaded_degree_node_count": graph_counts.get("node_degree", ""),
        "filtered_node_count": filtered_counts.get("filtered_node_all", ""),
        "filtered_edge_count": filtered_counts.get("filtered_edge_all", ""),
        "filtered_degree_node_count": filtered_counts.get("filtered_node_degree", ""),
        "node_mapping_count": node_mapping_count,
        "nodes_count": nodes_count,
        "context_edge_count": context_edge_count,
        "target_positive_edge_count": split_counts.get("target_positive_edges", ""),
        "positive_train_count": split_counts.get("positive_edges_train", ""),
        "positive_val_count": split_counts.get("positive_edges_val", ""),
        "positive_test_count": split_counts.get("positive_edges_test", ""),
        "negative_train_count": negative_sampling_summary.get("negative_split_counts", {}).get(
            "negative_edges_train",
            "",
        ),
        "negative_val_count": negative_sampling_summary.get("negative_split_counts", {}).get(
            "negative_edges_val",
            "",
        ),
        "negative_test_count": negative_sampling_summary.get("negative_split_counts", {}).get(
            "negative_edges_test",
            "",
        ),
        "labeled_train_count": labeled_counts.get("labeled_edges_train", ""),
        "labeled_val_count": labeled_counts.get("labeled_edges_val", ""),
        "labeled_test_count": labeled_counts.get("labeled_edges_test", ""),
    }

    artifact_rows = _write_dataset_artifacts_and_reports(
        connection,
        output_dir=output_dir,
        parameters=parameters,
        graph_counts=report_graph_counts,
        filtering_summary=filtering_summary,
        node_type_filtering=node_type_filtering,
        edge_table_filtering=edge_table_filtering,
        positive_split_summary=positive_split_summary,
        negative_split_summary=negative_split_summary,
        negative_sampling_summary=negative_sampling_summary,
        overwrite=True,
    )

    return {
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "target_edge_table": target_edge_table,
        "parameters": parameters,
        "graph_counts": graph_counts,
        "filtered_counts": filtered_counts,
        "filtering_summary": filtering_summary,
        "split_counts": split_counts,
        "negative_sampling_summary": negative_sampling_summary,
        "node_mapping_count": node_mapping_count,
        "nodes_count": nodes_count,
        "context_edge_count": context_edge_count,
        "labeled_counts": labeled_counts,
        "artifact_rows": artifact_rows,
        "dataset_manifest": str(output_dir / DATASET_OUTPUT_FILES["dataset_manifest"]),
        "dataset_report": str(output_dir / DATASET_OUTPUT_FILES["dataset_report"]),
    }