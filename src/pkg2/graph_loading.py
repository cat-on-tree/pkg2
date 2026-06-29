"""Graph loading utilities for extracted or aggregated PKG2 carrier graphs.

This module loads graph directories produced by scripts 06-10 into DuckDB
views and unified temporary tables.

The current graph is a Knowledge Carrier Graph prototype. It contains carrier
nodes such as Patent, Paper, ClinicalTrial, and Project, plus BioEntity nodes
used as evidence for later Knowledge Unit construction.

Main responsibilities:
- Validate that a graph directory contains required Parquet files.
- Create DuckDB temporary views for node and edge Parquet files.
- Build a unified node table with a common schema.
- Build a unified edge table with typed endpoints and edge weights.
- Build a node-degree table used by downstream filtering and diagnostics.

Default unified node table: node_all

Columns:
- node_type
- node_id
- label
- bioentity_type

Default unified edge table: edge_all

Columns:
- edge_table
- edge_type
- source_type
- source_id
- target_type
- target_id
- edge_weight
- mention_count

For BioEntity edges aggregated by scripts/09_aggregate_mention_edges.py,
edge_weight uses mention_count when available. Non-BioEntity edges use weight
1.0 by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .graph_schema import (
    EDGE_SCHEMAS,
    NODE_SCHEMAS,
    EdgeSchema,
    NodeSchema,
    get_required_graph_files,
)
from .io import (
    count_rows,
    fetch_dicts,
    get_columns,
    qident,
    require_paths_exist,
    scan_sql,
    sql_literal,
)


def require_graph_files(graph_dir: str | Path) -> None:
    """Validate that a graph directory contains all required node and edge files."""

    graph_dir = Path(graph_dir)
    required_paths = [graph_dir / file_name for file_name in get_required_graph_files()]
    require_paths_exist(required_paths)


def create_node_views(connection: Any, graph_dir: str | Path) -> None:
    """Create DuckDB temporary views for all node Parquet files."""

    graph_dir = Path(graph_dir)

    for schema in NODE_SCHEMAS.values():
        file_path = graph_dir / schema.file_name
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(schema.table_name)} AS
            SELECT *
            FROM {scan_sql(file_path)}
            """
        )


def create_edge_views(connection: Any, graph_dir: str | Path) -> None:
    """Create DuckDB temporary views for all edge Parquet files."""

    graph_dir = Path(graph_dir)

    for schema in EDGE_SCHEMAS.values():
        file_path = graph_dir / schema.file_name
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(schema.edge_table)} AS
            SELECT *
            FROM {scan_sql(file_path)}
            """
        )


def create_graph_views(
    connection: Any,
    graph_dir: str | Path,
    *,
    require_files: bool = True,
) -> None:
    """Validate and load all node and edge files as DuckDB temporary views."""

    graph_dir = Path(graph_dir)

    if require_files:
        require_graph_files(graph_dir)

    create_node_views(connection, graph_dir)
    create_edge_views(connection, graph_dir)


def _nullif_trim_cast(alias: str, column: str) -> str:
    """Return a SQL expression for a non-empty string column value."""

    return f"NULLIF(TRIM(CAST({alias}.{qident(column)} AS VARCHAR)), '')"


def label_expr_for_node_schema(
    connection: Any,
    schema: NodeSchema,
    *,
    alias: str = "n",
) -> str:
    """Build a robust label expression for a node table.

    The expression uses the first available non-empty value among schema-defined
    label columns and falls back to node_id.
    """

    columns = get_columns(connection, schema.table_name)

    pieces = []
    for column in schema.label_columns:
        if column in columns:
            pieces.append(_nullif_trim_cast(alias, column))

    pieces.append(f"CAST({alias}.{qident(schema.id_column)} AS VARCHAR)")

    return "COALESCE(" + ", ".join(pieces) + ")"


def bioentity_type_expr_for_node_schema(
    connection: Any,
    schema: NodeSchema,
    *,
    alias: str = "n",
) -> str:
    """Build a BioEntity type expression.

    Only BioEntity nodes are expected to have a meaningful Type column.
    Other node types receive NULL.
    """

    if schema.node_type != "BioEntity":
        return "CAST(NULL AS VARCHAR)"

    columns = get_columns(connection, schema.table_name)

    if "Type" in columns:
        return f"CAST({alias}.{qident('Type')} AS VARCHAR)"

    return "CAST(NULL AS VARCHAR)"


def create_node_union(
    connection: Any,
    *,
    output_table: str = "node_all",
    replace: bool = True,
) -> None:
    """Create a unified node table.

    Parameters
    ----------
    connection:
        DuckDB connection with node views already loaded.
    output_table:
        Name of the unified node table to create.
    replace:
        If true, create or replace the output table.
    """

    union_parts = []

    for schema in NODE_SCHEMAS.values():
        label_expr = label_expr_for_node_schema(connection, schema, alias="n")
        bioentity_type_expr = bioentity_type_expr_for_node_schema(connection, schema, alias="n")

        union_parts.append(
            f"""
            SELECT
                {sql_literal(schema.node_type)} AS node_type,
                CAST(n.{qident(schema.id_column)} AS VARCHAR) AS node_id,
                {label_expr} AS label,
                {bioentity_type_expr} AS bioentity_type
            FROM {qident(schema.table_name)} n
            WHERE n.{qident(schema.id_column)} IS NOT NULL
            """
        )

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        {" UNION ALL ".join(union_parts)}
        """
    )


def _edge_type_expr_for_edge_schema(
    connection: Any,
    schema: EdgeSchema,
    *,
    alias: str = "e",
) -> str:
    """Return SQL expression for edge_type."""

    columns = get_columns(connection, schema.edge_table)

    if schema.edge_type_column in columns:
        return f"CAST({alias}.{qident(schema.edge_type_column)} AS VARCHAR)"

    return sql_literal(schema.edge_table)


def _edge_type_not_null_condition(
    connection: Any,
    schema: EdgeSchema,
    *,
    alias: str = "e",
) -> str:
    """Return an edge_type non-null condition when the column exists."""

    columns = get_columns(connection, schema.edge_table)

    if schema.edge_type_column in columns:
        return f"AND {alias}.{qident(schema.edge_type_column)} IS NOT NULL"

    return ""


def _edge_weight_expr_for_edge_schema(
    connection: Any,
    schema: EdgeSchema,
    *,
    alias: str = "e",
    use_mention_count_weight: bool = True,
) -> str:
    """Return SQL expression for edge_weight."""

    columns = get_columns(connection, schema.edge_table)

    if (
        use_mention_count_weight
        and schema.weight_column is not None
        and schema.weight_column in columns
    ):
        return (
            "COALESCE("
            f"TRY_CAST({alias}.{qident(schema.weight_column)} AS DOUBLE), "
            f"{float(schema.default_weight)}"
            ")"
        )

    return str(float(schema.default_weight))


def _mention_count_expr_for_edge_schema(
    connection: Any,
    schema: EdgeSchema,
    *,
    alias: str = "e",
) -> str:
    """Return SQL expression for mention_count if available."""

    columns = get_columns(connection, schema.edge_table)

    if "mention_count" in columns:
        return f"TRY_CAST({alias}.{qident('mention_count')} AS BIGINT)"

    return "CAST(NULL AS BIGINT)"


def create_edge_union(
    connection: Any,
    *,
    output_table: str = "edge_all",
    replace: bool = True,
    use_mention_count_weight: bool = True,
    distinct_basic_edges: bool = False,
) -> None:
    """Create a unified edge table.

    Parameters
    ----------
    connection:
        DuckDB connection with edge views already loaded.
    output_table:
        Name of the unified edge table to create.
    replace:
        If true, create or replace the output table.
    use_mention_count_weight:
        If true, BioEntity edges with mention_count use it as edge_weight.
    distinct_basic_edges:
        If true, collapse exact duplicate rows by unified edge columns.
        For the first link prediction dataset, this can remain false because
        target-edge deduplication is handled by the splitting module.
    """

    union_parts = []

    for schema in EDGE_SCHEMAS.values():
        edge_type_expr = _edge_type_expr_for_edge_schema(connection, schema, alias="e")
        edge_type_not_null = _edge_type_not_null_condition(connection, schema, alias="e")
        edge_weight_expr = _edge_weight_expr_for_edge_schema(
            connection,
            schema,
            alias="e",
            use_mention_count_weight=use_mention_count_weight,
        )
        mention_count_expr = _mention_count_expr_for_edge_schema(connection, schema, alias="e")

        union_parts.append(
            f"""
            SELECT
                {sql_literal(schema.edge_table)} AS edge_table,
                {edge_type_expr} AS edge_type,
                {sql_literal(schema.source_type)} AS source_type,
                CAST(e.{qident(schema.source_column)} AS VARCHAR) AS source_id,
                {sql_literal(schema.target_type)} AS target_type,
                CAST(e.{qident(schema.target_column)} AS VARCHAR) AS target_id,
                {edge_weight_expr} AS edge_weight,
                {mention_count_expr} AS mention_count
            FROM {qident(schema.edge_table)} e
            WHERE e.{qident(schema.source_column)} IS NOT NULL
              AND e.{qident(schema.target_column)} IS NOT NULL
              {edge_type_not_null}
            """
        )

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"
    distinct_clause = "SELECT DISTINCT * FROM" if distinct_basic_edges else "SELECT * FROM"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        {distinct_clause} (
            {" UNION ALL ".join(union_parts)}
        )
        """
    )


def create_node_degree_table(
    connection: Any,
    *,
    edge_table: str = "edge_all",
    output_table: str = "node_degree",
    replace: bool = True,
) -> None:
    """Create a node degree table from a unified edge table.

    The resulting table contains both unweighted and weighted degrees.
    It is useful for isolated-node filtering and graph diagnostics.
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            node_type,
            node_id,
            SUM(out_degree) AS out_degree,
            SUM(in_degree) AS in_degree,
            SUM(total_degree) AS total_degree,
            SUM(weighted_out_degree) AS weighted_out_degree,
            SUM(weighted_in_degree) AS weighted_in_degree,
            SUM(weighted_total_degree) AS weighted_total_degree,
            COUNT(DISTINCT edge_table) AS incident_edge_table_count
        FROM (
            SELECT
                source_type AS node_type,
                source_id AS node_id,
                COUNT(*) AS out_degree,
                0 AS in_degree,
                COUNT(*) AS total_degree,
                SUM(edge_weight) AS weighted_out_degree,
                0.0 AS weighted_in_degree,
                SUM(edge_weight) AS weighted_total_degree,
                edge_table
            FROM {qident(edge_table)}
            GROUP BY source_type, source_id, edge_table

            UNION ALL

            SELECT
                target_type AS node_type,
                target_id AS node_id,
                0 AS out_degree,
                COUNT(*) AS in_degree,
                COUNT(*) AS total_degree,
                0.0 AS weighted_out_degree,
                SUM(edge_weight) AS weighted_in_degree,
                SUM(edge_weight) AS weighted_total_degree,
                edge_table
            FROM {qident(edge_table)}
            GROUP BY target_type, target_id, edge_table
        )
        GROUP BY node_type, node_id
        """
    )


def load_carrier_graph(
    connection: Any,
    graph_dir: str | Path,
    *,
    node_table: str = "node_all",
    edge_table: str = "edge_all",
    degree_table: str = "node_degree",
    require_files: bool = True,
    use_mention_count_weight: bool = True,
    distinct_basic_edges: bool = False,
    create_degree_table: bool = True,
) -> dict[str, int]:
    """Load a carrier graph directory and create unified tables.

    This is the main convenience function used by downstream scripts.

    Returns
    -------
    dict[str, int]
        Basic row counts for the created unified tables.
    """

    graph_dir = Path(graph_dir)

    create_graph_views(connection, graph_dir, require_files=require_files)
    create_node_union(connection, output_table=node_table)
    create_edge_union(
        connection,
        output_table=edge_table,
        use_mention_count_weight=use_mention_count_weight,
        distinct_basic_edges=distinct_basic_edges,
    )

    counts = {
        node_table: count_rows(connection, node_table),
        edge_table: count_rows(connection, edge_table),
    }

    if create_degree_table:
        create_node_degree_table(
            connection,
            edge_table=edge_table,
            output_table=degree_table,
        )
        counts[degree_table] = count_rows(connection, degree_table)

    return counts


def summarize_node_types(
    connection: Any,
    *,
    node_table: str = "node_all",
) -> list[dict[str, Any]]:
    """Return node counts by node type."""

    return fetch_dicts(
        connection,
        f"""
        SELECT
            node_type,
            COUNT(*) AS node_count
        FROM {qident(node_table)}
        GROUP BY node_type
        ORDER BY node_type
        """,
    )


def summarize_edge_tables(
    connection: Any,
    *,
    edge_table: str = "edge_all",
) -> list[dict[str, Any]]:
    """Return edge counts by edge table and edge type."""

    return fetch_dicts(
        connection,
        f"""
        SELECT
            edge_table,
            edge_type,
            source_type,
            target_type,
            COUNT(*) AS edge_count,
            COUNT(DISTINCT source_id) AS distinct_source_count,
            COUNT(DISTINCT target_id) AS distinct_target_count,
            SUM(edge_weight) AS total_edge_weight
        FROM {qident(edge_table)}
        GROUP BY edge_table, edge_type, source_type, target_type
        ORDER BY edge_table, edge_type
        """,
    )


def summarize_loaded_graph(
    connection: Any,
    *,
    node_table: str = "node_all",
    edge_table: str = "edge_all",
) -> dict[str, Any]:
    """Return a compact summary for a loaded unified graph."""

    node_type_counts = summarize_node_types(connection, node_table=node_table)
    edge_table_counts = summarize_edge_tables(connection, edge_table=edge_table)

    total_nodes = sum(int(row["node_count"]) for row in node_type_counts)
    total_edges = sum(int(row["edge_count"]) for row in edge_table_counts)

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_type_counts": node_type_counts,
        "edge_table_counts": edge_table_counts,
    }