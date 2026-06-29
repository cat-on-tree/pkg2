"""Graph filtering utilities for algorithm-stage datasets.

This module filters unified carrier graph tables produced by
``pkg2.graph_loading``.

The current primary use case is constructing carrier-level link prediction
datasets from the diabetes Knowledge Carrier Graph prototype.

Main responsibilities:
- Filter isolated nodes.
- Optionally exclude BioEntity nodes by type, such as species.
- Optionally exclude specific BioEntity node IDs, such as major species hubs.
- Keep only edges whose source and target nodes are both retained.
- Recompute node degrees after filtering.
- Return compact filtering summaries for dataset reports.

Default input tables:
- node_all
- edge_all
- node_degree

Default output tables:
- filtered_node_all
- filtered_edge_all
- filtered_node_degree
"""

from __future__ import annotations

from typing import Any, Iterable

from .graph_loading import create_node_degree_table
from .io import count_rows, fetch_dicts, qident, sql_literal


def _string_list_sql(values: Iterable[str]) -> str:
    """Return a SQL tuple list for string values.

    Example:
        ["species", "gene"] -> ('species', 'gene')
    """

    values = [str(value).strip() for value in values if str(value).strip()]

    if not values:
        raise ValueError("values must contain at least one non-empty string")

    return "(" + ", ".join(sql_literal(value) for value in values) + ")"


def create_filtered_nodes(
    connection: Any,
    *,
    source_node_table: str = "node_all",
    source_degree_table: str = "node_degree",
    output_table: str = "filtered_node_all",
    filter_isolated_nodes: bool = True,
    exclude_bioentity_types: Iterable[str] | None = None,
    exclude_bioentity_ids: Iterable[str] | None = None,
    replace: bool = True,
) -> None:
    """Create a filtered node table.

    Filtering is applied in this order:
    1. Optionally remove isolated nodes.
    2. Optionally remove BioEntity nodes whose bioentity_type is excluded.
    3. Optionally remove BioEntity nodes whose node_id is excluded.

    Parameters
    ----------
    connection:
        DuckDB connection.
    source_node_table:
        Unified node table, usually ``node_all``.
    source_degree_table:
        Degree table created from the unfiltered graph, usually ``node_degree``.
    output_table:
        Name of the filtered node table to create.
    filter_isolated_nodes:
        If true, keep only nodes that appear in the degree table.
    exclude_bioentity_types:
        Optional BioEntity types to exclude, for example ``["species"]``.
    exclude_bioentity_ids:
        Optional BioEntity node IDs to exclude.
    replace:
        If true, create or replace the output table.
    """

    where_conditions = ["1 = 1"]

    if filter_isolated_nodes:
        where_conditions.append("d.node_id IS NOT NULL")

    exclude_bioentity_types = [
        str(value).strip()
        for value in (exclude_bioentity_types or [])
        if str(value).strip()
    ]

    if exclude_bioentity_types:
        type_list = _string_list_sql(exclude_bioentity_types)
        where_conditions.append(
            f"""
            NOT (
                n.node_type = 'BioEntity'
                AND COALESCE(n.bioentity_type, '') IN {type_list}
            )
            """
        )

    exclude_bioentity_ids = [
        str(value).strip()
        for value in (exclude_bioentity_ids or [])
        if str(value).strip()
    ]

    if exclude_bioentity_ids:
        id_list = _string_list_sql(exclude_bioentity_ids)
        where_conditions.append(
            f"""
            NOT (
                n.node_type = 'BioEntity'
                AND n.node_id IN {id_list}
            )
            """
        )

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"
    where_sql = "\n          AND ".join(where_conditions)

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            n.node_type,
            n.node_id,
            n.label,
            n.bioentity_type
        FROM {qident(source_node_table)} n
        LEFT JOIN {qident(source_degree_table)} d
            ON n.node_type = d.node_type
           AND n.node_id = d.node_id
        WHERE {where_sql}
        """
    )


def create_filtered_edges(
    connection: Any,
    *,
    source_edge_table: str = "edge_all",
    filtered_node_table: str = "filtered_node_all",
    output_table: str = "filtered_edge_all",
    replace: bool = True,
) -> None:
    """Create a filtered edge table.

    Only edges whose source and target nodes are both present in
    ``filtered_node_table`` are retained.
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            e.edge_table,
            e.edge_type,
            e.source_type,
            e.source_id,
            e.target_type,
            e.target_id,
            e.edge_weight,
            e.mention_count
        FROM {qident(source_edge_table)} e
        INNER JOIN {qident(filtered_node_table)} s
            ON e.source_type = s.node_type
           AND e.source_id = s.node_id
        INNER JOIN {qident(filtered_node_table)} t
            ON e.target_type = t.node_type
           AND e.target_id = t.node_id
        """
    )


def create_filtered_graph(
    connection: Any,
    *,
    source_node_table: str = "node_all",
    source_edge_table: str = "edge_all",
    source_degree_table: str = "node_degree",
    filtered_node_table: str = "filtered_node_all",
    filtered_edge_table: str = "filtered_edge_all",
    filtered_degree_table: str = "filtered_node_degree",
    filter_isolated_nodes: bool = True,
    exclude_bioentity_types: Iterable[str] | None = None,
    exclude_bioentity_ids: Iterable[str] | None = None,
) -> dict[str, int]:
    """Create filtered node, edge, and degree tables.

    Returns
    -------
    dict[str, int]
        Row counts for filtered tables.
    """

    create_filtered_nodes(
        connection,
        source_node_table=source_node_table,
        source_degree_table=source_degree_table,
        output_table=filtered_node_table,
        filter_isolated_nodes=filter_isolated_nodes,
        exclude_bioentity_types=exclude_bioentity_types,
        exclude_bioentity_ids=exclude_bioentity_ids,
    )

    create_filtered_edges(
        connection,
        source_edge_table=source_edge_table,
        filtered_node_table=filtered_node_table,
        output_table=filtered_edge_table,
    )

    create_node_degree_table(
        connection,
        edge_table=filtered_edge_table,
        output_table=filtered_degree_table,
    )

    return {
        filtered_node_table: count_rows(connection, filtered_node_table),
        filtered_edge_table: count_rows(connection, filtered_edge_table),
        filtered_degree_table: count_rows(connection, filtered_degree_table),
    }


def summarize_filtering(
    connection: Any,
    *,
    source_node_table: str = "node_all",
    source_edge_table: str = "edge_all",
    filtered_node_table: str = "filtered_node_all",
    filtered_edge_table: str = "filtered_edge_all",
) -> dict[str, Any]:
    """Return before/after counts for graph filtering."""

    source_node_count = count_rows(connection, source_node_table)
    source_edge_count = count_rows(connection, source_edge_table)
    filtered_node_count = count_rows(connection, filtered_node_table)
    filtered_edge_count = count_rows(connection, filtered_edge_table)

    return {
        "source_node_count": source_node_count,
        "source_edge_count": source_edge_count,
        "filtered_node_count": filtered_node_count,
        "filtered_edge_count": filtered_edge_count,
        "removed_node_count": source_node_count - filtered_node_count,
        "removed_edge_count": source_edge_count - filtered_edge_count,
        "retained_node_ratio": filtered_node_count / source_node_count if source_node_count else 0.0,
        "retained_edge_ratio": filtered_edge_count / source_edge_count if source_edge_count else 0.0,
    }


def summarize_filtered_node_types(
    connection: Any,
    *,
    source_node_table: str = "node_all",
    filtered_node_table: str = "filtered_node_all",
) -> list[dict[str, Any]]:
    """Return before/after node counts by node type."""

    return fetch_dicts(
        connection,
        f"""
        WITH source_counts AS (
            SELECT
                node_type,
                COUNT(*) AS source_node_count
            FROM {qident(source_node_table)}
            GROUP BY node_type
        ),
        filtered_counts AS (
            SELECT
                node_type,
                COUNT(*) AS filtered_node_count
            FROM {qident(filtered_node_table)}
            GROUP BY node_type
        )
        SELECT
            COALESCE(s.node_type, f.node_type) AS node_type,
            COALESCE(s.source_node_count, 0) AS source_node_count,
            COALESCE(f.filtered_node_count, 0) AS filtered_node_count,
            COALESCE(s.source_node_count, 0) - COALESCE(f.filtered_node_count, 0) AS removed_node_count
        FROM source_counts s
        FULL OUTER JOIN filtered_counts f
            ON s.node_type = f.node_type
        ORDER BY node_type
        """,
    )


def summarize_filtered_edge_tables(
    connection: Any,
    *,
    source_edge_table: str = "edge_all",
    filtered_edge_table: str = "filtered_edge_all",
) -> list[dict[str, Any]]:
    """Return before/after edge counts by edge table."""

    return fetch_dicts(
        connection,
        f"""
        WITH source_counts AS (
            SELECT
                edge_table,
                edge_type,
                COUNT(*) AS source_edge_count
            FROM {qident(source_edge_table)}
            GROUP BY edge_table, edge_type
        ),
        filtered_counts AS (
            SELECT
                edge_table,
                edge_type,
                COUNT(*) AS filtered_edge_count
            FROM {qident(filtered_edge_table)}
            GROUP BY edge_table, edge_type
        )
        SELECT
            COALESCE(s.edge_table, f.edge_table) AS edge_table,
            COALESCE(s.edge_type, f.edge_type) AS edge_type,
            COALESCE(s.source_edge_count, 0) AS source_edge_count,
            COALESCE(f.filtered_edge_count, 0) AS filtered_edge_count,
            COALESCE(s.source_edge_count, 0) - COALESCE(f.filtered_edge_count, 0) AS removed_edge_count
        FROM source_counts s
        FULL OUTER JOIN filtered_counts f
            ON s.edge_table = f.edge_table
           AND s.edge_type = f.edge_type
        ORDER BY edge_table, edge_type
        """,
    )


def summarize_removed_bioentity_types(
    connection: Any,
    *,
    source_node_table: str = "node_all",
    filtered_node_table: str = "filtered_node_all",
) -> list[dict[str, Any]]:
    """Return removed BioEntity counts by BioEntity type."""

    return fetch_dicts(
        connection,
        f"""
        SELECT
            COALESCE(s.bioentity_type, '') AS bioentity_type,
            COUNT(*) AS removed_node_count
        FROM {qident(source_node_table)} s
        LEFT JOIN {qident(filtered_node_table)} f
            ON s.node_type = f.node_type
           AND s.node_id = f.node_id
        WHERE s.node_type = 'BioEntity'
          AND f.node_id IS NULL
        GROUP BY COALESCE(s.bioentity_type, '')
        ORDER BY removed_node_count DESC, bioentity_type
        """,
    )


def summarize_isolated_nodes(
    connection: Any,
    *,
    source_node_table: str = "node_all",
    source_degree_table: str = "node_degree",
) -> list[dict[str, Any]]:
    """Return isolated node counts by node type in the source graph."""

    return fetch_dicts(
        connection,
        f"""
        SELECT
            n.node_type,
            COUNT(*) AS isolated_node_count
        FROM {qident(source_node_table)} n
        LEFT JOIN {qident(source_degree_table)} d
            ON n.node_type = d.node_type
           AND n.node_id = d.node_id
        WHERE d.node_id IS NULL
        GROUP BY n.node_type
        ORDER BY n.node_type
        """,
    )