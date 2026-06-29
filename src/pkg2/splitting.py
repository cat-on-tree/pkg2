"""Edge splitting utilities for link prediction datasets.

This module creates positive target-edge tables and train / validation / test
splits for carrier-level link prediction tasks.

The current primary use case is Patent-Paper link prediction, where the target
edge table is:

    edges_patent_paper

The functions are intentionally generic enough to support future carrier-level
or knowledge-unit-level link prediction tasks.
"""

from __future__ import annotations

from typing import Any

from .graph_schema import DEFAULT_TARGET_EDGE_TABLE, get_edge_schema
from .io import count_rows, qident, sql_literal


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    tolerance: float = 1e-8,
) -> None:
    """Validate train / validation / test split ratios."""

    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    for name, value in ratios.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > tolerance:
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Got train={train_ratio}, val={val_ratio}, test={test_ratio}, sum={total}"
        )

    if train_ratio <= 0:
        raise ValueError("train_ratio must be greater than 0")

    if val_ratio <= 0:
        raise ValueError("val_ratio must be greater than 0")

    if test_ratio <= 0:
        raise ValueError("test_ratio must be greater than 0")


def create_distinct_target_edges(
    connection: Any,
    *,
    source_edge_table: str = "filtered_edge_all",
    target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE,
    output_table: str = "target_positive_edges",
    replace: bool = True,
) -> int:
    """Create a deduplicated positive target-edge table.

    Parameters
    ----------
    connection:
        DuckDB connection.
    source_edge_table:
        Unified edge table to read from, usually ``filtered_edge_all``.
    target_edge_table:
        Edge table name to use as prediction target, for example
        ``edges_patent_paper``.
    output_table:
        Output table containing distinct positive target edges.
    replace:
        If true, create or replace the output table.

    Returns
    -------
    int
        Number of distinct positive target edges.
    """

    schema = get_edge_schema(target_edge_table)
    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT DISTINCT
            {sql_literal(schema.source_type)} AS source_type,
            source_id,
            {sql_literal(schema.target_type)} AS target_type,
            target_id,
            edge_type,
            {sql_literal(target_edge_table)} AS edge_table,
            1 AS label
        FROM {qident(source_edge_table)}
        WHERE edge_table = {sql_literal(target_edge_table)}
          AND source_type = {sql_literal(schema.source_type)}
          AND target_type = {sql_literal(schema.target_type)}
          AND source_id IS NOT NULL
          AND target_id IS NOT NULL
        """
    )

    return count_rows(connection, output_table)


def create_randomized_target_edges(
    connection: Any,
    *,
    source_table: str = "target_positive_edges",
    output_table: str = "target_positive_edges_randomized",
    seed: int = 42,
    replace: bool = True,
) -> int:
    """Create a randomized copy of target edges with row numbers.

    DuckDB's random() can be seeded with ``setseed``. The seed is converted to
    a deterministic floating point value in [0, 1).
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    seed_value = (abs(int(seed)) % 1_000_000) / 1_000_000.0
    connection.execute(f"SELECT setseed({seed_value})")

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY random(), source_id, target_id) AS row_number,
            COUNT(*) OVER () AS total_count,
            source_type,
            source_id,
            target_type,
            target_id,
            edge_type,
            edge_table,
            label
        FROM {qident(source_table)}
        """
    )

    return count_rows(connection, output_table)


def create_positive_edge_splits(
    connection: Any,
    *,
    randomized_table: str = "target_positive_edges_randomized",
    train_table: str = "positive_edges_train",
    val_table: str = "positive_edges_val",
    test_table: str = "positive_edges_test",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    replace: bool = True,
) -> dict[str, int]:
    """Create train / validation / test positive-edge split tables."""

    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    total_count = count_rows(connection, randomized_table)

    if total_count == 0:
        raise ValueError(f"No rows available for splitting in table: {randomized_table}")

    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError(
            "Split counts must all be positive. "
            f"Got train={train_count}, val={val_count}, test={test_count}, total={total_count}"
        )

    train_end = train_count
    val_end = train_count + val_count

    split_specs = [
        (train_table, "train", f"row_number <= {train_end}"),
        (val_table, "val", f"row_number > {train_end} AND row_number <= {val_end}"),
        (test_table, "test", f"row_number > {val_end}"),
    ]

    for table_name, split_name, condition in split_specs:
        connection.execute(
            f"""
            {create_clause} {qident(table_name)} AS
            SELECT
                source_type,
                source_id,
                target_type,
                target_id,
                edge_type,
                edge_table,
                {sql_literal(split_name)} AS split,
                1 AS label
            FROM {qident(randomized_table)}
            WHERE {condition}
            """
        )

    return {
        train_table: count_rows(connection, train_table),
        val_table: count_rows(connection, val_table),
        test_table: count_rows(connection, test_table),
    }


def create_target_edge_splits(
    connection: Any,
    *,
    source_edge_table: str = "filtered_edge_all",
    target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE,
    target_positive_table: str = "target_positive_edges",
    randomized_table: str = "target_positive_edges_randomized",
    train_table: str = "positive_edges_train",
    val_table: str = "positive_edges_val",
    test_table: str = "positive_edges_test",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Create deduplicated positive target edges and split them.

    This is the main convenience function used by link prediction dataset
    construction.
    """

    positive_count = create_distinct_target_edges(
        connection,
        source_edge_table=source_edge_table,
        target_edge_table=target_edge_table,
        output_table=target_positive_table,
    )

    randomized_count = create_randomized_target_edges(
        connection,
        source_table=target_positive_table,
        output_table=randomized_table,
        seed=seed,
    )

    split_counts = create_positive_edge_splits(
        connection,
        randomized_table=randomized_table,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    result = {
        target_positive_table: positive_count,
        randomized_table: randomized_count,
    }
    result.update(split_counts)

    return result


def summarize_positive_splits(
    connection: Any,
    *,
    train_table: str = "positive_edges_train",
    val_table: str = "positive_edges_val",
    test_table: str = "positive_edges_test",
) -> list[dict[str, Any]]:
    """Return positive edge counts by split."""

    rows = []

    for split_name, table_name in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        rows.append(
            {
                "split": split_name,
                "table_name": table_name,
                "positive_edge_count": count_rows(connection, table_name),
            }
        )

    return rows