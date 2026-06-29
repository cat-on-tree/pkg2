"""Negative sampling utilities for bipartite link prediction datasets.

The current primary use case is Patent-Paper link prediction.

Negative samples are generated from the Cartesian product of source-type nodes
and target-type nodes, excluding all known positive target edges.

Important leakage rule:
Negative samples must exclude all known positives, not only positives from the
same split. For example, train negatives must not include validation or test
positive pairs.
"""

from __future__ import annotations

from typing import Any

from .graph_schema import DEFAULT_TARGET_EDGE_TABLE, get_edge_schema
from .io import count_rows, qident, sql_literal


def count_bipartite_candidate_space(
    connection: Any,
    *,
    node_table: str = "filtered_node_all",
    source_node_type: str,
    target_node_type: str,
) -> dict[str, int]:
    """Count source nodes, target nodes, and Cartesian candidate pairs."""

    row = connection.execute(
        f"""
        SELECT
            SUM(CASE WHEN node_type = {sql_literal(source_node_type)} THEN 1 ELSE 0 END) AS source_node_count,
            SUM(CASE WHEN node_type = {sql_literal(target_node_type)} THEN 1 ELSE 0 END) AS target_node_count
        FROM {qident(node_table)}
        """
    ).fetchone()

    source_count = int(row[0] or 0)
    target_count = int(row[1] or 0)

    return {
        "source_node_count": source_count,
        "target_node_count": target_count,
        "candidate_pair_count": source_count * target_count,
    }


def create_all_positive_pair_table(
    connection: Any,
    *,
    positive_edge_table: str = "target_positive_edges",
    output_table: str = "all_positive_pairs",
    replace: bool = True,
) -> int:
    """Create a compact table of all known positive pairs.

    The resulting table is used to exclude positives during negative sampling.
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT DISTINCT
            source_id,
            target_id
        FROM {qident(positive_edge_table)}
        WHERE source_id IS NOT NULL
          AND target_id IS NOT NULL
        """
    )

    return count_rows(connection, output_table)


def create_negative_candidate_table(
    connection: Any,
    *,
    node_table: str = "filtered_node_all",
    all_positive_pair_table: str = "all_positive_pairs",
    source_node_type: str,
    target_node_type: str,
    output_table: str = "negative_candidate_pairs",
    replace: bool = True,
) -> int:
    """Create all possible negative candidate pairs.

    A candidate pair is any source-type node paired with any target-type node
    that is not present in all known positive target edges.
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            s.node_id AS source_id,
            t.node_id AS target_id
        FROM (
            SELECT node_id
            FROM {qident(node_table)}
            WHERE node_type = {sql_literal(source_node_type)}
        ) s
        CROSS JOIN (
            SELECT node_id
            FROM {qident(node_table)}
            WHERE node_type = {sql_literal(target_node_type)}
        ) t
        LEFT JOIN {qident(all_positive_pair_table)} p
            ON s.node_id = p.source_id
           AND t.node_id = p.target_id
        WHERE p.source_id IS NULL
        """
    )

    return count_rows(connection, output_table)


def create_randomized_negative_candidates(
    connection: Any,
    *,
    candidate_table: str = "negative_candidate_pairs",
    output_table: str = "negative_candidate_pairs_randomized",
    seed: int = 42,
    replace: bool = True,
) -> int:
    """Create randomized negative candidates with row numbers."""

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    seed_value = (abs(int(seed)) % 1_000_000) / 1_000_000.0
    connection.execute(f"SELECT setseed({seed_value})")

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY random(), source_id, target_id) AS row_number,
            source_id,
            target_id
        FROM {qident(candidate_table)}
        """
    )

    return count_rows(connection, output_table)


def sample_negative_edges_for_split(
    connection: Any,
    *,
    randomized_candidate_table: str = "negative_candidate_pairs_randomized",
    positive_split_table: str,
    output_table: str,
    source_node_type: str,
    target_node_type: str,
    target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE,
    negative_ratio: float = 1.0,
    split_name: str,
    row_offset: int = 0,
    replace: bool = True,
) -> dict[str, int]:
    """Sample negative edges for one split.

    Parameters
    ----------
    randomized_candidate_table:
        Randomized table of all negative candidates.
    positive_split_table:
        Positive split table used to determine how many negatives to sample.
    output_table:
        Output negative edge table.
    source_node_type:
        Source node type, for example Patent.
    target_node_type:
        Target node type, for example Paper.
    target_edge_table:
        Target edge table name, for example edges_patent_paper.
    negative_ratio:
        Number of negatives per positive.
    split_name:
        Split name: train, val, or test.
    row_offset:
        Offset into the randomized negative candidate table. This prevents
        train / validation / test negatives from overlapping.
    replace:
        If true, create or replace the output table.

    Returns
    -------
    dict[str, int]
        Sample counts and row range information.
    """

    if negative_ratio <= 0:
        raise ValueError(f"negative_ratio must be greater than 0, got {negative_ratio}")

    positive_count = count_rows(connection, positive_split_table)
    requested_negative_count = int(round(positive_count * negative_ratio))

    if requested_negative_count <= 0:
        raise ValueError(
            "Requested negative count must be positive. "
            f"positive_count={positive_count}, negative_ratio={negative_ratio}"
        )

    available_count = count_rows(connection, randomized_candidate_table)

    start_row = row_offset + 1
    end_row = row_offset + requested_negative_count

    if end_row > available_count:
        raise ValueError(
            "Not enough negative candidates. "
            f"requested end_row={end_row}, available={available_count}"
        )

    schema = get_edge_schema(target_edge_table)
    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            {sql_literal(source_node_type)} AS source_type,
            source_id,
            {sql_literal(target_node_type)} AS target_type,
            target_id,
            {sql_literal(schema.edge_table)} AS edge_table,
            {sql_literal(schema.edge_table + '_negative')} AS edge_type,
            {sql_literal(split_name)} AS split,
            0 AS label
        FROM {qident(randomized_candidate_table)}
        WHERE row_number BETWEEN {start_row} AND {end_row}
        """
    )

    sampled_count = count_rows(connection, output_table)

    return {
        "positive_count": positive_count,
        "requested_negative_count": requested_negative_count,
        "sampled_negative_count": sampled_count,
        "row_offset": row_offset,
        "start_row": start_row,
        "end_row": end_row,
    }


def sample_negative_edges_for_splits(
    connection: Any,
    *,
    node_table: str = "filtered_node_all",
    target_positive_table: str = "target_positive_edges",
    train_positive_table: str = "positive_edges_train",
    val_positive_table: str = "positive_edges_val",
    test_positive_table: str = "positive_edges_test",
    train_negative_table: str = "negative_edges_train",
    val_negative_table: str = "negative_edges_val",
    test_negative_table: str = "negative_edges_test",
    all_positive_pair_table: str = "all_positive_pairs",
    candidate_table: str = "negative_candidate_pairs",
    randomized_candidate_table: str = "negative_candidate_pairs_randomized",
    target_edge_table: str = DEFAULT_TARGET_EDGE_TABLE,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Sample train / validation / test negative edges.

    This is the main convenience function used by link prediction dataset
    construction.
    """

    schema = get_edge_schema(target_edge_table)
    source_node_type = schema.source_type
    target_node_type = schema.target_type

    candidate_space_counts = count_bipartite_candidate_space(
        connection,
        node_table=node_table,
        source_node_type=source_node_type,
        target_node_type=target_node_type,
    )

    all_positive_count = create_all_positive_pair_table(
        connection,
        positive_edge_table=target_positive_table,
        output_table=all_positive_pair_table,
    )

    negative_candidate_count = create_negative_candidate_table(
        connection,
        node_table=node_table,
        all_positive_pair_table=all_positive_pair_table,
        source_node_type=source_node_type,
        target_node_type=target_node_type,
        output_table=candidate_table,
    )

    randomized_candidate_count = create_randomized_negative_candidates(
        connection,
        candidate_table=candidate_table,
        output_table=randomized_candidate_table,
        seed=seed,
    )

    split_results = {}

    offset = 0

    train_result = sample_negative_edges_for_split(
        connection,
        randomized_candidate_table=randomized_candidate_table,
        positive_split_table=train_positive_table,
        output_table=train_negative_table,
        source_node_type=source_node_type,
        target_node_type=target_node_type,
        target_edge_table=target_edge_table,
        negative_ratio=negative_ratio,
        split_name="train",
        row_offset=offset,
    )
    split_results["train"] = train_result
    offset = train_result["end_row"]

    val_result = sample_negative_edges_for_split(
        connection,
        randomized_candidate_table=randomized_candidate_table,
        positive_split_table=val_positive_table,
        output_table=val_negative_table,
        source_node_type=source_node_type,
        target_node_type=target_node_type,
        target_edge_table=target_edge_table,
        negative_ratio=negative_ratio,
        split_name="val",
        row_offset=offset,
    )
    split_results["val"] = val_result
    offset = val_result["end_row"]

    test_result = sample_negative_edges_for_split(
        connection,
        randomized_candidate_table=randomized_candidate_table,
        positive_split_table=test_positive_table,
        output_table=test_negative_table,
        source_node_type=source_node_type,
        target_node_type=target_node_type,
        target_edge_table=target_edge_table,
        negative_ratio=negative_ratio,
        split_name="test",
        row_offset=offset,
    )
    split_results["test"] = test_result

    return {
        "target_edge_table": target_edge_table,
        "source_node_type": source_node_type,
        "target_node_type": target_node_type,
        "negative_ratio": negative_ratio,
        "candidate_space": candidate_space_counts,
        "all_positive_pair_count": all_positive_count,
        "negative_candidate_count": negative_candidate_count,
        "randomized_candidate_count": randomized_candidate_count,
        "split_results": split_results,
        "negative_split_counts": {
            train_negative_table: count_rows(connection, train_negative_table),
            val_negative_table: count_rows(connection, val_negative_table),
            test_negative_table: count_rows(connection, test_negative_table),
        },
    }


def summarize_negative_splits(
    connection: Any,
    *,
    train_table: str = "negative_edges_train",
    val_table: str = "negative_edges_val",
    test_table: str = "negative_edges_test",
) -> list[dict[str, Any]]:
    """Return negative edge counts by split."""

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
                "negative_edge_count": count_rows(connection, table_name),
            }
        )

    return rows