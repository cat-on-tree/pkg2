"""Baseline feature builders and scoring utilities for link prediction.

This module contains reusable baseline logic for carrier-level link prediction
datasets produced by ``pkg2.link_prediction``.

The first supported baseline is BioEntity overlap for Patent-Paper link
prediction.

Core idea
---------
For a candidate Patent-Paper pair, compute how much overlap exists between:

- BioEntity nodes mentioned by the patent
- BioEntity nodes mentioned by the paper

The resulting features can be used directly as heuristic scores or as input
features for later machine-learning baselines such as logistic regression,
random forest, or XGBoost.

Expected dataset directory
--------------------------
The dataset directory should contain at least:

- nodes.parquet
- context_edges.parquet
- labeled_edges_train.parquet
- labeled_edges_val.parquet
- labeled_edges_test.parquet

The context graph should already exclude the target edge table, for example
``edges_patent_paper``, to avoid target-edge leakage.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .io import (
    count_rows,
    fetch_dicts,
    qident,
    require_paths_exist,
    scan_sql,
    sql_literal,
)


DEFAULT_SPLITS = ("train", "val", "test")

BIOENTITY_SCORE_COLUMNS = {
    "shared_bioentity_count",
    "bioentity_jaccard",
    "weighted_shared_bioentity_min",
    "weighted_bioentity_jaccard",
    "weighted_bioentity_cosine",
    "source_bioentity_count",
    "target_bioentity_count",
    "source_context_degree",
    "target_context_degree",
    "source_target_degree_product",
}


def _validate_identifier(identifier: str) -> None:
    """Validate a simple SQL identifier used as a table or column name."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")


def _string_list_sql(values: Iterable[str]) -> str:
    """Return a SQL tuple list for string values."""

    cleaned = [str(value).strip() for value in values if str(value).strip()]

    if not cleaned:
        raise ValueError("values must contain at least one non-empty string")

    return "(" + ", ".join(sql_literal(value) for value in cleaned) + ")"


def labeled_edge_table_name(split: str) -> str:
    """Return the standard labeled edge table name for a split."""

    split = str(split).strip()
    _validate_identifier(split)
    return f"labeled_edges_{split}"


def prediction_table_name(split: str, *, prefix: str = "bioentity_overlap_predictions") -> str:
    """Return the standard prediction table name for a split."""

    split = str(split).strip()
    prefix = str(prefix).strip()
    _validate_identifier(split)
    _validate_identifier(prefix)
    return f"{prefix}_{split}"


def feature_table_name(split: str, *, prefix: str = "bioentity_overlap_features") -> str:
    """Return the standard feature table name for a split."""

    split = str(split).strip()
    prefix = str(prefix).strip()
    _validate_identifier(split)
    _validate_identifier(prefix)
    return f"{prefix}_{split}"


def require_link_prediction_dataset_files(
    dataset_dir: str | Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> None:
    """Validate that a link prediction dataset directory has required files."""

    dataset_dir = Path(dataset_dir)

    required_paths = [
        dataset_dir / "nodes.parquet",
        dataset_dir / "context_edges.parquet",
    ]

    for split in splits:
        required_paths.append(dataset_dir / f"labeled_edges_{split}.parquet")

    require_paths_exist(required_paths)


def load_link_prediction_dataset_views(
    connection: Any,
    dataset_dir: str | Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    require_files: bool = True,
) -> None:
    """Load a link prediction dataset directory as DuckDB temporary views.

    Created views:

    - nodes
    - context_edges
    - labeled_edges_train
    - labeled_edges_val
    - labeled_edges_test
    """

    dataset_dir = Path(dataset_dir)
    split_list = [str(split).strip() for split in splits if str(split).strip()]

    if require_files:
        require_link_prediction_dataset_files(dataset_dir, splits=split_list)

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW nodes AS
        SELECT *
        FROM {scan_sql(dataset_dir / "nodes.parquet")}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW context_edges AS
        SELECT *
        FROM {scan_sql(dataset_dir / "context_edges.parquet")}
        """
    )

    for split in split_list:
        table_name = labeled_edge_table_name(split)
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(table_name)} AS
            SELECT *
            FROM {scan_sql(dataset_dir / f"{table_name}.parquet")}
            """
        )


def create_context_degree_table(
    connection: Any,
    *,
    context_edge_table: str = "context_edges",
    output_table: str = "context_node_degree",
    replace: bool = True,
) -> int:
    """Create node degree features from context edges.

    The target edge table should already be excluded from context_edges.

    Output columns:

    - node_type
    - node_id
    - context_out_degree
    - context_in_degree
    - context_total_degree
    - weighted_context_out_degree
    - weighted_context_in_degree
    - weighted_context_total_degree
    """

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            node_type,
            node_id,
            SUM(context_out_degree) AS context_out_degree,
            SUM(context_in_degree) AS context_in_degree,
            SUM(context_total_degree) AS context_total_degree,
            SUM(weighted_context_out_degree) AS weighted_context_out_degree,
            SUM(weighted_context_in_degree) AS weighted_context_in_degree,
            SUM(weighted_context_total_degree) AS weighted_context_total_degree
        FROM (
            SELECT
                source_type AS node_type,
                source_id AS node_id,
                COUNT(*) AS context_out_degree,
                0 AS context_in_degree,
                COUNT(*) AS context_total_degree,
                SUM(COALESCE(edge_weight, 1.0)) AS weighted_context_out_degree,
                0.0 AS weighted_context_in_degree,
                SUM(COALESCE(edge_weight, 1.0)) AS weighted_context_total_degree
            FROM {qident(context_edge_table)}
            WHERE source_id IS NOT NULL
            GROUP BY source_type, source_id

            UNION ALL

            SELECT
                target_type AS node_type,
                target_id AS node_id,
                0 AS context_out_degree,
                COUNT(*) AS context_in_degree,
                COUNT(*) AS context_total_degree,
                0.0 AS weighted_context_out_degree,
                SUM(COALESCE(edge_weight, 1.0)) AS weighted_context_in_degree,
                SUM(COALESCE(edge_weight, 1.0)) AS weighted_context_total_degree
            FROM {qident(context_edge_table)}
            WHERE target_id IS NOT NULL
            GROUP BY target_type, target_id
        )
        GROUP BY node_type, node_id
        """
    )

    return count_rows(connection, output_table)


def create_bioentity_incidence_table(
    connection: Any,
    *,
    context_edge_table: str = "context_edges",
    node_table: str = "nodes",
    output_table: str = "carrier_bioentity_incidence",
    carrier_node_types: Iterable[str] = ("Patent", "Paper"),
    exclude_bioentity_types: Iterable[str] | None = None,
    replace: bool = True,
) -> int:
    """Create a carrier-to-BioEntity incidence table.

    This table represents which BioEntity nodes are connected to each carrier
    node through context edges.

    Output columns:

    - carrier_type
    - carrier_id
    - bioentity_id
    - bioentity_type
    - bioentity_weight
    - mention_edge_count
    """

    carrier_node_types = [
        str(value).strip()
        for value in carrier_node_types
        if str(value).strip()
    ]

    if not carrier_node_types:
        raise ValueError("carrier_node_types must not be empty")

    carrier_type_sql = _string_list_sql(carrier_node_types)

    exclude_bioentity_types = [
        str(value).strip()
        for value in (exclude_bioentity_types or [])
        if str(value).strip()
    ]

    exclude_type_condition = ""
    if exclude_bioentity_types:
        exclude_type_condition = (
            "AND COALESCE(b.bioentity_type, '') NOT IN "
            + _string_list_sql(exclude_bioentity_types)
        )

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        WITH raw_incidence AS (
            SELECT
                source_type AS carrier_type,
                source_id AS carrier_id,
                target_id AS bioentity_id,
                COALESCE(edge_weight, 1.0) AS edge_weight
            FROM {qident(context_edge_table)}
            WHERE source_type IN {carrier_type_sql}
              AND target_type = 'BioEntity'
              AND source_id IS NOT NULL
              AND target_id IS NOT NULL

            UNION ALL

            SELECT
                target_type AS carrier_type,
                target_id AS carrier_id,
                source_id AS bioentity_id,
                COALESCE(edge_weight, 1.0) AS edge_weight
            FROM {qident(context_edge_table)}
            WHERE target_type IN {carrier_type_sql}
              AND source_type = 'BioEntity'
              AND target_id IS NOT NULL
              AND source_id IS NOT NULL
        )
        SELECT
            r.carrier_type,
            r.carrier_id,
            r.bioentity_id,
            b.bioentity_type,
            SUM(r.edge_weight) AS bioentity_weight,
            COUNT(*) AS mention_edge_count
        FROM raw_incidence r
        LEFT JOIN {qident(node_table)} b
            ON b.node_type = 'BioEntity'
           AND b.node_id = r.bioentity_id
        WHERE 1 = 1
          {exclude_type_condition}
        GROUP BY
            r.carrier_type,
            r.carrier_id,
            r.bioentity_id,
            b.bioentity_type
        """
    )

    return count_rows(connection, output_table)


def create_bioentity_count_table(
    connection: Any,
    *,
    incidence_table: str = "carrier_bioentity_incidence",
    output_table: str = "carrier_bioentity_counts",
    replace: bool = True,
) -> int:
    """Create per-carrier BioEntity count and weight summaries."""

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            carrier_type,
            carrier_id,
            COUNT(*) AS bioentity_count,
            SUM(bioentity_weight) AS bioentity_weight_sum,
            SUM(bioentity_weight * bioentity_weight) AS bioentity_weight_square_sum,
            SQRT(SUM(bioentity_weight * bioentity_weight)) AS bioentity_weight_l2
        FROM {qident(incidence_table)}
        GROUP BY carrier_type, carrier_id
        """
    )

    return count_rows(connection, output_table)


def create_bioentity_overlap_feature_table(
    connection: Any,
    *,
    labeled_edge_table: str,
    output_table: str,
    incidence_table: str = "carrier_bioentity_incidence",
    count_table: str = "carrier_bioentity_counts",
    degree_table: str = "context_node_degree",
    score_column: str = "shared_bioentity_count",
    replace: bool = True,
) -> int:
    """Create BioEntity overlap features for one labeled edge table.

    The output table contains original labeled examples plus feature columns
    and a generic ``score`` column.

    Supported score columns:

    - shared_bioentity_count
    - bioentity_jaccard
    - weighted_shared_bioentity_min
    - weighted_bioentity_jaccard
    - weighted_bioentity_cosine
    - source_bioentity_count
    - target_bioentity_count
    - source_context_degree
    - target_context_degree
    - source_target_degree_product
    """

    if score_column not in BIOENTITY_SCORE_COLUMNS:
        raise ValueError(
            f"Unsupported score_column: {score_column!r}. "
            f"Supported values: {sorted(BIOENTITY_SCORE_COLUMNS)}"
        )

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            *,
            CAST({qident(score_column)} AS DOUBLE) AS score
        FROM (
            WITH shared AS (
                SELECT
                    l.source_type,
                    l.source_id,
                    l.target_type,
                    l.target_id,
                    COUNT(*) AS shared_bioentity_count,
                    SUM(LEAST(p.bioentity_weight, t.bioentity_weight)) AS weighted_shared_bioentity_min,
                    SUM(p.bioentity_weight * t.bioentity_weight) AS weighted_shared_bioentity_product
                FROM {qident(labeled_edge_table)} l
                INNER JOIN {qident(incidence_table)} p
                    ON p.carrier_type = l.source_type
                   AND p.carrier_id = l.source_id
                INNER JOIN {qident(incidence_table)} t
                    ON t.carrier_type = l.target_type
                   AND t.carrier_id = l.target_id
                   AND t.bioentity_id = p.bioentity_id
                GROUP BY
                    l.source_type,
                    l.source_id,
                    l.target_type,
                    l.target_id
            )
            SELECT
                l.source_index,
                l.target_index,
                l.source_type,
                l.source_id,
                l.target_type,
                l.target_id,
                l.edge_table,
                l.edge_type,
                l.split,
                l.label,

                COALESCE(pc.bioentity_count, 0) AS source_bioentity_count,
                COALESCE(tc.bioentity_count, 0) AS target_bioentity_count,
                COALESCE(s.shared_bioentity_count, 0) AS shared_bioentity_count,

                (
                    COALESCE(pc.bioentity_count, 0)
                    + COALESCE(tc.bioentity_count, 0)
                    - COALESCE(s.shared_bioentity_count, 0)
                ) AS bioentity_union_count,

                CASE
                    WHEN (
                        COALESCE(pc.bioentity_count, 0)
                        + COALESCE(tc.bioentity_count, 0)
                        - COALESCE(s.shared_bioentity_count, 0)
                    ) > 0
                    THEN
                        COALESCE(s.shared_bioentity_count, 0)::DOUBLE
                        / (
                            COALESCE(pc.bioentity_count, 0)
                            + COALESCE(tc.bioentity_count, 0)
                            - COALESCE(s.shared_bioentity_count, 0)
                        )::DOUBLE
                    ELSE 0.0
                END AS bioentity_jaccard,

                COALESCE(pc.bioentity_weight_sum, 0.0) AS source_bioentity_weight_sum,
                COALESCE(tc.bioentity_weight_sum, 0.0) AS target_bioentity_weight_sum,
                COALESCE(pc.bioentity_weight_l2, 0.0) AS source_bioentity_weight_l2,
                COALESCE(tc.bioentity_weight_l2, 0.0) AS target_bioentity_weight_l2,

                COALESCE(s.weighted_shared_bioentity_min, 0.0) AS weighted_shared_bioentity_min,
                COALESCE(s.weighted_shared_bioentity_product, 0.0) AS weighted_shared_bioentity_product,

                (
                    COALESCE(pc.bioentity_weight_sum, 0.0)
                    + COALESCE(tc.bioentity_weight_sum, 0.0)
                    - COALESCE(s.weighted_shared_bioentity_min, 0.0)
                ) AS weighted_bioentity_union_minmax,

                CASE
                    WHEN (
                        COALESCE(pc.bioentity_weight_sum, 0.0)
                        + COALESCE(tc.bioentity_weight_sum, 0.0)
                        - COALESCE(s.weighted_shared_bioentity_min, 0.0)
                    ) > 0
                    THEN
                        COALESCE(s.weighted_shared_bioentity_min, 0.0)
                        / (
                            COALESCE(pc.bioentity_weight_sum, 0.0)
                            + COALESCE(tc.bioentity_weight_sum, 0.0)
                            - COALESCE(s.weighted_shared_bioentity_min, 0.0)
                        )
                    ELSE 0.0
                END AS weighted_bioentity_jaccard,

                CASE
                    WHEN COALESCE(pc.bioentity_weight_l2, 0.0) > 0
                     AND COALESCE(tc.bioentity_weight_l2, 0.0) > 0
                    THEN
                        COALESCE(s.weighted_shared_bioentity_product, 0.0)
                        / (
                            COALESCE(pc.bioentity_weight_l2, 0.0)
                            * COALESCE(tc.bioentity_weight_l2, 0.0)
                        )
                    ELSE 0.0
                END AS weighted_bioentity_cosine,

                COALESCE(sd.context_total_degree, 0) AS source_context_degree,
                COALESCE(td.context_total_degree, 0) AS target_context_degree,
                COALESCE(sd.weighted_context_total_degree, 0.0) AS source_weighted_context_degree,
                COALESCE(td.weighted_context_total_degree, 0.0) AS target_weighted_context_degree,

                (
                    COALESCE(sd.context_total_degree, 0)
                    * COALESCE(td.context_total_degree, 0)
                ) AS source_target_degree_product

            FROM {qident(labeled_edge_table)} l

            LEFT JOIN shared s
                ON s.source_type = l.source_type
               AND s.source_id = l.source_id
               AND s.target_type = l.target_type
               AND s.target_id = l.target_id

            LEFT JOIN {qident(count_table)} pc
                ON pc.carrier_type = l.source_type
               AND pc.carrier_id = l.source_id

            LEFT JOIN {qident(count_table)} tc
                ON tc.carrier_type = l.target_type
               AND tc.carrier_id = l.target_id

            LEFT JOIN {qident(degree_table)} sd
                ON sd.node_type = l.source_type
               AND sd.node_id = l.source_id

            LEFT JOIN {qident(degree_table)} td
                ON td.node_type = l.target_type
               AND td.node_id = l.target_id
        )
        """
    )

    return count_rows(connection, output_table)


def create_bioentity_overlap_feature_tables(
    connection: Any,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    labeled_table_prefix: str = "labeled_edges",
    feature_table_prefix: str = "bioentity_overlap_features",
    incidence_table: str = "carrier_bioentity_incidence",
    count_table: str = "carrier_bioentity_counts",
    degree_table: str = "context_node_degree",
    score_column: str = "shared_bioentity_count",
) -> dict[str, int]:
    """Create BioEntity overlap feature tables for multiple splits."""

    counts = {}

    for split in splits:
        split = str(split).strip()
        labeled_table = f"{labeled_table_prefix}_{split}"
        output_table = feature_table_name(split, prefix=feature_table_prefix)

        counts[output_table] = create_bioentity_overlap_feature_table(
            connection,
            labeled_edge_table=labeled_table,
            output_table=output_table,
            incidence_table=incidence_table,
            count_table=count_table,
            degree_table=degree_table,
            score_column=score_column,
        )

    return counts


def create_prediction_table_from_features(
    connection: Any,
    *,
    feature_table: str,
    output_table: str,
    score_column: str = "score",
    replace: bool = True,
) -> int:
    """Create a compact prediction table from a feature table."""

    _validate_identifier(score_column)

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            source_index,
            target_index,
            source_type,
            source_id,
            target_type,
            target_id,
            split,
            label,
            CAST({qident(score_column)} AS DOUBLE) AS score
        FROM {qident(feature_table)}
        """
    )

    return count_rows(connection, output_table)


def create_prediction_tables_from_features(
    connection: Any,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    feature_table_prefix: str = "bioentity_overlap_features",
    prediction_table_prefix: str = "bioentity_overlap_predictions",
    score_column: str = "score",
) -> dict[str, int]:
    """Create compact prediction tables for multiple splits."""

    counts = {}

    for split in splits:
        split = str(split).strip()
        feature_table = feature_table_name(split, prefix=feature_table_prefix)
        output_table = prediction_table_name(split, prefix=prediction_table_prefix)

        counts[output_table] = create_prediction_table_from_features(
            connection,
            feature_table=feature_table,
            output_table=output_table,
            score_column=score_column,
        )

    return counts


def create_random_score_prediction_table(
    connection: Any,
    *,
    labeled_edge_table: str,
    output_table: str,
    seed: int = 42,
    replace: bool = True,
) -> int:
    """Create a random-score baseline prediction table for one split."""

    seed_value = (abs(int(seed)) % 1_000_000) / 1_000_000.0
    connection.execute(f"SELECT setseed({seed_value})")

    create_clause = "CREATE OR REPLACE TEMP TABLE" if replace else "CREATE TEMP TABLE"

    connection.execute(
        f"""
        {create_clause} {qident(output_table)} AS
        SELECT
            source_index,
            target_index,
            source_type,
            source_id,
            target_type,
            target_id,
            split,
            label,
            random() AS score
        FROM {qident(labeled_edge_table)}
        """
    )

    return count_rows(connection, output_table)


def create_random_score_prediction_tables(
    connection: Any,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    labeled_table_prefix: str = "labeled_edges",
    prediction_table_prefix: str = "random_score_predictions",
    seed: int = 42,
) -> dict[str, int]:
    """Create random-score prediction tables for multiple splits."""

    counts = {}

    for index, split in enumerate(splits):
        split = str(split).strip()
        labeled_table = f"{labeled_table_prefix}_{split}"
        output_table = prediction_table_name(split, prefix=prediction_table_prefix)

        counts[output_table] = create_random_score_prediction_table(
            connection,
            labeled_edge_table=labeled_table,
            output_table=output_table,
            seed=seed + index,
        )

    return counts


def build_bioentity_overlap_baseline_tables(
    connection: Any,
    *,
    dataset_dir: str | Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
    score_column: str = "shared_bioentity_count",
    exclude_bioentity_types: Iterable[str] | None = None,
    load_views: bool = True,
) -> dict[str, Any]:
    """Build all tables needed for the BioEntity overlap baseline.

    This is the main convenience function used by the BioEntity overlap CLI
    script.

    Created tables include:

    - context_node_degree
    - carrier_bioentity_incidence
    - carrier_bioentity_counts
    - bioentity_overlap_features_train
    - bioentity_overlap_features_val
    - bioentity_overlap_features_test
    - bioentity_overlap_predictions_train
    - bioentity_overlap_predictions_val
    - bioentity_overlap_predictions_test
    """

    split_list = [str(split).strip() for split in splits if str(split).strip()]

    if load_views:
        load_link_prediction_dataset_views(
            connection,
            dataset_dir,
            splits=split_list,
            require_files=True,
        )

    degree_count = create_context_degree_table(
        connection,
        context_edge_table="context_edges",
        output_table="context_node_degree",
    )

    incidence_count = create_bioentity_incidence_table(
        connection,
        context_edge_table="context_edges",
        node_table="nodes",
        output_table="carrier_bioentity_incidence",
        carrier_node_types=("Patent", "Paper"),
        exclude_bioentity_types=exclude_bioentity_types,
    )

    carrier_count = create_bioentity_count_table(
        connection,
        incidence_table="carrier_bioentity_incidence",
        output_table="carrier_bioentity_counts",
    )

    feature_counts = create_bioentity_overlap_feature_tables(
        connection,
        splits=split_list,
        labeled_table_prefix="labeled_edges",
        feature_table_prefix="bioentity_overlap_features",
        incidence_table="carrier_bioentity_incidence",
        count_table="carrier_bioentity_counts",
        degree_table="context_node_degree",
        score_column=score_column,
    )

    prediction_counts = create_prediction_tables_from_features(
        connection,
        splits=split_list,
        feature_table_prefix="bioentity_overlap_features",
        prediction_table_prefix="bioentity_overlap_predictions",
        score_column="score",
    )

    return {
        "dataset_dir": str(dataset_dir),
        "score_column": score_column,
        "degree_count": degree_count,
        "incidence_count": incidence_count,
        "carrier_count": carrier_count,
        "feature_counts": feature_counts,
        "prediction_counts": prediction_counts,
    }


def fetch_prediction_rows(
    connection: Any,
    *,
    prediction_table: str,
) -> list[dict[str, Any]]:
    """Fetch prediction rows from a prediction table."""

    return fetch_dicts(
        connection,
        f"""
        SELECT
            source_index,
            target_index,
            source_type,
            source_id,
            target_type,
            target_id,
            split,
            label,
            score
        FROM {qident(prediction_table)}
        ORDER BY split, score DESC, source_id, target_id
        """,
    )


def fetch_prediction_rows_for_splits(
    connection: Any,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    prediction_table_prefix: str = "bioentity_overlap_predictions",
) -> list[dict[str, Any]]:
    """Fetch prediction rows from multiple split prediction tables."""

    rows: list[dict[str, Any]] = []

    for split in splits:
        split = str(split).strip()
        table_name = prediction_table_name(split, prefix=prediction_table_prefix)
        rows.extend(fetch_prediction_rows(connection, prediction_table=table_name))

    return rows


def summarize_bioentity_overlap_features(
    connection: Any,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    feature_table_prefix: str = "bioentity_overlap_features",
) -> list[dict[str, Any]]:
    """Return compact feature summaries by split and label."""

    union_parts = []

    for split in splits:
        split = str(split).strip()
        table_name = feature_table_name(split, prefix=feature_table_prefix)

        union_parts.append(
            f"""
            SELECT
                {sql_literal(split)} AS split,
                label,
                COUNT(*) AS example_count,
                AVG(shared_bioentity_count) AS avg_shared_bioentity_count,
                AVG(bioentity_jaccard) AS avg_bioentity_jaccard,
                AVG(weighted_shared_bioentity_min) AS avg_weighted_shared_bioentity_min,
                AVG(weighted_bioentity_jaccard) AS avg_weighted_bioentity_jaccard,
                AVG(weighted_bioentity_cosine) AS avg_weighted_bioentity_cosine,
                AVG(source_bioentity_count) AS avg_source_bioentity_count,
                AVG(target_bioentity_count) AS avg_target_bioentity_count,
                AVG(source_context_degree) AS avg_source_context_degree,
                AVG(target_context_degree) AS avg_target_context_degree
            FROM {qident(table_name)}
            GROUP BY label
            """
        )

    return fetch_dicts(
        connection,
        f"""
        {" UNION ALL ".join(union_parts)}
        ORDER BY split, label DESC
        """,
    )


def summarize_score_distribution(
    connection: Any,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    prediction_table_prefix: str = "bioentity_overlap_predictions",
) -> list[dict[str, Any]]:
    """Return score distribution summaries by split and label."""

    union_parts = []

    for split in splits:
        split = str(split).strip()
        table_name = prediction_table_name(split, prefix=prediction_table_prefix)

        union_parts.append(
            f"""
            SELECT
                {sql_literal(split)} AS split,
                label,
                COUNT(*) AS example_count,
                MIN(score) AS score_min,
                MAX(score) AS score_max,
                AVG(score) AS score_mean,
                QUANTILE_CONT(score, 0.25) AS score_q25,
                QUANTILE_CONT(score, 0.50) AS score_median,
                QUANTILE_CONT(score, 0.75) AS score_q75
            FROM {qident(table_name)}
            GROUP BY label
            """
        )

    return fetch_dicts(
        connection,
        f"""
        {" UNION ALL ".join(union_parts)}
        ORDER BY split, label DESC
        """,
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


def export_prediction_tables(
    connection: Any,
    *,
    output_dir: str | Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
    prediction_table_prefix: str = "bioentity_overlap_predictions",
    file_prefix: str = "predictions",
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Export prediction tables for multiple splits."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []

    for split in splits:
        split = str(split).strip()
        table_name = prediction_table_name(split, prefix=prediction_table_prefix)
        file_name = f"{file_prefix}_{split}.parquet"
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
                "path": str(output_path),
                "description": f"Prediction scores for the {split} split.",
                "row_count": count_rows(connection, table_name),
                "file_size_bytes": output_path.stat().st_size if output_path.exists() else "",
            }
        )

    return artifacts


def export_feature_tables(
    connection: Any,
    *,
    output_dir: str | Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
    feature_table_prefix: str = "bioentity_overlap_features",
    file_prefix: str = "features",
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Export feature tables for multiple splits."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []

    for split in splits:
        split = str(split).strip()
        table_name = feature_table_name(split, prefix=feature_table_prefix)
        file_name = f"{file_prefix}_{split}.parquet"
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
                "path": str(output_path),
                "description": f"BioEntity overlap features for the {split} split.",
                "row_count": count_rows(connection, table_name),
                "file_size_bytes": output_path.stat().st_size if output_path.exists() else "",
            }
        )

    return artifacts