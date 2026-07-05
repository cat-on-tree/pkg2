#!/usr/bin/env python
"""Build Knowledge Units from a carrier graph using DuckDB out-of-core SQL.

This script is a memory-safer alternative to scripts/20_build_knowledge_units.py
for large long-window carrier graphs.

It builds V1 Knowledge Units:

    Knowledge Unit = typed BioEntity pair co-mentioned on a carrier

Default target:

    data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated
      -> data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_streaming

Supported main translational pair types:

    chemical-disease
    gene-disease
    chemical-gene

Design
------
The script uses DuckDB to scan Parquet files and create intermediate temp tables.
DuckDB can spill to temp_directory when memory_limit is reached, making this much
safer than a pandas-only all-in-memory build.

Core outputs match the Stage 04/05 KU dataset shape:

    knowledge_units.parquet
    carrier_knowledge_unit_edges.parquet
    knowledge_unit_type_summary.csv
    knowledge_unit_summary.json
    knowledge_unit_report.md
    run_config.json
    dataset_manifest.csv

Typical run
-----------
python scripts/28_build_knowledge_units_streaming.py \
  --graph-dir data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated \
  --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational \
  --exclude-bioentity-types species \
  --pair-types chemical-disease gene-disease chemical-gene \
  --min-carrier-count 3 \
  --threads 4 \
  --memory-limit 24GB \
  --overwrite

If memory pressure is high, use:

python scripts/28_build_knowledge_units_streaming.py \
  --graph-dir data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated \
  --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational \
  --exclude-bioentity-types species \
  --pair-types chemical-disease gene-disease chemical-gene \
  --min-carrier-count 3 \
  --threads 2 \
  --memory-limit 16GB \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


MENTION_EDGE_FILES = {
    "paper": {
        "file_name": "edges_paper_bioentity.parquet",
        "carrier_id_column": "PMID",
    },
    "patent": {
        "file_name": "edges_patent_bioentity.parquet",
        "carrier_id_column": "PatentId",
    },
    "trial": {
        "file_name": "edges_trial_bioentity.parquet",
        "carrier_id_column": "nct_id",
    },
}

NODE_FILES = {
    "bioentity": "nodes_bioentity.parquet",
}

SUPPORTED_PAIR_TYPES = {
    "chemical-disease",
    "gene-disease",
    "chemical-gene",
}

SUPPORTED_PAIR_WEIGHT_METHODS = {
    "sqrt_product",
    "min",
    "product",
    "sum",
    "mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Knowledge Units from carrier graph using DuckDB."
    )

    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Processed aggregated carrier graph directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output Knowledge Unit dataset directory.",
    )
    parser.add_argument(
        "--carrier-types",
        nargs="*",
        default=["paper", "patent", "trial"],
        help="Carrier types to use. Default: paper patent trial.",
    )
    parser.add_argument(
        "--exclude-bioentity-types",
        nargs="*",
        default=["species"],
        help="Normalized BioEntity types to exclude. Default: species.",
    )
    parser.add_argument(
        "--pair-types",
        nargs="+",
        default=["chemical-disease", "gene-disease", "chemical-gene"],
        help="Allowed typed BioEntity pair types.",
    )
    parser.add_argument(
        "--min-carrier-count",
        type=int,
        default=3,
        help="Minimum number of distinct carriers supporting a KU. Default: 3.",
    )
    parser.add_argument(
        "--min-paper-count",
        type=int,
        default=0,
        help="Minimum number of paper carriers supporting a KU. Default: 0.",
    )
    parser.add_argument(
        "--min-patent-count",
        type=int,
        default=0,
        help="Minimum number of patent carriers supporting a KU. Default: 0.",
    )
    parser.add_argument(
        "--min-trial-count",
        type=int,
        default=0,
        help="Minimum number of trial carriers supporting a KU. Default: 0.",
    )
    parser.add_argument(
        "--pair-weight-method",
        default="sqrt_product",
        choices=sorted(SUPPORTED_PAIR_WEIGHT_METHODS),
        help="Pair evidence weight function. Default: sqrt_product.",
    )
    parser.add_argument(
        "--ku-id-prefix",
        default="KU",
        help="Prefix for stable Knowledge Unit IDs. Default: KU.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="DuckDB thread count. Default: 4.",
    )
    parser.add_argument(
        "--memory-limit",
        default="24GB",
        help="DuckDB memory limit. Default: 24GB.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/interim/duckdb_tmp"),
        help="DuckDB temp directory for spilling. Default: data/interim/duckdb_tmp.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression. Default: zstd.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top KUs in summary/report. Default: 25.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress logs.",
    )

    return parser.parse_args()


def log(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, flush=True)


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def qident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def scan_sql(path: Path) -> str:
    return f"read_parquet({sql_literal(path)})"


def normalize_token(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def normalize_pair_type(value: str) -> str:
    parts = [normalize_entity_type_token(part) for part in str(value).split("-")]
    parts = [part for part in parts if part]

    if len(parts) != 2:
        raise ValueError(f"Invalid pair type: {value!r}")

    return f"{parts[0]}-{parts[1]}"


def normalize_entity_type_token(value: str) -> str:
    token = str(value).strip().lower().replace("_", "-")

    aliases = {
        "chemical-substance": "chemical",
        "drug": "chemical",
        "compound": "chemical",
        "gene-or-gene-product": "gene",
        "protein": "gene",
        "disease-or-phenotypic-feature": "disease",
        "disease-phenotype": "disease",
        "organism": "species",
        "taxon": "species",
        "ncbi-taxon": "species",
    }

    return aliases.get(token, token)


def allowed_entity_types_from_pair_types(pair_types: list[str]) -> set[str]:
    out: set[str] = set()

    for pair_type in pair_types:
        left, right = normalize_pair_type(pair_type).split("-", maxsplit=1)
        out.add(left)
        out.add(right)

    return out


def sql_string_list(values: list[str] | set[str]) -> str:
    clean = [str(v) for v in values]
    if not clean:
        return "''"
    return ", ".join(sql_literal(v) for v in clean)


def connect_duckdb(args: argparse.Namespace):
    import duckdb

    args.temp_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET threads={int(args.threads)}")
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute(f"SET temp_directory={sql_literal(args.temp_dir)}")

    return connection


def execute(connection: Any, sql: str) -> None:
    connection.execute(sql)


def fetchone_int(connection: Any, sql: str) -> int:
    return int(connection.execute(sql).fetchone()[0])


def count_table(connection: Any, table_name: str) -> int:
    return fetchone_int(connection, f"SELECT COUNT(*) FROM {qident(table_name)}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def require_files(graph_dir: Path, carrier_types: list[str]) -> None:
    missing: list[str] = []

    bioentity_path = graph_dir / NODE_FILES["bioentity"]
    if not bioentity_path.exists():
        missing.append(str(bioentity_path))

    for carrier_type in carrier_types:
        if carrier_type not in MENTION_EDGE_FILES:
            raise ValueError(f"Unsupported carrier type: {carrier_type!r}")

        path = graph_dir / MENTION_EDGE_FILES[carrier_type]["file_name"]
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError("Missing required graph files:\n" + "\n".join(missing))


def run_step(args: argparse.Namespace, index: int, total: int, label: str, fn) -> Any:
    log(args, f"[{index}/{total}] {label} ...")
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    log(args, f"[{index}/{total}] {label} done in {elapsed:.1f}s")
    return result


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        return "\n".join([header, sep])

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"

    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if value is None:
                text = ""
            elif isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            text = text.replace("\n", " ").replace("|", "\\|")
            values.append(text)
        body.append("| " + " | ".join(values) + " |")

    return "\n".join([header, sep, *body])


def copy_table_to_parquet(
    connection: Any,
    table_name: str,
    path: Path,
    compression: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression_sql = compression.upper()

    execute(
        connection,
        f"""
        COPY (
            SELECT * FROM {qident(table_name)}
        )
        TO {sql_literal(path)}
        (FORMAT PARQUET, COMPRESSION {compression_sql})
        """,
    )


def copy_table_to_csv(
    connection: Any,
    table_name: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    execute(
        connection,
        f"""
        COPY (
            SELECT * FROM {qident(table_name)}
        )
        TO {sql_literal(path)}
        (HEADER, DELIMITER ',')
        """,
    )


def normalize_entity_type_sql(expr: str) -> str:
    """Return SQL CASE expression that normalizes BioEntity type."""

    token = f"LOWER(REPLACE(TRIM(CAST({expr} AS VARCHAR)), '_', '-'))"

    return f"""
    CASE
        WHEN {token} IN ('chemical-substance', 'chemical', 'drug', 'compound') THEN 'chemical'
        WHEN {token} IN ('gene-or-gene-product', 'gene', 'protein') THEN 'gene'
        WHEN {token} IN ('disease-or-phenotypic-feature', 'disease-phenotype', 'disease') THEN 'disease'
        WHEN {token} IN ('phenotypic-feature', 'phenotype') THEN 'phenotype'
        WHEN {token} IN ('organism', 'taxon', 'ncbi-taxon', 'species') THEN 'species'
        WHEN {token} IN ('cell-type', 'cell-line') THEN {token}
        WHEN {expr} IS NULL THEN 'unknown'
        WHEN TRIM(CAST({expr} AS VARCHAR)) = '' THEN 'unknown'
        ELSE {token}
    END
    """


def create_bioentity_table(connection: Any, args: argparse.Namespace) -> None:
    graph_dir = args.graph_dir
    bio_path = graph_dir / NODE_FILES["bioentity"]

    type_norm_expr = normalize_entity_type_sql("Type")

    execute(
        connection,
        f"""
        CREATE TEMP TABLE bioentities AS
        SELECT
            NULLIF(TRIM(CAST(EntityId AS VARCHAR)), '') AS bioentity_id,
            ANY_VALUE(NULLIF(TRIM(CAST(Mention AS VARCHAR)), '')) AS bioentity_name,
            ANY_VALUE(NULLIF(TRIM(CAST(Type AS VARCHAR)), '')) AS bioentity_type_raw,
            ANY_VALUE({type_norm_expr}) AS bioentity_type
        FROM {scan_sql(bio_path)}
        WHERE NULLIF(TRIM(CAST(EntityId AS VARCHAR)), '') IS NOT NULL
        GROUP BY bioentity_id
        """,
    )


def mention_union_sql(args: argparse.Namespace) -> str:
    selects: list[str] = []

    for carrier_type in args.carrier_types:
        meta = MENTION_EDGE_FILES[carrier_type]
        path = args.graph_dir / meta["file_name"]
        carrier_col = meta["carrier_id_column"]

        selects.append(
            f"""
            SELECT
                {sql_literal(carrier_type)} AS carrier_type,
                NULLIF(TRIM(CAST({qident(carrier_col)} AS VARCHAR)), '') AS carrier_id,
                {sql_literal(carrier_type)} || ':' || NULLIF(TRIM(CAST({qident(carrier_col)} AS VARCHAR)), '') AS carrier_key,
                TRY_CAST(year AS BIGINT) AS year,
                NULLIF(TRIM(CAST(EntityId AS VARCHAR)), '') AS bioentity_id,
                COALESCE(TRY_CAST(mention_count AS DOUBLE), 1.0) AS mention_count,
                {sql_literal(meta["file_name"])} AS source_edge_table
            FROM {scan_sql(path)}
            WHERE NULLIF(TRIM(CAST({qident(carrier_col)} AS VARCHAR)), '') IS NOT NULL
              AND NULLIF(TRIM(CAST(EntityId AS VARCHAR)), '') IS NOT NULL
              AND TRY_CAST(year AS BIGINT) IS NOT NULL
            """
        )

    return "\nUNION ALL\n".join(selects)


def create_filtered_mentions(connection: Any, args: argparse.Namespace) -> dict[str, Any]:
    allowed_entity_types = allowed_entity_types_from_pair_types(args.pair_types)
    excluded_entity_types = {
        normalize_entity_type_token(value) for value in args.exclude_bioentity_types
    }

    allowed_sql = sql_string_list(allowed_entity_types)
    excluded_sql = sql_string_list(excluded_entity_types) if excluded_entity_types else None

    union_sql = mention_union_sql(args)

    exclude_clause = ""
    if excluded_entity_types:
        exclude_clause = f"AND bioentity_type NOT IN ({excluded_sql})"

    execute(
        connection,
        f"""
        CREATE TEMP TABLE raw_mentions AS
        WITH unioned AS (
            {union_sql}
        )
        SELECT
            u.carrier_type,
            u.carrier_id,
            u.carrier_key,
            u.year,
            u.bioentity_id,
            COALESCE(b.bioentity_name, u.bioentity_id) AS bioentity_name,
            COALESCE(b.bioentity_type, 'unknown') AS bioentity_type,
            COALESCE(b.bioentity_type_raw, 'unknown') AS bioentity_type_raw,
            u.mention_count,
            u.source_edge_table
        FROM unioned u
        LEFT JOIN bioentities b
            ON u.bioentity_id = b.bioentity_id
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE filtered_mentions AS
        SELECT
            carrier_type,
            carrier_id,
            carrier_key,
            year,
            bioentity_id,
            ANY_VALUE(bioentity_name) AS bioentity_name,
            bioentity_type,
            ANY_VALUE(bioentity_type_raw) AS bioentity_type_raw,
            SUM(mention_count) AS mention_count,
            STRING_AGG(DISTINCT source_edge_table, '|') AS source_edge_table
        FROM raw_mentions
        WHERE bioentity_type IN ({allowed_sql})
          {exclude_clause}
        GROUP BY
            carrier_type,
            carrier_id,
            carrier_key,
            year,
            bioentity_id,
            bioentity_type
        """,
    )

    raw_summary = summarize_mentions(connection, "raw_mentions")
    filtered_summary = summarize_mentions(connection, "filtered_mentions")

    return {
        "raw_summary": raw_summary,
        "filtered_summary": filtered_summary,
        "allowed_entity_types_from_pair_types": sorted(allowed_entity_types),
        "excluded_bioentity_types": sorted(excluded_entity_types),
    }


def summarize_mentions(connection: Any, table_name: str) -> dict[str, Any]:
    row = connection.execute(
        f"""
        SELECT
            COUNT(*) AS mention_edge_count,
            COUNT(DISTINCT carrier_key) AS carrier_count,
            COUNT(DISTINCT bioentity_id) AS bioentity_count,
            MIN(year) AS year_min,
            MAX(year) AS year_max,
            SUM(CASE WHEN year IS NULL THEN 1 ELSE 0 END) AS missing_year_count,
            SUM(mention_count) AS mention_count_sum,
            AVG(mention_count) AS mention_count_mean,
            MAX(mention_count) AS mention_count_max
        FROM {qident(table_name)}
        """
    ).fetchone()

    carrier_types = [
        r[0]
        for r in connection.execute(
            f"""
            SELECT DISTINCT carrier_type
            FROM {qident(table_name)}
            ORDER BY carrier_type
            """
        ).fetchall()
    ]

    bioentity_types = [
        r[0]
        for r in connection.execute(
            f"""
            SELECT DISTINCT bioentity_type
            FROM {qident(table_name)}
            ORDER BY bioentity_type
            """
        ).fetchall()
    ]

    return {
        "mention_edge_count": int(row[0] or 0),
        "carrier_count": int(row[1] or 0),
        "bioentity_count": int(row[2] or 0),
        "carrier_types": carrier_types,
        "bioentity_types": bioentity_types,
        "year_min": int(row[3]) if row[3] is not None else None,
        "year_max": int(row[4]) if row[4] is not None else None,
        "missing_year_count": int(row[5] or 0),
        "mention_count_sum": float(row[6] or 0.0),
        "mention_count_mean": float(row[7] or 0.0),
        "mention_count_max": float(row[8] or 0.0),
    }


def parse_allowed_pairs(pair_types: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    for pair_type in pair_types:
        normalized = normalize_pair_type(pair_type)

        if normalized not in SUPPORTED_PAIR_TYPES:
            raise ValueError(
                f"Unsupported pair type for streaming builder: {pair_type!r}. "
                f"Supported: {sorted(SUPPORTED_PAIR_TYPES)}"
            )

        left, right = normalized.split("-", maxsplit=1)
        out.append((left, right))

    return out


def build_pair_case_sql(allowed_pairs: list[tuple[str, str]]) -> str:
    cases = []

    for left, right in allowed_pairs:
        pair_type = f"{left}-{right}"
        cases.append(
            f"""
            WHEN (
                (m1.bioentity_type = {sql_literal(left)} AND m2.bioentity_type = {sql_literal(right)})
                OR
                (m1.bioentity_type = {sql_literal(right)} AND m2.bioentity_type = {sql_literal(left)})
            )
            THEN {sql_literal(pair_type)}
            """
        )

    return "CASE " + "\n".join(cases) + "\nELSE NULL END"


def build_pair_filter_sql(allowed_pairs: list[tuple[str, str]]) -> str:
    filters = []

    for left, right in allowed_pairs:
        filters.append(
            f"""
            (
                (m1.bioentity_type = {sql_literal(left)} AND m2.bioentity_type = {sql_literal(right)})
                OR
                (m1.bioentity_type = {sql_literal(right)} AND m2.bioentity_type = {sql_literal(left)})
            )
            """
        )

    return " OR ".join(filters)


def oriented_expr(
    *,
    allowed_pairs: list[tuple[str, str]],
    left_value: str,
    right_value: str,
    output_side: str,
) -> str:
    """Build orientation CASE expression.

    output_side:
        'a' means return value from entity matching pair left type.
        'b' means return value from entity matching pair right type.
    """

    cases = []

    for left, right in allowed_pairs:
        if output_side == "a":
            cases.append(
                f"""
                WHEN pair_type = {sql_literal(left + '-' + right)}
                 AND m1_type = {sql_literal(left)}
                THEN {left_value}
                """
            )
            cases.append(
                f"""
                WHEN pair_type = {sql_literal(left + '-' + right)}
                 AND m2_type = {sql_literal(left)}
                THEN {right_value}
                """
            )
        elif output_side == "b":
            cases.append(
                f"""
                WHEN pair_type = {sql_literal(left + '-' + right)}
                 AND m1_type = {sql_literal(right)}
                THEN {left_value}
                """
            )
            cases.append(
                f"""
                WHEN pair_type = {sql_literal(left + '-' + right)}
                 AND m2_type = {sql_literal(right)}
                THEN {right_value}
                """
            )
        else:
            raise ValueError(f"Invalid output_side: {output_side}")

    return "CASE " + "\n".join(cases) + "\nELSE NULL END"


def pair_weight_expr(method: str) -> str:
    if method == "sqrt_product":
        return "SQRT(GREATEST(mention_count_a_raw, 0.0) * GREATEST(mention_count_b_raw, 0.0))"
    if method == "min":
        return "LEAST(mention_count_a_raw, mention_count_b_raw)"
    if method == "product":
        return "mention_count_a_raw * mention_count_b_raw"
    if method == "sum":
        return "mention_count_a_raw + mention_count_b_raw"
    if method == "mean":
        return "(mention_count_a_raw + mention_count_b_raw) / 2.0"

    raise ValueError(f"Unsupported pair weight method: {method}")


def create_carrier_ku_edges(connection: Any, args: argparse.Namespace) -> dict[str, Any]:
    allowed_pairs = parse_allowed_pairs(args.pair_types)

    pair_case = build_pair_case_sql(allowed_pairs)
    pair_filter = build_pair_filter_sql(allowed_pairs)

    entity_a_id_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_id",
        right_value="m2_id",
        output_side="a",
    )
    entity_b_id_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_id",
        right_value="m2_id",
        output_side="b",
    )

    entity_a_name_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_name",
        right_value="m2_name",
        output_side="a",
    )
    entity_b_name_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_name",
        right_value="m2_name",
        output_side="b",
    )

    entity_a_type_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_type",
        right_value="m2_type",
        output_side="a",
    )
    entity_b_type_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_type",
        right_value="m2_type",
        output_side="b",
    )

    entity_a_type_raw_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_type_raw",
        right_value="m2_type_raw",
        output_side="a",
    )
    entity_b_type_raw_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_type_raw",
        right_value="m2_type_raw",
        output_side="b",
    )

    mention_count_a_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_mention_count",
        right_value="m2_mention_count",
        output_side="a",
    )
    mention_count_b_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_mention_count",
        right_value="m2_mention_count",
        output_side="b",
    )

    source_a_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_source_edge_table",
        right_value="m2_source_edge_table",
        output_side="a",
    )
    source_b_expr = oriented_expr(
        allowed_pairs=allowed_pairs,
        left_value="m1_source_edge_table",
        right_value="m2_source_edge_table",
        output_side="b",
    )

    weight_sql = pair_weight_expr(args.pair_weight_method)

    execute(
        connection,
        f"""
        CREATE TEMP TABLE carrier_ku_edges_all AS
        WITH paired AS (
            SELECT
                m1.carrier_type,
                m1.carrier_id,
                m1.carrier_key,
                m1.year,

                m1.bioentity_id AS m1_id,
                m1.bioentity_name AS m1_name,
                m1.bioentity_type AS m1_type,
                m1.bioentity_type_raw AS m1_type_raw,
                m1.mention_count AS m1_mention_count,
                m1.source_edge_table AS m1_source_edge_table,

                m2.bioentity_id AS m2_id,
                m2.bioentity_name AS m2_name,
                m2.bioentity_type AS m2_type,
                m2.bioentity_type_raw AS m2_type_raw,
                m2.mention_count AS m2_mention_count,
                m2.source_edge_table AS m2_source_edge_table,

                {pair_case} AS pair_type
            FROM filtered_mentions m1
            INNER JOIN filtered_mentions m2
                ON m1.carrier_type = m2.carrier_type
               AND m1.carrier_id = m2.carrier_id
               AND m1.carrier_key = m2.carrier_key
               AND m1.year = m2.year
               AND m1.bioentity_id < m2.bioentity_id
            WHERE {pair_filter}
        ),
        oriented AS (
            SELECT
                pair_type,
                carrier_type,
                carrier_id,
                carrier_key,
                year,

                {entity_a_id_expr} AS entity_a_id,
                {entity_a_name_expr} AS entity_a_name,
                {entity_a_type_expr} AS entity_a_type,
                {entity_a_type_raw_expr} AS entity_a_type_raw,

                {entity_b_id_expr} AS entity_b_id,
                {entity_b_name_expr} AS entity_b_name,
                {entity_b_type_expr} AS entity_b_type,
                {entity_b_type_raw_expr} AS entity_b_type_raw,

                TRY_CAST({mention_count_a_expr} AS DOUBLE) AS mention_count_a_raw,
                TRY_CAST({mention_count_b_expr} AS DOUBLE) AS mention_count_b_raw,
                {source_a_expr} AS source_edge_table_a_raw,
                {source_b_expr} AS source_edge_table_b_raw
            FROM paired
            WHERE pair_type IS NOT NULL
        ),
        weighted AS (
            SELECT
                {sql_literal(args.ku_id_prefix)}
                    || '_'
                    || SUBSTR(MD5(pair_type || '||' || entity_a_id || '||' || entity_b_id), 1, 16)
                    AS ku_id,
                entity_a_id,
                entity_a_name,
                entity_a_type,
                entity_a_type_raw,
                entity_b_id,
                entity_b_name,
                entity_b_type,
                entity_b_type_raw,
                pair_type,
                carrier_type,
                carrier_id,
                carrier_key,
                year,
                mention_count_a_raw,
                mention_count_b_raw,
                {weight_sql} AS pair_weight,
                source_edge_table_a_raw,
                source_edge_table_b_raw,
                source_edge_table_a_raw || '|' || source_edge_table_b_raw AS source_edge_table_raw
            FROM oriented
            WHERE entity_a_id IS NOT NULL
              AND entity_b_id IS NOT NULL
              AND entity_a_id <> entity_b_id
        )
        SELECT
            ku_id,
            entity_a_id,
            ANY_VALUE(entity_a_name) AS entity_a_name,
            entity_a_type,
            ANY_VALUE(entity_a_type_raw) AS entity_a_type_raw,
            entity_b_id,
            ANY_VALUE(entity_b_name) AS entity_b_name,
            entity_b_type,
            ANY_VALUE(entity_b_type_raw) AS entity_b_type_raw,
            pair_type,
            carrier_type,
            carrier_id,
            carrier_key,
            year,
            SUM(mention_count_a_raw) AS mention_count_a,
            SUM(mention_count_b_raw) AS mention_count_b,
            SUM(pair_weight) AS pair_weight,
            STRING_AGG(DISTINCT source_edge_table_a_raw, '|') AS source_edge_table_a,
            STRING_AGG(DISTINCT source_edge_table_b_raw, '|') AS source_edge_table_b,
            STRING_AGG(DISTINCT source_edge_table_raw, '|') AS source_edge_table
        FROM weighted
        GROUP BY
            ku_id,
            entity_a_id,
            entity_a_type,
            entity_b_id,
            entity_b_type,
            pair_type,
            carrier_type,
            carrier_id,
            carrier_key,
            year
        """,
    )

    return {
        "raw_carrier_ku_edge_count": count_table(connection, "carrier_ku_edges_all")
    }


def create_knowledge_units_all(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE knowledge_units_all AS
        WITH base AS (
            SELECT
                ku_id,
                ANY_VALUE(entity_a_id) AS entity_a_id,
                ANY_VALUE(entity_a_name) AS entity_a_name,
                ANY_VALUE(entity_a_type) AS entity_a_type,
                ANY_VALUE(entity_a_type_raw) AS entity_a_type_raw,
                ANY_VALUE(entity_b_id) AS entity_b_id,
                ANY_VALUE(entity_b_name) AS entity_b_name,
                ANY_VALUE(entity_b_type) AS entity_b_type,
                ANY_VALUE(entity_b_type_raw) AS entity_b_type_raw,
                ANY_VALUE(pair_type) AS pair_type
            FROM carrier_ku_edges_all
            GROUP BY ku_id
        ),
        counts AS (
            SELECT
                ku_id,
                COUNT(DISTINCT carrier_key) AS carrier_count,
                SUM(pair_weight) AS total_pair_weight,
                AVG(pair_weight) AS mean_pair_weight,
                MAX(pair_weight) AS max_pair_weight,
                MIN(year) AS first_year,
                MAX(year) AS last_year,
                COUNT(DISTINCT CASE WHEN carrier_type = 'paper' THEN carrier_key ELSE NULL END) AS paper_count,
                COUNT(DISTINCT CASE WHEN carrier_type = 'patent' THEN carrier_key ELSE NULL END) AS patent_count,
                COUNT(DISTINCT CASE WHEN carrier_type = 'trial' THEN carrier_key ELSE NULL END) AS trial_count
            FROM carrier_ku_edges_all
            GROUP BY ku_id
        )
        SELECT
            b.ku_id,
            b.entity_a_id,
            b.entity_a_name,
            b.entity_a_type,
            b.entity_a_type_raw,
            b.entity_b_id,
            b.entity_b_name,
            b.entity_b_type,
            b.entity_b_type_raw,
            b.pair_type,
            c.carrier_count,
            c.total_pair_weight,
            c.mean_pair_weight,
            c.max_pair_weight,
            c.first_year,
            c.last_year,
            c.paper_count,
            c.patent_count,
            c.trial_count,
            c.paper_count > 0 AS has_paper_evidence,
            c.patent_count > 0 AS has_patent_evidence,
            c.trial_count > 0 AS has_trial_evidence
        FROM base b
        INNER JOIN counts c
            ON b.ku_id = c.ku_id
        """
    )


def apply_support_filter(connection: Any, args: argparse.Namespace) -> None:
    execute(
        connection,
        f"""
        CREATE TEMP TABLE kept_knowledge_unit_ids AS
        SELECT ku_id
        FROM knowledge_units_all
        WHERE carrier_count >= {int(args.min_carrier_count)}
          AND paper_count >= {int(args.min_paper_count)}
          AND patent_count >= {int(args.min_patent_count)}
          AND trial_count >= {int(args.min_trial_count)}
        """
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE carrier_knowledge_unit_edges AS
        SELECT e.*
        FROM carrier_ku_edges_all e
        INNER JOIN kept_knowledge_unit_ids k
            ON e.ku_id = k.ku_id
        """
    )

    # Re-aggregate final KU master table after edge filtering, matching the
    # semantics of scripts/20_build_knowledge_units.py.
    execute(
        connection,
        """
        CREATE TEMP TABLE knowledge_units AS
        WITH base AS (
            SELECT
                ku_id,
                ANY_VALUE(entity_a_id) AS entity_a_id,
                ANY_VALUE(entity_a_name) AS entity_a_name,
                ANY_VALUE(entity_a_type) AS entity_a_type,
                ANY_VALUE(entity_a_type_raw) AS entity_a_type_raw,
                ANY_VALUE(entity_b_id) AS entity_b_id,
                ANY_VALUE(entity_b_name) AS entity_b_name,
                ANY_VALUE(entity_b_type) AS entity_b_type,
                ANY_VALUE(entity_b_type_raw) AS entity_b_type_raw,
                ANY_VALUE(pair_type) AS pair_type
            FROM carrier_knowledge_unit_edges
            GROUP BY ku_id
        ),
        counts AS (
            SELECT
                ku_id,
                COUNT(DISTINCT carrier_key) AS carrier_count,
                SUM(pair_weight) AS total_pair_weight,
                AVG(pair_weight) AS mean_pair_weight,
                MAX(pair_weight) AS max_pair_weight,
                MIN(year) AS first_year,
                MAX(year) AS last_year,
                COUNT(DISTINCT CASE WHEN carrier_type = 'paper' THEN carrier_key ELSE NULL END) AS paper_count,
                COUNT(DISTINCT CASE WHEN carrier_type = 'patent' THEN carrier_key ELSE NULL END) AS patent_count,
                COUNT(DISTINCT CASE WHEN carrier_type = 'trial' THEN carrier_key ELSE NULL END) AS trial_count
            FROM carrier_knowledge_unit_edges
            GROUP BY ku_id
        )
        SELECT
            b.ku_id,
            b.entity_a_id,
            b.entity_a_name,
            b.entity_a_type,
            b.entity_a_type_raw,
            b.entity_b_id,
            b.entity_b_name,
            b.entity_b_type,
            b.entity_b_type_raw,
            b.pair_type,
            c.carrier_count,
            c.total_pair_weight,
            c.mean_pair_weight,
            c.max_pair_weight,
            c.first_year,
            c.last_year,
            c.paper_count,
            c.patent_count,
            c.trial_count,
            c.paper_count > 0 AS has_paper_evidence,
            c.patent_count > 0 AS has_patent_evidence,
            c.trial_count > 0 AS has_trial_evidence
        FROM base b
        INNER JOIN counts c
            ON b.ku_id = c.ku_id
        ORDER BY
            c.carrier_count DESC,
            c.total_pair_weight DESC,
            b.ku_id ASC
        """
    )


def create_type_summary(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE knowledge_unit_type_summary AS
        SELECT
            pair_type,
            COUNT(DISTINCT ku_id) AS knowledge_unit_count,
            SUM(carrier_count) AS carrier_count_sum,
            AVG(carrier_count) AS carrier_count_mean,
            SUM(paper_count) AS paper_count_sum,
            SUM(patent_count) AS patent_count_sum,
            SUM(trial_count) AS trial_count_sum,
            SUM(total_pair_weight) AS total_pair_weight_sum
        FROM knowledge_units
        GROUP BY pair_type
        ORDER BY knowledge_unit_count DESC, carrier_count_sum DESC
        """
    )


def summarize_build(
    connection: Any,
    args: argparse.Namespace,
    mention_summaries: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    raw_edges = count_table(connection, "carrier_ku_edges_all")
    filtered_edges = count_table(connection, "carrier_knowledge_unit_edges")
    knowledge_unit_count = count_table(connection, "knowledge_units")

    pair_type_count = fetchone_int(
        connection,
        "SELECT COUNT(DISTINCT pair_type) FROM knowledge_units",
    )

    carrier_count = fetchone_int(
        connection,
        "SELECT COUNT(DISTINCT carrier_key) FROM carrier_knowledge_unit_edges",
    )

    paper_carrier_count = fetchone_int(
        connection,
        """
        SELECT COUNT(DISTINCT carrier_key)
        FROM carrier_knowledge_unit_edges
        WHERE carrier_type = 'paper'
        """,
    )
    patent_carrier_count = fetchone_int(
        connection,
        """
        SELECT COUNT(DISTINCT carrier_key)
        FROM carrier_knowledge_unit_edges
        WHERE carrier_type = 'patent'
        """,
    )
    trial_carrier_count = fetchone_int(
        connection,
        """
        SELECT COUNT(DISTINCT carrier_key)
        FROM carrier_knowledge_unit_edges
        WHERE carrier_type = 'trial'
        """,
    )

    top_rows = connection.execute(
        f"""
        SELECT
            ku_id,
            entity_a_name,
            entity_a_type,
            entity_b_name,
            entity_b_type,
            pair_type,
            carrier_count,
            paper_count,
            patent_count,
            trial_count,
            total_pair_weight,
            first_year,
            last_year
        FROM knowledge_units
        ORDER BY carrier_count DESC, total_pair_weight DESC, ku_id ASC
        LIMIT {int(args.top_n)}
        """
    ).fetchall()

    top_knowledge_units = [
        {
            "ku_id": row[0],
            "entity_a_name": row[1],
            "entity_a_type": row[2],
            "entity_b_name": row[3],
            "entity_b_type": row[4],
            "pair_type": row[5],
            "carrier_count": int(row[6]),
            "paper_count": int(row[7]),
            "patent_count": int(row[8]),
            "trial_count": int(row[9]),
            "total_pair_weight": float(row[10]),
            "first_year": int(row[11]) if row[11] is not None else None,
            "last_year": int(row[12]) if row[12] is not None else None,
        }
        for row in top_rows
    ]

    stats_row = connection.execute(
        """
        SELECT
            MIN(first_year),
            MAX(last_year),
            MIN(carrier_count),
            MEDIAN(carrier_count),
            MAX(carrier_count),
            SUM(total_pair_weight)
        FROM knowledge_units
        """
    ).fetchone()

    type_summary_rows = [
        {
            "pair_type": row[0],
            "knowledge_unit_count": int(row[1]),
            "carrier_count_sum": int(row[2]),
            "carrier_count_mean": float(row[3]),
            "paper_count_sum": int(row[4]),
            "patent_count_sum": int(row[5]),
            "trial_count_sum": int(row[6]),
            "total_pair_weight_sum": float(row[7]),
        }
        for row in connection.execute(
            """
            SELECT
                pair_type,
                knowledge_unit_count,
                carrier_count_sum,
                carrier_count_mean,
                paper_count_sum,
                patent_count_sum,
                trial_count_sum,
                total_pair_weight_sum
            FROM knowledge_unit_type_summary
            ORDER BY knowledge_unit_count DESC, carrier_count_sum DESC
            """
        ).fetchall()
    ]

    summary = {
        "raw_mention_edge_count": mention_summaries["raw_summary"]["mention_edge_count"],
        "filtered_mention_edge_count": mention_summaries["filtered_summary"]["mention_edge_count"],
        "raw_carrier_ku_edge_count": raw_edges,
        "filtered_carrier_ku_edge_count": filtered_edges,
        "knowledge_unit_count": knowledge_unit_count,
        "pair_type_count": pair_type_count,
        "carrier_count": carrier_count,
        "paper_carrier_count": paper_carrier_count,
        "patent_carrier_count": patent_carrier_count,
        "trial_carrier_count": trial_carrier_count,
        "min_carrier_count": int(args.min_carrier_count),
        "min_paper_count": int(args.min_paper_count),
        "min_patent_count": int(args.min_patent_count),
        "min_trial_count": int(args.min_trial_count),
        "pair_weight_method": args.pair_weight_method,
        "max_entities_per_carrier": None,
        "excluded_bioentity_types": list(args.exclude_bioentity_types),
        "allowed_pair_types": list(args.pair_types),
        "allowed_entity_types_from_pair_types": mention_summaries["allowed_entity_types_from_pair_types"],
        "runtime_seconds": runtime_seconds,
        "first_year_min": int(stats_row[0]) if stats_row[0] is not None else None,
        "last_year_max": int(stats_row[1]) if stats_row[1] is not None else None,
        "carrier_count_min": int(stats_row[2]) if stats_row[2] is not None else None,
        "carrier_count_median": float(stats_row[3]) if stats_row[3] is not None else None,
        "carrier_count_max": int(stats_row[4]) if stats_row[4] is not None else None,
        "total_pair_weight_sum": float(stats_row[5]) if stats_row[5] is not None else 0.0,
        "top_knowledge_units": top_knowledge_units,
        "type_summary": type_summary_rows,
        "raw_summary": mention_summaries["raw_summary"],
        "filtered_summary": mention_summaries["filtered_summary"],
    }

    return summary


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/28_build_knowledge_units_streaming.py",
        "command": " ".join(sys.argv),
        "graph_dir": str(args.graph_dir),
        "output_dir": str(args.output_dir),
        "carrier_types": list(args.carrier_types),
        "exclude_bioentity_types": list(args.exclude_bioentity_types),
        "pair_types": list(args.pair_types),
        "min_carrier_count": int(args.min_carrier_count),
        "min_paper_count": int(args.min_paper_count),
        "min_patent_count": int(args.min_patent_count),
        "min_trial_count": int(args.min_trial_count),
        "pair_weight_method": args.pair_weight_method,
        "max_entities_per_carrier": None,
        "ku_id_prefix": args.ku_id_prefix,
        "threads": int(args.threads),
        "memory_limit": args.memory_limit,
        "temp_dir": str(args.temp_dir),
        "compression": args.compression,
        "created_at_unix": time.time(),
        "ku_definition": "typed_entity_pair",
        "ku_layer": "main_translational_entity_pair_layer",
        "relation_interpretation": (
            "Carrier co-mention evidence does not imply an asserted biomedical "
            "relation. KUs are typed entity-pair co-mention units, not "
            "relation-level claims."
        ),
        "disease_disease_policy": (
            "Disease-disease pairs are excluded from the main translational KU layer."
        ),
        "implementation": "duckdb_out_of_core_streaming_style_builder",
    }


def artifact_row(
    artifact: str,
    path: Path,
    description: str,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }


def write_outputs(
    connection: Any,
    args: argparse.Namespace,
    summary: dict[str, Any],
    run_config: dict[str, Any],
) -> list[dict[str, Any]]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    knowledge_units_path = output_dir / "knowledge_units.parquet"
    carrier_edges_path = output_dir / "carrier_knowledge_unit_edges.parquet"
    type_summary_path = output_dir / "knowledge_unit_type_summary.csv"
    summary_path = output_dir / "knowledge_unit_summary.json"
    run_config_path = output_dir / "run_config.json"

    copy_table_to_parquet(
        connection,
        "knowledge_units",
        knowledge_units_path,
        args.compression,
    )
    copy_table_to_parquet(
        connection,
        "carrier_knowledge_unit_edges",
        carrier_edges_path,
        args.compression,
    )
    copy_table_to_csv(
        connection,
        "knowledge_unit_type_summary",
        type_summary_path,
    )

    write_json(summary_path, summary)
    write_json(run_config_path, run_config)

    artifacts = [
        artifact_row(
            "knowledge_units",
            knowledge_units_path,
            "Knowledge Unit master table.",
            row_count=summary["knowledge_unit_count"],
        ),
        artifact_row(
            "carrier_knowledge_unit_edges",
            carrier_edges_path,
            "Carrier-to-Knowledge Unit co-mention evidence edges.",
            row_count=summary["filtered_carrier_ku_edge_count"],
        ),
        artifact_row(
            "knowledge_unit_type_summary",
            type_summary_path,
            "Knowledge Unit counts and evidence summary by pair type.",
            row_count=summary["pair_type_count"],
        ),
        artifact_row(
            "knowledge_unit_summary",
            summary_path,
            "JSON summary for Knowledge Unit construction.",
        ),
        artifact_row(
            "run_config",
            run_config_path,
            "Run configuration for streaming Knowledge Unit construction.",
        ),
    ]

    manifest_path = output_dir / "dataset_manifest.csv"
    artifacts.append(
        artifact_row(
            "dataset_manifest",
            manifest_path,
            "Manifest of Knowledge Unit dataset artifacts.",
            row_count=len(artifacts) + 1,
        )
    )

    write_csv(
        manifest_path,
        artifacts,
        ["artifact", "path", "description", "row_count", "file_size_bytes"],
    )

    return artifacts


def build_report(
    *,
    summary: dict[str, Any],
    run_config: dict[str, Any],
    output_dir: Path,
) -> str:
    type_rows = summary.get("type_summary", [])
    top_rows = summary.get("top_knowledge_units", [])

    return "\n".join(
        [
            "# Streaming Knowledge Unit Construction Report",
            "",
            "## 1. Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Raw mention edges | {summary.get('raw_mention_edge_count', 0)} |",
            f"| Filtered mention edges | {summary.get('filtered_mention_edge_count', 0)} |",
            f"| Raw carrier-KU edges | {summary.get('raw_carrier_ku_edge_count', 0)} |",
            f"| Filtered carrier-KU edges | {summary.get('filtered_carrier_ku_edge_count', 0)} |",
            f"| Knowledge Units | {summary.get('knowledge_unit_count', 0)} |",
            f"| Pair types | {summary.get('pair_type_count', 0)} |",
            f"| Carrier count | {summary.get('carrier_count', 0)} |",
            f"| Paper carriers | {summary.get('paper_carrier_count', 0)} |",
            f"| Patent carriers | {summary.get('patent_carrier_count', 0)} |",
            f"| Trial carriers | {summary.get('trial_carrier_count', 0)} |",
            f"| First year min | {summary.get('first_year_min')} |",
            f"| Last year max | {summary.get('last_year_max')} |",
            f"| Runtime seconds | {summary.get('runtime_seconds'):.1f} |",
            "",
            "## 2. Configuration",
            "",
            "```json",
            json.dumps(run_config, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 3. Pair type summary",
            "",
            markdown_table(
                type_rows,
                [
                    "pair_type",
                    "knowledge_unit_count",
                    "carrier_count_sum",
                    "paper_count_sum",
                    "patent_count_sum",
                    "trial_count_sum",
                    "total_pair_weight_sum",
                ],
            ),
            "",
            "## 4. Top Knowledge Units",
            "",
            markdown_table(
                top_rows,
                [
                    "ku_id",
                    "entity_a_name",
                    "entity_a_type",
                    "entity_b_name",
                    "entity_b_type",
                    "pair_type",
                    "carrier_count",
                    "paper_count",
                    "patent_count",
                    "trial_count",
                    "total_pair_weight",
                    "first_year",
                    "last_year",
                ],
            ),
            "",
            "## 5. Output files",
            "",
            "```text",
            str(output_dir),
            "```",
            "",
            "Core outputs:",
            "",
            "```text",
            "knowledge_units.parquet",
            "carrier_knowledge_unit_edges.parquet",
            "knowledge_unit_type_summary.csv",
            "knowledge_unit_summary.json",
            "run_config.json",
            "dataset_manifest.csv",
            "```",
            "",
            "## 6. Implementation note",
            "",
            "This run used DuckDB out-of-core SQL instead of pandas all-in-memory pair generation.",
            "",
        ]
    )


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    report_path = output_dir / "knowledge_unit_report.md"
    report_path.write_text(
        build_report(summary=summary, run_config=run_config, output_dir=output_dir),
        encoding="utf-8",
    )


def validate_args(args: argparse.Namespace) -> None:
    args.carrier_types = [normalize_token(v) for v in args.carrier_types]
    args.pair_types = [normalize_pair_type(v) for v in args.pair_types]
    args.exclude_bioentity_types = [
        normalize_entity_type_token(v) for v in args.exclude_bioentity_types
    ]

    for carrier_type in args.carrier_types:
        if carrier_type not in MENTION_EDGE_FILES:
            raise ValueError(
                f"Unsupported carrier type {carrier_type!r}. "
                f"Supported: {sorted(MENTION_EDGE_FILES)}"
            )

    for pair_type in args.pair_types:
        if pair_type not in SUPPORTED_PAIR_TYPES:
            raise ValueError(
                f"Unsupported pair type {pair_type!r}. "
                f"Supported: {sorted(SUPPORTED_PAIR_TYPES)}"
            )

    if args.pair_weight_method not in SUPPORTED_PAIR_WEIGHT_METHODS:
        raise ValueError(
            f"Unsupported pair weight method {args.pair_weight_method!r}. "
            f"Supported: {sorted(SUPPORTED_PAIR_WEIGHT_METHODS)}"
        )

    if args.threads < 1:
        raise ValueError("--threads must be >= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    start_time = time.time()

    args.graph_dir = args.graph_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()

    require_files(args.graph_dir, args.carrier_types)
    prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    run_config = build_run_config(args)
    write_json(args.output_dir / "run_config.json", run_config)

    connection = connect_duckdb(args)

    run_step(
        args,
        1,
        8,
        "Loading BioEntity metadata",
        lambda: create_bioentity_table(connection, args),
    )

    mention_summaries = run_step(
        args,
        2,
        8,
        "Creating filtered carrier-BioEntity mentions",
        lambda: create_filtered_mentions(connection, args),
    )

    ku_edge_summary = run_step(
        args,
        3,
        8,
        "Generating carrier-KU edges with DuckDB self-join",
        lambda: create_carrier_ku_edges(connection, args),
    )

    run_step(
        args,
        4,
        8,
        "Aggregating initial Knowledge Units",
        lambda: create_knowledge_units_all(connection),
    )

    run_step(
        args,
        5,
        8,
        "Applying Knowledge Unit support filters",
        lambda: apply_support_filter(connection, args),
    )

    run_step(
        args,
        6,
        8,
        "Creating pair-type summary",
        lambda: create_type_summary(connection),
    )

    runtime_seconds = time.time() - start_time

    summary = run_step(
        args,
        7,
        8,
        "Building JSON summary",
        lambda: summarize_build(
            connection,
            args,
            mention_summaries={
                **mention_summaries,
                **ku_edge_summary,
            },
            runtime_seconds=runtime_seconds,
        ),
    )

    def write_all_outputs() -> None:
        write_outputs(connection, args, summary, run_config)
        write_report(args.output_dir, summary, run_config)

    run_step(
        args,
        8,
        8,
        "Writing output artifacts",
        write_all_outputs,
    )

    print()
    print("Streaming Knowledge Unit construction complete.")
    print(f"Output directory: {args.output_dir}")
    print()
    print(f"Knowledge Units:          {summary.get('knowledge_unit_count'):,}")
    print(f"Carrier-KU edges:         {summary.get('filtered_carrier_ku_edge_count'):,}")
    print(f"Raw carrier-KU edges:     {summary.get('raw_carrier_ku_edge_count'):,}")
    print(f"Carrier count:            {summary.get('carrier_count'):,}")
    print(f"Paper carriers:           {summary.get('paper_carrier_count'):,}")
    print(f"Patent carriers:          {summary.get('patent_carrier_count'):,}")
    print(f"Trial carriers:           {summary.get('trial_carrier_count'):,}")
    print()
    print("Output files:")
    print(f"  {args.output_dir / 'knowledge_units.parquet'}")
    print(f"  {args.output_dir / 'carrier_knowledge_unit_edges.parquet'}")
    print(f"  {args.output_dir / 'knowledge_unit_type_summary.csv'}")
    print(f"  {args.output_dir / 'knowledge_unit_summary.json'}")
    print(f"  {args.output_dir / 'knowledge_unit_report.md'}")
    print(f"  {args.output_dir / 'dataset_manifest.csv'}")
    print()


if __name__ == "__main__":
    main()