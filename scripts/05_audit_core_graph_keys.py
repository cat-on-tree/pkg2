"""Audit join-key quality before extracting a patent-centered PKG24S4 subgraph.

This script audits the core node keys, edge endpoints, foreign-key coverage,
ProjectNumber join behavior, and BioEntity unmatched/CUI-less behavior for the
planned patent-centered biomedical translation graph.

It does not extract a graph, does not read raw data, and does not modify Parquet
files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


NODE_AUDIT_COLUMNS = [
    "node_type",
    "table_name",
    "key_column",
    "row_count",
    "non_null_count",
    "null_count",
    "null_fraction",
    "distinct_count",
    "duplicate_key_count",
    "exact_distinct_used",
    "sample_values_json",
    "warning",
    "status",
    "error_message",
]

EDGE_AUDIT_COLUMNS = [
    "edge_type",
    "table_name",
    "source_column",
    "target_column",
    "row_count",
    "source_non_null_count",
    "source_null_count",
    "source_null_fraction",
    "target_non_null_count",
    "target_null_count",
    "target_null_fraction",
    "distinct_source_count",
    "distinct_target_count",
    "duplicate_edge_count",
    "exact_duplicate_edges_used",
    "sample_edges_json",
    "warning",
    "status",
    "error_message",
]

FK_COVERAGE_COLUMNS = [
    "edge_type",
    "edge_table",
    "edge_column",
    "node_type",
    "node_table",
    "node_key_column",
    "edge_row_count",
    "edge_non_null_count",
    "matched_row_count",
    "unmatched_row_count",
    "row_match_rate",
    "distinct_edge_values",
    "matched_distinct_values",
    "unmatched_distinct_values",
    "distinct_match_rate",
    "coverage_mode",
    "sample_unmatched_values_json",
    "warning",
    "status",
    "error_message",
]

PROJECT_AUDIT_COLUMNS = [
    "link_table",
    "link_column",
    "project_table",
    "project_column",
    "link_row_count",
    "link_non_null_count",
    "matched_row_count",
    "row_match_rate",
    "distinct_link_values",
    "matched_distinct_values",
    "distinct_match_rate",
    "sample_link_values_json",
    "sample_unmatched_values_json",
    "normalization",
    "warning",
    "status",
    "error_message",
]

BIOENTITY_UNMATCHED_COLUMNS = [
    "source_table",
    "entity_type",
    "row_count",
    "approx_distinct_entity_ids",
    "mesh_non_null",
    "mim_non_null",
    "cl_non_null",
    "cellosaurus_non_null",
    "ncbi_taxon_non_null",
    "ncbi_gene_non_null",
    "chebi_non_null",
    "sample_entity_id",
    "sample_mention",
    "warning",
    "status",
    "error_message",
]

BIOENTITY_UNMATCHED_ENTITYID_COLUMNS = [
    "source_table",
    "entity_id",
    "row_count",
    "approx_distinct_mentions",
    "type_values_json",
    "sample_mentions_json",
    "recommendation",
    "warning",
    "status",
    "error_message",
]


NODES = [
    {"node_type": "Paper", "table_name": "C01_Papers", "key_column": "PMID"},
    {"node_type": "BioEntity", "table_name": "C23_BioEntities", "key_column": "EntityId"},
    {"node_type": "Project", "table_name": "B02_Projects", "key_column": "FULL_PROJECT_NUM"},
    {"node_type": "Project", "table_name": "B02_Projects", "key_column": "CORE_PROJECT_NUM"},
    {"node_type": "Project", "table_name": "B02_Projects", "key_column": "APPLICATION_ID"},
    {"node_type": "Project", "table_name": "B02_Projects", "key_column": "SUBPROJECT_ID"},
    {"node_type": "ClinicalTrial", "table_name": "C11_ClinicalTrials", "key_column": "nct_id"},
    {"node_type": "Patent", "table_name": "C15_Patents", "key_column": "PatentId"},
]

EDGES = [
    {
        "edge_type": "paper_mentions_bioentity",
        "table_name": "C06_Link_Papers_BioEntities",
        "source_column": "PMID",
        "target_column": "EntityId",
    },
    {
        "edge_type": "trial_mentions_bioentity",
        "table_name": "C13_Link_ClinicalTrials_BioEntities",
        "source_column": "nct_id",
        "target_column": "EntityId",
    },
    {
        "edge_type": "patent_mentions_bioentity",
        "table_name": "C18_Link_Patents_BioEntities",
        "source_column": "PatentId",
        "target_column": "EntityId",
    },
    {
        "edge_type": "paper_linked_trial",
        "table_name": "C12_Link_Papers_Clinicaltrials",
        "source_column": "PMID",
        "target_column": "nct_id",
    },
    {
        "edge_type": "patent_links_paper",
        "table_name": "C16_Link_Patents_Papers",
        "source_column": "PatentId",
        "target_column": "PMID",
    },
    {
        "edge_type": "paper_linked_project",
        "table_name": "B03_Link_Papers_Projects",
        "source_column": "PMID",
        "target_column": "ProjectNumber",
    },
    {
        "edge_type": "trial_linked_project",
        "table_name": "B04_Link_ClinicalTrials_Projects",
        "source_column": "nct_id",
        "target_column": "ProjectNumber",
    },
    {
        "edge_type": "patent_linked_project",
        "table_name": "B05_Link_Patents_Projects",
        "source_column": "PatentId",
        "target_column": "ProjectNumber",
    },
]

FK_CHECKS = [
    ("paper_mentions_bioentity", "C06_Link_Papers_BioEntities", "PMID", "Paper", "C01_Papers", "PMID"),
    ("paper_mentions_bioentity", "C06_Link_Papers_BioEntities", "EntityId", "BioEntity", "C23_BioEntities", "EntityId"),
    ("trial_mentions_bioentity", "C13_Link_ClinicalTrials_BioEntities", "nct_id", "ClinicalTrial", "C11_ClinicalTrials", "nct_id"),
    ("trial_mentions_bioentity", "C13_Link_ClinicalTrials_BioEntities", "EntityId", "BioEntity", "C23_BioEntities", "EntityId"),
    ("patent_mentions_bioentity", "C18_Link_Patents_BioEntities", "PatentId", "Patent", "C15_Patents", "PatentId"),
    ("patent_mentions_bioentity", "C18_Link_Patents_BioEntities", "EntityId", "BioEntity", "C23_BioEntities", "EntityId"),
    ("paper_linked_trial", "C12_Link_Papers_Clinicaltrials", "PMID", "Paper", "C01_Papers", "PMID"),
    ("paper_linked_trial", "C12_Link_Papers_Clinicaltrials", "nct_id", "ClinicalTrial", "C11_ClinicalTrials", "nct_id"),
    ("patent_links_paper", "C16_Link_Patents_Papers", "PatentId", "Patent", "C15_Patents", "PatentId"),
    ("patent_links_paper", "C16_Link_Patents_Papers", "PMID", "Paper", "C01_Papers", "PMID"),
    ("paper_linked_project", "B03_Link_Papers_Projects", "PMID", "Paper", "C01_Papers", "PMID"),
    ("trial_linked_project", "B04_Link_ClinicalTrials_Projects", "nct_id", "ClinicalTrial", "C11_ClinicalTrials", "nct_id"),
    ("patent_linked_project", "B05_Link_Patents_Projects", "PatentId", "Patent", "C15_Patents", "PatentId"),
]

PROJECT_LINKS = [
    ("B03_Link_Papers_Projects", "ProjectNumber"),
    ("B04_Link_ClinicalTrials_Projects", "ProjectNumber"),
    ("B05_Link_Patents_Projects", "ProjectNumber"),
]

PROJECT_COLUMNS = ["FULL_PROJECT_NUM", "CORE_PROJECT_NUM", "APPLICATION_ID", "SUBPROJECT_ID"]

NORMALIZATIONS = ["raw_trim_upper", "remove_spaces", "remove_hyphen_space"]

NORMALIZATION_PRIORITY = {
    "raw_trim_upper": 0,
    "remove_spaces": 1,
    "remove_hyphen_space": 2,
}

BIOENTITY_LINK_TABLES = [
    "C06_Link_Papers_BioEntities",
    "C13_Link_ClinicalTrials_BioEntities",
    "C18_Link_Patents_BioEntities",
]

BIOENTITY_LINK_COLUMNS = [
    "EntityId",
    "Type",
    "Mention",
    "mesh",
    "mim",
    "CL",
    "cellosaurus",
    "NCBITaxon",
    "NCBIGene",
    "CHEBI",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PKG24S4 core graph join-key quality.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/interim/parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/profile"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="20GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("data/interim/duckdb_tmp"))
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--exact-distinct", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--exact-duplicate-edges", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--coverage-mode", choices=("row", "distinct", "both"), default="row")
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def qident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_path(parquet_dir: Path, table_name: str) -> Path:
    return parquet_dir / f"{table_name}.parquet"


def scan_sql(path: Path) -> str:
    return f"read_parquet({sql_literal(path)})"


def key_expr(column: str, alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    return f"NULLIF(TRIM(CAST({prefix}{qident(column)} AS VARCHAR)), '')"


def normalized_expr(column: str, normalization: str, alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    base = f"UPPER(TRIM(CAST({prefix}{qident(column)} AS VARCHAR)))"
    if normalization == "raw_trim_upper":
        return f"NULLIF({base}, '')"
    if normalization == "remove_spaces":
        return f"NULLIF(REPLACE({base}, ' ', ''), '')"
    if normalization == "remove_hyphen_space":
        return f"NULLIF(REPLACE(REPLACE({base}, ' ', ''), '-', ''), '')"
    raise ValueError(f"Unsupported normalization: {normalization}")


def json_dumps(values: Any) -> str:
    return json.dumps(values, ensure_ascii=False)


def to_int(value: Any) -> int:
    if value is None or value == "" or value == "not_computed":
        return 0
    return int(float(value))


def to_float(value: Any) -> float:
    if value is None or value == "" or value == "not_computed":
        return 0.0
    return float(value)


def rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{numerator / denominator:.6f}"


def is_cuiless(value: Any) -> bool:
    return str(value or "").strip().upper() == "CUI-LESS"


def connect_duckdb(args: argparse.Namespace):
    import duckdb

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute(f"SET temp_directory={sql_literal(args.temp_dir)}")
    return connection


def get_columns(connection: Any, path: Path) -> set[str]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {scan_sql(path)}").fetchall()
    return {str(row[0]) for row in rows}


def validate_table_and_columns(
    connection: Any,
    parquet_dir: Path,
    table_name: str,
    columns: list[str],
) -> tuple[Path, str]:
    path = table_path(parquet_dir, table_name)
    if not path.exists():
        return path, f"missing_table:{path}"
    try:
        available = get_columns(connection, path)
    except Exception as error:
        return path, f"unreadable_table:{error}"
    missing = [column for column in columns if column not in available]
    if missing:
        return path, "missing_columns:" + ",".join(missing)
    return path, ""


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sample_values_sql(path: Path, column: str, sample_limit: int) -> str:
    expr = key_expr(column)
    return (
        f"SELECT {expr} AS value FROM {scan_sql(path)} "
        f"WHERE {expr} IS NOT NULL LIMIT {sample_limit}"
    )


def audit_node_key(
    connection: Any,
    parquet_dir: Path,
    node: dict[str, str],
    exact_distinct: bool,
    sample_limit: int,
) -> dict[str, Any]:
    base = {
        "node_type": node["node_type"],
        "table_name": node["table_name"],
        "key_column": node["key_column"],
        "row_count": "",
        "non_null_count": "",
        "null_count": "",
        "null_fraction": "",
        "distinct_count": "",
        "duplicate_key_count": "not_computed" if not exact_distinct else "",
        "exact_distinct_used": str(exact_distinct),
        "sample_values_json": "[]",
        "warning": "",
        "status": "ok",
        "error_message": "",
    }

    path, error = validate_table_and_columns(
        connection,
        parquet_dir,
        node["table_name"],
        [node["key_column"]],
    )
    if error:
        base.update({"status": "error", "warning": error, "error_message": error})
        return base

    expr = key_expr(node["key_column"])
    distinct_func = "COUNT(DISTINCT key)" if exact_distinct else "APPROX_COUNT_DISTINCT(key)"
    query = f"""
        WITH normalized AS (
            SELECT {expr} AS key
            FROM {scan_sql(path)}
        )
        SELECT
            COUNT(*) AS row_count,
            COUNT(key) AS non_null_count,
            COUNT(*) - COUNT(key) AS null_count,
            {distinct_func} AS distinct_count
        FROM normalized
    """

    try:
        row_count, non_null_count, null_count, distinct_count = connection.execute(query).fetchone()
        samples = [
            row[0]
            for row in connection.execute(
                sample_values_sql(path, node["key_column"], sample_limit)
            ).fetchall()
        ]
    except Exception as error:
        base.update({"status": "error", "warning": "query_failed", "error_message": str(error)})
        return base

    duplicate_key_count = (
        str(to_int(non_null_count) - to_int(distinct_count)) if exact_distinct else "not_computed"
    )
    null_fraction = to_int(null_count) / to_int(row_count) if to_int(row_count) else 0.0

    warnings = []
    if to_int(non_null_count) == 0:
        warnings.append("all_key_values_missing")
    if exact_distinct and to_int(duplicate_key_count) > 0:
        warnings.append("duplicate_keys")
    if null_fraction > 0:
        warnings.append("key_has_missing_values")

    base.update(
        {
            "row_count": row_count,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_fraction": f"{null_fraction:.6f}",
            "distinct_count": distinct_count,
            "duplicate_key_count": duplicate_key_count,
            "sample_values_json": json_dumps(samples),
            "warning": ";".join(warnings),
        }
    )
    return base


def audit_edge_endpoint(
    connection: Any,
    parquet_dir: Path,
    edge: dict[str, str],
    exact_duplicate_edges: bool,
    sample_limit: int,
) -> dict[str, Any]:
    base = {
        "edge_type": edge["edge_type"],
        "table_name": edge["table_name"],
        "source_column": edge["source_column"],
        "target_column": edge["target_column"],
        "row_count": "",
        "source_non_null_count": "",
        "source_null_count": "",
        "source_null_fraction": "",
        "target_non_null_count": "",
        "target_null_count": "",
        "target_null_fraction": "",
        "distinct_source_count": "",
        "distinct_target_count": "",
        "duplicate_edge_count": "not_computed" if not exact_duplicate_edges else "",
        "exact_duplicate_edges_used": str(exact_duplicate_edges),
        "sample_edges_json": "[]",
        "warning": "",
        "status": "ok",
        "error_message": "",
    }

    path, error = validate_table_and_columns(
        connection,
        parquet_dir,
        edge["table_name"],
        [edge["source_column"], edge["target_column"]],
    )
    if error:
        base.update({"status": "error", "warning": error, "error_message": error})
        return base

    source_expr = key_expr(edge["source_column"])
    target_expr = key_expr(edge["target_column"])

    query = f"""
        WITH normalized AS (
            SELECT {source_expr} AS source_key, {target_expr} AS target_key
            FROM {scan_sql(path)}
        )
        SELECT
            COUNT(*) AS row_count,
            COUNT(source_key) AS source_non_null_count,
            COUNT(*) - COUNT(source_key) AS source_null_count,
            COUNT(target_key) AS target_non_null_count,
            COUNT(*) - COUNT(target_key) AS target_null_count,
            APPROX_COUNT_DISTINCT(source_key) AS distinct_source_count,
            APPROX_COUNT_DISTINCT(target_key) AS distinct_target_count
        FROM normalized
    """

    try:
        (
            row_count,
            source_non_null,
            source_null,
            target_non_null,
            target_null,
            distinct_source,
            distinct_target,
        ) = connection.execute(query).fetchone()

        sample_query = f"""
            WITH normalized AS (
                SELECT {source_expr} AS source_key, {target_expr} AS target_key
                FROM {scan_sql(path)}
            )
            SELECT source_key, target_key
            FROM normalized
            WHERE source_key IS NOT NULL OR target_key IS NOT NULL
            LIMIT {sample_limit}
        """
        samples = [
            {"source": row[0], "target": row[1]}
            for row in connection.execute(sample_query).fetchall()
        ]

        duplicate_edge_count = "not_computed"
        if exact_duplicate_edges:
            duplicate_query = f"""
                WITH normalized AS (
                    SELECT {source_expr} AS source_key, {target_expr} AS target_key
                    FROM {scan_sql(path)}
                ),
                valid_pairs AS (
                    SELECT source_key, target_key
                    FROM normalized
                    WHERE source_key IS NOT NULL AND target_key IS NOT NULL
                ),
                distinct_pairs AS (
                    SELECT DISTINCT source_key, target_key
                    FROM valid_pairs
                )
                SELECT
                    (SELECT COUNT(*) FROM valid_pairs)
                    - (SELECT COUNT(*) FROM distinct_pairs)
            """
            duplicate_edge_count = str(connection.execute(duplicate_query).fetchone()[0])
    except Exception as error:
        base.update({"status": "error", "warning": "query_failed", "error_message": str(error)})
        return base

    source_null_fraction = to_int(source_null) / to_int(row_count) if to_int(row_count) else 0.0
    target_null_fraction = to_int(target_null) / to_int(row_count) if to_int(row_count) else 0.0

    warnings = []
    if to_int(source_non_null) == 0:
        warnings.append("all_source_values_missing")
    if to_int(target_non_null) == 0:
        warnings.append("all_target_values_missing")
    if source_null_fraction > 0:
        warnings.append("source_has_missing_values")
    if target_null_fraction > 0:
        warnings.append("target_has_missing_values")
    if exact_duplicate_edges and to_int(duplicate_edge_count) > 0:
        warnings.append("duplicate_edges")

    base.update(
        {
            "row_count": row_count,
            "source_non_null_count": source_non_null,
            "source_null_count": source_null,
            "source_null_fraction": f"{source_null_fraction:.6f}",
            "target_non_null_count": target_non_null,
            "target_null_count": target_null,
            "target_null_fraction": f"{target_null_fraction:.6f}",
            "distinct_source_count": distinct_source,
            "distinct_target_count": distinct_target,
            "duplicate_edge_count": duplicate_edge_count,
            "sample_edges_json": json_dumps(samples),
            "warning": ";".join(warnings),
        }
    )
    return base


def audit_fk_coverage(
    connection: Any,
    parquet_dir: Path,
    check: tuple[str, str, str, str, str, str],
    coverage_mode: str,
    sample_limit: int,
) -> dict[str, Any]:
    edge_type, edge_table, edge_column, node_type, node_table, node_key_column = check
    base = {
        "edge_type": edge_type,
        "edge_table": edge_table,
        "edge_column": edge_column,
        "node_type": node_type,
        "node_table": node_table,
        "node_key_column": node_key_column,
        "edge_row_count": "",
        "edge_non_null_count": "",
        "matched_row_count": "not_computed" if coverage_mode == "distinct" else "",
        "unmatched_row_count": "not_computed" if coverage_mode == "distinct" else "",
        "row_match_rate": "not_computed" if coverage_mode == "distinct" else "",
        "distinct_edge_values": "not_computed" if coverage_mode == "row" else "",
        "matched_distinct_values": "not_computed" if coverage_mode == "row" else "",
        "unmatched_distinct_values": "not_computed" if coverage_mode == "row" else "",
        "distinct_match_rate": "not_computed" if coverage_mode == "row" else "",
        "coverage_mode": coverage_mode,
        "sample_unmatched_values_json": "[]",
        "warning": "",
        "status": "ok",
        "error_message": "",
    }

    edge_path, edge_error = validate_table_and_columns(connection, parquet_dir, edge_table, [edge_column])
    node_path, node_error = validate_table_and_columns(connection, parquet_dir, node_table, [node_key_column])
    if edge_error or node_error:
        error = ";".join(value for value in [edge_error, node_error] if value)
        base.update({"status": "error", "warning": error, "error_message": error})
        return base

    edge_expr = key_expr(edge_column)
    node_expr = key_expr(node_key_column)
    warnings = []

    try:
        if coverage_mode in {"row", "both"}:
            row_query = f"""
                WITH edge_rows AS (
                    SELECT {edge_expr} AS key FROM {scan_sql(edge_path)}
                ),
                node_keys AS (
                    SELECT DISTINCT {node_expr} AS key
                    FROM {scan_sql(node_path)}
                    WHERE {node_expr} IS NOT NULL
                )
                SELECT
                    (SELECT COUNT(*) FROM edge_rows) AS edge_row_count,
                    COUNT(edge_rows.key) AS edge_non_null_count,
                    SUM(CASE WHEN edge_rows.key IS NOT NULL AND node_keys.key IS NOT NULL THEN 1 ELSE 0 END)
                        AS matched_row_count
                FROM edge_rows
                LEFT JOIN node_keys ON edge_rows.key = node_keys.key
            """
            edge_row_count, edge_non_null, matched_rows = connection.execute(row_query).fetchone()
            unmatched_rows = to_int(edge_non_null) - to_int(matched_rows)
            row_match_rate = rate(to_int(matched_rows), to_int(edge_non_null))

            base.update(
                {
                    "edge_row_count": edge_row_count,
                    "edge_non_null_count": edge_non_null,
                    "matched_row_count": matched_rows,
                    "unmatched_row_count": unmatched_rows,
                    "row_match_rate": row_match_rate,
                }
            )

            if to_int(edge_non_null) == 0:
                warnings.append("all_edge_values_missing")
            elif to_float(row_match_rate) < 0.95:
                warnings.append("coverage_below_0.95")

        if coverage_mode in {"distinct", "both"}:
            distinct_query = f"""
                WITH edge_values AS (
                    SELECT DISTINCT {edge_expr} AS key
                    FROM {scan_sql(edge_path)}
                    WHERE {edge_expr} IS NOT NULL
                ),
                node_keys AS (
                    SELECT DISTINCT {node_expr} AS key
                    FROM {scan_sql(node_path)}
                    WHERE {node_expr} IS NOT NULL
                )
                SELECT
                    COUNT(edge_values.key) AS distinct_edge_values,
                    SUM(CASE WHEN node_keys.key IS NOT NULL THEN 1 ELSE 0 END) AS matched_distinct_values
                FROM edge_values
                LEFT JOIN node_keys ON edge_values.key = node_keys.key
            """
            distinct_edge, matched_distinct = connection.execute(distinct_query).fetchone()
            unmatched_distinct = to_int(distinct_edge) - to_int(matched_distinct)
            distinct_match_rate = rate(to_int(matched_distinct), to_int(distinct_edge))

            base.update(
                {
                    "distinct_edge_values": distinct_edge,
                    "matched_distinct_values": matched_distinct,
                    "unmatched_distinct_values": unmatched_distinct,
                    "distinct_match_rate": distinct_match_rate,
                }
            )

            if to_int(distinct_edge) == 0:
                warnings.append("all_edge_values_missing")
            elif to_float(distinct_match_rate) < 0.95:
                warnings.append("coverage_below_0.95")

        if coverage_mode == "row":
            sample_query = f"""
                WITH edge_rows AS (
                    SELECT {edge_expr} AS key
                    FROM {scan_sql(edge_path)}
                    WHERE {edge_expr} IS NOT NULL
                ),
                node_keys AS (
                    SELECT DISTINCT {node_expr} AS key
                    FROM {scan_sql(node_path)}
                    WHERE {node_expr} IS NOT NULL
                )
                SELECT edge_rows.key
                FROM edge_rows
                LEFT JOIN node_keys ON edge_rows.key = node_keys.key
                WHERE node_keys.key IS NULL
                LIMIT {sample_limit}
            """
        else:
            sample_query = f"""
                WITH edge_values AS (
                    SELECT DISTINCT {edge_expr} AS key
                    FROM {scan_sql(edge_path)}
                    WHERE {edge_expr} IS NOT NULL
                ),
                node_keys AS (
                    SELECT DISTINCT {node_expr} AS key
                    FROM {scan_sql(node_path)}
                    WHERE {node_expr} IS NOT NULL
                )
                SELECT edge_values.key
                FROM edge_values
                LEFT JOIN node_keys ON edge_values.key = node_keys.key
                WHERE node_keys.key IS NULL
                LIMIT {sample_limit}
            """

        samples = [row[0] for row in connection.execute(sample_query).fetchall()]
    except Exception as error:
        base.update({"status": "error", "warning": "query_failed", "error_message": str(error)})
        return base

    base.update(
        {
            "sample_unmatched_values_json": json_dumps(samples),
            "warning": ";".join(sorted(set(warnings))),
        }
    )
    return base


def audit_project_number_join(
    connection: Any,
    parquet_dir: Path,
    link_table: str,
    link_column: str,
    project_column: str,
    normalization: str,
    sample_limit: int,
) -> dict[str, Any]:
    base = {
        "link_table": link_table,
        "link_column": link_column,
        "project_table": "B02_Projects",
        "project_column": project_column,
        "link_row_count": "",
        "link_non_null_count": "",
        "matched_row_count": "",
        "row_match_rate": "",
        "distinct_link_values": "",
        "matched_distinct_values": "",
        "distinct_match_rate": "",
        "sample_link_values_json": "[]",
        "sample_unmatched_values_json": "[]",
        "normalization": normalization,
        "warning": "",
        "status": "ok",
        "error_message": "",
    }

    link_path, link_error = validate_table_and_columns(connection, parquet_dir, link_table, [link_column])
    project_path, project_error = validate_table_and_columns(
        connection,
        parquet_dir,
        "B02_Projects",
        [project_column],
    )

    if link_error or project_error:
        error = ";".join(value for value in [link_error, project_error] if value)
        base.update({"status": "error", "warning": error, "error_message": error})
        return base

    link_expr = normalized_expr(link_column, normalization)
    project_expr = normalized_expr(project_column, normalization)

    try:
        row_query = f"""
            WITH link_rows AS (
                SELECT {link_expr} AS key FROM {scan_sql(link_path)}
            ),
            project_keys AS (
                SELECT DISTINCT {project_expr} AS key
                FROM {scan_sql(project_path)}
                WHERE {project_expr} IS NOT NULL
            )
            SELECT
                (SELECT COUNT(*) FROM link_rows) AS link_row_count,
                COUNT(link_rows.key) AS link_non_null_count,
                SUM(CASE WHEN link_rows.key IS NOT NULL AND project_keys.key IS NOT NULL THEN 1 ELSE 0 END)
                    AS matched_row_count
            FROM link_rows
            LEFT JOIN project_keys ON link_rows.key = project_keys.key
        """
        link_row_count, link_non_null, matched_rows = connection.execute(row_query).fetchone()

        distinct_query = f"""
            WITH link_values AS (
                SELECT DISTINCT {link_expr} AS key
                FROM {scan_sql(link_path)}
                WHERE {link_expr} IS NOT NULL
            ),
            project_keys AS (
                SELECT DISTINCT {project_expr} AS key
                FROM {scan_sql(project_path)}
                WHERE {project_expr} IS NOT NULL
            )
            SELECT
                COUNT(link_values.key) AS distinct_link_values,
                SUM(CASE WHEN project_keys.key IS NOT NULL THEN 1 ELSE 0 END) AS matched_distinct_values
            FROM link_values
            LEFT JOIN project_keys ON link_values.key = project_keys.key
        """
        distinct_link_values, matched_distinct = connection.execute(distinct_query).fetchone()

        sample_link_query = f"""
            SELECT DISTINCT {link_expr} AS key
            FROM {scan_sql(link_path)}
            WHERE {link_expr} IS NOT NULL
            LIMIT {sample_limit}
        """
        sample_unmatched_query = f"""
            WITH link_values AS (
                SELECT DISTINCT {link_expr} AS key
                FROM {scan_sql(link_path)}
                WHERE {link_expr} IS NOT NULL
            ),
            project_keys AS (
                SELECT DISTINCT {project_expr} AS key
                FROM {scan_sql(project_path)}
                WHERE {project_expr} IS NOT NULL
            )
            SELECT link_values.key
            FROM link_values
            LEFT JOIN project_keys ON link_values.key = project_keys.key
            WHERE project_keys.key IS NULL
            LIMIT {sample_limit}
        """

        sample_link = [row[0] for row in connection.execute(sample_link_query).fetchall()]
        sample_unmatched = [row[0] for row in connection.execute(sample_unmatched_query).fetchall()]
    except Exception as error:
        base.update({"status": "error", "warning": "query_failed", "error_message": str(error)})
        return base

    row_match_rate = rate(to_int(matched_rows), to_int(link_non_null))
    distinct_match_rate = rate(to_int(matched_distinct), to_int(distinct_link_values))

    warnings = []
    if to_int(link_non_null) == 0:
        warnings.append("all_link_values_missing")
    if to_float(row_match_rate) < 0.95 and to_float(distinct_match_rate) < 0.95:
        warnings.append("no_good_project_key_match")

    base.update(
        {
            "link_row_count": link_row_count,
            "link_non_null_count": link_non_null,
            "matched_row_count": matched_rows,
            "row_match_rate": row_match_rate,
            "distinct_link_values": distinct_link_values,
            "matched_distinct_values": matched_distinct,
            "distinct_match_rate": distinct_match_rate,
            "sample_link_values_json": json_dumps(sample_link),
            "sample_unmatched_values_json": json_dumps(sample_unmatched),
            "warning": ";".join(warnings),
        }
    )
    return base


def bioentity_error_row(error: str) -> dict[str, Any]:
    return {
        "source_table": "",
        "entity_type": "",
        "row_count": "",
        "approx_distinct_entity_ids": "",
        "mesh_non_null": "",
        "mim_non_null": "",
        "cl_non_null": "",
        "cellosaurus_non_null": "",
        "ncbi_taxon_non_null": "",
        "ncbi_gene_non_null": "",
        "chebi_non_null": "",
        "sample_entity_id": "",
        "sample_mention": "",
        "warning": error,
        "status": "error",
        "error_message": error,
    }


def bioentity_entityid_error_row(error: str) -> dict[str, Any]:
    return {
        "source_table": "",
        "entity_id": "",
        "row_count": "",
        "approx_distinct_mentions": "",
        "type_values_json": "[]",
        "sample_mentions_json": "[]",
        "recommendation": "",
        "warning": error,
        "status": "error",
        "error_message": error,
    }


def validate_bioentity_inputs(connection: Any, parquet_dir: Path) -> tuple[Path | None, dict[str, Path], str]:
    c23_path, c23_error = validate_table_and_columns(
        connection,
        parquet_dir,
        "C23_BioEntities",
        ["EntityId"],
    )
    if c23_error:
        return None, {}, c23_error

    link_paths: dict[str, Path] = {}
    errors = []
    for table_name in BIOENTITY_LINK_TABLES:
        path, error = validate_table_and_columns(
            connection,
            parquet_dir,
            table_name,
            BIOENTITY_LINK_COLUMNS,
        )
        if error:
            errors.append(f"{table_name}:{error}")
        else:
            link_paths[table_name] = path

    if errors:
        return c23_path, link_paths, ";".join(errors)

    return c23_path, link_paths, ""


def build_bioentity_links_union_sql(link_paths: dict[str, Path]) -> str:
    def link_select(table_name: str) -> str:
        path = link_paths[table_name]
        return f"""
            SELECT
                {sql_literal(table_name)} AS source_table,
                {key_expr("EntityId")} AS EntityId,
                CAST({qident("Type")} AS VARCHAR) AS entity_type,
                CAST({qident("Mention")} AS VARCHAR) AS Mention,
                {key_expr("mesh")} AS mesh,
                {key_expr("mim")} AS mim,
                {key_expr("CL")} AS CL,
                {key_expr("cellosaurus")} AS cellosaurus,
                {key_expr("NCBITaxon")} AS NCBITaxon,
                {key_expr("NCBIGene")} AS NCBIGene,
                {key_expr("CHEBI")} AS CHEBI
            FROM {scan_sql(path)}
            WHERE {key_expr("EntityId")} IS NOT NULL
        """

    return "\nUNION ALL\n".join(link_select(table_name) for table_name in BIOENTITY_LINK_TABLES)


def audit_bioentity_unmatched(connection: Any, parquet_dir: Path) -> list[dict[str, Any]]:
    c23_path, link_paths, error = validate_bioentity_inputs(connection, parquet_dir)
    if error:
        return [bioentity_error_row(error)]
    assert c23_path is not None

    union_sql = build_bioentity_links_union_sql(link_paths)

    query = f"""
        WITH c23 AS (
            SELECT DISTINCT {key_expr("EntityId")} AS EntityId
            FROM {scan_sql(c23_path)}
            WHERE {key_expr("EntityId")} IS NOT NULL
        ),
        links AS (
            {union_sql}
        ),
        unmatched AS (
            SELECT links.*
            FROM links
            LEFT JOIN c23 USING (EntityId)
            WHERE c23.EntityId IS NULL
        )
        SELECT
            source_table,
            COALESCE(entity_type, '') AS entity_type,
            COUNT(*) AS row_count,
            APPROX_COUNT_DISTINCT(EntityId) AS approx_distinct_entity_ids,
            SUM(CASE WHEN mesh IS NOT NULL THEN 1 ELSE 0 END) AS mesh_non_null,
            SUM(CASE WHEN mim IS NOT NULL THEN 1 ELSE 0 END) AS mim_non_null,
            SUM(CASE WHEN CL IS NOT NULL THEN 1 ELSE 0 END) AS cl_non_null,
            SUM(CASE WHEN cellosaurus IS NOT NULL THEN 1 ELSE 0 END) AS cellosaurus_non_null,
            SUM(CASE WHEN NCBITaxon IS NOT NULL THEN 1 ELSE 0 END) AS ncbi_taxon_non_null,
            SUM(CASE WHEN NCBIGene IS NOT NULL THEN 1 ELSE 0 END) AS ncbi_gene_non_null,
            SUM(CASE WHEN CHEBI IS NOT NULL THEN 1 ELSE 0 END) AS chebi_non_null,
            ANY_VALUE(EntityId) AS sample_entity_id,
            ANY_VALUE(Mention) AS sample_mention
        FROM unmatched
        GROUP BY source_table, COALESCE(entity_type, '')
        ORDER BY row_count DESC
    """

    try:
        result = connection.execute(query)
        columns = [description[0] for description in result.description]
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    except Exception as error:
        return [bioentity_error_row(f"query_failed:{error}")]

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        if is_cuiless(row.get("sample_entity_id")):
            warning = "cuiless_unmatched_mentions"
        elif to_int(row.get("row_count")) > 0:
            warning = "non_cuiless_entity_ids_missing_from_C23"
        else:
            warning = ""

        output_rows.append(
            {
                "source_table": row.get("source_table", ""),
                "entity_type": row.get("entity_type", ""),
                "row_count": row.get("row_count", ""),
                "approx_distinct_entity_ids": row.get("approx_distinct_entity_ids", ""),
                "mesh_non_null": row.get("mesh_non_null", ""),
                "mim_non_null": row.get("mim_non_null", ""),
                "cl_non_null": row.get("cl_non_null", ""),
                "cellosaurus_non_null": row.get("cellosaurus_non_null", ""),
                "ncbi_taxon_non_null": row.get("ncbi_taxon_non_null", ""),
                "ncbi_gene_non_null": row.get("ncbi_gene_non_null", ""),
                "chebi_non_null": row.get("chebi_non_null", ""),
                "sample_entity_id": row.get("sample_entity_id", ""),
                "sample_mention": row.get("sample_mention", ""),
                "warning": warning,
                "status": "ok",
                "error_message": "",
            }
        )

    return output_rows


def audit_bioentity_unmatched_entityids(connection: Any, parquet_dir: Path, sample_limit: int) -> list[dict[str, Any]]:
    c23_path, link_paths, error = validate_bioentity_inputs(connection, parquet_dir)
    if error:
        return [bioentity_entityid_error_row(error)]
    assert c23_path is not None

    union_sql = build_bioentity_links_union_sql(link_paths)

    query = f"""
        WITH c23 AS (
            SELECT DISTINCT {key_expr("EntityId")} AS EntityId
            FROM {scan_sql(c23_path)}
            WHERE {key_expr("EntityId")} IS NOT NULL
        ),
        links AS (
            {union_sql}
        ),
        unmatched AS (
            SELECT links.*
            FROM links
            LEFT JOIN c23 USING (EntityId)
            WHERE c23.EntityId IS NULL
        ),
        grouped AS (
            SELECT
                source_table,
                EntityId AS entity_id,
                COUNT(*) AS row_count,
                APPROX_COUNT_DISTINCT(Mention) AS approx_distinct_mentions
            FROM unmatched
            GROUP BY source_table, EntityId
        )
        SELECT
            source_table,
            entity_id,
            row_count,
            approx_distinct_mentions
        FROM grouped
        ORDER BY row_count DESC
    """

    try:
        rows = connection.execute(query).fetchall()
    except Exception as error:
        return [bioentity_entityid_error_row(f"query_failed:{error}")]

    output_rows: list[dict[str, Any]] = []
    for source_table, entity_id, row_count, approx_distinct_mentions in rows:
        sample_query = f"""
            WITH c23 AS (
                SELECT DISTINCT {key_expr("EntityId")} AS EntityId
                FROM {scan_sql(c23_path)}
                WHERE {key_expr("EntityId")} IS NOT NULL
            ),
            links AS (
                {union_sql}
            ),
            unmatched AS (
                SELECT links.*
                FROM links
                LEFT JOIN c23 USING (EntityId)
                WHERE c23.EntityId IS NULL
            )
            SELECT DISTINCT entity_type, Mention
            FROM unmatched
            WHERE source_table = {sql_literal(source_table)}
              AND EntityId = {sql_literal(entity_id)}
            LIMIT {sample_limit}
        """
        try:
            sample_pairs = connection.execute(sample_query).fetchall()
        except Exception:
            sample_pairs = []

        type_values = sorted({str(pair[0]) for pair in sample_pairs if pair[0] is not None})
        sample_mentions = [pair[1] for pair in sample_pairs if pair[1] is not None]

        if is_cuiless(entity_id):
            recommendation = "filter_from_v1_graph; optionally summarize as ungrounded mention features"
            warning = "cuiless_should_not_be_graph_node"
        else:
            recommendation = "review_non_cuiless_unmatched_entity_id; possible union-based BioEntity candidate"
            warning = "non_cuiless_missing_from_C23"

        output_rows.append(
            {
                "source_table": source_table,
                "entity_id": entity_id,
                "row_count": row_count,
                "approx_distinct_mentions": approx_distinct_mentions,
                "type_values_json": json_dumps(type_values),
                "sample_mentions_json": json_dumps(sample_mentions),
                "recommendation": recommendation,
                "warning": warning,
                "status": "ok",
                "error_message": "",
            }
        )

    return output_rows


def best_project_candidates(project_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in project_rows:
        if row.get("status") != "ok":
            continue

        link_table = row["link_table"]
        current = best.get(link_table)
        if current is None:
            best[link_table] = row
            continue

        row_score = (
            to_float(row.get("distinct_match_rate")),
            to_float(row.get("row_match_rate")),
            -NORMALIZATION_PRIORITY.get(str(row.get("normalization", "")), 99),
        )
        current_score = (
            to_float(current.get("distinct_match_rate")),
            to_float(current.get("row_match_rate")),
            -NORMALIZATION_PRIORITY.get(str(current.get("normalization", "")), 99),
        )

        if row_score > current_score:
            best[link_table] = row

    return best


def bioentity_cuiless_decision(
    bioentity_rows: list[dict[str, Any]],
    bioentity_entityid_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ok_rows = [row for row in bioentity_entityid_rows if row.get("status") == "ok"]
    total = sum(to_int(row.get("row_count")) for row in ok_rows)
    cuiless_total = sum(to_int(row.get("row_count")) for row in ok_rows if is_cuiless(row.get("entity_id")))
    non_cuiless_total = total - cuiless_total
    cuiless_fraction = cuiless_total / total if total else 0.0

    if not ok_rows and any(row.get("status") != "ok" for row in bioentity_rows + bioentity_entityid_rows):
        recommendation = "bioentity_unmatched_audit_failed"
    elif total == 0:
        recommendation = "c23_covers_all_bioentity_link_entityids"
    elif cuiless_fraction >= 0.99 and non_cuiless_total == 0:
        recommendation = "filter_cuiless_use_grounded_c23"
    elif cuiless_fraction >= 0.99:
        recommendation = "filter_cuiless_review_small_non_cuiless_unmatched"
    else:
        recommendation = "review_non_cuiless_unmatched_consider_union_nodes"

    return {
        "total_unmatched_rows": total,
        "cuiless_unmatched_rows": cuiless_total,
        "non_cuiless_unmatched_rows": non_cuiless_total,
        "cuiless_fraction": cuiless_fraction,
        "recommendation": recommendation,
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in rows:
        values = [
            str(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def write_summary(
    output_path: Path,
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    fk_rows: list[dict[str, Any]],
    project_rows: list[dict[str, Any]],
    bioentity_rows: list[dict[str, Any]],
    bioentity_entityid_rows: list[dict[str, Any]],
) -> None:
    best = best_project_candidates(project_rows)
    bio_decision = bioentity_cuiless_decision(bioentity_rows, bioentity_entityid_rows)

    parts = ["# PKG24S4 Core Graph Key Audit Summary\n"]

    parts.append("## 1. Node key audit\n")
    parts.append(
        markdown_table(
            node_rows,
            [
                "node_type",
                "table_name",
                "key_column",
                "row_count",
                "non_null_count",
                "distinct_count",
                "duplicate_key_count",
                "warning",
                "status",
            ],
        )
    )

    parts.append("\n## 2. Edge endpoint audit\n")
    parts.append(
        markdown_table(
            edge_rows,
            [
                "edge_type",
                "table_name",
                "row_count",
                "source_null_fraction",
                "target_null_fraction",
                "duplicate_edge_count",
                "warning",
                "status",
            ],
        )
    )

    parts.append("\n## 3. Foreign-key coverage\n")
    parts.append(
        markdown_table(
            fk_rows,
            [
                "edge_type",
                "edge_table",
                "edge_column",
                "node_table",
                "node_key_column",
                "row_match_rate",
                "distinct_match_rate",
                "warning",
                "status",
            ],
        )
    )

    parts.append("\n## 4. ProjectNumber join audit\n")
    best_rows = []
    for link_table in [link[0] for link in PROJECT_LINKS]:
        row = best.get(link_table, {})
        best_rows.append(
            {
                "link_table": link_table,
                "best_project_column": row.get("project_column", ""),
                "normalization": row.get("normalization", ""),
                "row_match_rate": row.get("row_match_rate", ""),
                "distinct_match_rate": row.get("distinct_match_rate", ""),
                "warning": row.get("warning", "no_good_project_key_match"),
                "status": row.get("status", "missing"),
            }
        )

    parts.append(
        markdown_table(
            best_rows,
            [
                "link_table",
                "best_project_column",
                "normalization",
                "row_match_rate",
                "distinct_match_rate",
                "warning",
                "status",
            ],
        )
    )

    parts.append("\n## 5. BioEntity unmatched audit by source/type\n")
    parts.append(
        "This section summarizes `EntityId` values that appear in paper/trial/patent BioEntity link tables "
        "but are not present in `C23_BioEntities`, grouped by source table and entity type.\n"
    )
    parts.append(
        markdown_table(
            bioentity_rows,
            [
                "source_table",
                "entity_type",
                "row_count",
                "approx_distinct_entity_ids",
                "sample_entity_id",
                "sample_mention",
                "warning",
                "status",
            ],
        )
    )

    parts.append("\n## 6. BioEntity unmatched audit by EntityId\n")
    parts.append(
        "This section directly checks which unmatched `EntityId` values cause BioEntity FK coverage below 0.95. "
        "`CUI-less` means an entity mention was detected but not grounded to a standard concept ID. "
        "`CUI-less` should not be used as a graph node.\n"
    )
    parts.append(
        markdown_table(
            bioentity_entityid_rows,
            [
                "source_table",
                "entity_id",
                "row_count",
                "approx_distinct_mentions",
                "type_values_json",
                "sample_mentions_json",
                "recommendation",
                "warning",
                "status",
            ],
        )
    )

    parts.append("\n## 7. Recommended decision\n")

    node_errors = [row for row in node_rows if row.get("status") != "ok"]
    edge_errors = [row for row in edge_rows if row.get("status") != "ok"]
    fk_errors = [row for row in fk_rows if row.get("status") != "ok"]
    project_errors = [row for row in project_rows if row.get("status") != "ok"]
    bioentity_errors = [
        row
        for row in bioentity_rows + bioentity_entityid_rows
        if row.get("status") != "ok"
    ]

    low_fk = [
        row
        for row in fk_rows
        if row.get("status") == "ok"
        and row.get("row_match_rate") not in {"", "not_computed"}
        and to_float(row.get("row_match_rate")) < 0.95
    ]

    weak_project = [
        row
        for row in best_rows
        if row.get("status") != "ok"
        or (
            to_float(row.get("row_match_rate")) < 0.95
            and to_float(row.get("distinct_match_rate")) < 0.95
        )
    ]

    if node_errors:
        parts.append("- Some node key checks failed; inspect `core_node_key_audit.csv` before extraction.\n")
    if edge_errors:
        parts.append("- Some edge endpoint checks failed; inspect `core_edge_endpoint_audit.csv` before extraction.\n")
    if fk_errors:
        parts.append("- Some FK coverage checks failed; inspect `core_fk_coverage.csv` before extraction.\n")
    elif not low_fk:
        parts.append("- Core FK row coverage is at least 0.95 for computed checks.\n")
    else:
        parts.append("- Some FK coverage checks are below 0.95; for BioEntity this is expected if unmatched rows are dominated by `CUI-less`.\n")

    if project_errors:
        parts.append("- Some ProjectNumber audit checks failed; inspect `project_number_join_audit.csv`.\n")
    if not weak_project:
        parts.append("- ProjectNumber matches `B02_Projects.CORE_PROJECT_NUM`; project edges can be considered for v1.\n")
    else:
        parts.append("- ProjectNumber matching is not reliable for all project link tables; consider excluding project edges from v1 or preserving ProjectNumber as string nodes until resolved.\n")

    if bioentity_errors:
        parts.append("- BioEntity unmatched audit failed; inspect `bioentity_unmatched_audit.csv` and `bioentity_unmatched_entityid_audit.csv`.\n")
    else:
        parts.append(
            f"- BioEntity unmatched row total: {bio_decision['total_unmatched_rows']}. "
            f"CUI-less unmatched rows: {bio_decision['cuiless_unmatched_rows']} "
            f"({bio_decision['cuiless_fraction']:.4f}). "
            f"Non-CUI-less unmatched rows: {bio_decision['non_cuiless_unmatched_rows']}.\n"
        )

        if bio_decision["recommendation"] in {
            "filter_cuiless_use_grounded_c23",
            "filter_cuiless_review_small_non_cuiless_unmatched",
        }:
            parts.append(
                "- BioEntity recommendation for v1: **do not create a BioEntity node for `CUI-less`**. "
                "Filter `EntityId = CUI-less` from BioEntity graph edges and use `C23_BioEntities` as the grounded BioEntity node table. "
                "Optionally summarize `CUI-less` mentions as carrier-level ungrounded mention features.\n"
            )
        elif bio_decision["recommendation"] == "c23_covers_all_bioentity_link_entityids":
            parts.append("- BioEntity recommendation for v1: `C23_BioEntities` covers all audited BioEntity link IDs.\n")
        else:
            parts.append(
                "- BioEntity recommendation for v1: unmatched non-CUI-less EntityIds exist; review them before deciding whether to use union-based BioEntity nodes.\n"
            )

    output_path.write_text("\n".join(parts), encoding="utf-8")


def print_terminal_summary(
    output_dir: Path,
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    fk_rows: list[dict[str, Any]],
    project_rows: list[dict[str, Any]],
    bioentity_rows: list[dict[str, Any]],
    bioentity_entityid_rows: list[dict[str, Any]],
) -> None:
    best = best_project_candidates(project_rows)
    bio_decision = bioentity_cuiless_decision(bioentity_rows, bioentity_entityid_rows)

    node_warnings = [row for row in node_rows if row.get("warning")]
    edge_warnings = [row for row in edge_rows if row.get("warning")]
    fk_warnings = [row for row in fk_rows if row.get("warning")]
    fk_errors = [row for row in fk_rows if row.get("status") != "ok"]
    project_errors = [row for row in project_rows if row.get("status") != "ok"]
    bioentity_errors = [
        row
        for row in bioentity_rows + bioentity_entityid_rows
        if row.get("status") != "ok"
    ]

    low_fk = [
        row
        for row in fk_rows
        if row.get("status") == "ok"
        and row.get("row_match_rate") not in {"", "not_computed"}
        and to_float(row.get("row_match_rate")) < 0.95
    ]

    print("PKG24S4 core graph key audit complete")
    print(f"Output directory: {output_dir}")
    print(f"Node key audit rows: {len(node_rows)}")
    print(f"Edge endpoint audit rows: {len(edge_rows)}")
    print(f"FK coverage rows: {len(fk_rows)}")
    print(f"ProjectNumber audit rows: {len(project_rows)}")
    print(f"BioEntity unmatched source/type rows: {len(bioentity_rows)}")
    print(f"BioEntity unmatched EntityId rows: {len(bioentity_entityid_rows)}")
    print(f"BioEntity unmatched row total: {bio_decision['total_unmatched_rows']}")
    print(f"BioEntity CUI-less unmatched rows: {bio_decision['cuiless_unmatched_rows']}")
    print(f"BioEntity non-CUI-less unmatched rows: {bio_decision['non_cuiless_unmatched_rows']}")
    print(f"BioEntity CUI-less fraction: {bio_decision['cuiless_fraction']:.6f}")

    for link_table in [link[0] for link in PROJECT_LINKS]:
        row = best.get(link_table, {})
        print(
            f"{link_table} best project key: "
            f"{row.get('project_column', 'not_found')} "
            f"normalization={row.get('normalization', '')} "
            f"row_match_rate={row.get('row_match_rate', '')} "
            f"distinct_match_rate={row.get('distinct_match_rate', '')}"
        )

    if node_warnings:
        print("Node key warnings:")
        for row in node_warnings:
            print(f"- {row['node_type']} {row['table_name']}.{row['key_column']}: {row['warning']}")

    if edge_warnings:
        print("Edge endpoint warnings:")
        for row in edge_warnings:
            print(f"- {row['edge_type']}: {row['warning']}")

    if fk_errors:
        print("FK coverage errors:")
        for row in fk_errors:
            print(
                f"- {row['edge_table']}.{row['edge_column']} -> "
                f"{row['node_table']}.{row['node_key_column']}: {row['error_message']}"
            )

    if fk_warnings:
        print("FK coverage warnings:")
        for row in fk_warnings:
            print(
                f"- {row['edge_table']}.{row['edge_column']} -> "
                f"{row['node_table']}.{row['node_key_column']}: "
                f"{row['warning']} row_match_rate={row.get('row_match_rate')}"
            )

    if low_fk:
        print("FK coverage below 0.95:")
        for row in low_fk:
            print(
                f"- {row['edge_table']}.{row['edge_column']} -> "
                f"{row['node_table']}.{row['node_key_column']}: {row['row_match_rate']}"
            )

    if project_errors:
        print("ProjectNumber audit errors:")
        for row in project_errors[:20]:
            print(
                f"- {row['link_table']}.{row['link_column']} -> "
                f"B02.{row['project_column']}: {row['error_message']}"
            )

    if bioentity_errors:
        print("BioEntity unmatched audit errors:")
        for row in bioentity_errors:
            print(f"- {row['error_message']}")
    elif bio_decision["recommendation"] in {
        "filter_cuiless_use_grounded_c23",
        "filter_cuiless_review_small_non_cuiless_unmatched",
    }:
        print(
            "BioEntity decision: filter EntityId=CUI-less from v1 graph edges; "
            "use C23_BioEntities as grounded BioEntity nodes; optionally summarize "
            "CUI-less mentions as carrier-level ungrounded mention features."
        )
    elif bio_decision["recommendation"] == "c23_covers_all_bioentity_link_entityids":
        print("BioEntity decision: C23_BioEntities covers all audited BioEntity link EntityId values.")
    else:
        print("BioEntity decision: review non-CUI-less unmatched EntityIds before extraction.")

    weak_project = [
        link
        for link in [link[0] for link in PROJECT_LINKS]
        if link not in best
        or (
            to_float(best[link].get("row_match_rate")) < 0.95
            and to_float(best[link].get("distinct_match_rate")) < 0.95
        )
    ]

    if fk_errors:
        print("Recommended next step: inspect core_fk_coverage.csv before extracting graph edges")
    elif bioentity_errors:
        print("Recommended next step: inspect BioEntity unmatched audit outputs")
    elif weak_project:
        print("Recommended next step: inspect project_number_join_audit.csv before extracting project edges")
    else:
        print(
            "Recommended next step: scripts/06_extract_patent_centered_subgraph.py "
            "with grounded BioEntity nodes and CUI-less filtering"
        )


def main() -> None:
    args = parse_args()

    if args.threads < 1:
        raise ValueError("--threads must be greater than or equal to 1")
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be greater than or equal to 1")

    args.parquet_dir = args.parquet_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = connect_duckdb(args)

    node_rows = [
        audit_node_key(connection, args.parquet_dir, node, args.exact_distinct, args.sample_limit)
        for node in NODES
    ]

    edge_rows = [
        audit_edge_endpoint(
            connection,
            args.parquet_dir,
            edge,
            args.exact_duplicate_edges,
            args.sample_limit,
        )
        for edge in EDGES
    ]

    fk_rows = [
        audit_fk_coverage(connection, args.parquet_dir, check, args.coverage_mode, args.sample_limit)
        for check in FK_CHECKS
    ]

    project_rows = [
        audit_project_number_join(
            connection,
            args.parquet_dir,
            link_table,
            link_column,
            project_column,
            normalization,
            args.sample_limit,
        )
        for link_table, link_column in PROJECT_LINKS
        for project_column in PROJECT_COLUMNS
        for normalization in NORMALIZATIONS
    ]

    bioentity_rows = audit_bioentity_unmatched(connection, args.parquet_dir)
    bioentity_entityid_rows = audit_bioentity_unmatched_entityids(
        connection,
        args.parquet_dir,
        args.sample_limit,
    )

    write_csv(args.output_dir / "core_node_key_audit.csv", node_rows, NODE_AUDIT_COLUMNS)
    write_csv(args.output_dir / "core_edge_endpoint_audit.csv", edge_rows, EDGE_AUDIT_COLUMNS)
    write_csv(args.output_dir / "core_fk_coverage.csv", fk_rows, FK_COVERAGE_COLUMNS)
    write_csv(args.output_dir / "project_number_join_audit.csv", project_rows, PROJECT_AUDIT_COLUMNS)
    write_csv(args.output_dir / "bioentity_unmatched_audit.csv", bioentity_rows, BIOENTITY_UNMATCHED_COLUMNS)
    write_csv(
        args.output_dir / "bioentity_unmatched_entityid_audit.csv",
        bioentity_entityid_rows,
        BIOENTITY_UNMATCHED_ENTITYID_COLUMNS,
    )

    write_summary(
        args.output_dir / "core_key_audit_summary.md",
        node_rows,
        edge_rows,
        fk_rows,
        project_rows,
        bioentity_rows,
        bioentity_entityid_rows,
    )

    print_terminal_summary(
        args.output_dir,
        node_rows,
        edge_rows,
        fk_rows,
        project_rows,
        bioentity_rows,
        bioentity_entityid_rows,
    )


if __name__ == "__main__":
    main()