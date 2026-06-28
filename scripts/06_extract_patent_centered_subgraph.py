"""Extract a first-pass patent-centered PKG24S4 biomedical translation subgraph.

This script reads already-converted PKG24S4 Parquet files from data/interim/parquet
and writes graph-ready node/edge Parquet files under data/processed/.

It does not read raw TSV files, does not modify intermediate Parquet files, and
does not implement knowledge units, GNNs, neural PDEs, or model training.

Audited schema decisions used here:
- ProjectNumber joins to B02_Projects.CORE_PROJECT_NUM after normalization.
- BioEntity graph nodes come from grounded C23_BioEntities.EntityId.
- EntityId = CUI-less is filtered from graph edges and is not a BioEntity node.

Important extraction behavior:
- Output directory is controlled by --output-dir.
- Paper nodes are de-duplicated by PMID.
- ClinicalTrial nodes are de-duplicated by nct_id.
- BioEntity nodes are de-duplicated by EntityId.
- Project nodes are aggregated by normalized CORE_PROJECT_NUM.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Any, Callable


CORE_TABLES = [
    "C15_Patents",
    "C01_Papers",
    "C11_ClinicalTrials",
    "B02_Projects",
    "C23_BioEntities",
    "C16_Link_Patents_Papers",
    "C18_Link_Patents_BioEntities",
    "B05_Link_Patents_Projects",
    "C12_Link_Papers_Clinicaltrials",
    "C06_Link_Papers_BioEntities",
    "B03_Link_Papers_Projects",
    "C13_Link_ClinicalTrials_BioEntities",
    "B04_Link_ClinicalTrials_Projects",
]

BIOENTITY_EDGE_ATTRS = [
    "Mention",
    "Type",
    "TextFrom",
    "StartPosition",
    "EndPosition",
    "is_neural_normalized",
    "prob",
    "mesh",
    "mim",
    "CL",
    "cellosaurus",
    "NCBITaxon",
    "NCBIGene",
    "CHEBI",
]

PATENT_NODE_ATTRS = [
    "GrantedDate",
    "Title",
    "Abstract",
    "Type",
    "Kind",
    "ClaimNum",
    "isWithdrawn",
    "has_citing_paper",
    "is_granted_by_NIH",
    "is_CPC_A61",
    "FileName",
]

PAPER_NODE_ATTRS = [
    "PubYear",
    "ArticleTitle",
    "AuthorNum",
    "CitedCount",
    "StdCitedCount",
    "CitedCount_ClinicalArticle",
    "CitedCount_ClinicalTrailStudy",
    "CitedCount_Patent",
    "IsClinicalArticle",
    "IsResearchArticle",
    "Human",
    "Animal",
    "MolecularCellular",
    "APT",
]

TRIAL_NODE_ATTRS = [
    "study_first_submitted_date",
    "last_update_posted_date",
    "start_date",
    "completion_date",
    "overall_status",
    "last_known_status",
    "brief_title",
    "official_title",
    "study_type",
    "phase",
    "source",
    "source_class",
    "keywords",
    "conditions",
]

PROJECT_SAMPLE_ATTRS = [
    "CORE_PROJECT_NUM",
    "PROJECT_TITLE",
    "Abstract",
    "ORG_NAME",
    "PI_NAMEs",
    "ACTIVITY",
    "IC_NAME",
]

OUTPUT_TABLES = [
    ("node", "nodes_patent", "nodes_patent.parquet", "C15_Patents", "Patent nodes selected by GrantedDate."),
    ("node", "nodes_paper", "nodes_paper.parquet", "C01_Papers,C16_Link_Patents_Papers", "Paper nodes linked from seed patents."),
    ("node", "nodes_clinicaltrial", "nodes_clinicaltrial.parquet", "C11_ClinicalTrials,C12_Link_Papers_Clinicaltrials", "ClinicalTrial nodes linked from selected papers."),
    ("node", "nodes_project", "nodes_project.parquet", "B02_Projects,B03/B04/B05 project link tables", "Project nodes aggregated by normalized CORE_PROJECT_NUM."),
    ("node", "nodes_bioentity", "nodes_bioentity.parquet", "C23_BioEntities,C06/C13/C18 BioEntity link tables", "Grounded BioEntity nodes from C23."),
    ("edge", "edges_patent_paper", "edges_patent_paper.parquet", "C16_Link_Patents_Papers", "Patent to paper edges."),
    ("edge", "edges_patent_bioentity", "edges_patent_bioentity.parquet", "C18_Link_Patents_BioEntities", "Patent to grounded BioEntity edges."),
    ("edge", "edges_patent_project", "edges_patent_project.parquet", "B05_Link_Patents_Projects", "Patent to Project edges."),
    ("edge", "edges_paper_trial", "edges_paper_trial.parquet", "C12_Link_Papers_Clinicaltrials", "Paper to ClinicalTrial edges."),
    ("edge", "edges_paper_project", "edges_paper_project.parquet", "B03_Link_Papers_Projects", "Paper to Project edges."),
    ("edge", "edges_paper_bioentity", "edges_paper_bioentity.parquet", "C06_Link_Papers_BioEntities", "Paper to grounded BioEntity edges."),
    ("edge", "edges_trial_project", "edges_trial_project.parquet", "B04_Link_ClinicalTrials_Projects", "ClinicalTrial to Project edges."),
    ("edge", "edges_trial_bioentity", "edges_trial_bioentity.parquet", "C13_Link_ClinicalTrials_BioEntities", "ClinicalTrial to grounded BioEntity edges."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a patent-centered PKG24S4 subgraph.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/interim/parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/patent_centered_subgraph"),
        help="Directory where extracted node/edge Parquet files and reports are written.",
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="20GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("data/interim/duckdb_tmp"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-patents", type=int, default=None)
    parser.add_argument("--include-paper-bioentities", type=parse_bool, default=True)
    parser.add_argument("--include-trial-expansion", type=parse_bool, default=True)
    parser.add_argument("--include-ungrounded-cuiless-features", type=parse_bool, default=False)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--no-progress", action="store_true")
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


def year_expr(column: str, alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    return f"TRY_CAST(REGEXP_EXTRACT(CAST({prefix}{qident(column)} AS VARCHAR), '([0-9]{{4}})', 1) AS INTEGER)"


def normalize_project_expr(column: str, alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    return (
        f"NULLIF(UPPER(REPLACE(REPLACE(TRIM(CAST({prefix}{qident(column)} AS VARCHAR)), ' ', ''), '-', '')), '')"
    )


def valid_entity_predicate(alias: str | None = None) -> str:
    expr = key_expr("EntityId", alias)
    prefix = f"{alias}." if alias else ""
    upper_expr = f"UPPER(TRIM(CAST({prefix}{qident('EntityId')} AS VARCHAR)))"
    return f"{expr} IS NOT NULL AND {upper_expr} <> 'CUI-LESS'"


def connect_duckdb(args: argparse.Namespace):
    import duckdb

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute(f"SET temp_directory={sql_literal(args.temp_dir)}")
    return connection


def get_columns(connection: Any, parquet_dir: Path, table_name: str) -> set[str]:
    path = table_path(parquet_dir, table_name)
    rows = connection.execute(f"DESCRIBE SELECT * FROM {scan_sql(path)}").fetchall()
    return {str(row[0]) for row in rows}


def require_files(parquet_dir: Path) -> None:
    missing = [
        str(table_path(parquet_dir, table_name))
        for table_name in CORE_TABLES
        if not table_path(parquet_dir, table_name).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required Parquet files:\n" + "\n".join(missing))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def optional_selects(
    available_columns: set[str],
    requested_columns: list[str],
    alias: str,
    default_type: str = "VARCHAR",
) -> list[str]:
    selects = []
    for column in requested_columns:
        if column in available_columns:
            selects.append(f"{alias}.{qident(column)} AS {qident(column)}")
        else:
            selects.append(f"CAST(NULL AS {default_type}) AS {qident(column)}")
    return selects


def optional_any_value_selects(
    requested_columns: list[str],
    default_type: str = "VARCHAR",
) -> list[str]:
    return [f"ANY_VALUE({qident(column)}) AS {qident(column)}" for column in requested_columns]


def optional_edge_attrs(available_columns: set[str], requested_columns: list[str], alias: str) -> list[str]:
    return optional_selects(available_columns, requested_columns, alias)


def execute(connection: Any, sql: str) -> None:
    connection.execute(sql)


def count_table(connection: Any, table_name: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {qident(table_name)}").fetchone()[0])


def copy_table_to_parquet(connection: Any, table_name: str, path: Path, compression: str) -> None:
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


def create_empty_table(connection: Any, table_name: str, schema_sql: str) -> None:
    execute(connection, f"CREATE TEMP TABLE {qident(table_name)} AS SELECT {schema_sql} WHERE FALSE")


def run_step(args: argparse.Namespace, index: int, total: int, label: str, fn: Callable[[], None]) -> None:
    if not args.no_progress:
        print(f"[{index}/{total}] {label} ...", flush=True)
    start = time.time()
    fn()
    elapsed = time.time() - start
    if not args.no_progress:
        print(f"[{index}/{total}] {label} done in {elapsed:.1f}s", flush=True)


def create_seed_patents(
    connection: Any,
    parquet_dir: Path,
    args: argparse.Namespace,
    columns_cache: dict[str, set[str]],
) -> None:
    table = "C15_Patents"
    path = table_path(parquet_dir, table)
    cols = columns_cache[table]
    attrs = optional_selects(cols, PATENT_NODE_ATTRS, "p")
    attr_aggs = optional_any_value_selects(PATENT_NODE_ATTRS)
    limit_clause = f"LIMIT {args.limit_patents}" if args.limit_patents is not None else ""

    execute(
        connection,
        f"""
        CREATE TEMP TABLE nodes_patent AS
        WITH base AS (
            SELECT
                {key_expr("PatentId", "p")} AS PatentId,
                {year_expr("GrantedDate", "p")} AS patent_year,
                {", ".join(attrs)}
            FROM {scan_sql(path)} p
            WHERE {key_expr("PatentId", "p")} IS NOT NULL
              AND {year_expr("GrantedDate", "p")} BETWEEN {args.start_year} AND {args.end_year}
        ),
        grouped AS (
            SELECT
                PatentId,
                MIN(patent_year) AS patent_year,
                {", ".join(attr_aggs)}
            FROM base
            GROUP BY PatentId
            ORDER BY PatentId
            {limit_clause}
        )
        SELECT
            PatentId AS node_id,
            *
        FROM grouped
        """,
    )


def create_patent_paper_edges_and_papers(
    connection: Any,
    parquet_dir: Path,
    columns_cache: dict[str, set[str]],
) -> None:
    c16 = table_path(parquet_dir, "C16_Link_Patents_Papers")
    c01 = table_path(parquet_dir, "C01_Papers")
    c16_cols = columns_cache["C16_Link_Patents_Papers"]
    c01_cols = columns_cache["C01_Papers"]

    edge_attrs = optional_edge_attrs(c16_cols, ["WhereFound", "ConfScore", "SelfCitation", "RefType"], "l")
    paper_attrs = optional_selects(c01_cols, PAPER_NODE_ATTRS, "p")
    paper_attr_aggs = optional_any_value_selects(PAPER_NODE_ATTRS)

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_patent_paper_raw AS
        SELECT
            {key_expr("PatentId", "l")} AS source_id,
            {key_expr("PMID", "l")} AS target_id,
            'patent_links_paper' AS edge_type,
            'C16_Link_Patents_Papers' AS source_table,
            {key_expr("PatentId", "l")} AS PatentId,
            {key_expr("PMID", "l")} AS PMID,
            {", ".join(edge_attrs)}
        FROM {scan_sql(c16)} l
        INNER JOIN nodes_patent p
            ON {key_expr("PatentId", "l")} = p.PatentId
        WHERE {key_expr("PMID", "l")} IS NOT NULL
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_paper_ids AS
        SELECT DISTINCT target_id AS PMID
        FROM edges_patent_paper_raw
        WHERE target_id IS NOT NULL
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE nodes_paper AS
        WITH paper_base AS (
            SELECT
                {key_expr("PMID", "p")} AS PMID,
                {", ".join(paper_attrs)}
            FROM {scan_sql(c01)} p
            INNER JOIN selected_paper_ids s
                ON {key_expr("PMID", "p")} = s.PMID
            WHERE {key_expr("PMID", "p")} IS NOT NULL
        ),
        grouped AS (
            SELECT
                PMID,
                {", ".join(paper_attr_aggs)}
            FROM paper_base
            GROUP BY PMID
        )
        SELECT
            PMID AS node_id,
            *
        FROM grouped
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_patent_paper AS
        SELECT r.*
        FROM edges_patent_paper_raw r
        INNER JOIN nodes_paper p
            ON r.target_id = p.PMID
        """,
    )


def create_trial_nodes_and_edges(
    connection: Any,
    parquet_dir: Path,
    args: argparse.Namespace,
    columns_cache: dict[str, set[str]],
) -> None:
    if not args.include_trial_expansion:
        create_empty_table(
            connection,
            "edges_paper_trial",
            "CAST(NULL AS VARCHAR) AS source_id, CAST(NULL AS VARCHAR) AS target_id, "
            "CAST(NULL AS VARCHAR) AS edge_type, CAST(NULL AS VARCHAR) AS source_table, "
            "CAST(NULL AS VARCHAR) AS PMID, CAST(NULL AS VARCHAR) AS nct_id, CAST(NULL AS VARCHAR) AS Type",
        )
        create_empty_table(
            connection,
            "nodes_clinicaltrial",
            "CAST(NULL AS VARCHAR) AS node_id, CAST(NULL AS VARCHAR) AS nct_id",
        )
        return

    c12 = table_path(parquet_dir, "C12_Link_Papers_Clinicaltrials")
    c11 = table_path(parquet_dir, "C11_ClinicalTrials")
    c12_cols = columns_cache["C12_Link_Papers_Clinicaltrials"]
    c11_cols = columns_cache["C11_ClinicalTrials"]
    trial_attrs = optional_selects(c11_cols, TRIAL_NODE_ATTRS, "t")
    trial_attr_aggs = optional_any_value_selects(TRIAL_NODE_ATTRS)
    type_expr = 'l."Type" AS "Type"' if "Type" in c12_cols else "CAST(NULL AS VARCHAR) AS Type"

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_paper_trial_raw AS
        SELECT
            {key_expr("PMID", "l")} AS source_id,
            {key_expr("nct_id", "l")} AS target_id,
            'paper_linked_trial' AS edge_type,
            'C12_Link_Papers_Clinicaltrials' AS source_table,
            {key_expr("PMID", "l")} AS PMID,
            {key_expr("nct_id", "l")} AS nct_id,
            {type_expr}
        FROM {scan_sql(c12)} l
        INNER JOIN nodes_paper p
            ON {key_expr("PMID", "l")} = p.PMID
        WHERE {key_expr("nct_id", "l")} IS NOT NULL
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_trial_ids AS
        SELECT DISTINCT target_id AS nct_id
        FROM edges_paper_trial_raw
        WHERE target_id IS NOT NULL
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE nodes_clinicaltrial AS
        WITH trial_base AS (
            SELECT
                {key_expr("nct_id", "t")} AS nct_id,
                {", ".join(trial_attrs)}
            FROM {scan_sql(c11)} t
            INNER JOIN selected_trial_ids s
                ON {key_expr("nct_id", "t")} = s.nct_id
            WHERE {key_expr("nct_id", "t")} IS NOT NULL
        ),
        grouped AS (
            SELECT
                nct_id,
                {", ".join(trial_attr_aggs)}
            FROM trial_base
            GROUP BY nct_id
        )
        SELECT
            nct_id AS node_id,
            *
        FROM grouped
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_paper_trial AS
        SELECT r.*
        FROM edges_paper_trial_raw r
        INNER JOIN nodes_clinicaltrial t
            ON r.target_id = t.nct_id
        """,
    )


def create_project_raw_edges(
    connection: Any,
    parquet_dir: Path,
    args: argparse.Namespace,
    columns_cache: dict[str, set[str]],
) -> None:
    b05 = table_path(parquet_dir, "B05_Link_Patents_Projects")
    b03 = table_path(parquet_dir, "B03_Link_Papers_Projects")
    b04 = table_path(parquet_dir, "B04_Link_ClinicalTrials_Projects")

    record_year_expr = (
        'l."RecordYear" AS RecordYear'
        if "RecordYear" in columns_cache["B03_Link_Papers_Projects"]
        else "CAST(NULL AS VARCHAR) AS RecordYear"
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_patent_project_raw AS
        SELECT
            {key_expr("PatentId", "l")} AS source_id,
            {normalize_project_expr("ProjectNumber", "l")} AS target_id,
            'patent_linked_project' AS edge_type,
            'B05_Link_Patents_Projects' AS source_table,
            {key_expr("PatentId", "l")} AS PatentId,
            {key_expr("ProjectNumber", "l")} AS ProjectNumber,
            {normalize_project_expr("ProjectNumber", "l")} AS project_id_norm
        FROM {scan_sql(b05)} l
        INNER JOIN nodes_patent p
            ON {key_expr("PatentId", "l")} = p.PatentId
        WHERE {normalize_project_expr("ProjectNumber", "l")} IS NOT NULL
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_paper_project_raw AS
        SELECT
            {key_expr("PMID", "l")} AS source_id,
            {normalize_project_expr("ProjectNumber", "l")} AS target_id,
            'paper_linked_project' AS edge_type,
            'B03_Link_Papers_Projects' AS source_table,
            {key_expr("PMID", "l")} AS PMID,
            {key_expr("ProjectNumber", "l")} AS ProjectNumber,
            {normalize_project_expr("ProjectNumber", "l")} AS project_id_norm,
            {record_year_expr}
        FROM {scan_sql(b03)} l
        INNER JOIN nodes_paper p
            ON {key_expr("PMID", "l")} = p.PMID
        WHERE {normalize_project_expr("ProjectNumber", "l")} IS NOT NULL
        """,
    )

    if args.include_trial_expansion:
        execute(
            connection,
            f"""
            CREATE TEMP TABLE edges_trial_project_raw AS
            SELECT
                {key_expr("nct_id", "l")} AS source_id,
                {normalize_project_expr("ProjectNumber", "l")} AS target_id,
                'trial_linked_project' AS edge_type,
                'B04_Link_ClinicalTrials_Projects' AS source_table,
                {key_expr("nct_id", "l")} AS nct_id,
                {key_expr("ProjectNumber", "l")} AS ProjectNumber,
                {normalize_project_expr("ProjectNumber", "l")} AS project_id_norm
            FROM {scan_sql(b04)} l
            INNER JOIN nodes_clinicaltrial t
                ON {key_expr("nct_id", "l")} = t.nct_id
            WHERE {normalize_project_expr("ProjectNumber", "l")} IS NOT NULL
            """,
        )
    else:
        create_empty_table(
            connection,
            "edges_trial_project_raw",
            "CAST(NULL AS VARCHAR) AS source_id, CAST(NULL AS VARCHAR) AS target_id, "
            "CAST(NULL AS VARCHAR) AS edge_type, CAST(NULL AS VARCHAR) AS source_table, "
            "CAST(NULL AS VARCHAR) AS nct_id, CAST(NULL AS VARCHAR) AS ProjectNumber, "
            "CAST(NULL AS VARCHAR) AS project_id_norm",
        )


def create_project_nodes_and_final_edges(
    connection: Any,
    parquet_dir: Path,
    columns_cache: dict[str, set[str]],
) -> None:
    b02 = table_path(parquet_dir, "B02_Projects")
    b02_cols = columns_cache["B02_Projects"]

    fiscal_year_expr = 'TRY_CAST("FiscalYear" AS INTEGER)' if "FiscalYear" in b02_cols else "CAST(NULL AS INTEGER)"
    application_expr = key_expr("APPLICATION_ID") if "APPLICATION_ID" in b02_cols else "CAST(NULL AS VARCHAR)"
    total_cost_expr = 'TRY_CAST("TOTAL_COST" AS DOUBLE)' if "TOTAL_COST" in b02_cols else "CAST(NULL AS DOUBLE)"
    direct_cost_expr = 'TRY_CAST("DIRECT_COST_AMT" AS DOUBLE)' if "DIRECT_COST_AMT" in b02_cols else "CAST(NULL AS DOUBLE)"
    indirect_cost_expr = 'TRY_CAST("INDIRECT_COST_AMT" AS DOUBLE)' if "INDIRECT_COST_AMT" in b02_cols else "CAST(NULL AS DOUBLE)"

    sample_exprs = []
    for column in PROJECT_SAMPLE_ATTRS:
        if column in b02_cols:
            sample_exprs.append(f"ANY_VALUE({qident(column)}) AS {qident(column)}")
        else:
            sample_exprs.append(f"CAST(NULL AS VARCHAR) AS {qident(column)}")

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_project_ids AS
        SELECT DISTINCT target_id AS project_id_norm FROM edges_patent_project_raw WHERE target_id IS NOT NULL
        UNION
        SELECT DISTINCT target_id AS project_id_norm FROM edges_paper_project_raw WHERE target_id IS NOT NULL
        UNION
        SELECT DISTINCT target_id AS project_id_norm FROM edges_trial_project_raw WHERE target_id IS NOT NULL
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE nodes_project AS
        WITH b02 AS (
            SELECT
                {normalize_project_expr("CORE_PROJECT_NUM")} AS core_project_num_norm,
                {fiscal_year_expr} AS fiscal_year,
                {application_expr} AS application_id,
                {total_cost_expr} AS total_cost,
                {direct_cost_expr} AS direct_cost,
                {indirect_cost_expr} AS indirect_cost,
                *
            FROM {scan_sql(b02)}
            WHERE {normalize_project_expr("CORE_PROJECT_NUM")} IS NOT NULL
        )
        SELECT
            core_project_num_norm AS node_id,
            core_project_num_norm,
            ANY_VALUE(CORE_PROJECT_NUM) AS core_project_num_sample,
            MIN(fiscal_year) AS min_fiscal_year,
            MAX(fiscal_year) AS max_fiscal_year,
            COUNT(DISTINCT fiscal_year) AS fiscal_year_count,
            COUNT(DISTINCT application_id) AS application_count,
            SUM(total_cost) AS total_cost_sum,
            SUM(direct_cost) AS direct_cost_sum,
            SUM(indirect_cost) AS indirect_cost_sum,
            {", ".join(sample_exprs)}
        FROM b02
        INNER JOIN selected_project_ids s
            ON b02.core_project_num_norm = s.project_id_norm
        GROUP BY core_project_num_norm
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_patent_project AS
        SELECT r.*
        FROM edges_patent_project_raw r
        INNER JOIN nodes_project p
            ON r.target_id = p.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_paper_project AS
        SELECT r.*
        FROM edges_paper_project_raw r
        INNER JOIN nodes_project p
            ON r.target_id = p.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_trial_project AS
        SELECT r.*
        FROM edges_trial_project_raw r
        INNER JOIN nodes_project p
            ON r.target_id = p.node_id
        """,
    )


def create_bioentity_edges_and_nodes(
    connection: Any,
    parquet_dir: Path,
    args: argparse.Namespace,
    columns_cache: dict[str, set[str]],
) -> None:
    c23 = table_path(parquet_dir, "C23_BioEntities")
    c18 = table_path(parquet_dir, "C18_Link_Patents_BioEntities")
    c06 = table_path(parquet_dir, "C06_Link_Papers_BioEntities")
    c13 = table_path(parquet_dir, "C13_Link_ClinicalTrials_BioEntities")

    c18_attrs = optional_edge_attrs(columns_cache["C18_Link_Patents_BioEntities"], BIOENTITY_EDGE_ATTRS, "l")
    c06_attrs = optional_edge_attrs(columns_cache["C06_Link_Papers_BioEntities"], BIOENTITY_EDGE_ATTRS + ["FileName"], "l")
    c13_attrs = optional_edge_attrs(columns_cache["C13_Link_ClinicalTrials_BioEntities"], BIOENTITY_EDGE_ATTRS, "l")

    execute(
        connection,
        f"""
        CREATE TEMP TABLE c23_entity_keys AS
        SELECT DISTINCT {key_expr("EntityId")} AS EntityId
        FROM {scan_sql(c23)}
        WHERE {key_expr("EntityId")} IS NOT NULL
          AND UPPER(TRIM(CAST({qident("EntityId")} AS VARCHAR))) <> 'CUI-LESS'
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_patent_bioentity AS
        SELECT
            {key_expr("PatentId", "l")} AS source_id,
            {key_expr("EntityId", "l")} AS target_id,
            'patent_mentions_bioentity' AS edge_type,
            'C18_Link_Patents_BioEntities' AS source_table,
            {key_expr("PatentId", "l")} AS PatentId,
            {key_expr("EntityId", "l")} AS EntityId,
            {", ".join(c18_attrs)}
        FROM {scan_sql(c18)} l
        INNER JOIN nodes_patent p
            ON {key_expr("PatentId", "l")} = p.PatentId
        INNER JOIN c23_entity_keys e
            ON {key_expr("EntityId", "l")} = e.EntityId
        WHERE {valid_entity_predicate("l")}
        """,
    )

    if args.include_paper_bioentities:
        execute(
            connection,
            f"""
            CREATE TEMP TABLE edges_paper_bioentity AS
            SELECT
                {key_expr("PMID", "l")} AS source_id,
                {key_expr("EntityId", "l")} AS target_id,
                'paper_mentions_bioentity' AS edge_type,
                'C06_Link_Papers_BioEntities' AS source_table,
                {key_expr("PMID", "l")} AS PMID,
                {key_expr("EntityId", "l")} AS EntityId,
                {", ".join(c06_attrs)}
            FROM {scan_sql(c06)} l
            INNER JOIN nodes_paper p
                ON {key_expr("PMID", "l")} = p.PMID
            INNER JOIN c23_entity_keys e
                ON {key_expr("EntityId", "l")} = e.EntityId
            WHERE {valid_entity_predicate("l")}
            """,
        )
    else:
        create_empty_table(
            connection,
            "edges_paper_bioentity",
            "CAST(NULL AS VARCHAR) AS source_id, CAST(NULL AS VARCHAR) AS target_id, "
            "CAST(NULL AS VARCHAR) AS edge_type, CAST(NULL AS VARCHAR) AS source_table, "
            "CAST(NULL AS VARCHAR) AS PMID, CAST(NULL AS VARCHAR) AS EntityId",
        )

    if args.include_trial_expansion:
        execute(
            connection,
            f"""
            CREATE TEMP TABLE edges_trial_bioentity AS
            SELECT
                {key_expr("nct_id", "l")} AS source_id,
                {key_expr("EntityId", "l")} AS target_id,
                'trial_mentions_bioentity' AS edge_type,
                'C13_Link_ClinicalTrials_BioEntities' AS source_table,
                {key_expr("nct_id", "l")} AS nct_id,
                {key_expr("EntityId", "l")} AS EntityId,
                {", ".join(c13_attrs)}
            FROM {scan_sql(c13)} l
            INNER JOIN nodes_clinicaltrial t
                ON {key_expr("nct_id", "l")} = t.nct_id
            INNER JOIN c23_entity_keys e
                ON {key_expr("EntityId", "l")} = e.EntityId
            WHERE {valid_entity_predicate("l")}
            """,
        )
    else:
        create_empty_table(
            connection,
            "edges_trial_bioentity",
            "CAST(NULL AS VARCHAR) AS source_id, CAST(NULL AS VARCHAR) AS target_id, "
            "CAST(NULL AS VARCHAR) AS edge_type, CAST(NULL AS VARCHAR) AS source_table, "
            "CAST(NULL AS VARCHAR) AS nct_id, CAST(NULL AS VARCHAR) AS EntityId",
        )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_bioentity_ids AS
        SELECT DISTINCT target_id AS EntityId FROM edges_patent_bioentity WHERE target_id IS NOT NULL
        UNION
        SELECT DISTINCT target_id AS EntityId FROM edges_paper_bioentity WHERE target_id IS NOT NULL
        UNION
        SELECT DISTINCT target_id AS EntityId FROM edges_trial_bioentity WHERE target_id IS NOT NULL
        """,
    )

    c23_cols = columns_cache["C23_BioEntities"]
    type_expr = 'b."Type" AS "Type"' if "Type" in c23_cols else "CAST(NULL AS VARCHAR) AS Type"
    mention_expr = 'b."Mention" AS "Mention"' if "Mention" in c23_cols else "CAST(NULL AS VARCHAR) AS Mention"

    execute(
        connection,
        f"""
        CREATE TEMP TABLE nodes_bioentity AS
        WITH bioentity_base AS (
            SELECT
                {key_expr("EntityId", "b")} AS EntityId,
                {type_expr},
                {mention_expr}
            FROM {scan_sql(c23)} b
            INNER JOIN selected_bioentity_ids s
                ON {key_expr("EntityId", "b")} = s.EntityId
            WHERE {key_expr("EntityId", "b")} IS NOT NULL
              AND UPPER(TRIM(CAST(b.{qident("EntityId")} AS VARCHAR))) <> 'CUI-LESS'
        ),
        grouped AS (
            SELECT
                EntityId,
                ANY_VALUE("Type") AS "Type",
                ANY_VALUE("Mention") AS "Mention"
            FROM bioentity_base
            GROUP BY EntityId
        )
        SELECT
            EntityId AS node_id,
            *
        FROM grouped
        """,
    )


def count_cuiless_rows(connection: Any, parquet_dir: Path, args: argparse.Namespace) -> dict[str, int]:
    counts: dict[str, int] = {}
    c18 = table_path(parquet_dir, "C18_Link_Patents_BioEntities")
    c06 = table_path(parquet_dir, "C06_Link_Papers_BioEntities")
    c13 = table_path(parquet_dir, "C13_Link_ClinicalTrials_BioEntities")

    counts["C18_Link_Patents_BioEntities"] = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {scan_sql(c18)} l
            INNER JOIN nodes_patent p
                ON {key_expr("PatentId", "l")} = p.PatentId
            WHERE UPPER(TRIM(CAST(l.{qident("EntityId")} AS VARCHAR))) = 'CUI-LESS'
            """
        ).fetchone()[0]
    )

    if args.include_paper_bioentities:
        counts["C06_Link_Papers_BioEntities"] = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {scan_sql(c06)} l
                INNER JOIN nodes_paper p
                    ON {key_expr("PMID", "l")} = p.PMID
                WHERE UPPER(TRIM(CAST(l.{qident("EntityId")} AS VARCHAR))) = 'CUI-LESS'
                """
            ).fetchone()[0]
        )
    else:
        counts["C06_Link_Papers_BioEntities"] = 0

    if args.include_trial_expansion:
        counts["C13_Link_ClinicalTrials_BioEntities"] = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {scan_sql(c13)} l
                INNER JOIN nodes_clinicaltrial t
                    ON {key_expr("nct_id", "l")} = t.nct_id
                WHERE UPPER(TRIM(CAST(l.{qident("EntityId")} AS VARCHAR))) = 'CUI-LESS'
                """
            ).fetchone()[0]
        )
    else:
        counts["C13_Link_ClinicalTrials_BioEntities"] = 0

    return counts


def create_ungrounded_cuiless_features(connection: Any, parquet_dir: Path, args: argparse.Namespace) -> None:
    if not args.include_ungrounded_cuiless_features:
        return

    c18 = table_path(parquet_dir, "C18_Link_Patents_BioEntities")
    c06 = table_path(parquet_dir, "C06_Link_Papers_BioEntities")
    c13 = table_path(parquet_dir, "C13_Link_ClinicalTrials_BioEntities")

    paper_sql = (
        f"""
        UNION ALL
        SELECT
            'paper' AS carrier_type,
            {key_expr("PMID", "l")} AS carrier_id,
            CAST(l.{qident("Type")} AS VARCHAR) AS entity_type,
            COUNT(*) AS cuiless_mention_count,
            APPROX_COUNT_DISTINCT(CAST(l.{qident("Mention")} AS VARCHAR)) AS approx_distinct_mentions,
            ANY_VALUE(CAST(l.{qident("Mention")} AS VARCHAR)) AS sample_mention,
            'C06_Link_Papers_BioEntities' AS source_table
        FROM {scan_sql(c06)} l
        INNER JOIN nodes_paper p
            ON {key_expr("PMID", "l")} = p.PMID
        WHERE UPPER(TRIM(CAST(l.{qident("EntityId")} AS VARCHAR))) = 'CUI-LESS'
        GROUP BY carrier_type, carrier_id, entity_type, source_table
        """
        if args.include_paper_bioentities
        else ""
    )

    trial_sql = (
        f"""
        UNION ALL
        SELECT
            'trial' AS carrier_type,
            {key_expr("nct_id", "l")} AS carrier_id,
            CAST(l.{qident("Type")} AS VARCHAR) AS entity_type,
            COUNT(*) AS cuiless_mention_count,
            APPROX_COUNT_DISTINCT(CAST(l.{qident("Mention")} AS VARCHAR)) AS approx_distinct_mentions,
            ANY_VALUE(CAST(l.{qident("Mention")} AS VARCHAR)) AS sample_mention,
            'C13_Link_ClinicalTrials_BioEntities' AS source_table
        FROM {scan_sql(c13)} l
        INNER JOIN nodes_clinicaltrial t
            ON {key_expr("nct_id", "l")} = t.nct_id
        WHERE UPPER(TRIM(CAST(l.{qident("EntityId")} AS VARCHAR))) = 'CUI-LESS'
        GROUP BY carrier_type, carrier_id, entity_type, source_table
        """
        if args.include_trial_expansion
        else ""
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE ungrounded_cuiless_mentions AS
        SELECT
            'patent' AS carrier_type,
            {key_expr("PatentId", "l")} AS carrier_id,
            CAST(l.{qident("Type")} AS VARCHAR) AS entity_type,
            COUNT(*) AS cuiless_mention_count,
            APPROX_COUNT_DISTINCT(CAST(l.{qident("Mention")} AS VARCHAR)) AS approx_distinct_mentions,
            ANY_VALUE(CAST(l.{qident("Mention")} AS VARCHAR)) AS sample_mention,
            'C18_Link_Patents_BioEntities' AS source_table
        FROM {scan_sql(c18)} l
        INNER JOIN nodes_patent p
            ON {key_expr("PatentId", "l")} = p.PatentId
        WHERE UPPER(TRIM(CAST(l.{qident("EntityId")} AS VARCHAR))) = 'CUI-LESS'
        GROUP BY carrier_type, carrier_id, entity_type, source_table
        {paper_sql}
        {trial_sql}
        """,
    )


def count_outputs(connection: Any, include_ungrounded: bool) -> dict[str, int]:
    counts = {name: count_table(connection, name) for _, name, _, _, _ in OUTPUT_TABLES}
    if include_ungrounded:
        counts["ungrounded_cuiless_mentions"] = count_table(connection, "ungrounded_cuiless_mentions")
    return counts


def write_outputs(
    connection: Any,
    output_dir: Path,
    compression: str,
    include_ungrounded: bool,
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []

    for artifact_type, table_name, file_name, source_tables, description in OUTPUT_TABLES:
        path = output_dir / file_name
        copy_table_to_parquet(connection, table_name, path, compression)
        row_count = count_table(connection, table_name)
        manifest_rows.append(
            {
                "artifact_type": artifact_type,
                "name": table_name,
                "path": str(path),
                "row_count": row_count,
                "source_tables": source_tables,
                "description": description,
            }
        )

    if include_ungrounded:
        path = output_dir / "ungrounded_cuiless_mentions.parquet"
        copy_table_to_parquet(connection, "ungrounded_cuiless_mentions", path, compression)
        row_count = count_table(connection, "ungrounded_cuiless_mentions")
        manifest_rows.append(
            {
                "artifact_type": "optional",
                "name": "ungrounded_cuiless_mentions",
                "path": str(path),
                "row_count": row_count,
                "source_tables": "C06/C13/C18 BioEntity link tables",
                "description": "Carrier-level summary of CUI-less ungrounded mentions.",
            }
        )

    return manifest_rows


def write_manifest(output_dir: Path, manifest_rows: list[dict[str, Any]]) -> None:
    path = output_dir / "subgraph_manifest.csv"
    columns = ["artifact_type", "name", "path", "row_count", "source_tables", "description"]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest_rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines) + "\n"


def project_match_stats(connection: Any) -> dict[str, dict[str, int]]:
    stats = {}
    pairs = [
        ("B05_Link_Patents_Projects", "edges_patent_project_raw", "edges_patent_project"),
        ("B03_Link_Papers_Projects", "edges_paper_project_raw", "edges_paper_project"),
        ("B04_Link_ClinicalTrials_Projects", "edges_trial_project_raw", "edges_trial_project"),
    ]
    for label, raw_table, final_table in pairs:
        raw_count = count_table(connection, raw_table)
        matched_count = count_table(connection, final_table)
        stats[label] = {
            "raw_project_edge_rows": raw_count,
            "matched_project_edge_rows": matched_count,
            "unmatched_project_edge_rows": raw_count - matched_count,
        }
    return stats


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    manifest_rows: list[dict[str, Any]],
    counts: dict[str, int],
    cuiless_counts: dict[str, int],
    project_stats: dict[str, dict[str, int]],
    runtime_seconds: float,
) -> None:
    node_rows = [
        {"node_type": "Patent", "count": counts.get("nodes_patent", 0)},
        {"node_type": "Paper", "count": counts.get("nodes_paper", 0)},
        {"node_type": "ClinicalTrial", "count": counts.get("nodes_clinicaltrial", 0)},
        {"node_type": "Project", "count": counts.get("nodes_project", 0)},
        {"node_type": "BioEntity", "count": counts.get("nodes_bioentity", 0)},
    ]
    edge_rows = [
        {"edge_type": "patent_paper", "count": counts.get("edges_patent_paper", 0)},
        {"edge_type": "patent_bioentity", "count": counts.get("edges_patent_bioentity", 0)},
        {"edge_type": "patent_project", "count": counts.get("edges_patent_project", 0)},
        {"edge_type": "paper_trial", "count": counts.get("edges_paper_trial", 0)},
        {"edge_type": "paper_project", "count": counts.get("edges_paper_project", 0)},
        {"edge_type": "paper_bioentity", "count": counts.get("edges_paper_bioentity", 0)},
        {"edge_type": "trial_project", "count": counts.get("edges_trial_project", 0)},
        {"edge_type": "trial_bioentity", "count": counts.get("edges_trial_bioentity", 0)},
    ]
    cuiless_rows = [
        {"source_table": key, "filtered_cuiless_rows": value}
        for key, value in cuiless_counts.items()
    ]
    project_rows = [
        {"source_table": key, **value}
        for key, value in project_stats.items()
    ]

    parts = ["# Patent-centered Subgraph Extraction Report\n"]
    parts.append("## 1. Parameters\n")
    parts.append(
        markdown_table(
            [
                {"parameter": "parquet_dir", "value": str(args.parquet_dir)},
                {"parameter": "output_dir", "value": str(args.output_dir)},
                {"parameter": "start_year", "value": args.start_year},
                {"parameter": "end_year", "value": args.end_year},
                {"parameter": "limit_patents", "value": args.limit_patents},
                {"parameter": "include_paper_bioentities", "value": args.include_paper_bioentities},
                {"parameter": "include_trial_expansion", "value": args.include_trial_expansion},
                {"parameter": "include_ungrounded_cuiless_features", "value": args.include_ungrounded_cuiless_features},
                {"parameter": "threads", "value": args.threads},
                {"parameter": "memory_limit", "value": args.memory_limit},
                {"parameter": "compression", "value": args.compression},
                {"parameter": "runtime_seconds", "value": f"{runtime_seconds:.1f}"},
            ],
            ["parameter", "value"],
        )
    )

    parts.append("\n## 2. Seed patents\n")
    parts.append(f"- Selected patent count: **{counts.get('nodes_patent', 0):,}**\n")

    parts.append("\n## 3. Node counts\n")
    parts.append(markdown_table(node_rows, ["node_type", "count"]))

    parts.append("\n## 4. Edge counts\n")
    parts.append(markdown_table(edge_rows, ["edge_type", "count"]))

    parts.append("\n## 5. CUI-less filtering\n")
    parts.append(
        "`CUI-less` means an entity mention was detected but not grounded to a standard concept ID. "
        "`CUI-less` is not a graph node in this subgraph.\n"
    )
    parts.append(markdown_table(cuiless_rows, ["source_table", "filtered_cuiless_rows"]))

    parts.append("\n## 6. ProjectNumber matching\n")
    parts.append(
        "ProjectNumber is joined to normalized `B02_Projects.CORE_PROJECT_NUM` using "
        "`UPPER(REPLACE(REPLACE(TRIM(CAST(value AS VARCHAR)), ' ', ''), '-', ''))`.\n"
    )
    parts.append(
        markdown_table(
            project_rows,
            ["source_table", "raw_project_edge_rows", "matched_project_edge_rows", "unmatched_project_edge_rows"],
        )
    )

    parts.append("\n## 7. Orphan checks\n")
    parts.append(
        "This extraction uses inner joins from edge endpoints to selected node tables where appropriate. "
        "Dropped project-link rows are reported in the ProjectNumber matching section. "
        "BioEntity edges are restricted to grounded `C23_BioEntities` and exclude `CUI-less`.\n"
    )

    parts.append("\n## 8. Output artifacts\n")
    parts.append(markdown_table(manifest_rows, ["artifact_type", "name", "path", "row_count", "source_tables", "description"]))

    parts.append("\n## 9. Next recommended step\n")
    parts.append(
        "- Inspect node and edge counts.\n"
        "- Validate that graph size is reasonable for local analysis.\n"
        "- Run `scripts/07_validate_extracted_subgraph.py` on this output directory.\n"
        "- If validation passes, proceed to mention-edge aggregation or temporal snapshot design.\n"
        "- Do not start model training yet.\n"
    )

    (output_dir / "subgraph_extraction_report.md").write_text("\n".join(parts), encoding="utf-8")


def print_summary(
    output_dir: Path,
    counts: dict[str, int],
    cuiless_counts: dict[str, int],
    project_stats: dict[str, dict[str, int]],
) -> None:
    print("Patent-centered subgraph extraction complete")
    print(f"Output directory: {output_dir}")

    print("Node counts:")
    for key in ["nodes_patent", "nodes_paper", "nodes_clinicaltrial", "nodes_project", "nodes_bioentity"]:
        print(f"- {key}: {counts.get(key, 0):,}")

    print("Edge counts:")
    for key in [
        "edges_patent_paper",
        "edges_patent_bioentity",
        "edges_patent_project",
        "edges_paper_trial",
        "edges_paper_project",
        "edges_paper_bioentity",
        "edges_trial_project",
        "edges_trial_bioentity",
    ]:
        print(f"- {key}: {counts.get(key, 0):,}")

    print("CUI-less rows filtered:")
    for key, value in cuiless_counts.items():
        print(f"- {key}: {value:,}")

    print("ProjectNumber matching:")
    for key, value in project_stats.items():
        print(
            f"- {key}: raw={value['raw_project_edge_rows']:,}, "
            f"matched={value['matched_project_edge_rows']:,}, "
            f"unmatched={value['unmatched_project_edge_rows']:,}"
        )

    print("Report: subgraph_extraction_report.md")
    print("Manifest: subgraph_manifest.csv")


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")
    if args.threads < 1:
        raise ValueError("--threads must be greater than or equal to 1")
    if args.limit_patents is not None and args.limit_patents < 1:
        raise ValueError("--limit-patents must be greater than or equal to 1")

    args.parquet_dir = args.parquet_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()

    require_files(args.parquet_dir)
    prepare_output_dir(args.output_dir, args.overwrite)

    connection = connect_duckdb(args)

    run_step(
        args,
        1,
        10,
        "Reading table schemas",
        lambda: None,
    )
    columns_cache = {table_name: get_columns(connection, args.parquet_dir, table_name) for table_name in CORE_TABLES}

    run_step(
        args,
        2,
        10,
        "Creating seed patent nodes",
        lambda: create_seed_patents(connection, args.parquet_dir, args, columns_cache),
    )
    run_step(
        args,
        3,
        10,
        "Creating patent-paper edges and paper nodes",
        lambda: create_patent_paper_edges_and_papers(connection, args.parquet_dir, columns_cache),
    )
    run_step(
        args,
        4,
        10,
        "Creating paper-trial edges and clinical trial nodes",
        lambda: create_trial_nodes_and_edges(connection, args.parquet_dir, args, columns_cache),
    )
    run_step(
        args,
        5,
        10,
        "Creating raw project edges",
        lambda: create_project_raw_edges(connection, args.parquet_dir, args, columns_cache),
    )
    run_step(
        args,
        6,
        10,
        "Creating project nodes and final project edges",
        lambda: create_project_nodes_and_final_edges(connection, args.parquet_dir, columns_cache),
    )
    run_step(
        args,
        7,
        10,
        "Creating BioEntity edges and nodes",
        lambda: create_bioentity_edges_and_nodes(connection, args.parquet_dir, args, columns_cache),
    )

    run_step(
        args,
        8,
        10,
        "Counting CUI-less rows",
        lambda: None,
    )
    cuiless_counts = count_cuiless_rows(connection, args.parquet_dir, args)

    if args.include_ungrounded_cuiless_features:
        run_step(
            args,
            9,
            10,
            "Creating ungrounded CUI-less features",
            lambda: create_ungrounded_cuiless_features(connection, args.parquet_dir, args),
        )
    elif not args.no_progress:
        print("[9/10] Skipping ungrounded CUI-less features", flush=True)

    counts = count_outputs(connection, args.include_ungrounded_cuiless_features)
    project_stats = project_match_stats(connection)

    run_step(
        args,
        10,
        10,
        "Writing Parquet outputs",
        lambda: None,
    )
    manifest_rows = write_outputs(
        connection,
        args.output_dir,
        args.compression,
        args.include_ungrounded_cuiless_features,
    )
    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "subgraph_manifest",
            "path": str(args.output_dir / "subgraph_manifest.csv"),
            "row_count": len(manifest_rows) + 2,
            "source_tables": "generated",
            "description": "Manifest of subgraph output artifacts.",
        }
    )
    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "subgraph_extraction_report",
            "path": str(args.output_dir / "subgraph_extraction_report.md"),
            "row_count": "",
            "source_tables": "generated",
            "description": "Human-readable extraction report.",
        }
    )

    write_manifest(args.output_dir, manifest_rows)

    runtime_seconds = time.time() - start_time
    write_report(
        args.output_dir,
        args,
        manifest_rows,
        counts,
        cuiless_counts,
        project_stats,
        runtime_seconds,
    )

    print_summary(args.output_dir, counts, cuiless_counts, project_stats)


if __name__ == "__main__":
    main()