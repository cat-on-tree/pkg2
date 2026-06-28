"""Extract a diabetes-specific subgraph from a validated patent-centered mother graph.

This script reads graph-ready outputs produced by scripts/06_extract_patent_centered_subgraph.py
and extracts a domain-specific induced subgraph.

Current supported domain:
- diabetes

Recommended input:
data/processed/patent_centered_subgraph_2018_2019_full_v1_1/

Recommended outputs:
data/processed/patent_centered_subgraph_2018_2019_diabetes_seed_only_v2/
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2/
data/processed/patent_centered_subgraph_2018_2019_diabetes_expanded_v2/

Important v2 changes:
- Adds --carrier-expansion-mode: seed-only, conservative, expanded.
- Defaults to conservative expansion.
- Uses boundary-aware regex matching instead of substring LIKE matching.
- Prevents short terms such as MODY, A1C, T1D, and T2D from matching inside larger words.
- Does not let seed projects reverse-expand into large numbers of papers/patents in conservative mode.

This script does not read raw PKG TSV files and does not rescan the original
large PKG Parquet tables. It works only from the already extracted mother graph.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable


NODE_FILES = {
    "patent": "nodes_patent.parquet",
    "paper": "nodes_paper.parquet",
    "clinicaltrial": "nodes_clinicaltrial.parquet",
    "project": "nodes_project.parquet",
    "bioentity": "nodes_bioentity.parquet",
}

EDGE_FILES = {
    "patent_paper": "edges_patent_paper.parquet",
    "patent_bioentity": "edges_patent_bioentity.parquet",
    "patent_project": "edges_patent_project.parquet",
    "paper_trial": "edges_paper_trial.parquet",
    "paper_project": "edges_paper_project.parquet",
    "paper_bioentity": "edges_paper_bioentity.parquet",
    "trial_project": "edges_trial_project.parquet",
    "trial_bioentity": "edges_trial_bioentity.parquet",
}

OUTPUT_TABLES = [
    ("node", "nodes_patent", "nodes_patent.parquet", "Patent nodes selected for the domain subgraph."),
    ("node", "nodes_paper", "nodes_paper.parquet", "Paper nodes selected for the domain subgraph."),
    ("node", "nodes_clinicaltrial", "nodes_clinicaltrial.parquet", "ClinicalTrial nodes selected for the domain subgraph."),
    ("node", "nodes_project", "nodes_project.parquet", "Project nodes selected for the domain subgraph."),
    ("node", "nodes_bioentity", "nodes_bioentity.parquet", "BioEntity nodes selected for the domain subgraph."),
    ("edge", "edges_patent_paper", "edges_patent_paper.parquet", "Patent to paper edges in the domain subgraph."),
    ("edge", "edges_patent_bioentity", "edges_patent_bioentity.parquet", "Patent to BioEntity edges in the domain subgraph."),
    ("edge", "edges_patent_project", "edges_patent_project.parquet", "Patent to Project edges in the domain subgraph."),
    ("edge", "edges_paper_trial", "edges_paper_trial.parquet", "Paper to ClinicalTrial edges in the domain subgraph."),
    ("edge", "edges_paper_project", "edges_paper_project.parquet", "Paper to Project edges in the domain subgraph."),
    ("edge", "edges_paper_bioentity", "edges_paper_bioentity.parquet", "Paper to BioEntity edges in the domain subgraph."),
    ("edge", "edges_trial_project", "edges_trial_project.parquet", "ClinicalTrial to Project edges in the domain subgraph."),
    ("edge", "edges_trial_bioentity", "edges_trial_bioentity.parquet", "ClinicalTrial to BioEntity edges in the domain subgraph."),
    ("audit", "domain_seed_carriers", "domain_seed_carriers.parquet", "Domain seed carriers and evidence counts."),
    ("audit", "domain_keyword_hits", "domain_keyword_hits.parquet", "Keyword/evidence hits used for domain seed selection."),
]

# High-precision terms. These terms are allowed to create domain seed carriers.
DIABETES_TIER1_TERMS = [
    "diabetes",
    "diabetic",
    "diabetes mellitus",
    "type 1 diabetes",
    "type 2 diabetes",
    "type i diabetes",
    "type ii diabetes",
    "t1d",
    "t2d",
    "gestational diabetes",
    "prediabetes",
    "pre-diabetes",
    "insulin resistance",
    "hyperglycemia",
    "hypoglycemia",
    "hba1c",
    "hemoglobin a1c",
    "a1c",
    "diabetic nephropathy",
    "diabetic retinopathy",
    "diabetic neuropathy",
    "diabetic ketoacidosis",
    "diabetic foot",
    "diabetic kidney disease",
    "diabetic macular edema",
    "maturity onset diabetes",
    "maturity-onset diabetes",
    "maturity onset diabetes of the young",
    "mody",
]

# Relevant but broader terms. By default these are reported but do not create seeds
# unless --include-tier2-seeds true is used.
DIABETES_TIER2_TERMS = [
    "metformin",
    "glp-1",
    "glp1",
    "glp-1 receptor",
    "glp1 receptor",
    "glp-1 receptor agonist",
    "liraglutide",
    "semaglutide",
    "exenatide",
    "dulaglutide",
    "sglt2",
    "sglt-2",
    "sglt2 inhibitor",
    "sglt-2 inhibitor",
    "empagliflozin",
    "dapagliflozin",
    "canagliflozin",
    "ertugliflozin",
    "dpp-4",
    "dpp4",
    "dpp-4 inhibitor",
    "dpp4 inhibitor",
    "sitagliptin",
    "linagliptin",
    "saxagliptin",
    "alogliptin",
    "insulin receptor",
    "pancreatic beta cell",
    "beta cell",
    "islet",
    "islets",
]

# Broad metabolic terms. These are intentionally not used as seeds by default.
# They are included in the domain term table for auditing if needed later.
DIABETES_TIER3_TERMS = [
    "glucose",
    "insulin",
    "metabolism",
    "obesity",
    "adipose",
    "lipid",
    "pancreas",
    "inflammation",
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a domain-specific subgraph from a patent-centered mother graph.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/patent_centered_subgraph_2018_2019_full_v1_1"),
        help="Input mother graph directory produced by scripts/06_extract_patent_centered_subgraph.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2"),
        help="Output directory for the extracted domain subgraph.",
    )
    parser.add_argument("--domain", default="diabetes", choices=["diabetes"])
    parser.add_argument(
        "--carrier-expansion-mode",
        choices=["seed-only", "conservative", "expanded"],
        default="conservative",
        help=(
            "seed-only keeps only directly matched carriers; "
            "conservative adds limited patent/paper/trial/project context without project reverse-expansion; "
            "expanded reproduces broad v1-style neighborhood expansion."
        ),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="20GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("data/interim/duckdb_tmp"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--bioentity-mode",
        choices=["all", "matched"],
        default="all",
        help=(
            "'all' keeps all BioEntity edges from selected carriers. "
            "'matched' keeps only BioEntity edges whose target BioEntity had a domain keyword hit."
        ),
    )
    parser.add_argument(
        "--include-tier2-seeds",
        type=parse_bool,
        default=False,
        help="If true, Tier 2 diabetes-related terms can create seed carriers. Default false.",
    )
    parser.add_argument(
        "--include-tier3-hits",
        type=parse_bool,
        default=False,
        help="If true, broad Tier 3 terms are included in keyword hit auditing. They never create seeds.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def qident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def scan_sql(path: Path) -> str:
    return f"read_parquet({sql_literal(path)})"


def connect_duckdb(args: argparse.Namespace):
    import duckdb

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute(f"SET temp_directory={sql_literal(args.temp_dir)}")
    return connection


def require_files(input_dir: Path) -> None:
    required = list(NODE_FILES.values()) + list(EDGE_FILES.values())
    missing = [str(input_dir / file_name) for file_name in required if not (input_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError("Missing required mother graph files:\n" + "\n".join(missing))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_step(args: argparse.Namespace, index: int, total: int, label: str, fn: Callable[[], None]) -> None:
    if not args.no_progress:
        print(f"[{index}/{total}] {label} ...", flush=True)
    start = time.time()
    fn()
    elapsed = time.time() - start
    if not args.no_progress:
        print(f"[{index}/{total}] {label} done in {elapsed:.1f}s", flush=True)


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


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def regex_for_term(term: str) -> str:
    """Create a boundary-aware, lowercase regex for DuckDB regexp_matches.

    Spaces and hyphens inside terms are made flexible so that e.g.
    "type 2 diabetes" can match "type-2 diabetes", and "pre-diabetes"
    can match "pre diabetes". Terms are still bounded by non-alphanumeric
    characters to avoid false positives such as MODY matching hemodynamic.
    """
    normalized = term.strip().lower()
    pieces = re.split(r"[\s\-]+", normalized)
    escaped_pieces = [re.escape(piece) for piece in pieces if piece]
    core = r"[\s\-]+".join(escaped_pieces)
    return rf"(^|[^a-z0-9]){core}([^a-z0-9]|$)"


def create_views(connection: Any, input_dir: Path) -> None:
    for logical_name, file_name in NODE_FILES.items():
        view_name = f"mother_nodes_{logical_name}"
        execute(
            connection,
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(view_name)} AS
            SELECT * FROM {scan_sql(input_dir / file_name)}
            """,
        )

    for logical_name, file_name in EDGE_FILES.items():
        view_name = f"mother_edges_{logical_name}"
        execute(
            connection,
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(view_name)} AS
            SELECT * FROM {scan_sql(input_dir / file_name)}
            """,
        )


def create_domain_terms(connection: Any, args: argparse.Namespace) -> None:
    terms: list[tuple[str, str, str, bool]] = []

    for term in DIABETES_TIER1_TERMS:
        terms.append(("tier1", term.lower(), regex_for_term(term), True))

    for term in DIABETES_TIER2_TERMS:
        terms.append(("tier2", term.lower(), regex_for_term(term), bool(args.include_tier2_seeds)))

    if args.include_tier3_hits:
        for term in DIABETES_TIER3_TERMS:
            terms.append(("tier3", term.lower(), regex_for_term(term), False))

    values_sql = ",\n".join(
        f"({sql_literal(tier)}, {sql_literal(term)}, {sql_literal(pattern)}, {str(seed_enabled).upper()})"
        for tier, term, pattern, seed_enabled in terms
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE domain_terms AS
        SELECT *
        FROM (
            VALUES
            {values_sql}
        ) AS t(tier, term, regex_pattern, seed_enabled)
        """,
    )


def create_carrier_text_sources(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE carrier_text_sources AS
        SELECT
            'Patent' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_patent' AS source_table,
            'Title' AS field_name,
            CAST("Title" AS VARCHAR) AS text_value
        FROM mother_nodes_patent
        WHERE "Title" IS NOT NULL

        UNION ALL
        SELECT
            'Patent' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_patent' AS source_table,
            'Abstract' AS field_name,
            CAST("Abstract" AS VARCHAR) AS text_value
        FROM mother_nodes_patent
        WHERE "Abstract" IS NOT NULL

        UNION ALL
        SELECT
            'Paper' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_paper' AS source_table,
            'ArticleTitle' AS field_name,
            CAST("ArticleTitle" AS VARCHAR) AS text_value
        FROM mother_nodes_paper
        WHERE "ArticleTitle" IS NOT NULL

        UNION ALL
        SELECT
            'ClinicalTrial' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_clinicaltrial' AS source_table,
            'brief_title' AS field_name,
            CAST("brief_title" AS VARCHAR) AS text_value
        FROM mother_nodes_clinicaltrial
        WHERE "brief_title" IS NOT NULL

        UNION ALL
        SELECT
            'ClinicalTrial' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_clinicaltrial' AS source_table,
            'official_title' AS field_name,
            CAST("official_title" AS VARCHAR) AS text_value
        FROM mother_nodes_clinicaltrial
        WHERE "official_title" IS NOT NULL

        UNION ALL
        SELECT
            'ClinicalTrial' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_clinicaltrial' AS source_table,
            'conditions' AS field_name,
            CAST("conditions" AS VARCHAR) AS text_value
        FROM mother_nodes_clinicaltrial
        WHERE "conditions" IS NOT NULL

        UNION ALL
        SELECT
            'ClinicalTrial' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_clinicaltrial' AS source_table,
            'keywords' AS field_name,
            CAST("keywords" AS VARCHAR) AS text_value
        FROM mother_nodes_clinicaltrial
        WHERE "keywords" IS NOT NULL

        UNION ALL
        SELECT
            'Project' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_project' AS source_table,
            'PROJECT_TITLE' AS field_name,
            CAST("PROJECT_TITLE" AS VARCHAR) AS text_value
        FROM mother_nodes_project
        WHERE "PROJECT_TITLE" IS NOT NULL

        UNION ALL
        SELECT
            'Project' AS carrier_type,
            CAST(node_id AS VARCHAR) AS carrier_id,
            'nodes_project' AS source_table,
            'Abstract' AS field_name,
            CAST("Abstract" AS VARCHAR) AS text_value
        FROM mother_nodes_project
        WHERE "Abstract" IS NOT NULL
        """,
    )


def create_keyword_hits(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE domain_keyword_hits AS
        SELECT
            c.carrier_type,
            c.carrier_id,
            c.source_table,
            c.field_name,
            t.tier,
            t.term AS matched_term,
            t.seed_enabled,
            'text' AS evidence_type,
            CAST(NULL AS VARCHAR) AS bioentity_id,
            CAST(NULL AS VARCHAR) AS bioentity_type,
            CAST(NULL AS VARCHAR) AS edge_table
        FROM carrier_text_sources c
        INNER JOIN domain_terms t
            ON regexp_matches(LOWER(c.text_value), t.regex_pattern)

        UNION ALL
        SELECT
            'Patent' AS carrier_type,
            CAST(e.source_id AS VARCHAR) AS carrier_id,
            'edges_patent_bioentity' AS source_table,
            'Mention' AS field_name,
            t.tier,
            t.term AS matched_term,
            t.seed_enabled,
            'bioentity_mention' AS evidence_type,
            CAST(e.target_id AS VARCHAR) AS bioentity_id,
            CAST(e."Type" AS VARCHAR) AS bioentity_type,
            'edges_patent_bioentity' AS edge_table
        FROM mother_edges_patent_bioentity e
        INNER JOIN domain_terms t
            ON regexp_matches(LOWER(CAST(e."Mention" AS VARCHAR)), t.regex_pattern)
        WHERE e."Mention" IS NOT NULL

        UNION ALL
        SELECT
            'Paper' AS carrier_type,
            CAST(e.source_id AS VARCHAR) AS carrier_id,
            'edges_paper_bioentity' AS source_table,
            'Mention' AS field_name,
            t.tier,
            t.term AS matched_term,
            t.seed_enabled,
            'bioentity_mention' AS evidence_type,
            CAST(e.target_id AS VARCHAR) AS bioentity_id,
            CAST(e."Type" AS VARCHAR) AS bioentity_type,
            'edges_paper_bioentity' AS edge_table
        FROM mother_edges_paper_bioentity e
        INNER JOIN domain_terms t
            ON regexp_matches(LOWER(CAST(e."Mention" AS VARCHAR)), t.regex_pattern)
        WHERE e."Mention" IS NOT NULL

        UNION ALL
        SELECT
            'ClinicalTrial' AS carrier_type,
            CAST(e.source_id AS VARCHAR) AS carrier_id,
            'edges_trial_bioentity' AS source_table,
            'Mention' AS field_name,
            t.tier,
            t.term AS matched_term,
            t.seed_enabled,
            'bioentity_mention' AS evidence_type,
            CAST(e.target_id AS VARCHAR) AS bioentity_id,
            CAST(e."Type" AS VARCHAR) AS bioentity_type,
            'edges_trial_bioentity' AS edge_table
        FROM mother_edges_trial_bioentity e
        INNER JOIN domain_terms t
            ON regexp_matches(LOWER(CAST(e."Mention" AS VARCHAR)), t.regex_pattern)
        WHERE e."Mention" IS NOT NULL
        """,
    )


def create_seed_carriers(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE domain_seed_carriers AS
        SELECT
            carrier_type,
            carrier_id,
            COUNT(*) AS hit_count,
            COUNT(DISTINCT matched_term) AS distinct_term_count,
            COUNT(*) FILTER (WHERE tier = 'tier1') AS tier1_hit_count,
            COUNT(*) FILTER (WHERE tier = 'tier2') AS tier2_hit_count,
            COUNT(*) FILTER (WHERE tier = 'tier3') AS tier3_hit_count,
            COUNT(*) FILTER (WHERE evidence_type = 'text') AS text_hit_count,
            COUNT(*) FILTER (WHERE evidence_type = 'bioentity_mention') AS bioentity_hit_count,
            ANY_VALUE(matched_term) AS sample_term,
            ANY_VALUE(evidence_type) AS sample_evidence_type
        FROM domain_keyword_hits
        WHERE seed_enabled
        GROUP BY carrier_type, carrier_id
        """,
    )


def create_selected_carriers_seed_only(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE selected_patent_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Patent'
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_paper_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Paper'
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_trial_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'ClinicalTrial'
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_project_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Project'
        """,
    )


def create_selected_carriers_conservative(connection: Any) -> None:
    # Conservative mode intentionally prevents seed projects from reverse-expanding
    # into many patents and papers. Projects are kept as context/funding nodes.
    execute(
        connection,
        """
        CREATE TEMP TABLE selected_patent_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Patent'

        UNION

        SELECT DISTINCT CAST(e.source_id AS VARCHAR) AS node_id
        FROM mother_edges_patent_paper e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'Paper'
           AND CAST(e.target_id AS VARCHAR) = s.carrier_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_paper_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Paper'

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_patent_paper e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'Patent'
           AND CAST(e.source_id AS VARCHAR) = s.carrier_id

        UNION

        SELECT DISTINCT CAST(e.source_id AS VARCHAR) AS node_id
        FROM mother_edges_paper_trial e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'ClinicalTrial'
           AND CAST(e.target_id AS VARCHAR) = s.carrier_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_trial_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'ClinicalTrial'

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_paper_trial e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'Paper'
           AND CAST(e.source_id AS VARCHAR) = s.carrier_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_project_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Project'

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_patent_project e
        INNER JOIN selected_patent_ids p
            ON CAST(e.source_id AS VARCHAR) = p.node_id

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_paper_project e
        INNER JOIN selected_paper_ids p
            ON CAST(e.source_id AS VARCHAR) = p.node_id

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_trial_project e
        INNER JOIN selected_trial_ids t
            ON CAST(e.source_id AS VARCHAR) = t.node_id
        """,
    )


def create_selected_carriers_expanded(connection: Any) -> None:
    # Expanded mode reproduces broad v1-style neighborhood expansion.
    execute(
        connection,
        """
        CREATE TEMP TABLE selected_patent_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Patent'

        UNION

        SELECT DISTINCT CAST(e.source_id AS VARCHAR) AS node_id
        FROM mother_edges_patent_paper e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'Paper'
           AND CAST(e.target_id AS VARCHAR) = s.carrier_id

        UNION

        SELECT DISTINCT CAST(e.source_id AS VARCHAR) AS node_id
        FROM mother_edges_patent_project e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'Project'
           AND CAST(e.target_id AS VARCHAR) = s.carrier_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_paper_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Paper'

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_patent_paper e
        INNER JOIN selected_patent_ids p
            ON CAST(e.source_id AS VARCHAR) = p.node_id

        UNION

        SELECT DISTINCT CAST(e.source_id AS VARCHAR) AS node_id
        FROM mother_edges_paper_trial e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'ClinicalTrial'
           AND CAST(e.target_id AS VARCHAR) = s.carrier_id

        UNION

        SELECT DISTINCT CAST(e.source_id AS VARCHAR) AS node_id
        FROM mother_edges_paper_project e
        INNER JOIN domain_seed_carriers s
            ON s.carrier_type = 'Project'
           AND CAST(e.target_id AS VARCHAR) = s.carrier_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_trial_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'ClinicalTrial'

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_paper_trial e
        INNER JOIN selected_paper_ids p
            ON CAST(e.source_id AS VARCHAR) = p.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE selected_project_ids AS
        SELECT DISTINCT carrier_id AS node_id
        FROM domain_seed_carriers
        WHERE carrier_type = 'Project'

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_patent_project e
        INNER JOIN selected_patent_ids p
            ON CAST(e.source_id AS VARCHAR) = p.node_id

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_paper_project e
        INNER JOIN selected_paper_ids p
            ON CAST(e.source_id AS VARCHAR) = p.node_id

        UNION

        SELECT DISTINCT CAST(e.target_id AS VARCHAR) AS node_id
        FROM mother_edges_trial_project e
        INNER JOIN selected_trial_ids t
            ON CAST(e.source_id AS VARCHAR) = t.node_id
        """,
    )


def create_selected_carriers(connection: Any, args: argparse.Namespace) -> None:
    if args.carrier_expansion_mode == "seed-only":
        create_selected_carriers_seed_only(connection)
    elif args.carrier_expansion_mode == "conservative":
        create_selected_carriers_conservative(connection)
    elif args.carrier_expansion_mode == "expanded":
        create_selected_carriers_expanded(connection)
    else:
        raise ValueError(f"Unsupported carrier expansion mode: {args.carrier_expansion_mode}")


def create_domain_edges(connection: Any, args: argparse.Namespace) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE edges_patent_paper AS
        SELECT e.*
        FROM mother_edges_patent_paper e
        INNER JOIN selected_patent_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        INNER JOIN selected_paper_ids t
            ON CAST(e.target_id AS VARCHAR) = t.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_patent_project AS
        SELECT e.*
        FROM mother_edges_patent_project e
        INNER JOIN selected_patent_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        INNER JOIN selected_project_ids t
            ON CAST(e.target_id AS VARCHAR) = t.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_paper_trial AS
        SELECT e.*
        FROM mother_edges_paper_trial e
        INNER JOIN selected_paper_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        INNER JOIN selected_trial_ids t
            ON CAST(e.target_id AS VARCHAR) = t.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_paper_project AS
        SELECT e.*
        FROM mother_edges_paper_project e
        INNER JOIN selected_paper_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        INNER JOIN selected_project_ids t
            ON CAST(e.target_id AS VARCHAR) = t.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE edges_trial_project AS
        SELECT e.*
        FROM mother_edges_trial_project e
        INNER JOIN selected_trial_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        INNER JOIN selected_project_ids t
            ON CAST(e.target_id AS VARCHAR) = t.node_id
        """,
    )

    if args.bioentity_mode == "matched":
        execute(
            connection,
            """
            CREATE TEMP TABLE matched_domain_bioentity_ids AS
            SELECT DISTINCT bioentity_id AS node_id
            FROM domain_keyword_hits
            WHERE bioentity_id IS NOT NULL
              AND seed_enabled
            """,
        )

        bioentity_join = """
            INNER JOIN matched_domain_bioentity_ids b
                ON CAST(e.target_id AS VARCHAR) = b.node_id
        """
    else:
        bioentity_join = ""

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_patent_bioentity AS
        SELECT e.*
        FROM mother_edges_patent_bioentity e
        INNER JOIN selected_patent_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        {bioentity_join}
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_paper_bioentity AS
        SELECT e.*
        FROM mother_edges_paper_bioentity e
        INNER JOIN selected_paper_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        {bioentity_join}
        """,
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE edges_trial_bioentity AS
        SELECT e.*
        FROM mother_edges_trial_bioentity e
        INNER JOIN selected_trial_ids s
            ON CAST(e.source_id AS VARCHAR) = s.node_id
        {bioentity_join}
        """,
    )


def create_domain_nodes(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE selected_bioentity_ids AS
        SELECT DISTINCT CAST(target_id AS VARCHAR) AS node_id FROM edges_patent_bioentity
        UNION
        SELECT DISTINCT CAST(target_id AS VARCHAR) AS node_id FROM edges_paper_bioentity
        UNION
        SELECT DISTINCT CAST(target_id AS VARCHAR) AS node_id FROM edges_trial_bioentity
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE nodes_patent AS
        SELECT n.*
        FROM mother_nodes_patent n
        INNER JOIN selected_patent_ids s
            ON CAST(n.node_id AS VARCHAR) = s.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE nodes_paper AS
        SELECT n.*
        FROM mother_nodes_paper n
        INNER JOIN selected_paper_ids s
            ON CAST(n.node_id AS VARCHAR) = s.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE nodes_clinicaltrial AS
        SELECT n.*
        FROM mother_nodes_clinicaltrial n
        INNER JOIN selected_trial_ids s
            ON CAST(n.node_id AS VARCHAR) = s.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE nodes_project AS
        SELECT n.*
        FROM mother_nodes_project n
        INNER JOIN selected_project_ids s
            ON CAST(n.node_id AS VARCHAR) = s.node_id
        """,
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE nodes_bioentity AS
        SELECT n.*
        FROM mother_nodes_bioentity n
        INNER JOIN selected_bioentity_ids s
            ON CAST(n.node_id AS VARCHAR) = s.node_id
        """,
    )


def count_outputs(connection: Any) -> dict[str, int]:
    return {table_name: count_table(connection, table_name) for _, table_name, _, _ in OUTPUT_TABLES}


def get_keyword_summary(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            tier,
            matched_term,
            evidence_type,
            carrier_type,
            COUNT(*) AS hit_count,
            COUNT(DISTINCT carrier_id) AS distinct_carrier_count
        FROM domain_keyword_hits
        GROUP BY tier, matched_term, evidence_type, carrier_type
        ORDER BY hit_count DESC, tier, matched_term
        LIMIT 200
        """
    ).fetchall()

    return [
        {
            "tier": row[0],
            "matched_term": row[1],
            "evidence_type": row[2],
            "carrier_type": row[3],
            "hit_count": row[4],
            "distinct_carrier_count": row[5],
        }
        for row in rows
    ]


def get_seed_summary(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            carrier_type,
            COUNT(*) AS seed_carrier_count,
            SUM(hit_count) AS total_hits,
            SUM(tier1_hit_count) AS tier1_hits,
            SUM(tier2_hit_count) AS tier2_hits,
            SUM(text_hit_count) AS text_hits,
            SUM(bioentity_hit_count) AS bioentity_hits
        FROM domain_seed_carriers
        GROUP BY carrier_type
        ORDER BY carrier_type
        """
    ).fetchall()

    return [
        {
            "carrier_type": row[0],
            "seed_carrier_count": row[1],
            "total_hits": row[2],
            "tier1_hits": row[3],
            "tier2_hits": row[4],
            "text_hits": row[5],
            "bioentity_hits": row[6],
        }
        for row in rows
    ]


def get_top_bioentities(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            COALESCE(CAST(n."Type" AS VARCHAR), '') AS bioentity_type,
            CAST(n.node_id AS VARCHAR) AS node_id,
            COALESCE(CAST(n."Mention" AS VARCHAR), '') AS mention,
            COUNT(*) AS edge_count
        FROM nodes_bioentity n
        INNER JOIN (
            SELECT target_id FROM edges_patent_bioentity
            UNION ALL
            SELECT target_id FROM edges_paper_bioentity
            UNION ALL
            SELECT target_id FROM edges_trial_bioentity
        ) e
            ON CAST(n.node_id AS VARCHAR) = CAST(e.target_id AS VARCHAR)
        GROUP BY bioentity_type, node_id, mention
        ORDER BY edge_count DESC
        LIMIT 100
        """
    ).fetchall()

    return [
        {
            "bioentity_type": row[0],
            "node_id": row[1],
            "mention": row[2],
            "edge_count": row[3],
        }
        for row in rows
    ]


def write_outputs(connection: Any, args: argparse.Namespace, counts: dict[str, int]) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []

    for artifact_type, table_name, file_name, description in OUTPUT_TABLES:
        path = args.output_dir / file_name
        copy_table_to_parquet(connection, table_name, path, args.compression)
        manifest_rows.append(
            {
                "artifact_type": artifact_type,
                "name": table_name,
                "path": str(path),
                "row_count": counts.get(table_name, ""),
                "description": description,
            }
        )

    return manifest_rows


def write_manifest(output_dir: Path, manifest_rows: list[dict[str, Any]]) -> None:
    path = output_dir / "domain_manifest.csv"
    columns = ["artifact_type", "name", "path", "row_count", "description"]
    write_csv(path, manifest_rows, columns)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    counts: dict[str, int],
    seed_summary: list[dict[str, Any]],
    keyword_summary: list[dict[str, Any]],
    top_bioentities: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
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

    parts = ["# Diabetes Domain Subgraph Extraction Report\n"]

    parts.append("## 1. Parameters\n")
    parts.append(
        markdown_table(
            [
                {"parameter": "domain", "value": args.domain},
                {"parameter": "input_dir", "value": str(args.input_dir)},
                {"parameter": "output_dir", "value": str(args.output_dir)},
                {"parameter": "carrier_expansion_mode", "value": args.carrier_expansion_mode},
                {"parameter": "bioentity_mode", "value": args.bioentity_mode},
                {"parameter": "include_tier2_seeds", "value": args.include_tier2_seeds},
                {"parameter": "include_tier3_hits", "value": args.include_tier3_hits},
                {"parameter": "threads", "value": args.threads},
                {"parameter": "memory_limit", "value": args.memory_limit},
                {"parameter": "compression", "value": args.compression},
                {"parameter": "runtime_seconds", "value": f"{runtime_seconds:.1f}"},
            ],
            ["parameter", "value"],
        )
    )

    parts.append("\n## 2. Domain definition\n")
    parts.append(
        "The diabetes v2 filter uses high-precision Tier 1 diabetes terms as seed evidence. "
        "Keyword matching is boundary-aware regex matching, not substring matching, so short terms "
        "such as `MODY`, `A1C`, `T1D`, and `T2D` should not match inside unrelated words. "
        "Tier 2 pharmacologic/mechanistic terms are reported but do not create seed carriers unless "
        "`--include-tier2-seeds true` is used. Broad Tier 3 metabolic terms are disabled by default "
        "and never create seeds.\n"
    )
    parts.append(
        f"The carrier expansion mode is `{args.carrier_expansion_mode}`. "
        "`seed-only` keeps only direct keyword/evidence seed carriers. "
        "`conservative` adds limited patent/paper/trial context and project context without allowing "
        "seed projects to reverse-expand into large paper/patent neighborhoods. "
        "`expanded` reproduces broad v1-style neighborhood expansion.\n"
    )

    parts.append("\n## 3. Seed carrier summary\n")
    parts.append(
        markdown_table(
            seed_summary,
            [
                "carrier_type",
                "seed_carrier_count",
                "total_hits",
                "tier1_hits",
                "tier2_hits",
                "text_hits",
                "bioentity_hits",
            ],
        )
    )

    parts.append("\n## 4. Node counts\n")
    parts.append(markdown_table(node_rows, ["node_type", "count"]))

    parts.append("\n## 5. Edge counts\n")
    parts.append(markdown_table(edge_rows, ["edge_type", "count"]))

    parts.append("\n## 6. Top keyword hits\n")
    parts.append(
        markdown_table(
            keyword_summary[:50],
            ["tier", "matched_term", "evidence_type", "carrier_type", "hit_count", "distinct_carrier_count"],
        )
    )

    parts.append("\n## 7. Top BioEntities in domain subgraph\n")
    parts.append(
        markdown_table(
            top_bioentities[:50],
            ["bioentity_type", "node_id", "mention", "edge_count"],
        )
    )

    parts.append("\n## 8. Output artifacts\n")
    parts.append(markdown_table(manifest_rows, ["artifact_type", "name", "path", "row_count", "description"]))

    parts.append("\n## 9. Recommended next step\n")
    parts.append(
        "- Run `scripts/07_validate_extracted_subgraph.py` on this domain output directory.\n"
        "- Inspect `domain_keyword_hits.parquet` and `domain_seed_carriers.parquet` for false positives.\n"
        "- Confirm that `MODY`, `A1C`, `T1D`, and `T2D` no longer create substring false positives.\n"
        "- Review top BioEntities for overly broad hubs such as generic species, cancer, or metabolism terms.\n"
        "- If the domain is too broad, try `--carrier-expansion-mode seed-only` or `--bioentity-mode matched`.\n"
        "- If the domain is too narrow, consider enabling selected Tier 2 seed terms manually in a later version.\n"
    )

    (output_dir / "domain_filter_report.md").write_text("\n".join(parts), encoding="utf-8")


def print_summary(counts: dict[str, int], seed_summary: list[dict[str, Any]], output_dir: Path) -> None:
    print("Domain subgraph extraction complete")
    print(f"Output directory: {output_dir}")

    print("Seed carriers:")
    for row in seed_summary:
        print(f"- {row['carrier_type']}: {row['seed_carrier_count']:,}")

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

    print("Report: domain_filter_report.md")
    print("Manifest: domain_manifest.csv")


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if args.threads < 1:
        raise ValueError("--threads must be greater than or equal to 1")

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()

    require_files(args.input_dir)
    prepare_output_dir(args.output_dir, args.overwrite)

    connection = connect_duckdb(args)

    run_step(args, 1, 9, "Creating mother graph views", lambda: create_views(connection, args.input_dir))
    run_step(args, 2, 9, "Creating domain term table", lambda: create_domain_terms(connection, args))
    run_step(args, 3, 9, "Creating carrier text sources", lambda: create_carrier_text_sources(connection))
    run_step(args, 4, 9, "Creating boundary-aware keyword and BioEntity mention hits", lambda: create_keyword_hits(connection))
    run_step(args, 5, 9, "Creating domain seed carriers", lambda: create_seed_carriers(connection))
    run_step(args, 6, 9, f"Selecting domain carriers ({args.carrier_expansion_mode})", lambda: create_selected_carriers(connection, args))
    run_step(args, 7, 9, "Creating domain edges", lambda: create_domain_edges(connection, args))
    run_step(args, 8, 9, "Creating domain nodes", lambda: create_domain_nodes(connection))

    counts = count_outputs(connection)
    seed_summary = get_seed_summary(connection)
    keyword_summary = get_keyword_summary(connection)
    top_bioentities = get_top_bioentities(connection)

    run_step(args, 9, 9, "Writing domain outputs", lambda: None)
    manifest_rows = write_outputs(connection, args, counts)

    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "domain_manifest",
            "path": str(args.output_dir / "domain_manifest.csv"),
            "row_count": len(manifest_rows) + 2,
            "description": "Manifest of domain subgraph output artifacts.",
        }
    )
    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "domain_filter_report",
            "path": str(args.output_dir / "domain_filter_report.md"),
            "row_count": "",
            "description": "Human-readable domain extraction report.",
        }
    )

    write_manifest(args.output_dir, manifest_rows)

    runtime_seconds = time.time() - start_time
    write_report(
        args.output_dir,
        args,
        counts,
        seed_summary,
        keyword_summary,
        top_bioentities,
        manifest_rows,
        runtime_seconds,
    )

    print_summary(counts, seed_summary, args.output_dir)


if __name__ == "__main__":
    main()