"""Analyze PKG24S4 profile CSVs and write rule-based Markdown decision reports.

This script summarizes the profiling outputs from scripts/03_profile_parquet_tables.py
and writes two Markdown reports:

- data/interim/profile/profile_analysis.md
- data/interim/profile/recommended_subgraph_plan.md

Important:
This is a decision-support script. It does not extract a graph, does not read raw data,
and does not modify Parquet files.

The first-pass graph recommendation is deliberately rule-based because purely
heuristic profile outputs can misclassify Link tables as node tables and attribute
tables as edge tables.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


PROFILE_FILES = {
    "table": "table_profile.csv",
    "column": "column_profile.csv",
    "link": "link_table_profile.csv",
    "entity": "entity_candidate_tables.csv",
    "join": "join_candidate_edges.csv",
}

CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}


CORE_NODE_RULES = [
    {
        "node_type": "Paper",
        "source_table": "C01_Papers",
        "id_column": "PMID",
        "label_columns": "ArticleTitle",
        "year_columns": "PubYear",
        "include_in_v1": "yes",
        "reason": "Core paper node. Use PMID as the graph key; internal id is table-local.",
    },
    {
        "node_type": "BioEntity",
        "source_table": "C23_BioEntities",
        "id_column": "EntityId",
        "label_columns": "Mention, Type",
        "year_columns": "",
        "include_in_v1": "yes",
        "reason": "Core biomedical entity node linked from papers, clinical trials, and patents.",
    },
    {
        "node_type": "Project",
        "source_table": "B02_Projects",
        "id_column": "needs audit: ProjectNumber likely joins to FULL_PROJECT_NUM or CORE_PROJECT_NUM",
        "label_columns": "PROJECT_TITLE, Abstract, ORG_NAME, PI_NAMEs",
        "year_columns": "FiscalYear, SUPPORT_YEAR, PROJECT_START, PROJECT_END",
        "include_in_v1": "yes",
        "reason": "Core NIH/project funding node. ProjectNumber join key must be audited before extraction.",
    },
    {
        "node_type": "ClinicalTrial",
        "source_table": "C11_ClinicalTrials",
        "id_column": "nct_id",
        "label_columns": "brief_title, official_title, keywords, conditions",
        "year_columns": "study_first_submitted_date, start_date, completion_date",
        "include_in_v1": "yes",
        "reason": "Core clinical translation node.",
    },
    {
        "node_type": "Patent",
        "source_table": "C15_Patents",
        "id_column": "PatentId",
        "label_columns": "Title, Abstract",
        "year_columns": "GrantedDate",
        "include_in_v1": "yes",
        "reason": "Core translational output node for patent-centered analysis.",
    },
]


CORE_EDGE_RULES = [
    {
        "edge_type": "paper_mentions_bioentity",
        "source_table": "C06_Link_Papers_BioEntities",
        "source_id_column": "PMID",
        "target_id_column": "EntityId",
        "include_in_v1": "yes_filtered",
        "reason": "Core paper-bioentity edge. Very large table; extract only for selected PMIDs or selected year range.",
    },
    {
        "edge_type": "trial_mentions_bioentity",
        "source_table": "C13_Link_ClinicalTrials_BioEntities",
        "source_id_column": "nct_id",
        "target_id_column": "EntityId",
        "include_in_v1": "yes",
        "reason": "Connects clinical trials to normalized biomedical entities.",
    },
    {
        "edge_type": "patent_mentions_bioentity",
        "source_table": "C18_Link_Patents_BioEntities",
        "source_id_column": "PatentId",
        "target_id_column": "EntityId",
        "include_in_v1": "yes",
        "reason": "Connects patents to normalized biomedical entities.",
    },
    {
        "edge_type": "paper_linked_trial",
        "source_table": "C12_Link_Papers_Clinicaltrials",
        "source_id_column": "PMID",
        "target_id_column": "nct_id",
        "include_in_v1": "yes",
        "reason": "Connects papers to clinical trials.",
    },
    {
        "edge_type": "patent_links_paper",
        "source_table": "C16_Link_Patents_Papers",
        "source_id_column": "PatentId",
        "target_id_column": "PMID",
        "include_in_v1": "yes",
        "reason": "Core patent-paper linkage; useful for patent-centered translational analysis.",
    },
    {
        "edge_type": "paper_linked_project",
        "source_table": "B03_Link_Papers_Projects",
        "source_id_column": "PMID",
        "target_id_column": "ProjectNumber",
        "include_in_v1": "yes_after_project_key_audit",
        "reason": "Connects papers to projects. Must audit ProjectNumber against B02_Projects keys first.",
    },
    {
        "edge_type": "trial_linked_project",
        "source_table": "B04_Link_ClinicalTrials_Projects",
        "source_id_column": "nct_id",
        "target_id_column": "ProjectNumber",
        "include_in_v1": "yes_after_project_key_audit",
        "reason": "Connects clinical trials to projects. Must audit ProjectNumber against B02_Projects keys first.",
    },
    {
        "edge_type": "patent_linked_project",
        "source_table": "B05_Link_Patents_Projects",
        "source_id_column": "PatentId",
        "target_id_column": "ProjectNumber",
        "include_in_v1": "yes_after_project_key_audit",
        "reason": "Connects patents to projects. Must audit ProjectNumber against B02_Projects keys first.",
    },
]


ATTRIBUTE_TABLE_RULES = [
    {
        "table_name": "A01_Articles",
        "entity": "Paper",
        "use": "Paper metadata",
        "join_key": "PMID",
        "include_in_v1": "optional_attributes",
        "reason": "Rich PubMed article metadata. Use as Paper attributes, not as an edge table.",
    },
    {
        "table_name": "A04_Abstract",
        "entity": "Paper",
        "use": "Paper abstract text",
        "join_key": "PMID",
        "include_in_v1": "optional_attributes",
        "reason": "Abstract text can be added as Paper attributes or used for text embeddings. Not an edge table.",
    },
    {
        "table_name": "A03_KeywordList",
        "entity": "Paper",
        "use": "Paper keywords",
        "join_key": "PMID",
        "include_in_v1": "optional_attributes",
        "reason": "Useful paper descriptors. Can be attributes or v2 keyword nodes.",
    },
    {
        "table_name": "A05_GrantList",
        "entity": "Paper/Grant",
        "use": "PubMed grant metadata",
        "join_key": "PMID, GrantID",
        "include_in_v1": "optional",
        "reason": "Grant metadata from PubMed. Do not treat as the main Project node table; use B02_Projects for projects.",
    },
    {
        "table_name": "C10_Link_Papers_Journals",
        "entity": "Paper/Journal",
        "use": "Journal metadata or paper-journal edge",
        "join_key": "PMID",
        "include_in_v1": "optional",
        "reason": "Can be Paper attributes or a Journal node in v2.",
    },
]


DEFERRED_TABLE_RULES = [
    {
        "table_name": "C04_ReferenceList_Papers",
        "category": "Citation",
        "reason": "807M paper-paper citation edges. Defer or restrict to the selected seed subgraph.",
    },
    {
        "table_name": "A14_ReferenceList",
        "category": "Citation metadata",
        "reason": "Large reference metadata table. Prefer C04_ReferenceList_Papers for citation edges if needed.",
    },
    {
        "table_name": "C02_Link_Papers_Authors",
        "category": "Author graph",
        "reason": "171M paper-author links. Useful but large; defer to v2 or restrict to selected PMIDs.",
    },
    {
        "table_name": "C07_Authors",
        "category": "Author graph",
        "reason": "Author node table. Useful for v2 author/inventor analysis.",
    },
    {
        "table_name": "C03_Affiliations",
        "category": "Organization/Affiliation",
        "reason": "Useful for application value, but affiliation normalization can be complex. Add in v1.5.",
    },
    {
        "table_name": "C17_Assignees",
        "category": "Patent organization",
        "reason": "Useful for patent ownership and organization analysis. Add after core patent graph is stable.",
    },
    {
        "table_name": "C19_Inventors",
        "category": "Patent inventor",
        "reason": "Useful for inventor/author linkage. Add after core patent graph is stable.",
    },
    {
        "table_name": "C20_Link_Authors_Inventors",
        "category": "Author-inventor linkage",
        "reason": "High application value, but should be added after Author and Inventor nodes are validated.",
    },
    {
        "table_name": "A06_MeshHeadingList",
        "category": "MeSH",
        "reason": "331M paper-MeSH rows. Useful as features or v2 descriptor nodes; too large for first pass.",
    },
    {
        "table_name": "B01_Descriptor",
        "category": "MeSH descriptor",
        "reason": "Descriptor vocabulary table. Use with A06 if MeSH graph/features are added.",
    },
    {
        "table_name": "A08_ChemicalList",
        "category": "Chemical metadata",
        "reason": "Potentially useful but may overlap with BioEntity. Defer until BioEntity strategy is clear.",
    },
    {
        "table_name": "C21_Bioentity_Relationships",
        "category": "BioEntity relationship",
        "reason": "Potentially valuable, but inspect schema and semantics before using as BioEntity-BioEntity edges.",
    },
]


PROJECT_JOIN_AUDIT_RULES = [
    {
        "link_table": "B03_Link_Papers_Projects",
        "link_column": "ProjectNumber",
        "candidate_project_columns": "FULL_PROJECT_NUM, CORE_PROJECT_NUM, APPLICATION_ID, SUBPROJECT_ID",
        "reason": "Paper-project links depend on resolving ProjectNumber to a B02_Projects key.",
    },
    {
        "link_table": "B04_Link_ClinicalTrials_Projects",
        "link_column": "ProjectNumber",
        "candidate_project_columns": "FULL_PROJECT_NUM, CORE_PROJECT_NUM, APPLICATION_ID, SUBPROJECT_ID",
        "reason": "Clinical trial-project links depend on resolving ProjectNumber to a B02_Projects key.",
    },
    {
        "link_table": "B05_Link_Patents_Projects",
        "link_column": "ProjectNumber",
        "candidate_project_columns": "FULL_PROJECT_NUM, CORE_PROJECT_NUM, APPLICATION_ID, SUBPROJECT_ID",
        "reason": "Patent-project links depend on resolving ProjectNumber to a B02_Projects key.",
    },
]


CORE_FK_AUDIT_RULES = [
    "C06_Link_Papers_BioEntities.PMID -> C01_Papers.PMID",
    "C06_Link_Papers_BioEntities.EntityId -> C23_BioEntities.EntityId",
    "C13_Link_ClinicalTrials_BioEntities.nct_id -> C11_ClinicalTrials.nct_id",
    "C13_Link_ClinicalTrials_BioEntities.EntityId -> C23_BioEntities.EntityId",
    "C18_Link_Patents_BioEntities.PatentId -> C15_Patents.PatentId",
    "C18_Link_Patents_BioEntities.EntityId -> C23_BioEntities.EntityId",
    "C12_Link_Papers_Clinicaltrials.PMID -> C01_Papers.PMID",
    "C12_Link_Papers_Clinicaltrials.nct_id -> C11_ClinicalTrials.nct_id",
    "C16_Link_Patents_Papers.PatentId -> C15_Patents.PatentId",
    "C16_Link_Patents_Papers.PMID -> C01_Papers.PMID",
    "B03_Link_Papers_Projects.PMID -> C01_Papers.PMID",
    "B04_Link_ClinicalTrials_Projects.nct_id -> C11_ClinicalTrials.nct_id",
    "B05_Link_Patents_Projects.PatentId -> C15_Patents.PatentId",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PKG24S4 profile CSV outputs.")
    parser.add_argument("--profile-dir", type=Path, default=Path("data/interim/profile"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/profile/profile_analysis.md"),
    )
    parser.add_argument(
        "--subgraph-plan-output",
        type=Path,
        default=Path("data/interim/profile/recommended_subgraph_plan.md"),
    )
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if not path.exists():
        return [], f"Missing profile file: {path}"
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file)), None


def load_profiles(profile_dir: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    profiles: dict[str, list[dict[str, str]]] = {}
    missing = []
    for key, file_name in PROFILE_FILES.items():
        rows, warning = read_csv_if_exists(profile_dir / file_name)
        profiles[key] = rows
        if warning:
            missing.append(warning)
    return profiles, missing


def to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(str(value)))
    except ValueError:
        return 0


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value))
    except ValueError:
        return 0.0


def fmt_int(value: Any) -> str:
    return f"{to_int(value):,}"


def fmt_mb(value: Any) -> str:
    return f"{to_float(value):,.1f}"


def truncate(value: Any, max_len: int = 220) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def md_escape(value: Any) -> str:
    return truncate(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def table_lookup(table_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("table_name", ""): row for row in table_rows}


def add_table_stats(
    rows: list[dict[str, Any]],
    table_rows: list[dict[str, str]],
    source_key: str = "source_table",
) -> list[dict[str, Any]]:
    lookup = table_lookup(table_rows)
    output = []
    for row in rows:
        table_name = row.get(source_key, "")
        table_row = lookup.get(str(table_name), {})
        enriched = dict(row)
        enriched["row_count"] = fmt_int(table_row.get("row_count", ""))
        enriched["parquet_size_mb"] = fmt_mb(table_row.get("parquet_size_mb", ""))
        enriched["column_count"] = table_row.get("column_count", "")
        output.append(enriched)
    return output


def build_column_warnings(column_rows: list[dict[str, str]], top_n: int) -> list[dict[str, Any]]:
    warnings = []
    for row in column_rows:
        null_fraction = to_float(row.get("null_fraction", ""))
        kind = row.get("inferred_kind", "")
        n_unique = row.get("sample_n_unique") or row.get("approx_n_unique", "")
        reason = ""

        if null_fraction >= 0.9:
            reason = "high_null_fraction"
        if kind == "id" and null_fraction >= 0.5:
            reason = "id_like_high_null_fraction"
        if kind in {"year", "date"} and null_fraction >= 0.5:
            reason = "temporal_high_null_fraction"

        if reason:
            warnings.append(
                {
                    "table_name": row.get("table_name", ""),
                    "column_name": row.get("column_name", ""),
                    "inferred_kind": kind,
                    "null_fraction": f"{null_fraction:.3f}" if row.get("null_fraction") else "",
                    "null_fraction_basis": row.get("null_fraction_basis", ""),
                    "unique": n_unique,
                    "sample_non_null_values_json": row.get("sample_non_null_values_json", ""),
                    "reason": reason,
                }
            )

    return sorted(warnings, key=lambda item: to_float(item["null_fraction"]), reverse=True)[:top_n]


def top_join_candidates(join_rows: list[dict[str, str]], top_n: int) -> list[dict[str, str]]:
    return sorted(
        join_rows,
        key=lambda row: CONFIDENCE_ORDER.get(row.get("confidence", "").lower(), 0),
        reverse=True,
    )[:top_n]


def summarize_id_systems(join_rows: list[dict[str, str]]) -> list[str]:
    id_systems = {
        "PMID / PaperId": ["pmid", "paperid", "paper"],
        "EntityId": ["entityid", "entity"],
        "ProjectNumber / ProjectId": ["projectnumber", "projectid", "project"],
        "ClinicalTrialId / NCTId": ["clinicaltrial", "nct", "trial"],
        "PatentId": ["patent"],
        "AID / S2AID": ["aid", "s2aid", "author"],
        "NlmUniqueID / Journal": ["nlmuniqueid", "journal"],
    }
    lines = []
    for label, tokens in id_systems.items():
        count = sum(
            1
            for row in join_rows
            if any(
                token
                in " ".join(
                    [
                        row.get("source_table", ""),
                        row.get("source_column", ""),
                        row.get("target_table", ""),
                        row.get("target_column", ""),
                        row.get("reason", ""),
                    ]
                ).lower()
                for token in tokens
            )
        )
        lines.append(f"- **{label}**: {count} candidate join edges")
    return lines


def build_overview_section(table_rows: list[dict[str, str]], top_n: int) -> str:
    largest_tables = sorted(table_rows, key=lambda row: to_int(row.get("row_count", "")), reverse=True)[:top_n]
    largest_files = sorted(
        table_rows, key=lambda row: to_float(row.get("parquet_size_mb", "")), reverse=True
    )[:top_n]
    total_rows = sum(to_int(row.get("row_count", "")) for row in table_rows)
    total_size = sum(to_float(row.get("parquet_size_mb", "")) for row in table_rows)

    parts = []
    parts.append("## 1. Dataset scale overview\n")
    parts.append(f"- Total tables: **{len(table_rows):,}**")
    parts.append(f"- Total rows: **{total_rows:,}**")
    parts.append(f"- Total Parquet size: **{total_size:,.1f} MB**")
    parts.append("\nLargest tables by row_count:\n")
    parts.append(markdown_table(largest_tables, ["table_name", "file_name", "row_count", "parquet_size_mb"]))
    parts.append("\nLargest files by parquet_size_mb:\n")
    parts.append(markdown_table(largest_files, ["table_name", "file_name", "parquet_size_mb", "row_count"]))
    parts.append("\n## 2. Largest tables\n")
    parts.append(
        markdown_table(
            largest_tables,
            ["table_name", "file_name", "row_count", "parquet_size_mb", "column_count", "actual_engine"],
        )
    )
    return "\n".join(parts)


def build_rule_based_recommendation_sections(table_rows: list[dict[str, str]]) -> str:
    core_nodes = add_table_stats(CORE_NODE_RULES, table_rows)
    core_edges = add_table_stats(CORE_EDGE_RULES, table_rows)
    attributes = add_table_stats(ATTRIBUTE_TABLE_RULES, table_rows, source_key="table_name")
    deferred = add_table_stats(DEFERRED_TABLE_RULES, table_rows, source_key="table_name")

    parts = []
    parts.append("## 3. Rule-based first-pass graph recommendation\n")
    parts.append(
        "The following recommendation is manually curated from the profile outputs. "
        "It intentionally overrides purely heuristic classifications that may label Link tables as nodes "
        "or attribute tables as edges.\n"
    )

    parts.append("### 3.1 Core node tables\n")
    parts.append(
        markdown_table(
            core_nodes,
            [
                "node_type",
                "source_table",
                "id_column",
                "label_columns",
                "year_columns",
                "row_count",
                "parquet_size_mb",
                "include_in_v1",
                "reason",
            ],
        )
    )

    parts.append("\n### 3.2 Core edge tables\n")
    parts.append(
        markdown_table(
            core_edges,
            [
                "edge_type",
                "source_table",
                "source_id_column",
                "target_id_column",
                "row_count",
                "parquet_size_mb",
                "include_in_v1",
                "reason",
            ],
        )
    )

    parts.append("\n### 3.3 Attribute tables, not primary edge tables\n")
    parts.append(
        markdown_table(
            attributes,
            [
                "table_name",
                "entity",
                "use",
                "join_key",
                "row_count",
                "parquet_size_mb",
                "include_in_v1",
                "reason",
            ],
        )
    )

    parts.append("\n### 3.4 Deferred or v2 tables\n")
    parts.append(
        markdown_table(
            deferred,
            ["table_name", "category", "row_count", "parquet_size_mb", "reason"],
        )
    )

    return "\n".join(parts)


def build_profile_diagnostics_sections(
    profiles: dict[str, list[dict[str, str]]],
    top_n: int,
) -> str:
    table_rows = profiles["table"]
    link_rows = profiles["link"]
    entity_rows = profiles["entity"]
    join_rows = profiles["join"]
    column_rows = profiles["column"]

    link_display = sorted(link_rows, key=lambda row: to_int(row.get("row_count", "")), reverse=True)[:top_n]
    entity_display = sorted(entity_rows, key=lambda row: to_int(row.get("row_count", "")), reverse=True)[:top_n]
    warnings = build_column_warnings(column_rows, top_n)
    join_top = top_join_candidates(join_rows, top_n)

    parts = []
    parts.append("## 4. Heuristic profile outputs for manual review\n")
    parts.append(
        "These tables come directly from profiling heuristics and should be treated as review aids, "
        "not as the graph schema contract.\n"
    )

    parts.append("### 4.1 Largest heuristic entity candidates\n")
    parts.append(
        markdown_table(
            entity_display,
            [
                "table_name",
                "file_name",
                "row_count",
                "primary_id_candidates_json",
                "label_columns_json",
                "year_columns_json",
            ],
        )
    )

    parts.append("\n### 4.2 Largest heuristic link candidates\n")
    parts.append(
        markdown_table(
            link_display,
            [
                "table_name",
                "file_name",
                "row_count",
                "id_columns_json",
                "likely_source_columns_json",
                "likely_target_columns_json",
                "duplicate_row_count",
                "duplicate_count_basis",
            ],
        )
    )

    parts.append("\n### 4.3 Top join candidates from heuristic profiling\n")
    parts.append(
        markdown_table(
            join_top,
            ["source_table", "source_column", "target_table", "target_column", "confidence", "reason"],
        )
    )

    parts.append("\nImportant ID systems seen in join candidates:\n")
    parts.append("\n".join(summarize_id_systems(join_rows)) + "\n")

    parts.append("\n### 4.4 Column quality warnings\n")
    parts.append(
        markdown_table(
            warnings,
            [
                "table_name",
                "column_name",
                "inferred_kind",
                "null_fraction",
                "null_fraction_basis",
                "unique",
                "sample_non_null_values_json",
                "reason",
            ],
        )
    )

    return "\n".join(parts)


def build_audit_sections() -> str:
    parts = []
    parts.append("## 5. Required key audits before extraction\n")
    parts.append(
        "Before extracting any graph, audit node keys, edge endpoints, foreign-key coverage, "
        "duplicate edges, and especially the ProjectNumber join.\n"
    )

    parts.append("### 5.1 ProjectNumber join audit\n")
    parts.append(markdown_table(PROJECT_JOIN_AUDIT_RULES, ["link_table", "link_column", "candidate_project_columns", "reason"]))

    parts.append("\n### 5.2 Core foreign-key coverage checks\n")
    parts.append("\n".join(f"- `{rule}`" for rule in CORE_FK_AUDIT_RULES) + "\n")

    parts.append("\n### 5.3 Audit thresholds to inspect\n")
    parts.append(
        "\n".join(
            [
                "- Node key non-null rate should be close to 1.0.",
                "- Node key duplicate rate should be reviewed; some project keys may legitimately repeat by fiscal year.",
                "- Edge endpoint non-null rate should be close to 1.0.",
                "- Foreign-key coverage below 0.95 should be investigated before extraction.",
                "- Duplicate source-target pairs should be counted and either deduplicated or retained with explicit edge weights.",
                "- ProjectNumber should be matched against `FULL_PROJECT_NUM`, `CORE_PROJECT_NUM`, `APPLICATION_ID`, and `SUBPROJECT_ID` as strings.",
            ]
        )
        + "\n"
    )

    return "\n".join(parts)


def build_analysis(
    profiles: dict[str, list[dict[str, str]]],
    missing: list[str],
    top_n: int,
) -> str:
    table_rows = profiles["table"]

    parts = ["# PKG24S4 Profile Analysis\n"]
    parts.append(
        "This report combines profile statistics with a rule-based first-pass graph recommendation. "
        "The recommendation is focused on a patent-centered biomedical translation graph.\n"
    )

    if missing:
        parts.append("## Missing input files\n")
        parts.append("\n".join(f"- {warning}" for warning in missing) + "\n")

    parts.append(build_overview_section(table_rows, top_n))
    parts.append(build_rule_based_recommendation_sections(table_rows))
    parts.append(build_profile_diagnostics_sections(profiles, top_n))
    parts.append(build_audit_sections())

    parts.append("## 6. Risks and open questions\n")
    parts.append(
        "\n".join(
            [
                "- Confirm `ProjectNumber -> B02_Projects` join key before extracting project edges.",
                "- Decide whether the v1 graph is patent-centered, paper-centered, or project-centered. Current recommendation is patent-centered.",
                "- Decide the first test time window, for example 2018-2024 or a smaller 2020-2022 window.",
                "- For `C06_Link_Papers_BioEntities`, avoid scanning all 490M rows without restricting to selected PMIDs or years.",
                "- Decide whether repeated source-target pairs become edge weights or are deduplicated.",
                "- Decide whether BioEntity `Type` should be filtered for the first application scenario.",
                "- Add affiliation/organization/assignee tables after the core graph is stable if application analysis requires institutional translation metrics.",
                "- Do not use generic `id` columns for cross-table joins unless specifically audited.",
            ]
        )
        + "\n"
    )

    parts.append("## 7. Recommended next step\n")
    parts.append(
        "Create and run `scripts/05_audit_core_graph_keys.py` before writing any graph extraction script. "
        "The audit should determine ProjectNumber join behavior, endpoint coverage, duplicate keys, duplicate edges, "
        "and orphan-edge ratios.\n"
    )

    return "\n".join(parts)


def build_subgraph_plan(profiles: dict[str, list[dict[str, str]]]) -> str:
    table_rows = profiles["table"]
    core_nodes = add_table_stats(CORE_NODE_RULES, table_rows)
    core_edges = add_table_stats(CORE_EDGE_RULES, table_rows)
    attributes = add_table_stats(ATTRIBUTE_TABLE_RULES, table_rows, source_key="table_name")
    deferred = add_table_stats(DEFERRED_TABLE_RULES, table_rows, source_key="table_name")

    parts = ["# Recommended First-Pass Subgraph Plan\n"]
    parts.append("## Goal\n")
    parts.append(
        "Build a first-pass **patent-centered biomedical translation subgraph** for exploring how biomedical "
        "knowledge moves among papers, bioentities, NIH/project funding, clinical trials, and patents.\n"
    )

    parts.append("## Proposed core node types\n")
    parts.append(
        markdown_table(
            core_nodes,
            [
                "node_type",
                "source_table",
                "id_column",
                "label_columns",
                "year_columns",
                "row_count",
                "include_in_v1",
                "reason",
            ],
        )
    )

    parts.append("\n## Proposed core edge types\n")
    parts.append(
        markdown_table(
            core_edges,
            [
                "edge_type",
                "source_table",
                "source_id_column",
                "target_id_column",
                "row_count",
                "include_in_v1",
                "reason",
            ],
        )
    )

    parts.append("\n## Attribute tables\n")
    parts.append(
        markdown_table(
            attributes,
            ["table_name", "entity", "use", "join_key", "row_count", "include_in_v1", "reason"],
        )
    )

    parts.append("\n## Deferred / v2 tables\n")
    parts.append(
        markdown_table(
            deferred,
            ["table_name", "category", "row_count", "reason"],
        )
    )

    parts.append("\n## Suggested v1 extraction scope\n")
    parts.append(
        "\n".join(
            [
                "- Start with a patent-centered seed set.",
                "- Use `C15_Patents.GrantedDate` to select a manageable patent time window.",
                "- Suggested first test window: 2020-2022. Suggested broader v1 window: 2018-2024.",
                "- Select patents first, then expand to linked papers, projects, bioentities, and clinical trials.",
                "- For `C06_Link_Papers_BioEntities`, filter to PMIDs already selected from the Paper node set.",
                "- Do not load all 490M paper-bioentity rows unless the extraction engine is designed for it.",
                "- Remove edges with missing source or target IDs.",
                "- Deduplicate source-target pairs or convert duplicate rows into weighted edges.",
                "- Keep useful edge attributes such as `Type`, `prob`, `WhereFound`, `ConfScore`, `SelfCitation`, and `RefType` where applicable.",
            ]
        )
        + "\n"
    )

    parts.append("\n## Data quality checks before extraction\n")
    parts.append(
        "\n".join(
            [
                "- Node key non-null count and distinct count.",
                "- Duplicate node key count.",
                "- Edge source and target non-null counts.",
                "- Duplicate source-target pair count.",
                "- Foreign-key coverage from each edge endpoint to its node table.",
                "- Orphan edge ratio after joining to selected nodes.",
                "- ProjectNumber match rate against B02_Projects candidate keys.",
                "- PubYear / GrantedDate / FiscalYear coverage for temporal filtering.",
            ]
        )
        + "\n"
    )

    parts.append("\n## ProjectNumber open question\n")
    parts.append(
        "The project link tables use `ProjectNumber`, while `B02_Projects` contains several candidate identifiers: "
        "`FULL_PROJECT_NUM`, `CORE_PROJECT_NUM`, `APPLICATION_ID`, and `SUBPROJECT_ID`. "
        "Do not extract project edges until `ProjectNumber` match rates are audited.\n"
    )

    parts.append("\n## Next script recommendation\n")
    parts.append("`scripts/05_audit_core_graph_keys.py`\n")

    return "\n".join(parts)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be greater than or equal to 1")

    profile_dir = args.profile_dir.resolve()
    profiles, missing = load_profiles(profile_dir)

    analysis = build_analysis(profiles, missing, args.top_n)
    subgraph_plan = build_subgraph_plan(profiles)

    output_path = args.output.resolve()
    plan_path = args.subgraph_plan_output.resolve()
    write_text(output_path, analysis)
    write_text(plan_path, subgraph_plan)

    table_count = len(profiles["table"])
    total_rows = sum(to_int(row.get("row_count", "")) for row in profiles["table"])

    print("PKG24S4 rule-based profile analysis written")
    print(f"Profile directory: {profile_dir}")
    print(f"Tables analyzed: {table_count}")
    print(f"Total rows represented: {total_rows:,}")
    print(f"Missing profile files: {len(missing)}")
    print(f"Analysis report: {output_path}")
    print(f"Subgraph plan: {plan_path}")
    print("Recommended next step: scripts/05_audit_core_graph_keys.py")


if __name__ == "__main__":
    main()