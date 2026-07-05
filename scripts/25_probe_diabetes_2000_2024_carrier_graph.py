#!/usr/bin/env python
"""Streaming scale/schema probe for diabetes long-window carrier graph.

This script performs a read-only streaming probe over data/interim/parquet
to estimate the scale of a 2000-2024 domain-centered diabetes carrier graph.

It does NOT materialize the final processed graph.

Outputs:
- data/reports/source_coverage/diabetes_2000_2024_carrier_graph_probe_summary.json
- data/reports/source_coverage/diabetes_2000_2024_carrier_graph_probe_report.md

Typical run:
python scripts/25_probe_diabetes_2000_2024_carrier_graph.py

Smoke test:
python scripts/25_probe_diabetes_2000_2024_carrier_graph.py --max-row-groups 1

Medium test:
python scripts/25_probe_diabetes_2000_2024_carrier_graph.py --max-row-groups 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


warnings.filterwarnings(
    "ignore",
    message="Could not infer format",
    category=UserWarning,
)


DEFAULT_SEED_TERMS = [
    "diabetes",
    "diabetic",
    "diabetes mellitus",
    "type 1 diabetes",
    "type 2 diabetes",
]


TABLES = {
    "bioentities": "C23_BioEntities.parquet",
    "papers": "C01_Papers.parquet",
    "patents": "C15_Patents.parquet",
    "trials": "C11_ClinicalTrials.parquet",
    "projects": "B02_Projects.parquet",
    "paper_bioentities": "C06_Link_Papers_BioEntities.parquet",
    "patent_bioentities": "C18_Link_Patents_BioEntities.parquet",
    "trial_bioentities": "C13_Link_ClinicalTrials_BioEntities.parquet",
    "patent_paper": "C16_Link_Patents_Papers.parquet",
    "paper_trial": "C12_Link_Papers_Clinicaltrials.parquet",
    "paper_project": "B03_Link_Papers_Projects.parquet",
    "trial_project": "B04_Link_ClinicalTrials_Projects.parquet",
    "patent_project": "B05_Link_Patents_Projects.parquet",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def normalize_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").astype(str)


def isin_large_set(series: pd.Series, values: set[str]) -> pd.Series:
    """Fast membership check for large Python sets.

    pandas.Series.isin(large_set) can be slow in repeated row-group streaming
    loops because it may rebuild internal lookup structures. Mapping
    set.__contains__ is often faster and keeps progress responsive.
    """

    if not values:
        return pd.Series(False, index=series.index)

    text = normalize_string_series(series)
    contains = values.__contains__
    return text.map(contains)


def parse_year_series(series: pd.Series) -> pd.Series:
    """Parse numeric or date-like values into nullable year values.

    Optimized for streaming:
    - if most values are numeric years, return numeric parsing directly;
    - otherwise parse date-like strings only as fallback.
    """

    text = series.astype("string").str.strip()
    text = text.mask(
        text.isin(["", "NA", "N/A", "NULL", "None", "none", "nan", "NaN", "<NA>"]),
        pd.NA,
    )

    numeric = pd.to_numeric(text, errors="coerce")
    numeric_year = numeric.where((numeric >= 1500) & (numeric <= 2100))

    # Fast path for PubYear / FiscalYear style columns.
    valid_numeric_fraction = (
        float(numeric_year.notna().mean()) if len(numeric_year) else 0.0
    )
    if valid_numeric_fraction >= 0.8:
        return numeric_year

    # Date-like fallback for GrantedDate / start_date / PROJECT_START, etc.
    try:
        dt = pd.to_datetime(text, errors="coerce", utc=False, format="mixed")
    except TypeError:
        dt = pd.to_datetime(text, errors="coerce", utc=False)

    date_year = dt.dt.year.where((dt.dt.year >= 1500) & (dt.dt.year <= 2100))

    return numeric_year.fillna(date_year)


def read_row_group(
    parquet_file: pq.ParquetFile,
    row_group_idx: int,
    columns: list[str],
) -> pd.DataFrame:
    table = parquet_file.read_row_group(row_group_idx, columns=columns)
    return table.to_pandas()


def iter_row_groups(
    path: Path,
    columns: list[str],
    *,
    max_row_groups: int | None = None,
):
    parquet_file = pq.ParquetFile(path)
    row_group_count = parquet_file.num_row_groups

    if max_row_groups is not None:
        row_group_count = min(row_group_count, max_row_groups)

    for row_group_idx in range(row_group_count):
        yield (
            row_group_idx,
            row_group_count,
            read_row_group(parquet_file, row_group_idx, columns),
        )


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    return int(value)


def find_diabetes_seed_entities(
    *,
    input_dir: Path,
    seed_terms: list[str],
) -> tuple[set[str], dict[str, Any]]:
    """Load C23_BioEntities and find diabetes-related EntityIds."""

    path = input_dir / TABLES["bioentities"]
    log(f"Loading BioEntities for seed matching: {path}")

    df = pd.read_parquet(path, columns=["EntityId", "Type", "Mention"])

    mention = normalize_string_series(df["Mention"]).str.lower()
    type_series = normalize_string_series(df["Type"])

    patterns = [re.escape(term.lower()) for term in seed_terms]
    pattern = "|".join(patterns)

    mask = mention.str.contains(pattern, regex=True, na=False)

    seeds = set(normalize_string_series(df.loc[mask, "EntityId"]))

    type_counts = Counter(type_series.loc[mask])
    mention_examples = (
        df.loc[mask, ["EntityId", "Type", "Mention"]]
        .drop_duplicates()
        .head(50)
        .to_dict("records")
    )

    summary = {
        "seed_terms": seed_terms,
        "seed_entity_count": len(seeds),
        "seed_entity_type_counts": counter_to_dict(type_counts),
        "seed_entity_examples": mention_examples,
    }

    log(f"Seed BioEntities: {len(seeds):,}")
    return seeds, summary


def collect_seed_carriers_from_bioentity_links(
    *,
    input_dir: Path,
    table_key: str,
    carrier_id_col: str,
    seed_entity_ids: set[str],
    max_row_groups: int | None = None,
    progress_every_row_groups: int = 25,
) -> tuple[set[str], dict[str, Any]]:
    """Stream carrier-BioEntity links and collect carriers mentioning seed entities."""

    path = input_dir / TABLES[table_key]
    log(f"Scanning seed carriers from {path.name}")
    log(f"  Seed Entity IDs: {len(seed_entity_ids):,}")

    carriers: set[str] = set()
    seed_edge_count = 0
    scanned_rows = 0
    matched_seed_entities: set[str] = set()

    for rg, total_rg, df in iter_row_groups(
        path,
        [carrier_id_col, "EntityId"],
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        mask = isin_large_set(df["EntityId"], seed_entity_ids)

        if mask.any():
            matched = df.loc[mask]
            carrier_values = normalize_string_series(matched[carrier_id_col])
            entity_values = normalize_string_series(matched["EntityId"])

            carriers.update(carrier_values[carrier_values != ""])
            matched_seed_entities.update(entity_values[entity_values != ""])
            seed_edge_count += int(mask.sum())

    summary = {
        "table": path.name,
        "carrier_id_col": carrier_id_col,
        "scanned_rows": scanned_rows,
        "seed_edge_count": seed_edge_count,
        "seed_carrier_count": len(carriers),
        "matched_seed_entity_count": len(matched_seed_entities),
    }

    log(f"  Seed carriers from {path.name}: {len(carriers):,}")
    return carriers, summary


def filter_carriers_by_year(
    *,
    input_dir: Path,
    table_key: str,
    id_col: str,
    year_col: str,
    candidate_ids: set[str],
    start_year: int,
    end_year: int,
    max_row_groups: int | None = None,
    progress_every_row_groups: int = 10,
) -> tuple[set[str], dict[str, Any], Counter]:
    """Filter candidate carrier IDs by carrier node year."""

    path = input_dir / TABLES[table_key]
    log(f"Filtering by year from {path.name}: {id_col}, {year_col}")
    log(f"  Candidate IDs: {len(candidate_ids):,}")

    kept: set[str] = set()
    seen_candidates: set[str] = set()
    year_counts: Counter[int] = Counter()

    scanned_rows = 0
    valid_year_candidate_rows = 0
    out_of_window_candidate_rows = 0
    missing_year_candidate_rows = 0

    for rg, total_rg, df in iter_row_groups(
        path,
        [id_col, year_col],
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        mask_candidate = isin_large_set(df[id_col], candidate_ids)

        if not mask_candidate.any():
            continue

        sub = df.loc[mask_candidate, [id_col, year_col]].copy()
        sub_ids = normalize_string_series(sub[id_col])
        seen_candidates.update(sub_ids[sub_ids != ""])

        years = parse_year_series(sub[year_col])
        valid_year_mask = years.notna()

        missing_year_candidate_rows += int((~valid_year_mask).sum())

        in_window_mask = valid_year_mask & (years >= start_year) & (years <= end_year)
        out_of_window_candidate_rows += int((valid_year_mask & ~in_window_mask).sum())
        valid_year_candidate_rows += int(in_window_mask.sum())

        if in_window_mask.any():
            kept_ids = sub_ids.loc[in_window_mask]
            kept.update(kept_ids[kept_ids != ""])

            years_int = years.loc[in_window_mask].astype(int)
            year_counts.update(years_int.value_counts().to_dict())

    summary = {
        "table": path.name,
        "id_col": id_col,
        "year_col": year_col,
        "candidate_count": len(candidate_ids),
        "candidate_seen_in_node_table": len(seen_candidates),
        "candidate_missing_from_node_table": len(candidate_ids - seen_candidates),
        "kept_in_year_window": len(kept),
        "start_year": start_year,
        "end_year": end_year,
        "scanned_rows": scanned_rows,
        "valid_year_candidate_rows": valid_year_candidate_rows,
        "out_of_window_candidate_rows": out_of_window_candidate_rows,
        "missing_year_candidate_rows": missing_year_candidate_rows,
        "year_counts": counter_to_dict(year_counts),
    }

    log(f"  Kept {len(kept):,}/{len(candidate_ids):,} from {path.name}")
    return kept, summary, year_counts


def collect_patent_paper_context(
    *,
    input_dir: Path,
    selected_patents: set[str],
    selected_papers: set[str],
    max_row_groups: int | None = None,
    progress_every_row_groups: int = 20,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Collect one-hop patent-paper context where either side is already selected."""

    path = input_dir / TABLES["patent_paper"]
    log(f"Collecting patent-paper context from {path.name}")
    log(f"  Selected patents: {len(selected_patents):,}")
    log(f"  Selected papers: {len(selected_papers):,}")

    context_patents: set[str] = set()
    context_papers: set[str] = set()
    matched_edges = 0
    scanned_rows = 0

    for rg, total_rg, df in iter_row_groups(
        path,
        ["PatentId", "PMID"],
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        mask = isin_large_set(df["PatentId"], selected_patents) | isin_large_set(
            df["PMID"],
            selected_papers,
        )

        if mask.any():
            matched_edges += int(mask.sum())

            patents = normalize_string_series(df.loc[mask, "PatentId"])
            papers = normalize_string_series(df.loc[mask, "PMID"])

            context_patents.update(patents[patents != ""])
            context_papers.update(papers[papers != ""])

    summary = {
        "table": path.name,
        "scanned_rows": scanned_rows,
        "matched_edges": matched_edges,
        "context_patent_count": len(context_patents),
        "context_paper_count": len(context_papers),
    }

    log(
        f"  Patent-paper context: {len(context_patents):,} patents, "
        f"{len(context_papers):,} papers, {matched_edges:,} edges"
    )
    return context_patents, context_papers, summary


def collect_paper_trial_context(
    *,
    input_dir: Path,
    selected_papers: set[str],
    selected_trials: set[str],
    max_row_groups: int | None = None,
    progress_every_row_groups: int = 20,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Collect one-hop paper-trial context where either side is already selected."""

    path = input_dir / TABLES["paper_trial"]
    log(f"Collecting paper-trial context from {path.name}")
    log(f"  Selected papers: {len(selected_papers):,}")
    log(f"  Selected trials: {len(selected_trials):,}")

    context_papers: set[str] = set()
    context_trials: set[str] = set()
    matched_edges = 0
    scanned_rows = 0

    for rg, total_rg, df in iter_row_groups(
        path,
        ["PMID", "nct_id"],
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        mask = isin_large_set(df["PMID"], selected_papers) | isin_large_set(
            df["nct_id"],
            selected_trials,
        )

        if mask.any():
            matched_edges += int(mask.sum())

            papers = normalize_string_series(df.loc[mask, "PMID"])
            trials = normalize_string_series(df.loc[mask, "nct_id"])

            context_papers.update(papers[papers != ""])
            context_trials.update(trials[trials != ""])

    summary = {
        "table": path.name,
        "scanned_rows": scanned_rows,
        "matched_edges": matched_edges,
        "context_paper_count": len(context_papers),
        "context_trial_count": len(context_trials),
    }

    log(
        f"  Paper-trial context: {len(context_papers):,} papers, "
        f"{len(context_trials):,} trials, {matched_edges:,} edges"
    )
    return context_papers, context_trials, summary


def collect_project_links(
    *,
    input_dir: Path,
    selected_papers: set[str],
    selected_patents: set[str],
    selected_trials: set[str],
    max_row_groups: int | None = None,
    progress_every_row_groups: int = 20,
) -> tuple[set[str], dict[str, Any]]:
    """Collect ProjectNumber values linked to selected carriers."""

    project_numbers: set[str] = set()
    summaries: dict[str, Any] = {}

    specs = [
        (
            "paper_project",
            "PMID",
            selected_papers,
            ["PMID", "ProjectNumber", "RecordYear"],
        ),
        (
            "patent_project",
            "PatentId",
            selected_patents,
            ["PatentId", "ProjectNumber"],
        ),
        (
            "trial_project",
            "nct_id",
            selected_trials,
            ["nct_id", "ProjectNumber"],
        ),
    ]

    for table_key, carrier_col, carrier_ids, columns in specs:
        path = input_dir / TABLES[table_key]
        log(f"Collecting project links from {path.name}")
        log(f"  Selected carrier IDs: {len(carrier_ids):,}")

        matched_edges = 0
        scanned_rows = 0
        local_projects: set[str] = set()
        record_year_counts: Counter[int] = Counter()

        for rg, total_rg, df in iter_row_groups(
            path,
            columns,
            max_row_groups=max_row_groups,
        ):
            if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
                log(f"  {path.name}: row group {rg + 1}/{total_rg}")

            scanned_rows += len(df)

            mask = isin_large_set(df[carrier_col], carrier_ids)

            if not mask.any():
                continue

            matched_edges += int(mask.sum())
            projects = normalize_string_series(df.loc[mask, "ProjectNumber"])
            local_projects.update(projects[projects != ""])

            if "RecordYear" in df.columns:
                years = parse_year_series(df.loc[mask, "RecordYear"]).dropna()
                if not years.empty:
                    record_year_counts.update(years.astype(int).value_counts().to_dict())

        project_numbers.update(local_projects)

        summaries[table_key] = {
            "table": path.name,
            "carrier_col": carrier_col,
            "selected_carrier_count": len(carrier_ids),
            "scanned_rows": scanned_rows,
            "matched_edges": matched_edges,
            "project_number_count": len(local_projects),
            "record_year_counts": counter_to_dict(record_year_counts),
        }

        log(f"  {path.name}: {len(local_projects):,} projects, {matched_edges:,} edges")

    return project_numbers, summaries


def probe_project_key_alignment(
    *,
    input_dir: Path,
    project_numbers: set[str],
    start_year: int,
    end_year: int,
    max_row_groups: int | None = None,
    progress_every_row_groups: int = 20,
) -> dict[str, Any]:
    """Probe how B03/B04/B05 ProjectNumber aligns to B02_Projects columns."""

    path = input_dir / TABLES["projects"]
    log(f"Probing ProjectNumber alignment against {path.name}")
    log(f"  Candidate ProjectNumbers: {len(project_numbers):,}")

    key_cols = ["APPLICATION_ID", "CORE_PROJECT_NUM", "FULL_PROJECT_NUM"]
    columns = key_cols + [
        "FiscalYear",
        "PROJECT_START",
        "PROJECT_END",
        "BUDGET_START",
        "BUDGET_END",
    ]

    matched_by_key: dict[str, set[str]] = {col: set() for col in key_cols}
    matched_in_window_by_key: dict[str, set[str]] = {col: set() for col in key_cols}

    scanned_rows = 0
    fiscal_year_counts: Counter[int] = Counter()

    for rg, total_rg, df in iter_row_groups(
        path,
        columns,
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        fiscal_year = parse_year_series(df["FiscalYear"])
        in_window = (
            fiscal_year.notna()
            & (fiscal_year >= start_year)
            & (fiscal_year <= end_year)
        )

        if in_window.any():
            fiscal_year_counts.update(
                fiscal_year.loc[in_window].astype(int).value_counts().to_dict()
            )

        for col in key_cols:
            mask = isin_large_set(df[col], project_numbers)

            if mask.any():
                values = normalize_string_series(df.loc[mask, col])
                matched_by_key[col].update(values[values != ""])

                mask_in_window = mask & in_window
                if mask_in_window.any():
                    values_in_window = normalize_string_series(df.loc[mask_in_window, col])
                    matched_in_window_by_key[col].update(
                        values_in_window[values_in_window != ""]
                    )

    summary = {
        "table": path.name,
        "project_number_candidate_count": len(project_numbers),
        "scanned_rows": scanned_rows,
        "key_columns": key_cols,
        "matched_project_numbers_by_key": {
            col: len(values) for col, values in matched_by_key.items()
        },
        "matched_project_numbers_in_window_by_key": {
            col: len(values) for col, values in matched_in_window_by_key.items()
        },
        "unmatched_project_numbers_by_key": {
            col: len(project_numbers - values) for col, values in matched_by_key.items()
        },
        "fiscal_year_counts_all_projects_in_window": counter_to_dict(
            fiscal_year_counts
        ),
    }

    return summary


def estimate_selected_carrier_mentions(
    *,
    input_dir: Path,
    table_key: str,
    carrier_id_col: str,
    selected_carriers: set[str],
    max_row_groups: int | None = None,
    progress_every_row_groups: int = 25,
) -> tuple[set[str], dict[str, Any]]:
    """Estimate mention-edge scale for selected carriers."""

    path = input_dir / TABLES[table_key]
    log(f"Estimating selected carrier BioEntity mentions from {path.name}")
    log(f"  Selected carrier IDs: {len(selected_carriers):,}")

    scanned_rows = 0
    selected_edge_count = 0
    selected_carrier_ids: set[str] = set()
    bioentity_ids: set[str] = set()
    bioentity_type_counts: Counter[str] = Counter()

    columns = [carrier_id_col, "EntityId", "Type"]

    for rg, total_rg, df in iter_row_groups(
        path,
        columns,
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        mask = isin_large_set(df[carrier_id_col], selected_carriers)

        if not mask.any():
            continue

        selected_edge_count += int(mask.sum())

        matched_carriers = normalize_string_series(df.loc[mask, carrier_id_col])
        selected_carrier_ids.update(matched_carriers[matched_carriers != ""])

        entities = normalize_string_series(df.loc[mask, "EntityId"])
        bioentity_ids.update(entities[entities != ""])

        types = normalize_string_series(df.loc[mask, "Type"])
        bioentity_type_counts.update(types.value_counts().to_dict())

    summary = {
        "table": path.name,
        "carrier_id_col": carrier_id_col,
        "selected_carrier_input_count": len(selected_carriers),
        "selected_carrier_with_mentions_count": len(selected_carrier_ids),
        "scanned_rows": scanned_rows,
        "selected_mention_edge_count": selected_edge_count,
        "selected_bioentity_count": len(bioentity_ids),
        "bioentity_type_counts": counter_to_dict(bioentity_type_counts),
    }

    log(
        f"  {path.name}: {selected_edge_count:,} mention edges, "
        f"{len(bioentity_ids):,} BioEntities"
    )

    return bioentity_ids, summary


def build_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.extend(
        [
            "# Diabetes 2000-2024 Carrier Graph Streaming Probe",
            "",
            "## 1. Configuration",
            "",
            "```json",
            json.dumps(summary["config"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 2. Seed BioEntities",
            "",
            f"- Seed BioEntities: **{summary['seed_entities']['seed_entity_count']:,}**",
            f"- Type counts: `{summary['seed_entities']['seed_entity_type_counts']}`",
            "",
            "## 3. Direct seed carriers",
            "",
        ]
    )

    for key, value in summary["seed_carriers"].items():
        lines.append(
            f"- {key}: **{value['seed_carrier_count']:,}** carriers "
            f"from {value['seed_edge_count']:,} seed edges"
        )

    lines.extend(["", "## 4. Seed carriers after year filtering", ""])

    for key, value in summary["year_filtered_seed_carriers"].items():
        lines.append(
            f"- {key}: **{value['kept_in_year_window']:,}** kept "
            f"from {value['candidate_count']:,} candidates"
        )

    lines.extend(["", "## 5. Context expansion", ""])

    pp = summary["context_expansion"]["patent_paper"]
    pt = summary["context_expansion"]["paper_trial"]

    lines.extend(
        [
            f"- Patent-paper matched edges: **{pp['matched_edges']:,}**",
            f"- Patent-paper context patents: **{pp['context_patent_count']:,}**",
            f"- Patent-paper context papers: **{pp['context_paper_count']:,}**",
            f"- Paper-trial matched edges: **{pt['matched_edges']:,}**",
            f"- Paper-trial context papers: **{pt['context_paper_count']:,}**",
            f"- Paper-trial context trials: **{pt['context_trial_count']:,}**",
            "",
            "## 6. Expanded selected carriers after year filtering",
            "",
        ]
    )

    for key, value in summary["year_filtered_expanded_carriers"].items():
        lines.append(
            f"- {key}: **{value['kept_in_year_window']:,}** kept "
            f"from {value['candidate_count']:,} candidates"
        )

    lines.extend(["", "## 7. Project links", ""])

    for key, value in summary["project_links"].items():
        lines.append(
            f"- {key}: **{value['project_number_count']:,}** projects, "
            f"**{value['matched_edges']:,}** matched edges"
        )

    project_alignment = summary["project_key_alignment"]
    lines.extend(
        [
            "",
            "## 8. Project key alignment",
            "",
            f"- Candidate ProjectNumber count: **{project_alignment['project_number_candidate_count']:,}**",
            f"- Matched by key: `{project_alignment['matched_project_numbers_by_key']}`",
            f"- Matched in 2000-2024 by key: `{project_alignment['matched_project_numbers_in_window_by_key']}`",
            "",
            "## 9. Selected carrier BioEntity mention scale",
            "",
        ]
    )

    for key, value in summary["selected_mentions"].items():
        lines.append(
            f"- {key}: **{value['selected_mention_edge_count']:,}** mention edges, "
            f"**{value['selected_bioentity_count']:,}** BioEntities, "
            f"**{value['selected_carrier_with_mentions_count']:,}** carriers with mentions"
        )

    lines.extend(
        [
            "",
            "## 10. Final estimated graph scale",
            "",
            f"- Selected papers: **{summary['estimated_graph_scale']['selected_papers']:,}**",
            f"- Selected patents: **{summary['estimated_graph_scale']['selected_patents']:,}**",
            f"- Selected trials: **{summary['estimated_graph_scale']['selected_trials']:,}**",
            f"- Candidate project numbers: **{summary['estimated_graph_scale']['candidate_project_numbers']:,}**",
            f"- Selected BioEntities from mentions: **{summary['estimated_graph_scale']['selected_bioentities_from_mentions']:,}**",
            f"- Selected carrier-BioEntity mention edges: **{summary['estimated_graph_scale']['selected_carrier_bioentity_mention_edges']:,}**",
            "",
            "## 11. Interpretation",
            "",
            "This is a streaming scale/schema probe only. It does not materialize the final graph.",
            "",
            "If the selected carrier and mention-edge scale is acceptable, the next step is to materialize:",
            "",
            "```text",
            "data/processed/diabetes_2000_2024_v1_carrier_graph/",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/interim/parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports/source_coverage"),
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--seed-term", action="append", default=None)
    parser.add_argument("--max-row-groups", type=int, default=None)
    parser.add_argument(
        "--progress-every-row-groups",
        type=int,
        default=10,
        help="Print progress every N row groups for streaming scans.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_terms = args.seed_term if args.seed_term else DEFAULT_SEED_TERMS
    progress_every = max(int(args.progress_every_row_groups), 1)

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "seed_terms": seed_terms,
        "max_row_groups": args.max_row_groups,
        "progress_every_row_groups": progress_every,
        "domain": "diabetes",
        "probe_type": "streaming_scale_schema_probe",
        "materializes_graph": False,
    }

    log("Starting diabetes long-window carrier graph streaming probe")
    log(json.dumps(config, ensure_ascii=False, indent=2))

    seed_entity_ids, seed_entity_summary = find_diabetes_seed_entities(
        input_dir=input_dir,
        seed_terms=seed_terms,
    )

    seed_papers, seed_paper_summary = collect_seed_carriers_from_bioentity_links(
        input_dir=input_dir,
        table_key="paper_bioentities",
        carrier_id_col="PMID",
        seed_entity_ids=seed_entity_ids,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    seed_patents, seed_patent_summary = collect_seed_carriers_from_bioentity_links(
        input_dir=input_dir,
        table_key="patent_bioentities",
        carrier_id_col="PatentId",
        seed_entity_ids=seed_entity_ids,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    seed_trials, seed_trial_summary = collect_seed_carriers_from_bioentity_links(
        input_dir=input_dir,
        table_key="trial_bioentities",
        carrier_id_col="nct_id",
        seed_entity_ids=seed_entity_ids,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    selected_seed_papers, seed_paper_year_summary, _ = filter_carriers_by_year(
        input_dir=input_dir,
        table_key="papers",
        id_col="PMID",
        year_col="PubYear",
        candidate_ids=seed_papers,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    selected_seed_patents, seed_patent_year_summary, _ = filter_carriers_by_year(
        input_dir=input_dir,
        table_key="patents",
        id_col="PatentId",
        year_col="GrantedDate",
        candidate_ids=seed_patents,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    selected_seed_trials, seed_trial_year_summary, _ = filter_carriers_by_year(
        input_dir=input_dir,
        table_key="trials",
        id_col="nct_id",
        year_col="start_date",
        candidate_ids=seed_trials,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    context_patents_pp, context_papers_pp, patent_paper_context_summary = (
        collect_patent_paper_context(
            input_dir=input_dir,
            selected_patents=selected_seed_patents,
            selected_papers=selected_seed_papers,
            max_row_groups=args.max_row_groups,
            progress_every_row_groups=progress_every,
        )
    )

    context_papers_pt, context_trials_pt, paper_trial_context_summary = (
        collect_paper_trial_context(
            input_dir=input_dir,
            selected_papers=selected_seed_papers | context_papers_pp,
            selected_trials=selected_seed_trials,
            max_row_groups=args.max_row_groups,
            progress_every_row_groups=progress_every,
        )
    )

    expanded_paper_candidates = selected_seed_papers | context_papers_pp | context_papers_pt
    expanded_patent_candidates = selected_seed_patents | context_patents_pp
    expanded_trial_candidates = selected_seed_trials | context_trials_pt

    selected_papers, expanded_paper_year_summary, _ = filter_carriers_by_year(
        input_dir=input_dir,
        table_key="papers",
        id_col="PMID",
        year_col="PubYear",
        candidate_ids=expanded_paper_candidates,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    selected_patents, expanded_patent_year_summary, _ = filter_carriers_by_year(
        input_dir=input_dir,
        table_key="patents",
        id_col="PatentId",
        year_col="GrantedDate",
        candidate_ids=expanded_patent_candidates,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    selected_trials, expanded_trial_year_summary, _ = filter_carriers_by_year(
        input_dir=input_dir,
        table_key="trials",
        id_col="nct_id",
        year_col="start_date",
        candidate_ids=expanded_trial_candidates,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    project_numbers, project_link_summaries = collect_project_links(
        input_dir=input_dir,
        selected_papers=selected_papers,
        selected_patents=selected_patents,
        selected_trials=selected_trials,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    project_key_alignment = probe_project_key_alignment(
        input_dir=input_dir,
        project_numbers=project_numbers,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    paper_bioentities, paper_mentions_summary = estimate_selected_carrier_mentions(
        input_dir=input_dir,
        table_key="paper_bioentities",
        carrier_id_col="PMID",
        selected_carriers=selected_papers,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    patent_bioentities, patent_mentions_summary = estimate_selected_carrier_mentions(
        input_dir=input_dir,
        table_key="patent_bioentities",
        carrier_id_col="PatentId",
        selected_carriers=selected_patents,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )
    trial_bioentities, trial_mentions_summary = estimate_selected_carrier_mentions(
        input_dir=input_dir,
        table_key="trial_bioentities",
        carrier_id_col="nct_id",
        selected_carriers=selected_trials,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    selected_bioentities = paper_bioentities | patent_bioentities | trial_bioentities

    selected_mention_edges = (
        paper_mentions_summary["selected_mention_edge_count"]
        + patent_mentions_summary["selected_mention_edge_count"]
        + trial_mentions_summary["selected_mention_edge_count"]
    )

    summary = {
        "config": config,
        "seed_entities": seed_entity_summary,
        "seed_carriers": {
            "paper": seed_paper_summary,
            "patent": seed_patent_summary,
            "trial": seed_trial_summary,
        },
        "year_filtered_seed_carriers": {
            "paper": seed_paper_year_summary,
            "patent": seed_patent_year_summary,
            "trial": seed_trial_year_summary,
        },
        "context_expansion": {
            "patent_paper": patent_paper_context_summary,
            "paper_trial": paper_trial_context_summary,
        },
        "year_filtered_expanded_carriers": {
            "paper": expanded_paper_year_summary,
            "patent": expanded_patent_year_summary,
            "trial": expanded_trial_year_summary,
        },
        "project_links": project_link_summaries,
        "project_key_alignment": project_key_alignment,
        "selected_mentions": {
            "paper": paper_mentions_summary,
            "patent": patent_mentions_summary,
            "trial": trial_mentions_summary,
        },
        "estimated_graph_scale": {
            "selected_papers": len(selected_papers),
            "selected_patents": len(selected_patents),
            "selected_trials": len(selected_trials),
            "candidate_project_numbers": len(project_numbers),
            "selected_bioentities_from_mentions": len(selected_bioentities),
            "selected_carrier_bioentity_mention_edges": selected_mention_edges,
        },
    }

    summary_path = output_dir / "diabetes_2000_2024_carrier_graph_probe_summary.json"
    report_path = output_dir / "diabetes_2000_2024_carrier_graph_probe_report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_path.write_text(build_markdown_report(summary), encoding="utf-8")

    log("Probe complete.")
    log(f"Wrote: {summary_path}")
    log(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()