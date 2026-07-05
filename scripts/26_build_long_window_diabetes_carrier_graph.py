#!/usr/bin/env python
"""Build a long-window diabetes carrier graph from PKG2 interim Parquet files.

Stage 05A:
    Build diabetes_2000_2024_v1 long-window Knowledge Carrier Graph.

This script materializes a domain-centered diabetes carrier graph from:

    data/interim/parquet/

Default output:

    data/processed/diabetes_2000_2024_v1_carrier_graph/

Design
------
The graph is built in a streaming / row-group manner where large tables are
involved. It avoids loading large edge tables fully into memory.

Default domain setup:
    domain = diabetes
    start_year = 2000
    end_year = 2024
    seed terms = diabetes / diabetic / diabetes mellitus / type 1 diabetes / type 2 diabetes
    seed entity type = disease
    context expansion = one-hop patent-paper and paper-trial
    project key alignment = ProjectNumber -> B02_Projects.CORE_PROJECT_NUM

Core outputs
------------
nodes_paper.parquet
nodes_patent.parquet
nodes_clinicaltrial.parquet
nodes_project.parquet
nodes_bioentity.parquet

edges_paper_bioentity.parquet
edges_patent_bioentity.parquet
edges_trial_bioentity.parquet

edges_patent_paper.parquet
edges_paper_trial.parquet

edges_paper_project.parquet
edges_patent_project.parquet
edges_trial_project.parquet

graph_summary.json
graph_manifest.csv
run_config.json

Typical run
-----------
python scripts/25_build_long_window_diabetes_carrier_graph.py --overwrite

Smoke test
----------
python scripts/25_build_long_window_diabetes_carrier_graph.py \
  --output-dir data/processed/diabetes_2000_2024_v1_carrier_graph_smoke \
  --max-row-groups 2 \
  --overwrite

Important
---------
If --max-row-groups is used, the output graph is incomplete and should only be
used for smoke testing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
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


# ---------------------------------------------------------------------------
# Logging / generic helpers
# ---------------------------------------------------------------------------


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def normalize_string_series(series: pd.Series) -> pd.Series:
    """Convert a Series to stable non-null string values."""

    return series.astype("string").fillna("").astype(str)


def isin_large_set(series: pd.Series, values: set[str]) -> pd.Series:
    """Fast membership check for large Python sets in row-group loops."""

    if not values:
        return pd.Series(False, index=series.index)

    text = normalize_string_series(series)
    contains = values.__contains__
    return text.map(contains)


def parse_year_series(series: pd.Series) -> pd.Series:
    """Parse numeric or date-like values into nullable year values.

    Optimized for streaming:
    - numeric-year columns return quickly;
    - date-like parsing only happens as fallback.
    """

    text = series.astype("string").str.strip()
    text = text.mask(
        text.isin(["", "NA", "N/A", "NULL", "None", "none", "nan", "NaN", "<NA>"]),
        pd.NA,
    )

    numeric = pd.to_numeric(text, errors="coerce")
    numeric_year = numeric.where((numeric >= 1500) & (numeric <= 2100))

    valid_numeric_fraction = (
        float(numeric_year.notna().mean()) if len(numeric_year) else 0.0
    )
    if valid_numeric_fraction >= 0.8:
        return numeric_year

    try:
        dt = pd.to_datetime(text, errors="coerce", utc=False, format="mixed")
    except TypeError:
        dt = pd.to_datetime(text, errors="coerce", utc=False)

    date_year = dt.dt.year.where((dt.dt.year >= 1500) & (dt.dt.year <= 2100))
    return numeric_year.fillna(date_year)


def normalize_bioentity_type(value: Any) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[\s_]+", "-", token)
    token = re.sub(r"-+", "-", token)

    aliases = {
        "disease-or-phenotypic-feature": "disease",
        "disease-phenotype": "disease",
        "phenotypic-feature": "phenotype",
        "phenotype": "phenotype",
        "chemical-substance": "chemical",
        "chemical": "chemical",
        "drug": "chemical",
        "gene-or-gene-product": "gene",
        "gene": "gene",
        "protein": "gene",
        "organism": "species",
        "taxon": "species",
        "ncbi-taxon": "species",
        "species": "species",
    }

    return aliases.get(token, token)


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def dataframe_memory_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(deep=True).sum() / 1024 / 1024)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, Counter):
        return counter_to_dict(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None

    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except Exception:
        pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, ensure_ascii=False, indent=2)


def read_row_group(
    parquet_file: pq.ParquetFile,
    row_group_idx: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    table = parquet_file.read_row_group(row_group_idx, columns=columns)
    return table.to_pandas()


def iter_row_groups(
    path: Path,
    columns: list[str] | None = None,
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


# ---------------------------------------------------------------------------
# Streaming Parquet writer
# ---------------------------------------------------------------------------


class StreamingParquetWriter:
    """Small wrapper around pyarrow.parquet.ParquetWriter."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0
        self.schema: pa.Schema | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return

        table = pa.Table.from_pandas(frame, preserve_index=False)

        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema)

        assert self.writer is not None
        self.writer.write_table(table)
        self.row_count += len(frame)

    def close(self, empty_columns: list[str] | None = None) -> int:
        if self.writer is not None:
            self.writer.close()
            return self.row_count

        if empty_columns is None:
            empty_columns = []

        pd.DataFrame(columns=empty_columns).to_parquet(self.path, index=False)
        return 0


# ---------------------------------------------------------------------------
# Stage 1: seed entity and seed carrier collection
# ---------------------------------------------------------------------------


def find_diabetes_seed_entities(
    *,
    input_dir: Path,
    seed_terms: list[str],
    seed_entity_types: set[str] | None,
) -> tuple[set[str], dict[str, Any]]:
    """Load C23_BioEntities and find diabetes-related seed EntityIds."""

    path = input_dir / TABLES["bioentities"]
    log(f"Loading BioEntities for seed matching: {path}")

    df = pd.read_parquet(path, columns=["EntityId", "Type", "Mention"])

    mention = normalize_string_series(df["Mention"]).str.lower()
    type_raw = normalize_string_series(df["Type"])
    type_norm = type_raw.map(normalize_bioentity_type)

    patterns = [re.escape(term.lower()) for term in seed_terms]
    pattern = "|".join(patterns)

    mask = mention.str.contains(pattern, regex=True, na=False)

    if seed_entity_types:
        normalized_allowed = {normalize_bioentity_type(v) for v in seed_entity_types}
        mask &= type_norm.isin(normalized_allowed)

    seeds = set(normalize_string_series(df.loc[mask, "EntityId"]))

    type_counts = Counter(type_norm.loc[mask])
    mention_examples = (
        df.loc[mask, ["EntityId", "Type", "Mention"]]
        .drop_duplicates()
        .head(100)
        .to_dict("records")
    )

    summary = {
        "seed_terms": seed_terms,
        "seed_entity_types": sorted(seed_entity_types) if seed_entity_types else None,
        "seed_entity_count": len(seeds),
        "seed_entity_type_counts": counter_to_dict(type_counts),
        "seed_entity_examples": mention_examples,
    }

    log(f"Seed BioEntities: {len(seeds):,}")
    log(f"Seed BioEntity type counts: {counter_to_dict(type_counts)}")
    return seeds, summary


def collect_seed_carriers_from_bioentity_links(
    *,
    input_dir: Path,
    table_key: str,
    carrier_id_col: str,
    seed_entity_ids: set[str],
    max_row_groups: int | None,
    progress_every_row_groups: int,
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
    max_row_groups: int | None,
    progress_every_row_groups: int,
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


# ---------------------------------------------------------------------------
# Stage 2: context expansion
# ---------------------------------------------------------------------------


def collect_patent_paper_context(
    *,
    input_dir: Path,
    selected_patents: set[str],
    selected_papers: set[str],
    max_row_groups: int | None,
    progress_every_row_groups: int,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Collect one-hop patent-paper context where either side is selected."""

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
    max_row_groups: int | None,
    progress_every_row_groups: int,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Collect one-hop paper-trial context where either side is selected."""

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


# ---------------------------------------------------------------------------
# Stage 3: project links and project-node alignment
# ---------------------------------------------------------------------------


def collect_project_numbers(
    *,
    input_dir: Path,
    selected_papers: set[str],
    selected_patents: set[str],
    selected_trials: set[str],
    max_row_groups: int | None,
    progress_every_row_groups: int,
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


def materialize_project_nodes(
    *,
    input_dir: Path,
    output_dir: Path,
    project_numbers: set[str],
    start_year: int,
    end_year: int,
    max_row_groups: int | None,
    progress_every_row_groups: int,
) -> tuple[set[str], dict[str, Any]]:
    """Materialize project nodes aligned by CORE_PROJECT_NUM.

    Link tables use ProjectNumber, and the probe showed that this aligns with
    B02_Projects.CORE_PROJECT_NUM. We aggregate B02 rows to one node per
    CORE_PROJECT_NUM / ProjectNumber.
    """

    path = input_dir / TABLES["projects"]
    output_path = output_dir / "nodes_project.parquet"

    log(f"Materializing project nodes from {path.name}")
    log("  Alignment: ProjectNumber -> CORE_PROJECT_NUM")
    log(f"  Candidate ProjectNumbers: {len(project_numbers):,}")

    parquet_file = pq.ParquetFile(path)
    all_columns = list(parquet_file.schema.names)

    selected_chunks: list[pd.DataFrame] = []
    scanned_rows = 0
    matched_rows = 0

    for rg, total_rg, df in iter_row_groups(
        path,
        all_columns,
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        project_values = normalize_string_series(df["CORE_PROJECT_NUM"])
        years = parse_year_series(df["FiscalYear"])
        mask = (
            project_values.isin(project_numbers)
            & years.notna()
            & (years >= start_year)
            & (years <= end_year)
        )

        if not mask.any():
            continue

        out = df.loc[mask].copy()
        out["ProjectNumber"] = normalize_string_series(out["CORE_PROJECT_NUM"])
        out["year"] = years.loc[mask].astype("Int64")
        matched_rows += len(out)
        selected_chunks.append(out)

    if not selected_chunks:
        empty = pd.DataFrame(columns=all_columns + ["ProjectNumber", "year"])
        empty.to_parquet(output_path, index=False)
        return set(), {
            "table": path.name,
            "output": str(output_path),
            "alignment_key": "CORE_PROJECT_NUM",
            "candidate_project_number_count": len(project_numbers),
            "selected_project_count": 0,
            "selected_project_record_rows": 0,
            "scanned_rows": scanned_rows,
            "output_rows": 0,
        }

    selected = pd.concat(selected_chunks, ignore_index=True)
    selected["ProjectNumber"] = normalize_string_series(selected["ProjectNumber"])
    selected["year"] = selected["year"].astype("Int64")

    # One node per ProjectNumber. Keep earliest row as representative and add
    # temporal span/count columns.
    selected = selected.sort_values(["ProjectNumber", "year"], ascending=[True, True])

    first_rows = (
        selected.drop_duplicates(subset=["ProjectNumber"], keep="first")
        .copy()
        .reset_index(drop=True)
    )

    span = (
        selected.groupby("ProjectNumber", as_index=False)
        .agg(
            project_year_min=("year", "min"),
            project_year_max=("year", "max"),
            project_record_count=("ProjectNumber", "size"),
            project_fiscal_year_count=("year", "nunique"),
        )
        .reset_index(drop=True)
    )

    nodes = first_rows.merge(span, on="ProjectNumber", how="left")
    nodes["year"] = nodes["project_year_min"].astype("Int64")
    nodes["project_year_min"] = nodes["project_year_min"].astype("Int64")
    nodes["project_year_max"] = nodes["project_year_max"].astype("Int64")

    nodes.to_parquet(output_path, index=False)

    selected_project_numbers = set(normalize_string_series(nodes["ProjectNumber"]))

    summary = {
        "table": path.name,
        "output": str(output_path),
        "alignment_key": "CORE_PROJECT_NUM",
        "candidate_project_number_count": len(project_numbers),
        "selected_project_count": len(selected_project_numbers),
        "selected_project_record_rows": int(matched_rows),
        "scanned_rows": int(scanned_rows),
        "output_rows": int(len(nodes)),
        "memory_mb_selected_records": dataframe_memory_mb(selected),
        "memory_mb_output_nodes": dataframe_memory_mb(nodes),
    }

    log(f"  Project nodes: {len(nodes):,}")
    return selected_project_numbers, summary


# ---------------------------------------------------------------------------
# Stage 4: materialization helpers
# ---------------------------------------------------------------------------


def materialize_carrier_nodes(
    *,
    input_dir: Path,
    output_dir: Path,
    table_key: str,
    output_name: str,
    id_col: str,
    year_col: str,
    selected_ids: set[str],
    start_year: int,
    end_year: int,
    extra_year_columns: list[str],
    max_row_groups: int | None,
    progress_every_row_groups: int,
) -> dict[str, Any]:
    """Materialize selected paper/patent/trial nodes."""

    path = input_dir / TABLES[table_key]
    output_path = output_dir / output_name

    log(f"Materializing carrier nodes: {output_name}")
    log(f"  Input: {path.name}")
    log(f"  Selected IDs: {len(selected_ids):,}")

    parquet_file = pq.ParquetFile(path)
    all_columns = list(parquet_file.schema.names)

    writer = StreamingParquetWriter(output_path)

    scanned_rows = 0
    output_rows = 0
    year_counts: Counter[int] = Counter()

    for rg, total_rg, df in iter_row_groups(
        path,
        all_columns,
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        ids = normalize_string_series(df[id_col])
        years = parse_year_series(df[year_col])

        mask = (
            ids.isin(selected_ids)
            & years.notna()
            & (years >= start_year)
            & (years <= end_year)
        )

        if not mask.any():
            continue

        out = df.loc[mask].copy()
        out["year"] = years.loc[mask].astype("Int64")

        for col in extra_year_columns:
            out[col] = out["year"]

        writer.write(out)
        output_rows += len(out)

        year_counts.update(out["year"].astype(int).value_counts().to_dict())

    writer.close(empty_columns=all_columns + ["year", *extra_year_columns])

    summary = {
        "input": str(path),
        "output": str(output_path),
        "id_col": id_col,
        "year_col": year_col,
        "selected_id_count": len(selected_ids),
        "scanned_rows": scanned_rows,
        "output_rows": output_rows,
        "year_counts": counter_to_dict(year_counts),
    }

    log(f"  Wrote {output_rows:,} rows -> {output_path}")
    return summary


def materialize_bioentity_edges(
    *,
    input_dir: Path,
    output_dir: Path,
    table_key: str,
    output_name: str,
    carrier_id_col: str,
    selected_carriers: set[str],
    max_row_groups: int | None,
    progress_every_row_groups: int,
) -> tuple[set[str], dict[str, Any]]:
    """Materialize selected carrier-BioEntity mention edges."""

    path = input_dir / TABLES[table_key]
    output_path = output_dir / output_name

    log(f"Materializing BioEntity edges: {output_name}")
    log(f"  Input: {path.name}")
    log(f"  Selected carriers: {len(selected_carriers):,}")

    parquet_file = pq.ParquetFile(path)
    all_columns = list(parquet_file.schema.names)

    writer = StreamingParquetWriter(output_path)

    scanned_rows = 0
    output_rows = 0
    carriers_with_edges: set[str] = set()
    bioentity_ids: set[str] = set()
    bioentity_type_counts: Counter[str] = Counter()

    for rg, total_rg, df in iter_row_groups(
        path,
        all_columns,
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        mask = isin_large_set(df[carrier_id_col], selected_carriers)

        if not mask.any():
            continue

        out = df.loc[mask].copy()
        writer.write(out)

        output_rows += len(out)

        carrier_values = normalize_string_series(out[carrier_id_col])
        carriers_with_edges.update(carrier_values[carrier_values != ""])

        entity_values = normalize_string_series(out["EntityId"])
        bioentity_ids.update(entity_values[entity_values != ""])

        if "Type" in out.columns:
            type_values = normalize_string_series(out["Type"]).map(normalize_bioentity_type)
            bioentity_type_counts.update(type_values.value_counts().to_dict())

    writer.close(empty_columns=all_columns)

    summary = {
        "input": str(path),
        "output": str(output_path),
        "carrier_id_col": carrier_id_col,
        "selected_carrier_input_count": len(selected_carriers),
        "selected_carrier_with_edges_count": len(carriers_with_edges),
        "selected_bioentity_count": len(bioentity_ids),
        "bioentity_type_counts": counter_to_dict(bioentity_type_counts),
        "scanned_rows": scanned_rows,
        "output_rows": output_rows,
    }

    log(f"  Wrote {output_rows:,} rows -> {output_path}")
    return bioentity_ids, summary


def materialize_bioentity_nodes(
    *,
    input_dir: Path,
    output_dir: Path,
    selected_bioentities: set[str],
    max_row_groups: int | None,
    progress_every_row_groups: int,
) -> dict[str, Any]:
    """Materialize selected BioEntity nodes from C23_BioEntities."""

    path = input_dir / TABLES["bioentities"]
    output_path = output_dir / "nodes_bioentity.parquet"

    log("Materializing BioEntity nodes")
    log(f"  Selected BioEntities: {len(selected_bioentities):,}")

    parquet_file = pq.ParquetFile(path)
    all_columns = list(parquet_file.schema.names)

    writer = StreamingParquetWriter(output_path)

    scanned_rows = 0
    output_rows = 0
    type_counts: Counter[str] = Counter()

    for rg, total_rg, df in iter_row_groups(
        path,
        all_columns,
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)
        mask = isin_large_set(df["EntityId"], selected_bioentities)

        if not mask.any():
            continue

        out = df.loc[mask].copy()
        writer.write(out)
        output_rows += len(out)

        type_values = normalize_string_series(out["Type"]).map(normalize_bioentity_type)
        type_counts.update(type_values.value_counts().to_dict())

    writer.close(empty_columns=all_columns)

    summary = {
        "input": str(path),
        "output": str(output_path),
        "selected_bioentity_input_count": len(selected_bioentities),
        "scanned_rows": scanned_rows,
        "output_rows": output_rows,
        "bioentity_type_counts": counter_to_dict(type_counts),
    }

    log(f"  Wrote {output_rows:,} rows -> {output_path}")
    return summary


def materialize_pair_edges(
    *,
    input_dir: Path,
    output_dir: Path,
    table_key: str,
    output_name: str,
    left_col: str,
    right_col: str,
    selected_left: set[str],
    selected_right: set[str],
    max_row_groups: int | None,
    progress_every_row_groups: int,
) -> dict[str, Any]:
    """Materialize binary context edges where both endpoints are selected."""

    path = input_dir / TABLES[table_key]
    output_path = output_dir / output_name

    log(f"Materializing context edges: {output_name}")
    log(f"  Input: {path.name}")
    log(f"  Selected left ({left_col}): {len(selected_left):,}")
    log(f"  Selected right ({right_col}): {len(selected_right):,}")

    parquet_file = pq.ParquetFile(path)
    all_columns = list(parquet_file.schema.names)

    writer = StreamingParquetWriter(output_path)

    scanned_rows = 0
    output_rows = 0
    left_nodes_with_edges: set[str] = set()
    right_nodes_with_edges: set[str] = set()

    for rg, total_rg, df in iter_row_groups(
        path,
        all_columns,
        max_row_groups=max_row_groups,
    ):
        if rg % progress_every_row_groups == 0 or rg == total_rg - 1:
            log(f"  {path.name}: row group {rg + 1}/{total_rg}")

        scanned_rows += len(df)

        mask = isin_large_set(df[left_col], selected_left) & isin_large_set(
            df[right_col],
            selected_right,
        )

        if not mask.any():
            continue

        out = df.loc[mask].copy()
        writer.write(out)
        output_rows += len(out)

        left_values = normalize_string_series(out[left_col])
        right_values = normalize_string_series(out[right_col])
        left_nodes_with_edges.update(left_values[left_values != ""])
        right_nodes_with_edges.update(right_values[right_values != ""])

    writer.close(empty_columns=all_columns)

    summary = {
        "input": str(path),
        "output": str(output_path),
        "left_col": left_col,
        "right_col": right_col,
        "selected_left_count": len(selected_left),
        "selected_right_count": len(selected_right),
        "left_nodes_with_edges_count": len(left_nodes_with_edges),
        "right_nodes_with_edges_count": len(right_nodes_with_edges),
        "scanned_rows": scanned_rows,
        "output_rows": output_rows,
    }

    log(f"  Wrote {output_rows:,} rows -> {output_path}")
    return summary


# ---------------------------------------------------------------------------
# Reports / manifest
# ---------------------------------------------------------------------------


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


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = output_dir / "graph_manifest.csv"
    rows.append(
        artifact_row(
            "graph_manifest",
            manifest_path,
            "Manifest of long-window diabetes carrier graph artifacts.",
            row_count=len(rows) + 1,
        )
    )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def build_markdown_report(summary: dict[str, Any]) -> str:
    scale = summary["graph_scale"]

    lines = [
        "# Long-window Diabetes Carrier Graph Build Report",
        "",
        "## 1. Configuration",
        "",
        "```json",
        json.dumps(summary["config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 2. Final graph scale",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Papers | {scale['paper_nodes']:,} |",
        f"| Patents | {scale['patent_nodes']:,} |",
        f"| Clinical trials | {scale['trial_nodes']:,} |",
        f"| Projects | {scale['project_nodes']:,} |",
        f"| BioEntities | {scale['bioentity_nodes']:,} |",
        f"| Paper-BioEntity edges | {scale['paper_bioentity_edges']:,} |",
        f"| Patent-BioEntity edges | {scale['patent_bioentity_edges']:,} |",
        f"| Trial-BioEntity edges | {scale['trial_bioentity_edges']:,} |",
        f"| Patent-Paper edges | {scale['patent_paper_edges']:,} |",
        f"| Paper-Trial edges | {scale['paper_trial_edges']:,} |",
        f"| Paper-Project edges | {scale['paper_project_edges']:,} |",
        f"| Patent-Project edges | {scale['patent_project_edges']:,} |",
        f"| Trial-Project edges | {scale['trial_project_edges']:,} |",
        "",
        "## 3. Seed entities",
        "",
        f"- Seed BioEntities: **{summary['seed_entities']['seed_entity_count']:,}**",
        f"- Type counts: `{summary['seed_entities']['seed_entity_type_counts']}`",
        "",
        "## 4. Output directory",
        "",
        "```text",
        summary["config"]["output_dir"],
        "```",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build long-window diabetes carrier graph from interim Parquet."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/interim/parquet"),
        help="Input directory containing PKG2 interim Parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/diabetes_2000_2024_v1_carrier_graph"),
        help="Output processed carrier graph directory.",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--seed-term", action="append", default=None)
    parser.add_argument(
        "--seed-entity-type",
        action="append",
        default=["disease"],
        help=(
            "Allowed normalized BioEntity type for domain seed matching. "
            "Can be passed multiple times. Default: disease. "
            "Use --seed-entity-type all to disable type filtering."
        ),
    )
    parser.add_argument(
        "--max-row-groups",
        type=int,
        default=None,
        help=(
            "Limit row groups per table for smoke testing. "
            "If set, output graph is incomplete."
        ),
    )
    parser.add_argument(
        "--progress-every-row-groups",
        type=int,
        default=10,
        help="Print progress every N row groups.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    seed_terms = args.seed_term if args.seed_term else DEFAULT_SEED_TERMS

    seed_entity_types_raw = args.seed_entity_type or ["disease"]
    if any(str(v).lower() == "all" for v in seed_entity_types_raw):
        seed_entity_types = None
    else:
        seed_entity_types = {normalize_bioentity_type(v) for v in seed_entity_types_raw}

    progress_every = max(int(args.progress_every_row_groups), 1)

    config = {
        "script": "scripts/25_build_long_window_diabetes_carrier_graph.py",
        "command": " ".join(sys.argv),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "domain": "diabetes",
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "seed_terms": seed_terms,
        "seed_entity_types": sorted(seed_entity_types) if seed_entity_types else None,
        "context_expansion": {
            "patent_paper": True,
            "paper_trial": True,
            "project_links": True,
            "depth": 1,
        },
        "project_key_alignment": "ProjectNumber -> B02_Projects.CORE_PROJECT_NUM",
        "max_row_groups": args.max_row_groups,
        "progress_every_row_groups": progress_every,
        "created_at_unix": time.time(),
        "materializes_graph": True,
    }

    log("Starting long-window diabetes carrier graph build")
    log(json.dumps(config, ensure_ascii=False, indent=2))

    if args.max_row_groups is not None:
        log(
            "WARNING: --max-row-groups is set. "
            "This output graph will be incomplete and should only be used for smoke testing."
        )

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    write_json(output_dir / "run_config.json", config)

    # ------------------------------------------------------------------
    # 1. Seed entities and direct seed carriers
    # ------------------------------------------------------------------

    seed_entity_ids, seed_entity_summary = find_diabetes_seed_entities(
        input_dir=input_dir,
        seed_terms=seed_terms,
        seed_entity_types=seed_entity_types,
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

    # ------------------------------------------------------------------
    # 2. Year-filter direct seed carriers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 3. One-hop context expansion
    # ------------------------------------------------------------------

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

    expanded_paper_candidates = (
        selected_seed_papers | context_papers_pp | context_papers_pt
    )
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

    # ------------------------------------------------------------------
    # 4. Project collection and project-node materialization
    # ------------------------------------------------------------------

    project_numbers, project_link_collection_summary = collect_project_numbers(
        input_dir=input_dir,
        selected_papers=selected_papers,
        selected_patents=selected_patents,
        selected_trials=selected_trials,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    selected_project_numbers, project_node_summary = materialize_project_nodes(
        input_dir=input_dir,
        output_dir=output_dir,
        project_numbers=project_numbers,
        start_year=args.start_year,
        end_year=args.end_year,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    # ------------------------------------------------------------------
    # 5. Materialize carrier nodes
    # ------------------------------------------------------------------

    paper_node_summary = materialize_carrier_nodes(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="papers",
        output_name="nodes_paper.parquet",
        id_col="PMID",
        year_col="PubYear",
        selected_ids=selected_papers,
        start_year=args.start_year,
        end_year=args.end_year,
        extra_year_columns=[],
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    patent_node_summary = materialize_carrier_nodes(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="patents",
        output_name="nodes_patent.parquet",
        id_col="PatentId",
        year_col="GrantedDate",
        selected_ids=selected_patents,
        start_year=args.start_year,
        end_year=args.end_year,
        extra_year_columns=["patent_year"],
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    trial_node_summary = materialize_carrier_nodes(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="trials",
        output_name="nodes_clinicaltrial.parquet",
        id_col="nct_id",
        year_col="start_date",
        selected_ids=selected_trials,
        start_year=args.start_year,
        end_year=args.end_year,
        extra_year_columns=["start_year"],
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    # ------------------------------------------------------------------
    # 6. Materialize carrier-BioEntity edges and collect BioEntity IDs
    # ------------------------------------------------------------------

    paper_bioentities, paper_bioedge_summary = materialize_bioentity_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="paper_bioentities",
        output_name="edges_paper_bioentity.parquet",
        carrier_id_col="PMID",
        selected_carriers=selected_papers,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    patent_bioentities, patent_bioedge_summary = materialize_bioentity_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="patent_bioentities",
        output_name="edges_patent_bioentity.parquet",
        carrier_id_col="PatentId",
        selected_carriers=selected_patents,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    trial_bioentities, trial_bioedge_summary = materialize_bioentity_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="trial_bioentities",
        output_name="edges_trial_bioentity.parquet",
        carrier_id_col="nct_id",
        selected_carriers=selected_trials,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    selected_bioentities = paper_bioentities | patent_bioentities | trial_bioentities

    bioentity_node_summary = materialize_bioentity_nodes(
        input_dir=input_dir,
        output_dir=output_dir,
        selected_bioentities=selected_bioentities,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    # ------------------------------------------------------------------
    # 7. Materialize context edges
    # ------------------------------------------------------------------

    patent_paper_edge_summary = materialize_pair_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="patent_paper",
        output_name="edges_patent_paper.parquet",
        left_col="PatentId",
        right_col="PMID",
        selected_left=selected_patents,
        selected_right=selected_papers,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    paper_trial_edge_summary = materialize_pair_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="paper_trial",
        output_name="edges_paper_trial.parquet",
        left_col="PMID",
        right_col="nct_id",
        selected_left=selected_papers,
        selected_right=selected_trials,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    paper_project_edge_summary = materialize_pair_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="paper_project",
        output_name="edges_paper_project.parquet",
        left_col="PMID",
        right_col="ProjectNumber",
        selected_left=selected_papers,
        selected_right=selected_project_numbers,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    patent_project_edge_summary = materialize_pair_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="patent_project",
        output_name="edges_patent_project.parquet",
        left_col="PatentId",
        right_col="ProjectNumber",
        selected_left=selected_patents,
        selected_right=selected_project_numbers,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    trial_project_edge_summary = materialize_pair_edges(
        input_dir=input_dir,
        output_dir=output_dir,
        table_key="trial_project",
        output_name="edges_trial_project.parquet",
        left_col="nct_id",
        right_col="ProjectNumber",
        selected_left=selected_trials,
        selected_right=selected_project_numbers,
        max_row_groups=args.max_row_groups,
        progress_every_row_groups=progress_every,
    )

    # ------------------------------------------------------------------
    # 8. Summary / manifest / report
    # ------------------------------------------------------------------

    graph_scale = {
        "paper_nodes": paper_node_summary["output_rows"],
        "patent_nodes": patent_node_summary["output_rows"],
        "trial_nodes": trial_node_summary["output_rows"],
        "project_nodes": project_node_summary["output_rows"],
        "bioentity_nodes": bioentity_node_summary["output_rows"],
        "paper_bioentity_edges": paper_bioedge_summary["output_rows"],
        "patent_bioentity_edges": patent_bioedge_summary["output_rows"],
        "trial_bioentity_edges": trial_bioedge_summary["output_rows"],
        "patent_paper_edges": patent_paper_edge_summary["output_rows"],
        "paper_trial_edges": paper_trial_edge_summary["output_rows"],
        "paper_project_edges": paper_project_edge_summary["output_rows"],
        "patent_project_edges": patent_project_edge_summary["output_rows"],
        "trial_project_edges": trial_project_edge_summary["output_rows"],
    }

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
        "project_link_collection": project_link_collection_summary,
        "node_outputs": {
            "paper": paper_node_summary,
            "patent": patent_node_summary,
            "trial": trial_node_summary,
            "project": project_node_summary,
            "bioentity": bioentity_node_summary,
        },
        "edge_outputs": {
            "paper_bioentity": paper_bioedge_summary,
            "patent_bioentity": patent_bioedge_summary,
            "trial_bioentity": trial_bioedge_summary,
            "patent_paper": patent_paper_edge_summary,
            "paper_trial": paper_trial_edge_summary,
            "paper_project": paper_project_edge_summary,
            "patent_project": patent_project_edge_summary,
            "trial_project": trial_project_edge_summary,
        },
        "graph_scale": graph_scale,
    }

    summary_path = output_dir / "graph_summary.json"
    report_path = output_dir / "graph_report.md"

    write_json(summary_path, summary)
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")

    manifest_rows = [
        artifact_row(
            "nodes_paper",
            output_dir / "nodes_paper.parquet",
            "Selected paper carrier nodes.",
            row_count=graph_scale["paper_nodes"],
        ),
        artifact_row(
            "nodes_patent",
            output_dir / "nodes_patent.parquet",
            "Selected patent carrier nodes.",
            row_count=graph_scale["patent_nodes"],
        ),
        artifact_row(
            "nodes_clinicaltrial",
            output_dir / "nodes_clinicaltrial.parquet",
            "Selected clinical trial carrier nodes.",
            row_count=graph_scale["trial_nodes"],
        ),
        artifact_row(
            "nodes_project",
            output_dir / "nodes_project.parquet",
            "Selected project nodes aligned by CORE_PROJECT_NUM.",
            row_count=graph_scale["project_nodes"],
        ),
        artifact_row(
            "nodes_bioentity",
            output_dir / "nodes_bioentity.parquet",
            "BioEntity nodes appearing on selected carriers.",
            row_count=graph_scale["bioentity_nodes"],
        ),
        artifact_row(
            "edges_paper_bioentity",
            output_dir / "edges_paper_bioentity.parquet",
            "Paper-BioEntity mention edges.",
            row_count=graph_scale["paper_bioentity_edges"],
        ),
        artifact_row(
            "edges_patent_bioentity",
            output_dir / "edges_patent_bioentity.parquet",
            "Patent-BioEntity mention edges.",
            row_count=graph_scale["patent_bioentity_edges"],
        ),
        artifact_row(
            "edges_trial_bioentity",
            output_dir / "edges_trial_bioentity.parquet",
            "ClinicalTrial-BioEntity mention edges.",
            row_count=graph_scale["trial_bioentity_edges"],
        ),
        artifact_row(
            "edges_patent_paper",
            output_dir / "edges_patent_paper.parquet",
            "Patent-Paper context edges.",
            row_count=graph_scale["patent_paper_edges"],
        ),
        artifact_row(
            "edges_paper_trial",
            output_dir / "edges_paper_trial.parquet",
            "Paper-ClinicalTrial context edges.",
            row_count=graph_scale["paper_trial_edges"],
        ),
        artifact_row(
            "edges_paper_project",
            output_dir / "edges_paper_project.parquet",
            "Paper-Project funding/context edges.",
            row_count=graph_scale["paper_project_edges"],
        ),
        artifact_row(
            "edges_patent_project",
            output_dir / "edges_patent_project.parquet",
            "Patent-Project funding/context edges.",
            row_count=graph_scale["patent_project_edges"],
        ),
        artifact_row(
            "edges_trial_project",
            output_dir / "edges_trial_project.parquet",
            "ClinicalTrial-Project funding/context edges.",
            row_count=graph_scale["trial_project_edges"],
        ),
        artifact_row(
            "graph_summary",
            summary_path,
            "JSON summary for long-window diabetes carrier graph build.",
        ),
        artifact_row(
            "graph_report",
            report_path,
            "Markdown report for long-window diabetes carrier graph build.",
        ),
        artifact_row(
            "run_config",
            output_dir / "run_config.json",
            "Run configuration.",
        ),
    ]

    write_manifest(output_dir, manifest_rows)

    log("Long-window diabetes carrier graph build complete.")
    log(f"Output directory: {output_dir}")
    log(f"Graph summary: {summary_path}")
    log(f"Graph report: {report_path}")
    log(f"Graph manifest: {output_dir / 'graph_manifest.csv'}")
    log("Final graph scale:")
    log(json.dumps(graph_scale, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()