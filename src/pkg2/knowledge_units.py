"""Knowledge Unit construction utilities.

This module implements the first Knowledge Unit MVP:

    Knowledge Unit = typed BioEntity pair

It builds Knowledge Units from carrier-level BioEntity co-mentions:

    Paper / Patent / Trial --mentions--> BioEntity

The main outputs are intended to be:

    knowledge_units.parquet
    carrier_knowledge_unit_edges.parquet
    knowledge_unit_type_summary.csv
    knowledge_unit_summary.json

PrimeKG support is intentionally not implemented in this first module.
When PrimeKG is added later, the recommended path is:

    1. Inspect PrimeKG node / edge schemas.
    2. Audit ID and name mapping coverage.
    3. Build entity-pair KU first.
    4. Add PrimeKG relation annotations as auxiliary evidence.
    5. Only then optionally derive relation-level KU candidates.

Design assumptions
------------------
1. The current graph is a carrier graph, not a Knowledge Unit graph.
2. Carrier-to-BioEntity mention edges provide co-mention evidence.
3. Co-mention does not imply a specific biomedical relation.
4. Therefore, V1 Knowledge Units are typed entity pairs rather than
   asserted relation triples.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CARRIER_TYPES = ("paper", "patent", "trial")

DEFAULT_EXCLUDED_BIOENTITY_TYPES = ("species",)

DEFAULT_PAIR_TYPES = (
    "chemical-disease",
    "gene-disease",
    "chemical-gene",
    # Keep phenotype-related pairs available for future datasets where
    # phenotype entities are reliably present.
    "disease-phenotype",
)

DEFAULT_BIOENTITY_NODE_FILES = (
    "nodes_bioentity.parquet",
    "nodes_bioentities.parquet",
    "bioentities.parquet",
    "bioentity_nodes.parquet",
)

DEFAULT_PAPER_NODE_FILES = (
    "nodes_paper.parquet",
    "nodes_papers.parquet",
    "papers.parquet",
    "paper_nodes.parquet",
)

DEFAULT_PATENT_NODE_FILES = (
    "nodes_patent.parquet",
    "nodes_patents.parquet",
    "patents.parquet",
    "patent_nodes.parquet",
)

DEFAULT_TRIAL_NODE_FILES = (
    "nodes_trial.parquet",
    "nodes_trials.parquet",
    "nodes_clinicaltrial.parquet",
    "nodes_clinical_trials.parquet",
    "clinical_trials.parquet",
    "trials.parquet",
    "trial_nodes.parquet",
)

DEFAULT_MENTION_EDGE_FILES = {
    "paper": (
        "edges_paper_bioentity.parquet",
        "paper_bioentity_edges.parquet",
        "edges_paper_bioentities.parquet",
    ),
    "patent": (
        "edges_patent_bioentity.parquet",
        "patent_bioentity_edges.parquet",
        "edges_patent_bioentities.parquet",
    ),
    "trial": (
        "edges_trial_bioentity.parquet",
        "trial_bioentity_edges.parquet",
        "edges_trial_bioentities.parquet",
    ),
}

BIOENTITY_ID_COLUMNS = (
    "bioentity_id",
    "entity_id",
    "EntityId",
    "EntityID",
    "node_id",
    "id",
    "BioEntityID",
    "identifier",
)

BIOENTITY_NAME_COLUMNS = (
    "bioentity_name",
    "entity_name",
    "Mention",
    "mention",
    "name",
    "label",
    "Name",
    "BioEntityName",
)

BIOENTITY_TYPE_COLUMNS = (
    "bioentity_type",
    "entity_type",
    "Type",
    "type",
    "category",
    "BioEntityType",
)

MENTION_COUNT_COLUMNS = (
    "mention_count",
    "mentions",
    "count",
    "weight",
    "edge_weight",
    "MentionCount",
    "distinct_mention_count",
)

YEAR_COLUMNS = (
    "year",
    "Year",
    "PubYear",
    "pub_year",
    "publication_year",
    "PublicationYear",
    "patent_year",
    "PatentYear",
    "filing_year",
    "FilingYear",
    "start_year",
    "StartYear",
    "StartDate",
    "start_date",
)

SOURCE_COLUMNS = (
    "source_id",
    "src_id",
    "head_id",
    "from_id",
    "source",
    "src",
    "head",
)

TARGET_COLUMNS = (
    "target_id",
    "dst_id",
    "tail_id",
    "to_id",
    "target",
    "dst",
    "tail",
)

CARRIER_ID_COLUMNS = {
    "paper": (
        "PMID",
        "pmid",
        "paper_id",
        "PaperID",
        "source_id",
        "src_id",
        "head_id",
        "id",
        "node_id",
    ),
    "patent": (
        "PatentId",
        "patent_id",
        "patent",
        "PatentID",
        "publication_number",
        "source_id",
        "src_id",
        "head_id",
        "id",
        "node_id",
    ),
    "trial": (
        "nct_id",
        "NCTId",
        "NCT",
        "trial_id",
        "ClinicalTrialID",
        "source_id",
        "src_id",
        "head_id",
        "id",
        "node_id",
    ),
}

EDGE_BIOENTITY_ID_COLUMNS = (
    "bioentity_id",
    "entity_id",
    "EntityId",
    "EntityID",
    "target_id",
    "dst_id",
    "tail_id",
    "BioEntityID",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeUnitBuildConfig:
    """Configuration for entity-pair Knowledge Unit construction."""

    graph_dir: str | None = None
    output_dir: str | None = None
    carrier_types: tuple[str, ...] = DEFAULT_CARRIER_TYPES
    exclude_bioentity_types: tuple[str, ...] = DEFAULT_EXCLUDED_BIOENTITY_TYPES
    pair_types: tuple[str, ...] = DEFAULT_PAIR_TYPES
    min_carrier_count: int = 3
    min_paper_count: int = 0
    min_patent_count: int = 0
    min_trial_count: int = 0
    pair_weight_method: str = "sqrt_product"
    max_entities_per_carrier: int | None = 100
    keep_unyearled_edges: bool = True
    ku_id_prefix: str = "KU"
    created_at_unix: float = field(default_factory=time.time)

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-safe config dictionary."""

        return asdict(self)


@dataclass
class KnowledgeUnitBuildResult:
    """Container for Knowledge Unit construction outputs."""

    knowledge_units: pd.DataFrame
    carrier_knowledge_unit_edges: pd.DataFrame
    type_summary: pd.DataFrame
    summary: dict[str, Any]
    run_config: dict[str, Any]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def normalize_token(value: Any) -> str:
    """Normalize a string-like token for matching."""

    if value is None:
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text


def normalize_carrier_type(value: Any) -> str:
    """Normalize carrier type names.

    The current aggregated graph uses values such as:

        Paper
        Patent
        ClinicalTrial

    This function maps them to canonical lower-case carrier types:

        paper
        patent
        trial
    """

    token = normalize_token(value)

    aliases = {
        "paper": "paper",
        "papers": "paper",
        "pubmed": "paper",
        "pmid": "paper",
        "article": "paper",
        "articles": "paper",
        "patent": "patent",
        "patents": "patent",
        "clinicaltrial": "trial",
        "clinical-trial": "trial",
        "clinical-trials": "trial",
        "trial": "trial",
        "trials": "trial",
        "nct": "trial",
    }

    return aliases.get(token, token)


def normalize_entity_type(value: Any) -> str:
    """Normalize BioEntity type names."""

    token = normalize_token(value)

    aliases = {
        "disease-or-phenotypic-feature": "disease",
        "disease-phenotype": "disease",
        "phenotypic-feature": "phenotype",
        "phenotype": "phenotype",
        "cell-type": "cell-type",
        "cell-type-or-cell-line": "cell-type",
        "cell-line": "cell-line",
        "chemical-substance": "chemical",
        "chemical": "chemical",
        "drug": "chemical",
        "compound": "chemical",
        "acid": "chemical",
        "gene-or-gene-product": "gene",
        "gene": "gene",
        "protein": "gene",
        "organism": "species",
        "taxon": "species",
        "ncbi-taxon": "species",
        "species": "species",
    }

    return aliases.get(token, token)


def normalize_pair_type(value: str) -> str:
    """Normalize pair type string such as chemical-disease."""

    parts = [normalize_entity_type(part) for part in str(value).split("-")]
    parts = [part for part in parts if part]

    if len(parts) != 2:
        raise ValueError(f"Invalid pair type {value!r}; expected '<type_a>-<type_b>'.")

    return f"{parts[0]}-{parts[1]}"


def stable_hash(parts: Sequence[Any], length: int = 16) -> str:
    """Create a stable short SHA1 hash from string-like parts."""

    payload = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def make_ku_id(
    *,
    entity_a_id: Any,
    entity_b_id: Any,
    pair_type: str,
    prefix: str = "KU",
) -> str:
    """Create a stable Knowledge Unit ID."""

    digest = stable_hash([pair_type, entity_a_id, entity_b_id], length=16)
    return f"{prefix}_{digest}"


def find_first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
    role: str = "column",
) -> str | None:
    """Return the first candidate column present in a DataFrame."""

    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise ValueError(
            f"Could not find {role}. "
            f"Looked for {list(candidates)} in columns {list(frame.columns)}."
        )

    return None


def find_existing_file(directory: Path, candidates: Sequence[str]) -> Path | None:
    """Find the first existing file under a directory."""

    for filename in candidates:
        path = directory / filename
        if path.exists():
            return path

    return None


def read_parquet_if_exists(path: Path | None) -> pd.DataFrame | None:
    """Read a parquet file if path exists."""

    if path is None:
        return None

    if not path.exists():
        return None

    return pd.read_parquet(path)


def ensure_string_id(series: pd.Series) -> pd.Series:
    """Convert ID series to stable string representation."""

    return series.astype("string").fillna("").astype(str)


def safe_numeric_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Convert series to numeric with default for missing / invalid values."""

    return pd.to_numeric(series, errors="coerce").fillna(default)


def safe_year_series(series: pd.Series | None) -> pd.Series:
    """Convert a year-like or date-like series to nullable integer years.

    Supports both numeric years:

        2018
        2018.0
        "2018"

    and date strings:

        "2018-03-15"
        "2018/03/15"
        "March 2018"

    This is important for clinical trial node tables, where start dates are
    usually stored as date strings rather than plain numeric years.
    """

    if series is None:
        return pd.Series(dtype="Int64")

    text = series.astype("string").str.strip()

    missing_tokens = {
        "",
        "NA",
        "N/A",
        "NULL",
        "None",
        "none",
        "nan",
        "NaN",
        "<NA>",
    }

    text = text.mask(text.isin(missing_tokens), pd.NA)

    # First try direct numeric years, e.g. 2018, 2018.0, "2018".
    numeric = pd.to_numeric(text, errors="coerce")
    numeric_year = numeric.where((numeric >= 1000) & (numeric <= 3000))

    # Then try date-like strings, e.g. "2018-03-15".
    parsed_dates = pd.to_datetime(text, errors="coerce")
    date_year = parsed_dates.dt.year.astype("float")

    year = numeric_year.fillna(date_year)
    year = year.where((year >= 1000) & (year <= 3000))

    return year.round().astype("Int64")


def dataframe_memory_mb(frame: pd.DataFrame) -> float:
    """Return approximate DataFrame memory size in MB."""

    return float(frame.memory_usage(deep=True).sum() / 1024 / 1024)


# ---------------------------------------------------------------------------
# Loading and standardization
# ---------------------------------------------------------------------------


def load_graph_bioentity_nodes(graph_dir: str | Path) -> pd.DataFrame:
    """Load BioEntity node table from a processed graph directory."""

    graph_dir = Path(graph_dir)
    path = find_existing_file(graph_dir, DEFAULT_BIOENTITY_NODE_FILES)

    if path is None:
        raise FileNotFoundError(
            "Could not find BioEntity node table. "
            f"Tried: {list(DEFAULT_BIOENTITY_NODE_FILES)} under {graph_dir}"
        )

    frame = pd.read_parquet(path)
    return standardize_bioentity_nodes(frame, source_path=path)


def load_carrier_nodes(
    graph_dir: str | Path,
    *,
    carrier_type: str,
) -> pd.DataFrame | None:
    """Load optional carrier node table for year metadata."""

    graph_dir = Path(graph_dir)
    carrier_type = normalize_carrier_type(carrier_type)

    file_candidates = {
        "paper": DEFAULT_PAPER_NODE_FILES,
        "patent": DEFAULT_PATENT_NODE_FILES,
        "trial": DEFAULT_TRIAL_NODE_FILES,
    }.get(carrier_type)

    if not file_candidates:
        raise ValueError(f"Unsupported carrier_type {carrier_type!r}.")

    path = find_existing_file(graph_dir, file_candidates)
    frame = read_parquet_if_exists(path)

    if frame is None:
        return None

    return standardize_carrier_nodes(frame, carrier_type=carrier_type, source_path=path)


def load_mention_edges(
    graph_dir: str | Path,
    *,
    carrier_type: str,
) -> pd.DataFrame | None:
    """Load raw carrier-BioEntity mention edges."""

    graph_dir = Path(graph_dir)
    carrier_type = normalize_carrier_type(carrier_type)

    file_candidates = DEFAULT_MENTION_EDGE_FILES.get(carrier_type)

    if not file_candidates:
        raise ValueError(f"Unsupported carrier_type {carrier_type!r}.")

    path = find_existing_file(graph_dir, file_candidates)
    frame = read_parquet_if_exists(path)

    if frame is None:
        return None

    return frame


def standardize_bioentity_nodes(
    frame: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
) -> pd.DataFrame:
    """Standardize BioEntity node table.

    Returns columns:

        bioentity_id
        bioentity_name
        bioentity_type
        bioentity_type_raw
    """

    id_col = find_first_existing_column(
        frame,
        BIOENTITY_ID_COLUMNS,
        role="BioEntity ID column",
    )
    name_col = find_first_existing_column(
        frame,
        BIOENTITY_NAME_COLUMNS,
        required=False,
        role="BioEntity name column",
    )
    type_col = find_first_existing_column(
        frame,
        BIOENTITY_TYPE_COLUMNS,
        required=False,
        role="BioEntity type column",
    )

    out = pd.DataFrame()
    out["bioentity_id"] = ensure_string_id(frame[id_col])

    if name_col is not None:
        out["bioentity_name"] = frame[name_col].astype("string").fillna("").astype(str)
    else:
        out["bioentity_name"] = out["bioentity_id"]

    if type_col is not None:
        out["bioentity_type_raw"] = frame[type_col].astype("string").fillna("").astype(str)
        out["bioentity_type"] = out["bioentity_type_raw"].map(normalize_entity_type)
    else:
        out["bioentity_type_raw"] = "unknown"
        out["bioentity_type"] = "unknown"

    if source_path is not None:
        out["source_node_table"] = str(source_path)

    out = out.drop_duplicates(subset=["bioentity_id"], keep="first").reset_index(drop=True)
    return out


def standardize_carrier_nodes(
    frame: pd.DataFrame,
    *,
    carrier_type: str,
    source_path: str | Path | None = None,
) -> pd.DataFrame:
    """Standardize optional carrier node table.

    Returns columns:

        carrier_type
        carrier_id
        year

    Important:
        Build the output DataFrame with the same index as the input frame.
        If we start from an empty DataFrame and assign a scalar column first,
        pandas creates a zero-length column; later assigning Series columns
        can leave the scalar column as NaN after index alignment.
    """

    carrier_type = normalize_carrier_type(carrier_type)

    id_col = find_first_existing_column(
        frame,
        CARRIER_ID_COLUMNS[carrier_type],
        role=f"{carrier_type} ID column",
    )
    year_col = find_first_existing_column(
        frame,
        YEAR_COLUMNS,
        required=False,
        role=f"{carrier_type} year column",
    )

    out = pd.DataFrame(index=frame.index)
    out["carrier_type"] = carrier_type
    out["carrier_id"] = ensure_string_id(frame[id_col])

    if year_col is not None:
        out["year"] = safe_year_series(frame[year_col])
    else:
        out["year"] = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Int64")

    if source_path is not None:
        out["source_node_table"] = str(source_path)

    out = out.drop_duplicates(subset=["carrier_type", "carrier_id"], keep="first")
    return out.reset_index(drop=True)


def standardize_mention_edges(
    frame: pd.DataFrame,
    *,
    carrier_type: str,
    bioentities: pd.DataFrame,
    carrier_nodes: pd.DataFrame | None = None,
    source_edge_table: str | Path | None = None,
) -> pd.DataFrame:
    """Standardize a carrier-BioEntity mention edge table.

    Returns columns:

        carrier_type
        carrier_id
        bioentity_id
        bioentity_name
        bioentity_type
        year
        mention_count
        source_edge_table
    """

    carrier_type = normalize_carrier_type(carrier_type)

    carrier_id_col = find_first_existing_column(
        frame,
        CARRIER_ID_COLUMNS[carrier_type],
        required=False,
        role=f"{carrier_type} ID edge column",
    )

    if carrier_id_col is None:
        carrier_id_col = find_first_existing_column(
            frame,
            SOURCE_COLUMNS,
            role=f"{carrier_type} source ID edge column",
        )

    bioentity_id_col = find_first_existing_column(
        frame,
        EDGE_BIOENTITY_ID_COLUMNS,
        required=False,
        role="BioEntity edge ID column",
    )

    if bioentity_id_col is None:
        bioentity_id_col = find_first_existing_column(
            frame,
            TARGET_COLUMNS,
            role="BioEntity target ID edge column",
        )

    mention_col = find_first_existing_column(
        frame,
        MENTION_COUNT_COLUMNS,
        required=False,
        role="mention count column",
    )

    year_col = find_first_existing_column(
        frame,
        YEAR_COLUMNS,
        required=False,
        role="edge year column",
    )

    out = pd.DataFrame()

    if "carrier_type" in frame.columns:
        out["carrier_type"] = frame["carrier_type"].map(normalize_carrier_type)
        out["carrier_type"] = (
            out["carrier_type"]
            .astype("string")
            .replace("", pd.NA)
            .fillna(carrier_type)
            .astype(str)
        )
    else:
        out["carrier_type"] = carrier_type

    out["carrier_id"] = ensure_string_id(frame[carrier_id_col])
    out["bioentity_id"] = ensure_string_id(frame[bioentity_id_col])

    if mention_col is not None:
        out["mention_count"] = safe_numeric_series(frame[mention_col], default=1.0)
    else:
        out["mention_count"] = 1.0

    if year_col is not None:
        out["year"] = safe_year_series(frame[year_col])
    else:
        out["year"] = pd.Series([pd.NA] * len(frame), dtype="Int64")

    if source_edge_table is not None:
        out["source_edge_table"] = Path(source_edge_table).name
    else:
        out["source_edge_table"] = f"edges_{carrier_type}_bioentity"

    # Join BioEntity metadata.
    out = out.merge(
        bioentities[
            [
                "bioentity_id",
                "bioentity_name",
                "bioentity_type",
                "bioentity_type_raw",
            ]
        ],
        on="bioentity_id",
        how="left",
    )

    out["bioentity_name"] = out["bioentity_name"].fillna(out["bioentity_id"])
    out["bioentity_type"] = out["bioentity_type"].fillna("unknown")
    out["bioentity_type_raw"] = out["bioentity_type_raw"].fillna("unknown")

    # Fill year from carrier node table if missing.
    if carrier_nodes is not None and "year" in carrier_nodes.columns:
        year_lookup = carrier_nodes[
            ["carrier_type", "carrier_id", "year"]
        ].drop_duplicates()

        out = out.merge(
            year_lookup.rename(columns={"year": "carrier_year"}),
            on=["carrier_type", "carrier_id"],
            how="left",
        )

        out["year"] = out["year"].fillna(out["carrier_year"])
        out = out.drop(columns=["carrier_year"])

    out["year"] = out["year"].astype("Int64")

    # Aggregate duplicate carrier-entity edges.
    group_cols = [
        "carrier_type",
        "carrier_id",
        "bioentity_id",
        "bioentity_name",
        "bioentity_type",
        "bioentity_type_raw",
        "year",
        "source_edge_table",
    ]

    out = (
        out.groupby(group_cols, dropna=False, as_index=False)
        .agg(mention_count=("mention_count", "sum"))
        .reset_index(drop=True)
    )

    return out


def load_standardized_carrier_bioentity_mentions(
    graph_dir: str | Path,
    *,
    carrier_types: Sequence[str] = DEFAULT_CARRIER_TYPES,
) -> pd.DataFrame:
    """Load and standardize all carrier-BioEntity mention edges.

    This function performs two year-joining passes:

    1. Inside ``standardize_mention_edges`` for each carrier type.
    2. After concatenating all carrier mention edges, merge carrier years again
       from node tables as a robust fallback.

    The second pass is important for processed graph schemas where paper years
    and patent years live only in node tables:

        nodes_paper.parquet:  PubYear
        nodes_patent.parquet: patent_year

    Trial years may remain missing if ``nodes_trial.parquet`` is unavailable.
    """

    graph_dir = Path(graph_dir)
    bioentities = load_graph_bioentity_nodes(graph_dir)

    frames: list[pd.DataFrame] = []
    carrier_year_frames: list[pd.DataFrame] = []

    for carrier_type in carrier_types:
        carrier_type = normalize_carrier_type(carrier_type)

        raw_edges = load_mention_edges(graph_dir, carrier_type=carrier_type)

        if raw_edges is None:
            continue

        edge_file = find_existing_file(
            graph_dir,
            DEFAULT_MENTION_EDGE_FILES[carrier_type],
        )

        carrier_nodes = load_carrier_nodes(graph_dir, carrier_type=carrier_type)

        if carrier_nodes is not None and not carrier_nodes.empty:
            carrier_year_frame = carrier_nodes[
                ["carrier_type", "carrier_id", "year"]
            ].copy()
            carrier_year_frame["carrier_type"] = carrier_year_frame[
                "carrier_type"
            ].map(normalize_carrier_type)
            carrier_year_frame["carrier_id"] = ensure_string_id(
                carrier_year_frame["carrier_id"]
            )
            carrier_year_frame["year"] = carrier_year_frame["year"].astype("Int64")
            carrier_year_frames.append(carrier_year_frame)

        standardized = standardize_mention_edges(
            raw_edges,
            carrier_type=carrier_type,
            bioentities=bioentities,
            carrier_nodes=carrier_nodes,
            source_edge_table=edge_file,
        )

        frames.append(standardized)

    if not frames:
        raise FileNotFoundError(
            f"No carrier-BioEntity mention edge tables found under {graph_dir}."
        )

    mentions = pd.concat(frames, ignore_index=True)

    mentions["carrier_type"] = mentions["carrier_type"].map(normalize_carrier_type)
    mentions["carrier_id"] = mentions["carrier_id"].astype("string").fillna("").astype(str)
    mentions["carrier_key"] = (
        mentions["carrier_type"].astype(str) + ":" + mentions["carrier_id"].astype(str)
    )

    if "year" not in mentions.columns:
        mentions["year"] = pd.Series([pd.NA] * len(mentions), dtype="Int64")
    else:
        mentions["year"] = mentions["year"].astype("Int64")

    # Robust fallback year merge after all carrier mention frames are combined.
    if carrier_year_frames:
        carrier_year_lookup = pd.concat(carrier_year_frames, ignore_index=True)
        carrier_year_lookup["carrier_type"] = carrier_year_lookup[
            "carrier_type"
        ].map(normalize_carrier_type)
        carrier_year_lookup["carrier_id"] = ensure_string_id(
            carrier_year_lookup["carrier_id"]
        )
        carrier_year_lookup["year"] = carrier_year_lookup["year"].astype("Int64")

        carrier_year_lookup = (
            carrier_year_lookup.dropna(subset=["year"])
            .drop_duplicates(subset=["carrier_type", "carrier_id"], keep="first")
            .rename(columns={"year": "carrier_year"})
        )

        if not carrier_year_lookup.empty:
            mentions = mentions.merge(
                carrier_year_lookup,
                on=["carrier_type", "carrier_id"],
                how="left",
            )

            mentions["year"] = mentions["year"].fillna(mentions["carrier_year"])
            mentions = mentions.drop(columns=["carrier_year"])

    mentions["year"] = mentions["year"].astype("Int64")

    return mentions


# ---------------------------------------------------------------------------
# Filtering and pair construction
# ---------------------------------------------------------------------------


def filter_mentions_for_knowledge_units(
    mentions: pd.DataFrame,
    *,
    exclude_bioentity_types: Sequence[str] = DEFAULT_EXCLUDED_BIOENTITY_TYPES,
    allowed_entity_types: Sequence[str] | None = None,
    keep_unyearled_edges: bool = True,
) -> pd.DataFrame:
    """Filter standardized mention edges before pair generation."""

    required = {
        "carrier_type",
        "carrier_id",
        "bioentity_id",
        "bioentity_type",
        "mention_count",
        "year",
    }
    missing = required - set(mentions.columns)

    if missing:
        raise ValueError(f"mentions missing required columns: {sorted(missing)}")

    out = mentions.copy()

    out["carrier_type"] = out["carrier_type"].map(normalize_carrier_type)
    out["carrier_id"] = out["carrier_id"].astype("string").fillna("").astype(str)
    out["bioentity_type"] = out["bioentity_type"].map(normalize_entity_type)

    excluded = {normalize_entity_type(value) for value in exclude_bioentity_types}

    if excluded:
        out = out[~out["bioentity_type"].isin(excluded)].copy()

    if allowed_entity_types is not None:
        allowed = {normalize_entity_type(value) for value in allowed_entity_types}
        out = out[out["bioentity_type"].isin(allowed)].copy()

    if not keep_unyearled_edges:
        out = out[out["year"].notna()].copy()

    out["carrier_key"] = (
        out["carrier_type"].astype(str) + ":" + out["carrier_id"].astype(str)
    )

    group_cols = [
        "carrier_type",
        "carrier_id",
        "carrier_key",
        "year",
        "bioentity_id",
        "bioentity_name",
        "bioentity_type",
        "bioentity_type_raw",
        "source_edge_table",
    ]

    out = (
        out.groupby(group_cols, dropna=False, as_index=False)
        .agg(mention_count=("mention_count", "sum"))
        .reset_index(drop=True)
    )

    return out


def parse_allowed_pair_types(pair_types: Sequence[str]) -> set[tuple[str, str]]:
    """Parse allowed pair type strings into tuples."""

    allowed: set[tuple[str, str]] = set()

    for pair_type in pair_types:
        normalized = normalize_pair_type(pair_type)
        left, right = normalized.split("-", maxsplit=1)
        allowed.add((left, right))

    return allowed

def parse_allowed_pair_type_rank(pair_types: Sequence[str]) -> dict[tuple[str, str], int]:
    """Return normalized allowed pair types with their first configured rank.

    The rank preserves user intent when both orientations are configured, e.g.
    disease-phenotype and phenotype-disease. The earlier configured pair type
    wins as the canonical orientation.
    """

    rank: dict[tuple[str, str], int] = {}

    for index, pair_type in enumerate(pair_types):
        normalized = normalize_pair_type(pair_type)
        left, right = normalized.split("-", maxsplit=1)
        key = (left, right)

        if key not in rank:
            rank[key] = index

    return rank


def allowed_entity_types_from_pair_types(pair_types: Sequence[str]) -> set[str]:
    """Return all entity types that appear in allowed pair types."""

    types: set[str] = set()

    for left, right in parse_allowed_pair_types(pair_types):
        types.add(left)
        types.add(right)

    return types


def entity_pair_sort_key(entity: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a stable sort key for same-type unordered entity pairs."""

    return (
        normalize_entity_type(entity.get("bioentity_type", "")),
        str(entity.get("bioentity_id", "")),
        str(entity.get("bioentity_name", "")),
    )


def orient_pair_by_allowed_types(
    entity_a: Mapping[str, Any],
    entity_b: Mapping[str, Any],
    *,
    allowed_pair_types: set[tuple[str, str]],
    allowed_pair_type_rank: Mapping[tuple[str, str], int] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str] | None:
    """Canonicalize and orient an entity pair according to allowed pair types.

    Knowledge Units are treated as unordered typed BioEntity pairs.

    Rules
    -----
    1. If the two entities have the same normalized type, e.g. disease-disease,
       sort them by stable entity key so A-B and B-A collapse to one KU.
    2. If the two entities have different types, orient them according to the
       configured allowed pair types.
    3. If both orientations are configured, use the earlier pair type in the
       user-provided pair_types list as the canonical orientation.
    """

    type_a = normalize_entity_type(entity_a["bioentity_type"])
    type_b = normalize_entity_type(entity_b["bioentity_type"])

    if str(entity_a.get("bioentity_id", "")) == str(entity_b.get("bioentity_id", "")):
        return None

    # Same-type pair, e.g. disease-disease, gene-gene, chemical-chemical.
    # Even if disease-disease is excluded from the main configuration, this
    # keeps the builder correct for exploratory same-type layers.
    if type_a == type_b:
        pair_key = (type_a, type_b)

        if pair_key not in allowed_pair_types:
            return None

        if entity_pair_sort_key(entity_a) <= entity_pair_sort_key(entity_b):
            return entity_a, entity_b, f"{type_a}-{type_b}"

        return entity_b, entity_a, f"{type_a}-{type_b}"

    direct_key = (type_a, type_b)
    reverse_key = (type_b, type_a)

    candidates: list[tuple[int, tuple[str, str]]] = []

    if direct_key in allowed_pair_types:
        candidates.append(
            (
                allowed_pair_type_rank.get(direct_key, 10**9)
                if allowed_pair_type_rank is not None
                else 0,
                direct_key,
            )
        )

    if reverse_key in allowed_pair_types:
        candidates.append(
            (
                allowed_pair_type_rank.get(reverse_key, 10**9)
                if allowed_pair_type_rank is not None
                else 0,
                reverse_key,
            )
        )

    if not candidates:
        return None

    _, chosen_key = min(candidates, key=lambda item: item[0])
    chosen_left, chosen_right = chosen_key
    pair_type = f"{chosen_left}-{chosen_right}"

    if chosen_key == direct_key:
        return entity_a, entity_b, pair_type

    return entity_b, entity_a, pair_type


def compute_pair_weight(
    mention_count_a: float,
    mention_count_b: float,
    *,
    method: str = "sqrt_product",
) -> float:
    """Compute pair evidence weight from two mention counts."""

    a = float(mention_count_a)
    b = float(mention_count_b)
    method = normalize_token(method)

    if method == "sqrt-product":
        return float(math.sqrt(max(a, 0.0) * max(b, 0.0)))

    if method == "min":
        return float(min(a, b))

    if method == "product":
        return float(a * b)

    if method == "sum":
        return float(a + b)

    if method == "mean":
        return float((a + b) / 2.0)

    raise ValueError(
        f"Unsupported pair_weight_method {method!r}. "
        "Supported: sqrt_product, min, product, sum, mean."
    )


def _carrier_mentions_to_pair_rows(
    group: pd.DataFrame,
    *,
    allowed_pair_types: set[tuple[str, str]],
    allowed_pair_type_rank: Mapping[tuple[str, str], int],
    pair_weight_method: str,
    ku_id_prefix: str,
    max_entities_per_carrier: int | None,
) -> list[dict[str, Any]]:

    if group.empty:
        return []

    entity_cols = [
        "bioentity_id",
        "bioentity_name",
        "bioentity_type",
        "bioentity_type_raw",
    ]

    entities = (
        group.groupby(entity_cols, dropna=False, as_index=False)
        .agg(
            mention_count=("mention_count", "sum"),
            source_edge_tables=(
                "source_edge_table",
                lambda values: "|".join(sorted({str(value) for value in values})),
            ),
        )
        .sort_values(["mention_count", "bioentity_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if max_entities_per_carrier is not None and len(entities) > max_entities_per_carrier:
        entities = entities.head(int(max_entities_per_carrier)).copy()

    if len(entities) < 2:
        return []

    first = group.iloc[0]
    carrier_type = normalize_carrier_type(first["carrier_type"])
    carrier_id = first["carrier_id"]
    carrier_key = first["carrier_key"]
    year = first["year"]

    rows: list[dict[str, Any]] = []
    records = entities.to_dict("records")

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            entity_i = records[i]
            entity_j = records[j]

            oriented = orient_pair_by_allowed_types(
                entity_i,
                entity_j,
                allowed_pair_types=allowed_pair_types,
                allowed_pair_type_rank=allowed_pair_type_rank,
            )

            if oriented is None:
                continue

            entity_a, entity_b, pair_type = oriented

            if str(entity_a["bioentity_id"]) == str(entity_b["bioentity_id"]):
                continue

            pair_weight = compute_pair_weight(
                entity_a["mention_count"],
                entity_b["mention_count"],
                method=pair_weight_method,
            )

            ku_id = make_ku_id(
                entity_a_id=entity_a["bioentity_id"],
                entity_b_id=entity_b["bioentity_id"],
                pair_type=pair_type,
                prefix=ku_id_prefix,
            )

            rows.append(
                {
                    "ku_id": ku_id,
                    "entity_a_id": str(entity_a["bioentity_id"]),
                    "entity_a_name": str(entity_a["bioentity_name"]),
                    "entity_a_type": normalize_entity_type(entity_a["bioentity_type"]),
                    "entity_a_type_raw": str(entity_a["bioentity_type_raw"]),
                    "entity_b_id": str(entity_b["bioentity_id"]),
                    "entity_b_name": str(entity_b["bioentity_name"]),
                    "entity_b_type": normalize_entity_type(entity_b["bioentity_type"]),
                    "entity_b_type_raw": str(entity_b["bioentity_type_raw"]),
                    "pair_type": pair_type,
                    "carrier_type": str(carrier_type),
                    "carrier_id": str(carrier_id),
                    "carrier_key": str(carrier_key),
                    "year": year,
                    "mention_count_a": float(entity_a["mention_count"]),
                    "mention_count_b": float(entity_b["mention_count"]),
                    "pair_weight": pair_weight,
                    "source_edge_table_a": str(entity_a["source_edge_tables"]),
                    "source_edge_table_b": str(entity_b["source_edge_tables"]),
                    "source_edge_table": (
                        f"{entity_a['source_edge_tables']}|{entity_b['source_edge_tables']}"
                    ),
                }
            )

    return rows


def generate_carrier_knowledge_unit_edges(
    mentions: pd.DataFrame,
    *,
    pair_types: Sequence[str] = DEFAULT_PAIR_TYPES,
    pair_weight_method: str = "sqrt_product",
    ku_id_prefix: str = "KU",
    max_entities_per_carrier: int | None = 100,
) -> pd.DataFrame:
    """Generate carrier-to-Knowledge Unit edges from filtered mentions."""

    required = {
        "carrier_type",
        "carrier_id",
        "carrier_key",
        "year",
        "bioentity_id",
        "bioentity_name",
        "bioentity_type",
        "bioentity_type_raw",
        "mention_count",
        "source_edge_table",
    }
    missing = required - set(mentions.columns)

    if missing:
        raise ValueError(f"mentions missing required columns: {sorted(missing)}")

    allowed_pair_types = parse_allowed_pair_types(pair_types)
    allowed_pair_type_rank = parse_allowed_pair_type_rank(pair_types)

    rows: list[dict[str, Any]] = []

    group_cols = ["carrier_type", "carrier_id", "carrier_key", "year"]

    for _, group in mentions.groupby(group_cols, dropna=False, sort=False):
        rows.extend(
            _carrier_mentions_to_pair_rows(
                group,
                allowed_pair_types=allowed_pair_types,
                allowed_pair_type_rank=allowed_pair_type_rank,
                pair_weight_method=pair_weight_method,
                ku_id_prefix=ku_id_prefix,
                max_entities_per_carrier=max_entities_per_carrier,
            )
        )

    if not rows:
        return empty_carrier_knowledge_unit_edges()

    edges = pd.DataFrame(rows)

    group_cols = [
        "ku_id",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_a_type_raw",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "entity_b_type_raw",
        "pair_type",
        "carrier_type",
        "carrier_id",
        "carrier_key",
        "year",
    ]

    edges = (
        edges.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            mention_count_a=("mention_count_a", "sum"),
            mention_count_b=("mention_count_b", "sum"),
            pair_weight=("pair_weight", "sum"),
            source_edge_table_a=(
                "source_edge_table_a",
                lambda values: "|".join(sorted({str(value) for value in values})),
            ),
            source_edge_table_b=(
                "source_edge_table_b",
                lambda values: "|".join(sorted({str(value) for value in values})),
            ),
            source_edge_table=(
                "source_edge_table",
                lambda values: "|".join(sorted({str(value) for value in values})),
            ),
        )
        .reset_index(drop=True)
    )

    edges["year"] = edges["year"].astype("Int64")

    return edges


def empty_carrier_knowledge_unit_edges() -> pd.DataFrame:
    """Return an empty carrier-KU edge table with canonical columns."""

    columns = [
        "ku_id",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_a_type_raw",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "entity_b_type_raw",
        "pair_type",
        "carrier_type",
        "carrier_id",
        "carrier_key",
        "year",
        "mention_count_a",
        "mention_count_b",
        "pair_weight",
        "source_edge_table_a",
        "source_edge_table_b",
        "source_edge_table",
    ]

    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Knowledge Unit aggregation and filtering
# ---------------------------------------------------------------------------


def aggregate_knowledge_units(edges: pd.DataFrame) -> pd.DataFrame:
    """Aggregate carrier-KU edges into a Knowledge Unit master table."""

    if edges.empty:
        return empty_knowledge_units()

    required = {
        "ku_id",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "pair_type",
        "carrier_type",
        "carrier_key",
        "year",
        "pair_weight",
    }
    missing = required - set(edges.columns)

    if missing:
        raise ValueError(f"edges missing required columns: {sorted(missing)}")

    base_cols = [
        "ku_id",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_a_type_raw",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "entity_b_type_raw",
        "pair_type",
    ]

    base = edges[base_cols].drop_duplicates(subset=["ku_id"], keep="first")

    counts = (
        edges.groupby("ku_id", as_index=False)
        .agg(
            carrier_count=("carrier_key", "nunique"),
            total_pair_weight=("pair_weight", "sum"),
            mean_pair_weight=("pair_weight", "mean"),
            max_pair_weight=("pair_weight", "max"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .reset_index(drop=True)
    )

    carrier_type_counts = (
        edges.groupby(["ku_id", "carrier_type"])["carrier_key"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    for carrier_type in DEFAULT_CARRIER_TYPES:
        if carrier_type not in carrier_type_counts.columns:
            carrier_type_counts[carrier_type] = 0

    carrier_type_counts = carrier_type_counts.rename(
        columns={
            "paper": "paper_count",
            "patent": "patent_count",
            "trial": "trial_count",
        }
    )

    units = base.merge(counts, on="ku_id", how="left").merge(
        carrier_type_counts[
            ["ku_id", "paper_count", "patent_count", "trial_count"]
        ],
        on="ku_id",
        how="left",
    )

    for column in ["paper_count", "patent_count", "trial_count"]:
        units[column] = units[column].fillna(0).astype(int)

    units["carrier_count"] = units["carrier_count"].fillna(0).astype(int)
    units["total_pair_weight"] = units["total_pair_weight"].fillna(0.0)
    units["mean_pair_weight"] = units["mean_pair_weight"].fillna(0.0)
    units["max_pair_weight"] = units["max_pair_weight"].fillna(0.0)
    units["first_year"] = units["first_year"].astype("Int64")
    units["last_year"] = units["last_year"].astype("Int64")

    units["has_paper_evidence"] = units["paper_count"] > 0
    units["has_patent_evidence"] = units["patent_count"] > 0
    units["has_trial_evidence"] = units["trial_count"] > 0

    units = units.sort_values(
        ["carrier_count", "total_pair_weight", "ku_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return units


def empty_knowledge_units() -> pd.DataFrame:
    """Return an empty Knowledge Unit master table with canonical columns."""

    columns = [
        "ku_id",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_a_type_raw",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "entity_b_type_raw",
        "pair_type",
        "carrier_count",
        "total_pair_weight",
        "mean_pair_weight",
        "max_pair_weight",
        "first_year",
        "last_year",
        "paper_count",
        "patent_count",
        "trial_count",
        "has_paper_evidence",
        "has_patent_evidence",
        "has_trial_evidence",
    ]

    return pd.DataFrame(columns=columns)


def filter_knowledge_units_by_support(
    knowledge_units: pd.DataFrame,
    *,
    min_carrier_count: int = 3,
    min_paper_count: int = 0,
    min_patent_count: int = 0,
    min_trial_count: int = 0,
) -> pd.DataFrame:
    """Filter KU master table by support thresholds."""

    if knowledge_units.empty:
        return knowledge_units.copy()

    mask = pd.Series(True, index=knowledge_units.index)

    if min_carrier_count > 0:
        mask &= knowledge_units["carrier_count"] >= int(min_carrier_count)

    if min_paper_count > 0:
        mask &= knowledge_units["paper_count"] >= int(min_paper_count)

    if min_patent_count > 0:
        mask &= knowledge_units["patent_count"] >= int(min_patent_count)

    if min_trial_count > 0:
        mask &= knowledge_units["trial_count"] >= int(min_trial_count)

    return knowledge_units[mask].copy().reset_index(drop=True)


def apply_knowledge_unit_support_filter(
    edges: pd.DataFrame,
    *,
    min_carrier_count: int = 3,
    min_paper_count: int = 0,
    min_patent_count: int = 0,
    min_trial_count: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter KUs and corresponding carrier-KU edges by support thresholds."""

    initial_units = aggregate_knowledge_units(edges)

    kept_units = filter_knowledge_units_by_support(
        initial_units,
        min_carrier_count=min_carrier_count,
        min_paper_count=min_paper_count,
        min_patent_count=min_patent_count,
        min_trial_count=min_trial_count,
    )

    kept_ids = set(kept_units["ku_id"].astype(str))

    if not kept_ids:
        return empty_knowledge_units(), empty_carrier_knowledge_unit_edges()

    filtered_edges = edges[edges["ku_id"].astype(str).isin(kept_ids)].copy()
    filtered_edges = filtered_edges.reset_index(drop=True)

    filtered_units = aggregate_knowledge_units(filtered_edges)

    return filtered_units, filtered_edges


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_knowledge_unit_types(knowledge_units: pd.DataFrame) -> pd.DataFrame:
    """Summarize Knowledge Units by pair type."""

    if knowledge_units.empty:
        return pd.DataFrame(
            columns=[
                "pair_type",
                "knowledge_unit_count",
                "carrier_count_sum",
                "carrier_count_mean",
                "paper_count_sum",
                "patent_count_sum",
                "trial_count_sum",
                "total_pair_weight_sum",
            ]
        )

    summary = (
        knowledge_units.groupby("pair_type", as_index=False)
        .agg(
            knowledge_unit_count=("ku_id", "nunique"),
            carrier_count_sum=("carrier_count", "sum"),
            carrier_count_mean=("carrier_count", "mean"),
            paper_count_sum=("paper_count", "sum"),
            patent_count_sum=("patent_count", "sum"),
            trial_count_sum=("trial_count", "sum"),
            total_pair_weight_sum=("total_pair_weight", "sum"),
        )
        .sort_values(
            ["knowledge_unit_count", "carrier_count_sum"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    return summary


def summarize_top_knowledge_units(
    knowledge_units: pd.DataFrame,
    *,
    top_n: int = 25,
) -> list[dict[str, Any]]:
    """Return top Knowledge Units as JSON-safe rows."""

    if knowledge_units.empty:
        return []

    columns = [
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
    ]

    existing = [column for column in columns if column in knowledge_units.columns]

    rows = (
        knowledge_units.sort_values(
            ["carrier_count", "total_pair_weight", "ku_id"],
            ascending=[False, False, True],
        )
        .head(top_n)[existing]
        .to_dict("records")
    )

    return json_safe(rows)


def summarize_knowledge_unit_build(
    *,
    raw_mentions: pd.DataFrame,
    filtered_mentions: pd.DataFrame,
    raw_edges: pd.DataFrame,
    filtered_edges: pd.DataFrame,
    knowledge_units: pd.DataFrame,
    type_summary: pd.DataFrame,
    config: KnowledgeUnitBuildConfig,
) -> dict[str, Any]:
    """Build a JSON summary for KU construction."""

    summary: dict[str, Any] = {
        "raw_mention_edge_count": int(len(raw_mentions)),
        "filtered_mention_edge_count": int(len(filtered_mentions)),
        "raw_carrier_ku_edge_count": int(len(raw_edges)),
        "filtered_carrier_ku_edge_count": int(len(filtered_edges)),
        "knowledge_unit_count": int(len(knowledge_units)),
        "pair_type_count": int(knowledge_units["pair_type"].nunique())
        if not knowledge_units.empty
        else 0,
        "carrier_count": int(filtered_edges["carrier_key"].nunique())
        if not filtered_edges.empty
        else 0,
        "paper_carrier_count": int(
            filtered_edges.loc[filtered_edges["carrier_type"] == "paper", "carrier_key"].nunique()
        )
        if not filtered_edges.empty
        else 0,
        "patent_carrier_count": int(
            filtered_edges.loc[filtered_edges["carrier_type"] == "patent", "carrier_key"].nunique()
        )
        if not filtered_edges.empty
        else 0,
        "trial_carrier_count": int(
            filtered_edges.loc[filtered_edges["carrier_type"] == "trial", "carrier_key"].nunique()
        )
        if not filtered_edges.empty
        else 0,
        "min_carrier_count": int(config.min_carrier_count),
        "min_paper_count": int(config.min_paper_count),
        "min_patent_count": int(config.min_patent_count),
        "min_trial_count": int(config.min_trial_count),
        "pair_weight_method": config.pair_weight_method,
        "max_entities_per_carrier": config.max_entities_per_carrier,
        "excluded_bioentity_types": list(config.exclude_bioentity_types),
        "allowed_pair_types": list(config.pair_types),
        "memory_mb": {
            "raw_mentions": dataframe_memory_mb(raw_mentions),
            "filtered_mentions": dataframe_memory_mb(filtered_mentions),
            "raw_carrier_ku_edges": dataframe_memory_mb(raw_edges),
            "filtered_carrier_ku_edges": dataframe_memory_mb(filtered_edges),
            "knowledge_units": dataframe_memory_mb(knowledge_units),
        },
        "top_knowledge_units": summarize_top_knowledge_units(knowledge_units, top_n=25),
        "type_summary": json_safe(type_summary.to_dict("records")),
    }

    if not knowledge_units.empty:
        summary.update(
            {
                "first_year_min": json_safe(knowledge_units["first_year"].min()),
                "last_year_max": json_safe(knowledge_units["last_year"].max()),
                "carrier_count_min": int(knowledge_units["carrier_count"].min()),
                "carrier_count_median": float(knowledge_units["carrier_count"].median()),
                "carrier_count_max": int(knowledge_units["carrier_count"].max()),
                "total_pair_weight_sum": float(
                    knowledge_units["total_pair_weight"].sum()
                ),
            }
        )

    return json_safe(summary)


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------


def build_knowledge_units_from_mentions(
    mentions: pd.DataFrame,
    *,
    config: KnowledgeUnitBuildConfig | None = None,
) -> KnowledgeUnitBuildResult:
    """Build entity-pair Knowledge Units from standardized mention edges."""

    config = config or KnowledgeUnitBuildConfig()

    allowed_entity_types = allowed_entity_types_from_pair_types(config.pair_types)

    filtered_mentions = filter_mentions_for_knowledge_units(
        mentions,
        exclude_bioentity_types=config.exclude_bioentity_types,
        allowed_entity_types=allowed_entity_types,
        keep_unyearled_edges=config.keep_unyearled_edges,
    )

    raw_edges = generate_carrier_knowledge_unit_edges(
        filtered_mentions,
        pair_types=config.pair_types,
        pair_weight_method=config.pair_weight_method,
        ku_id_prefix=config.ku_id_prefix,
        max_entities_per_carrier=config.max_entities_per_carrier,
    )

    knowledge_units, filtered_edges = apply_knowledge_unit_support_filter(
        raw_edges,
        min_carrier_count=config.min_carrier_count,
        min_paper_count=config.min_paper_count,
        min_patent_count=config.min_patent_count,
        min_trial_count=config.min_trial_count,
    )

    type_summary = summarize_knowledge_unit_types(knowledge_units)

    summary = summarize_knowledge_unit_build(
        raw_mentions=mentions,
        filtered_mentions=filtered_mentions,
        raw_edges=raw_edges,
        filtered_edges=filtered_edges,
        knowledge_units=knowledge_units,
        type_summary=type_summary,
        config=config,
    )

    return KnowledgeUnitBuildResult(
        knowledge_units=knowledge_units,
        carrier_knowledge_unit_edges=filtered_edges,
        type_summary=type_summary,
        summary=summary,
        run_config=config.to_json_dict(),
    )


def build_knowledge_units_from_graph_dir(
    graph_dir: str | Path,
    *,
    config: KnowledgeUnitBuildConfig | None = None,
) -> KnowledgeUnitBuildResult:
    """Load carrier-BioEntity mentions from graph_dir and build KUs."""

    graph_dir = Path(graph_dir)
    config = config or KnowledgeUnitBuildConfig(graph_dir=str(graph_dir))

    mentions = load_standardized_carrier_bioentity_mentions(
        graph_dir,
        carrier_types=config.carrier_types,
    )

    return build_knowledge_units_from_mentions(
        mentions,
        config=config,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Convert common pandas / numpy values into JSON-safe objects."""

    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is pd.NA:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    return value


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write JSON with UTF-8 encoding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, ensure_ascii=False, indent=2)


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame by file extension."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".json", ".jsonl"}:
        frame.to_json(path, orient="records", lines=suffix == ".jsonl", force_ascii=False)
    else:
        raise ValueError(f"Unsupported DataFrame output extension: {path.suffix}")


def write_knowledge_unit_outputs(
    result: KnowledgeUnitBuildResult,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Write standard KU output artifacts and return manifest rows."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []

    knowledge_units_path = output_dir / "knowledge_units.parquet"
    write_dataframe(result.knowledge_units, knowledge_units_path)
    artifacts.append(
        artifact_row(
            "knowledge_units",
            knowledge_units_path,
            "Knowledge Unit master table.",
            row_count=len(result.knowledge_units),
        )
    )

    carrier_edges_path = output_dir / "carrier_knowledge_unit_edges.parquet"
    write_dataframe(result.carrier_knowledge_unit_edges, carrier_edges_path)
    artifacts.append(
        artifact_row(
            "carrier_knowledge_unit_edges",
            carrier_edges_path,
            "Carrier-to-Knowledge Unit co-mention evidence edges.",
            row_count=len(result.carrier_knowledge_unit_edges),
        )
    )

    type_summary_path = output_dir / "knowledge_unit_type_summary.csv"
    write_dataframe(result.type_summary, type_summary_path)
    artifacts.append(
        artifact_row(
            "knowledge_unit_type_summary",
            type_summary_path,
            "Knowledge Unit counts and evidence summary by pair type.",
            row_count=len(result.type_summary),
        )
    )

    summary_path = output_dir / "knowledge_unit_summary.json"
    write_json(summary_path, result.summary)
    artifacts.append(
        artifact_row(
            "knowledge_unit_summary",
            summary_path,
            "JSON summary for Knowledge Unit construction.",
        )
    )

    run_config_path = output_dir / "run_config.json"
    write_json(run_config_path, result.run_config)
    artifacts.append(
        artifact_row(
            "run_config",
            run_config_path,
            "Run configuration for Knowledge Unit construction.",
        )
    )

    manifest = pd.DataFrame(artifacts)
    manifest_path = output_dir / "dataset_manifest.csv"
    write_dataframe(manifest, manifest_path)
    artifacts.append(
        artifact_row(
            "dataset_manifest",
            manifest_path,
            "Manifest of Knowledge Unit dataset artifacts.",
            row_count=len(manifest),
        )
    )

    write_dataframe(pd.DataFrame(artifacts), manifest_path)

    return artifacts


def artifact_row(
    artifact: str,
    path: str | Path,
    description: str,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Create an artifact manifest row."""

    path = Path(path)

    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }


# ---------------------------------------------------------------------------
# Lightweight report helpers
# ---------------------------------------------------------------------------


def build_knowledge_unit_report(
    *,
    result: KnowledgeUnitBuildResult,
    output_dir: str | Path,
    title: str = "Knowledge Unit Construction Report",
) -> str:
    """Build a compact Markdown report for KU construction."""

    output_dir = Path(output_dir)
    summary = result.summary

    top_rows = summary.get("top_knowledge_units", [])
    type_rows = result.type_summary.head(30).to_dict("records")

    return "\n".join(
        [
            f"# {title}",
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
            "",
            "## 2. Configuration",
            "",
            "```json",
            json.dumps(json_safe(result.run_config), ensure_ascii=False, indent=2),
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
        ]
    )


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Render a simple Markdown table."""

    if not rows:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        return "\n".join([header, sep])

    def fmt(value: Any) -> str:
        value = json_safe(value)

        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.6f}"

        text = str(value)
        text = text.replace("\n", " ").replace("|", "\\|")
        return text

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"

    body = []

    for row in rows:
        body.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")

    return "\n".join([header, sep, *body])


__all__ = [
    "KnowledgeUnitBuildConfig",
    "KnowledgeUnitBuildResult",
    "DEFAULT_PAIR_TYPES",
    "DEFAULT_EXCLUDED_BIOENTITY_TYPES",
    "normalize_carrier_type",
    "normalize_entity_type",
    "load_graph_bioentity_nodes",
    "load_carrier_nodes",
    "load_mention_edges",
    "load_standardized_carrier_bioentity_mentions",
    "standardize_bioentity_nodes",
    "standardize_carrier_nodes",
    "standardize_mention_edges",
    "filter_mentions_for_knowledge_units",
    "generate_carrier_knowledge_unit_edges",
    "aggregate_knowledge_units",
    "filter_knowledge_units_by_support",
    "apply_knowledge_unit_support_filter",
    "summarize_knowledge_unit_types",
    "summarize_knowledge_unit_build",
    "build_knowledge_units_from_mentions",
    "build_knowledge_units_from_graph_dir",
    "write_knowledge_unit_outputs",
    "build_knowledge_unit_report",
    "write_json",
    "write_dataframe",
    "json_safe",
    "markdown_table",
    "allowed_entity_types_from_pair_types",
    "parse_allowed_pair_types",
    "parse_allowed_pair_type_rank",
    "orient_pair_by_allowed_types",
    "entity_pair_sort_key",
]