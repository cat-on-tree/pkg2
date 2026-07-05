"""Build KU-level temporal modeling datasets from Stage 04 outputs.

This module implements Stage 05 of the PKG2 project.

Stage 04 produces the Knowledge Unit evidence layer:

    Carrier Graph
        -> Knowledge Units
        -> carrier-KU evidence edges
        -> KU-year temporal panel

Stage 05 prepares an algorithm-ready modeling dataset for later temporal
prediction models and ablation studies. It does not train an encoder and does
not define hand-crafted Knowledge State scores.

The output dataset provides:

- one-row-per-KU static covariates;
- one-row-per-KU-year temporal sequences;
- carrier-level KU evidence observations;
- metadata and reports for downstream modeling.

Later algorithms are responsible for defining their own encoders, prediction
heads, graph diffusion modules, reaction/source modules, and ablations.

Expected inputs
---------------
Knowledge Unit directory:

    knowledge_units.parquet
    carrier_knowledge_unit_edges.parquet

Temporal panel directory:

    knowledge_unit_temporal_features.parquet
    knowledge_unit_year_panel.parquet

Main outputs
------------
Output modeling dataset directory:

    knowledge_unit_static_features.parquet
    knowledge_unit_temporal_sequences.parquet
    knowledge_unit_carrier_observations.parquet
    knowledge_unit_modeling_summary.json
    knowledge_unit_modeling_manifest.csv
    knowledge_unit_modeling_report.md
    future_label_plan.json

Design principles
-----------------
1. Do not overwrite Stage 04 evidence outputs.
2. Do not train models or encoders in this stage.
3. Do not define the final Knowledge State with hand-crafted scores.
4. Do not generate final future labels from the current demo graph.
5. Preserve structured observations for downstream temporal modeling and
   leakage-controlled prediction datasets.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


KU_FILE_NAME = "knowledge_units.parquet"
CARRIER_KU_EDGE_FILE_NAME = "carrier_knowledge_unit_edges.parquet"
TEMPORAL_FEATURE_FILE_NAME = "knowledge_unit_temporal_features.parquet"
YEAR_PANEL_FILE_NAME = "knowledge_unit_year_panel.parquet"

STATIC_OUTPUT_FILE_NAME = "knowledge_unit_static_features.parquet"
SEQUENCE_OUTPUT_FILE_NAME = "knowledge_unit_temporal_sequences.parquet"
OBSERVATION_OUTPUT_FILE_NAME = "knowledge_unit_carrier_observations.parquet"
SUMMARY_OUTPUT_FILE_NAME = "knowledge_unit_modeling_summary.json"
MANIFEST_OUTPUT_FILE_NAME = "knowledge_unit_modeling_manifest.csv"
REPORT_OUTPUT_FILE_NAME = "knowledge_unit_modeling_report.md"
FUTURE_LABEL_PLAN_FILE_NAME = "future_label_plan.json"


@dataclass
class KnowledgeUnitModelingDatasetConfig:
    """Configuration for building a KU temporal modeling dataset."""

    # Optional year clipping. If None, use all years in the temporal panel.
    start_year: int | None = None
    end_year: int | None = None

    # If true, keep all rows from the dense / sparse panel. If false, keep only
    # rows with at least one annual observation.
    keep_zero_year_rows: bool = True

    # If true, add lightweight entity hub statistics based on KU degree.
    # These are structural covariates, not final state scores.
    add_entity_hub_features: bool = True

    # If true, add observed modality flags such as has_paper_evidence,
    # has_patent_evidence, and has_trial_evidence.
    add_modality_flags: bool = True

    # If true, add first-observed-year lag covariates such as paper_to_patent_lag.
    add_lag_features: bool = True

    # If true, ensure commonly expected sequence columns exist, filled with 0.
    ensure_default_sequence_columns: bool = True

    # If true, write future_label_plan.json as scaffold documentation.
    write_future_label_plan: bool = True

    # Preferred annual feature columns for temporal modeling sequences.
    preferred_sequence_columns: list[str] = field(
        default_factory=lambda: [
            "paper_count_year",
            "patent_count_year",
            "trial_count_year",
            "carrier_count_year",
            "pair_weight_year",
            "paper_count_cum",
            "patent_count_cum",
            "trial_count_cum",
            "carrier_count_cum",
            "pair_weight_cum",
            "has_paper_year",
            "has_patent_year",
            "has_trial_year",
            "has_translation_year",
            "has_paper_cum",
            "has_patent_cum",
            "has_trial_cum",
            "has_translation_cum",
        ]
    )


@dataclass
class KnowledgeUnitModelingDatasetResult:
    """Result object returned by ``build_knowledge_unit_modeling_dataset``."""

    output_dir: Path
    static_features: pd.DataFrame
    temporal_sequences: pd.DataFrame
    carrier_observations: pd.DataFrame
    summary: dict[str, Any]
    manifest: pd.DataFrame


def prepare_output_dir(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    allow_existing_empty: bool = True,
) -> Path:
    """Prepare an output directory."""

    output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            existing = list(output_dir.iterdir())
            if existing and not allow_existing_empty:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Use overwrite=True to replace it."
                )
            if existing:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Use overwrite=True to replace it."
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def json_safe(value: Any) -> Any:
    """Convert common NumPy / pandas values into JSON-safe values."""

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
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


def write_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(dict(data)), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame based on output suffix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".json":
        frame.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported DataFrame output suffix: {path}")

    return path


def read_required_parquet(path: str | Path) -> pd.DataFrame:
    """Read a required Parquet file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    return pd.read_parquet(path)


def require_columns(frame: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    """Validate required columns exist."""

    missing = [column for column in columns if column not in frame.columns]

    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def coerce_year(series: pd.Series) -> pd.Series:
    """Convert a series to nullable integer years."""

    year = pd.to_numeric(series, errors="coerce").round()
    year = year.where((year >= 0) & (year <= 3000))
    return year.astype("Int64")


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Coerce a series to numeric and fill missing values."""

    return pd.to_numeric(series, errors="coerce").fillna(default)


def normalize_bool_numeric(series: pd.Series) -> pd.Series:
    """Convert bool-like values to 0/1 integers."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")

    lowered = series.astype("string").str.lower().str.strip()
    mapped = lowered.map(
        {
            "true": 1,
            "1": 1,
            "yes": 1,
            "y": 1,
            "false": 0,
            "0": 0,
            "no": 0,
            "n": 0,
        }
    )

    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.fillna(numeric).fillna(0)
    return (out > 0).astype("int8")


def load_stage04_inputs(
    knowledge_unit_dir: str | Path,
    temporal_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Stage 04 KU and temporal panel inputs."""

    knowledge_unit_dir = Path(knowledge_unit_dir)
    temporal_dir = Path(temporal_dir)

    ku = read_required_parquet(knowledge_unit_dir / KU_FILE_NAME)
    edges = read_required_parquet(knowledge_unit_dir / CARRIER_KU_EDGE_FILE_NAME)
    temporal_features = read_required_parquet(temporal_dir / TEMPORAL_FEATURE_FILE_NAME)
    year_panel = read_required_parquet(temporal_dir / YEAR_PANEL_FILE_NAME)

    require_columns(
        ku,
        [
            "ku_id",
            "entity_a_id",
            "entity_a_name",
            "entity_a_type",
            "entity_b_id",
            "entity_b_name",
            "entity_b_type",
            "pair_type",
        ],
        KU_FILE_NAME,
    )

    require_columns(
        edges,
        [
            "ku_id",
            "carrier_id",
            "carrier_key",
            "carrier_type",
            "year",
            "pair_weight",
        ],
        CARRIER_KU_EDGE_FILE_NAME,
    )

    require_columns(
        temporal_features,
        ["ku_id"],
        TEMPORAL_FEATURE_FILE_NAME,
    )

    require_columns(
        year_panel,
        ["ku_id", "year"],
        YEAR_PANEL_FILE_NAME,
    )

    ku = ku.copy()
    edges = edges.copy()
    temporal_features = temporal_features.copy()
    year_panel = year_panel.copy()

    ku["ku_id"] = ku["ku_id"].astype(str)
    edges["ku_id"] = edges["ku_id"].astype(str)
    temporal_features["ku_id"] = temporal_features["ku_id"].astype(str)
    year_panel["ku_id"] = year_panel["ku_id"].astype(str)

    edges["year"] = coerce_year(edges["year"])
    year_panel["year"] = coerce_year(year_panel["year"])

    return ku, edges, temporal_features, year_panel


def compute_entity_hub_features(ku: pd.DataFrame) -> pd.DataFrame:
    """Compute entity-level KU degree covariates.

    These features are structural model inputs / diagnostics, not final
    Knowledge State scores.
    """

    left = ku[
        [
            "ku_id",
            "entity_a_id",
            "entity_a_name",
            "entity_a_type",
        ]
    ].rename(
        columns={
            "entity_a_id": "entity_id",
            "entity_a_name": "entity_name",
            "entity_a_type": "entity_type",
        }
    )
    left["entity_side"] = "a"

    right = ku[
        [
            "ku_id",
            "entity_b_id",
            "entity_b_name",
            "entity_b_type",
        ]
    ].rename(
        columns={
            "entity_b_id": "entity_id",
            "entity_b_name": "entity_name",
            "entity_b_type": "entity_type",
        }
    )
    right["entity_side"] = "b"

    long = pd.concat([left, right], ignore_index=True)
    long["entity_id"] = long["entity_id"].astype(str)

    entity_degree = (
        long.groupby("entity_id", as_index=False)
        .agg(
            entity_ku_degree=("ku_id", "nunique"),
            entity_name=("entity_name", "first"),
            entity_type=("entity_type", "first"),
        )
    )

    entity_degree["entity_ku_degree_percentile"] = entity_degree[
        "entity_ku_degree"
    ].rank(method="average", pct=True)

    a = entity_degree.rename(
        columns={
            "entity_id": "entity_a_id",
            "entity_ku_degree": "entity_a_ku_degree",
            "entity_ku_degree_percentile": "entity_a_ku_degree_percentile",
        }
    )[["entity_a_id", "entity_a_ku_degree", "entity_a_ku_degree_percentile"]]

    b = entity_degree.rename(
        columns={
            "entity_id": "entity_b_id",
            "entity_ku_degree": "entity_b_ku_degree",
            "entity_ku_degree_percentile": "entity_b_ku_degree_percentile",
        }
    )[["entity_b_id", "entity_b_ku_degree", "entity_b_ku_degree_percentile"]]

    out = ku[["ku_id", "entity_a_id", "entity_b_id"]].copy()
    out["entity_a_id"] = out["entity_a_id"].astype(str)
    out["entity_b_id"] = out["entity_b_id"].astype(str)

    out = out.merge(a, on="entity_a_id", how="left").merge(b, on="entity_b_id", how="left")

    for column in [
        "entity_a_ku_degree",
        "entity_b_ku_degree",
        "entity_a_ku_degree_percentile",
        "entity_b_ku_degree_percentile",
    ]:
        out[column] = safe_numeric(out[column], 0)

    out["entity_hub_degree_sum"] = out["entity_a_ku_degree"] + out["entity_b_ku_degree"]
    out["entity_hub_degree_max"] = out[["entity_a_ku_degree", "entity_b_ku_degree"]].max(axis=1)
    out["entity_hub_percentile_max"] = out[
        ["entity_a_ku_degree_percentile", "entity_b_ku_degree_percentile"]
    ].max(axis=1)
    out["entity_hub_log_sum"] = np.log1p(out["entity_a_ku_degree"]) + np.log1p(
        out["entity_b_ku_degree"]
    )

    return out.drop(columns=["entity_a_id", "entity_b_id"])


def compute_edge_derived_static_features(edges: pd.DataFrame) -> pd.DataFrame:
    """Compute static evidence covariates directly from carrier-KU edges."""

    work = edges.copy()
    work["year"] = coerce_year(work["year"])

    if "pair_weight" not in work.columns:
        work["pair_weight"] = 0.0

    work["pair_weight"] = safe_numeric(work["pair_weight"], 0.0)

    if "carrier_key" not in work.columns:
        work["carrier_key"] = work["carrier_type"].astype(str) + "::" + work["carrier_id"].astype(str)

    base = (
        work.groupby("ku_id", as_index=False)
        .agg(
            edge_row_count=("ku_id", "size"),
            distinct_carrier_count_from_edges=("carrier_key", "nunique"),
            dated_edge_count=("year", lambda s: int(s.notna().sum())),
            undated_edge_count=("year", lambda s: int(s.isna().sum())),
            total_pair_weight_from_edges=("pair_weight", "sum"),
            mean_pair_weight_from_edges=("pair_weight", "mean"),
            max_pair_weight_from_edges=("pair_weight", "max"),
            first_observed_year_from_edges=("year", "min"),
            last_observed_year_from_edges=("year", "max"),
        )
    )

    modality = (
        work.groupby(["ku_id", "carrier_type"])["carrier_key"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    for carrier_type in ["paper", "patent", "trial"]:
        if carrier_type not in modality.columns:
            modality[carrier_type] = 0

    modality = modality.rename(
        columns={
            "paper": "paper_carrier_count_from_edges",
            "patent": "patent_carrier_count_from_edges",
            "trial": "trial_carrier_count_from_edges",
        }
    )

    out = base.merge(
        modality[
            [
                "ku_id",
                "paper_carrier_count_from_edges",
                "patent_carrier_count_from_edges",
                "trial_carrier_count_from_edges",
            ]
        ],
        on="ku_id",
        how="left",
    )

    out["dated_edge_fraction"] = out["dated_edge_count"] / out["edge_row_count"].replace(0, np.nan)
    out["dated_edge_fraction"] = out["dated_edge_fraction"].fillna(0.0)

    out["active_year_span_from_edges"] = (
        safe_numeric(out["last_observed_year_from_edges"], 0)
        - safe_numeric(out["first_observed_year_from_edges"], 0)
    ).clip(lower=0)

    return out


def add_modality_and_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add observed modality flags and first-year lag covariates."""

    out = frame.copy()

    for column in ["paper_count", "patent_count", "trial_count", "carrier_count"]:
        if column in out.columns:
            out[column] = safe_numeric(out[column], 0)

    if "paper_count" in out.columns:
        out["has_paper_evidence"] = (out["paper_count"] > 0).astype("int8")
    if "patent_count" in out.columns:
        out["has_patent_evidence"] = (out["patent_count"] > 0).astype("int8")
    if "trial_count" in out.columns:
        out["has_trial_evidence"] = (out["trial_count"] > 0).astype("int8")

    if {"patent_count", "trial_count"}.issubset(out.columns):
        out["has_translation_evidence"] = (
            (out["patent_count"] > 0) | (out["trial_count"] > 0)
        ).astype("int8")

    if {"paper_count", "patent_count", "trial_count"}.issubset(out.columns):
        modality_count = (
            (out["paper_count"] > 0).astype(int)
            + (out["patent_count"] > 0).astype(int)
            + (out["trial_count"] > 0).astype(int)
        )
        out["modality_count"] = modality_count.astype("int8")
        out["is_cross_modal"] = (modality_count >= 2).astype("int8")
        out["is_paper_only"] = (
            (out["paper_count"] > 0)
            & (out["patent_count"] == 0)
            & (out["trial_count"] == 0)
        ).astype("int8")

    first_year_cols = [
        "first_year",
        "first_paper_year",
        "first_patent_year",
        "first_trial_year",
        "first_translation_year",
        "last_year",
        "last_observed_year",
    ]
    for column in first_year_cols:
        if column in out.columns:
            out[column] = coerce_year(out[column])

    if {"first_patent_year", "first_trial_year"}.issubset(out.columns):
        first_translation = pd.concat(
            [
                out["first_patent_year"].astype("Float64"),
                out["first_trial_year"].astype("Float64"),
            ],
            axis=1,
        ).min(axis=1, skipna=True)
        out["first_translation_year_derived"] = (
            first_translation.astype("Float64").round().astype("Int64")
        )

    if {"first_paper_year", "first_patent_year"}.issubset(out.columns):
        out["paper_to_patent_lag"] = (
            out["first_patent_year"].astype("Float64")
            - out["first_paper_year"].astype("Float64")
        )

    if {"first_paper_year", "first_trial_year"}.issubset(out.columns):
        out["paper_to_trial_lag"] = (
            out["first_trial_year"].astype("Float64")
            - out["first_paper_year"].astype("Float64")
        )

    translation_year_col = None
    if "first_translation_year" in out.columns:
        translation_year_col = "first_translation_year"
    elif "first_translation_year_derived" in out.columns:
        translation_year_col = "first_translation_year_derived"

    if translation_year_col and "first_paper_year" in out.columns:
        out["paper_to_translation_lag"] = (
            out[translation_year_col].astype("Float64")
            - out["first_paper_year"].astype("Float64")
        )

    if {"first_year", "last_year"}.issubset(out.columns):
        out["observed_year_span"] = (
            out["last_year"].astype("Float64") - out["first_year"].astype("Float64")
        ).clip(lower=0)

    return out


def build_static_features(
    ku: pd.DataFrame,
    edges: pd.DataFrame,
    temporal_features: pd.DataFrame,
    *,
    config: KnowledgeUnitModelingDatasetConfig,
) -> pd.DataFrame:
    """Build one-row-per-KU static covariate table."""

    static = ku.copy()

    temporal_extra_columns = [
        column
        for column in temporal_features.columns
        if column != "ku_id" and column not in static.columns
    ]

    if temporal_extra_columns:
        static = static.merge(
            temporal_features[["ku_id", *temporal_extra_columns]],
            on="ku_id",
            how="left",
        )

    edge_features = compute_edge_derived_static_features(edges)
    edge_extra_columns = [
        column for column in edge_features.columns if column != "ku_id" and column not in static.columns
    ]

    if edge_extra_columns:
        static = static.merge(
            edge_features[["ku_id", *edge_extra_columns]],
            on="ku_id",
            how="left",
        )

    if config.add_entity_hub_features:
        hub = compute_entity_hub_features(ku)
        static = static.merge(hub, on="ku_id", how="left")

    if config.add_modality_flags or config.add_lag_features:
        static = add_modality_and_lag_features(static)

    for column in static.columns:
        if column == "ku_id":
            continue
        if pd.api.types.is_numeric_dtype(static[column]):
            static[column] = static[column].fillna(0)

    return static


def ensure_sequence_columns(
    panel: pd.DataFrame,
    preferred_columns: Sequence[str],
) -> pd.DataFrame:
    """Ensure preferred temporal sequence columns exist when they can be derived."""

    out = panel.copy()

    for column in [
        "paper_count_year",
        "patent_count_year",
        "trial_count_year",
        "carrier_count_year",
        "pair_weight_year",
    ]:
        if column not in out.columns:
            out[column] = 0.0

    if "carrier_count_year" in out.columns:
        if safe_numeric(out["carrier_count_year"], 0).sum() == 0:
            modality_cols = [
                c
                for c in ["paper_count_year", "patent_count_year", "trial_count_year"]
                if c in out.columns
            ]
            if modality_cols:
                out["carrier_count_year"] = out[modality_cols].apply(
                    lambda s: safe_numeric(s, 0)
                ).sum(axis=1)

    for base_col, cum_col in [
        ("paper_count_year", "paper_count_cum"),
        ("patent_count_year", "patent_count_cum"),
        ("trial_count_year", "trial_count_cum"),
        ("carrier_count_year", "carrier_count_cum"),
        ("pair_weight_year", "pair_weight_cum"),
    ]:
        if cum_col not in out.columns:
            out[base_col] = safe_numeric(out[base_col], 0)
            out = out.sort_values(["ku_id", "year"])
            out[cum_col] = out.groupby("ku_id")[base_col].cumsum()

    for count_col, flag_col in [
        ("paper_count_year", "has_paper_year"),
        ("patent_count_year", "has_patent_year"),
        ("trial_count_year", "has_trial_year"),
    ]:
        if flag_col not in out.columns:
            out[flag_col] = (safe_numeric(out[count_col], 0) > 0).astype("int8")
        else:
            out[flag_col] = normalize_bool_numeric(out[flag_col])

    if "has_translation_year" not in out.columns:
        out["has_translation_year"] = (
            (safe_numeric(out["patent_count_year"], 0) > 0)
            | (safe_numeric(out["trial_count_year"], 0) > 0)
        ).astype("int8")
    else:
        out["has_translation_year"] = normalize_bool_numeric(out["has_translation_year"])

    for cum_col, flag_col in [
        ("paper_count_cum", "has_paper_cum"),
        ("patent_count_cum", "has_patent_cum"),
        ("trial_count_cum", "has_trial_cum"),
    ]:
        if flag_col not in out.columns:
            out[flag_col] = (safe_numeric(out[cum_col], 0) > 0).astype("int8")
        else:
            out[flag_col] = normalize_bool_numeric(out[flag_col])

    if "has_translation_cum" not in out.columns:
        out["has_translation_cum"] = (
            (safe_numeric(out["patent_count_cum"], 0) > 0)
            | (safe_numeric(out["trial_count_cum"], 0) > 0)
        ).astype("int8")
    else:
        out["has_translation_cum"] = normalize_bool_numeric(out["has_translation_cum"])

    for column in preferred_columns:
        if column not in out.columns:
            out[column] = 0.0

    return out


def build_temporal_sequences(
    year_panel: pd.DataFrame,
    *,
    config: KnowledgeUnitModelingDatasetConfig,
) -> pd.DataFrame:
    """Build one-row-per-KU-year temporal modeling sequences."""

    panel = year_panel.copy()
    panel["ku_id"] = panel["ku_id"].astype(str)
    panel["year"] = coerce_year(panel["year"])

    panel = panel[panel["year"].notna()].copy()
    panel["year"] = panel["year"].astype(int)

    if config.start_year is not None:
        panel = panel[panel["year"] >= int(config.start_year)].copy()

    if config.end_year is not None:
        panel = panel[panel["year"] <= int(config.end_year)].copy()

    if panel.empty:
        raise ValueError("No KU-year rows remain after year filtering")

    if config.ensure_default_sequence_columns:
        panel = ensure_sequence_columns(panel, config.preferred_sequence_columns)

    sequence_columns = [c for c in config.preferred_sequence_columns if c in panel.columns]

    excluded = {
        "ku_id",
        "year",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "pair_type",
    }

    for column in panel.columns:
        if column in excluded or column in sequence_columns:
            continue
        if pd.api.types.is_numeric_dtype(panel[column]):
            sequence_columns.append(column)

    out = panel[["ku_id", "year", *sequence_columns]].copy()

    for column in sequence_columns:
        out[column] = safe_numeric(out[column], 0.0)

    if not config.keep_zero_year_rows:
        value_sum = out[sequence_columns].abs().sum(axis=1)
        out = out[value_sum > 0].copy()

    out = out.sort_values(["ku_id", "year"]).reset_index(drop=True)
    return out


def build_carrier_observations(edges: pd.DataFrame) -> pd.DataFrame:
    """Build carrier-level KU evidence observation table."""

    keep_columns = [
        "ku_id",
        "carrier_id",
        "carrier_key",
        "carrier_type",
        "year",
        "pair_weight",
        "mention_count_a",
        "mention_count_b",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_b_id",
        "entity_b_name",
        "entity_b_type",
        "pair_type",
        "source_edge_table",
    ]

    existing = [column for column in keep_columns if column in edges.columns]
    out = edges[existing].copy()

    out["ku_id"] = out["ku_id"].astype(str)

    if "year" in out.columns:
        out["year"] = coerce_year(out["year"])

    if "pair_weight" in out.columns:
        out["pair_weight"] = safe_numeric(out["pair_weight"], 0.0)

    if "carrier_type" in out.columns:
        out["carrier_type"] = out["carrier_type"].astype(str).str.lower()

    out = out.sort_values(
        [c for c in ["ku_id", "year", "carrier_type", "carrier_key"] if c in out.columns]
    ).reset_index(drop=True)

    return out


def build_future_label_plan(
    *,
    knowledge_unit_dir: Path,
    temporal_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build scaffold documentation for future leakage-controlled labels."""

    return {
        "status": "scaffold_only",
        "has_future_labels": False,
        "reason": (
            "The current 2018-2019 patent-centered demo graph is useful for "
            "pipeline validation and representation/modeling input preparation, "
            "but it is not a final leakage-controlled long-window temporal "
            "prediction dataset."
        ),
        "current_inputs": {
            "knowledge_unit_dir": str(knowledge_unit_dir),
            "temporal_dir": str(temporal_dir),
            "modeling_dataset_dir": str(output_dir),
        },
        "future_plan": [
            "Construct a long-window graph, for example 2010-2025 or 2018-2025.",
            "Choose one or more cutoff years.",
            "Compute all model features using only evidence observed at or before each cutoff year.",
            "Define future patent emergence strictly after the cutoff year.",
            "Define future trial emergence strictly after the cutoff year.",
            "Keep future labels separate from input features.",
            "Create train/validation/test splits that respect temporal leakage control.",
            "Evaluate downstream models with AUROC, AUPRC, Precision@K, Recall@K, and NDCG@K.",
        ],
        "recommended_label_tables": [
            "knowledge_unit_future_patent_labels.parquet",
            "knowledge_unit_future_trial_labels.parquet",
            "knowledge_unit_temporal_splits.json",
        ],
    }


def build_manifest(output_dir: Path, artifact_descriptions: Mapping[str, str]) -> pd.DataFrame:
    """Build output artifact manifest."""

    rows = []

    for file_name, description in artifact_descriptions.items():
        path = output_dir / file_name
        rows.append(
            {
                "artifact": file_name,
                "path": str(path),
                "description": description,
                "exists": path.exists(),
                "file_size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    return pd.DataFrame(rows)


def build_summary(
    *,
    knowledge_unit_dir: Path,
    temporal_dir: Path,
    output_dir: Path,
    config: KnowledgeUnitModelingDatasetConfig,
    static_features: pd.DataFrame,
    temporal_sequences: pd.DataFrame,
    carrier_observations: pd.DataFrame,
) -> dict[str, Any]:
    """Build JSON summary for a KU temporal modeling dataset."""

    years = sorted(temporal_sequences["year"].dropna().astype(int).unique().tolist())

    summary: dict[str, Any] = {
        "dataset_type": "knowledge_unit_temporal_modeling_dataset",
        "stage": "Stage 04F",
        "trains_model": False,
        "defines_rule_based_state_scores": False,
        "has_future_labels": False,
        "knowledge_unit_dir": str(knowledge_unit_dir),
        "temporal_dir": str(temporal_dir),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "ku_count": int(static_features["ku_id"].nunique()),
        "static_feature_rows": int(len(static_features)),
        "temporal_sequence_rows": int(len(temporal_sequences)),
        "carrier_observation_rows": int(len(carrier_observations)),
        "year_min": int(min(years)) if years else None,
        "year_max": int(max(years)) if years else None,
        "year_count": int(len(years)),
        "temporal_sequence_feature_columns": [
            c for c in temporal_sequences.columns if c not in {"ku_id", "year"}
        ],
        "static_columns": list(static_features.columns),
        "carrier_observation_columns": list(carrier_observations.columns),
    }

    if "pair_type" in static_features.columns:
        summary["pair_type_counts"] = static_features["pair_type"].value_counts(dropna=False).to_dict()

    for flag in [
        "has_paper_evidence",
        "has_patent_evidence",
        "has_trial_evidence",
        "has_translation_evidence",
        "is_cross_modal",
        "is_paper_only",
    ]:
        if flag in static_features.columns:
            summary[f"{flag}_count"] = int((safe_numeric(static_features[flag], 0) > 0).sum())

    return summary


def build_markdown_report(summary: Mapping[str, Any]) -> str:
    """Build a compact Markdown report."""

    lines = [
        "# Knowledge Unit Temporal Modeling Dataset Report",
        "",
        "## 1. Positioning",
        "",
        "This dataset is a Stage 05 algorithm-ready modeling dataset derived from Stage 04 Knowledge Unit evidence outputs.",
        "",
        "It does not train an encoder, does not define rule-based Knowledge State scores, and does not create final future labels.",
        "",
        "Downstream algorithms are responsible for defining encoders, temporal dynamics, prediction heads, and ablation studies.",
        "",
        "## 2. Inputs",
        "",
        f"- Knowledge Unit directory: `{summary.get('knowledge_unit_dir')}`",
        f"- Temporal panel directory: `{summary.get('temporal_dir')}`",
        "",
        "## 3. Outputs",
        "",
        f"- Output directory: `{summary.get('output_dir')}`",
        "",
        "## 4. Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| KU count | {summary.get('ku_count')} |",
        f"| Static feature rows | {summary.get('static_feature_rows')} |",
        f"| Temporal sequence rows | {summary.get('temporal_sequence_rows')} |",
        f"| Carrier observation rows | {summary.get('carrier_observation_rows')} |",
        f"| Year min | {summary.get('year_min')} |",
        f"| Year max | {summary.get('year_max')} |",
        f"| Year count | {summary.get('year_count')} |",
        f"| Trains model | {summary.get('trains_model')} |",
        f"| Defines rule-based state scores | {summary.get('defines_rule_based_state_scores')} |",
        f"| Has future labels | {summary.get('has_future_labels')} |",
        "",
        "## 5. Temporal sequence feature columns",
        "",
        "```text",
        "\n".join(map(str, summary.get("temporal_sequence_feature_columns", []))),
        "```",
        "",
        "## 6. Notes",
        "",
        "- `knowledge_unit_static_features.parquet` contains one row per KU.",
        "- `knowledge_unit_temporal_sequences.parquet` contains one row per KU-year.",
        "- `knowledge_unit_carrier_observations.parquet` preserves carrier-level KU evidence observations.",
        "- `future_label_plan.json` documents how future leakage-controlled labels should be created from a long-window graph.",
        "- This Stage 05 dataset is an input preparation layer, not a modeling result.",
        "",
    ]

    return "\n".join(lines)


def build_knowledge_unit_modeling_dataset(
    knowledge_unit_dir: str | Path,
    temporal_dir: str | Path,
    output_dir: str | Path,
    *,
    config: KnowledgeUnitModelingDatasetConfig | None = None,
    overwrite: bool = False,
) -> KnowledgeUnitModelingDatasetResult:
    """Build a KU temporal modeling dataset from Stage 04 outputs.

    Parameters
    ----------
    knowledge_unit_dir:
        Stage 04 Knowledge Unit dataset directory.

    temporal_dir:
        Stage 04 temporal panel directory.

    output_dir:
        Destination for the Stage 05 modeling dataset.

    config:
        Dataset construction configuration.

    overwrite:
        If true, replace the output directory.

    Returns
    -------
    KnowledgeUnitModelingDatasetResult
        In-memory result containing output tables and metadata.
    """

    config = config or KnowledgeUnitModelingDatasetConfig()

    knowledge_unit_dir = Path(knowledge_unit_dir)
    temporal_dir = Path(temporal_dir)
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)

    ku, edges, temporal_features, year_panel = load_stage04_inputs(
        knowledge_unit_dir,
        temporal_dir,
    )

    static_features = build_static_features(
        ku,
        edges,
        temporal_features,
        config=config,
    )

    temporal_sequences = build_temporal_sequences(
        year_panel,
        config=config,
    )

    carrier_observations = build_carrier_observations(edges)

    write_dataframe(static_features, output_dir / STATIC_OUTPUT_FILE_NAME)
    write_dataframe(temporal_sequences, output_dir / SEQUENCE_OUTPUT_FILE_NAME)
    write_dataframe(carrier_observations, output_dir / OBSERVATION_OUTPUT_FILE_NAME)

    summary = build_summary(
        knowledge_unit_dir=knowledge_unit_dir,
        temporal_dir=temporal_dir,
        output_dir=output_dir,
        config=config,
        static_features=static_features,
        temporal_sequences=temporal_sequences,
        carrier_observations=carrier_observations,
    )

    write_json(output_dir / SUMMARY_OUTPUT_FILE_NAME, summary)

    report = build_markdown_report(summary)
    (output_dir / REPORT_OUTPUT_FILE_NAME).write_text(report, encoding="utf-8")

    if config.write_future_label_plan:
        future_label_plan = build_future_label_plan(
            knowledge_unit_dir=knowledge_unit_dir,
            temporal_dir=temporal_dir,
            output_dir=output_dir,
        )
        write_json(output_dir / FUTURE_LABEL_PLAN_FILE_NAME, future_label_plan)

    artifact_descriptions = {
        STATIC_OUTPUT_FILE_NAME: "One-row-per-KU static covariates for downstream models.",
        SEQUENCE_OUTPUT_FILE_NAME: "One-row-per-KU-year temporal evidence sequences for downstream models.",
        OBSERVATION_OUTPUT_FILE_NAME: "Carrier-level KU evidence observations for optional attribution or graph models.",
        SUMMARY_OUTPUT_FILE_NAME: "JSON summary for the KU temporal modeling dataset.",
        MANIFEST_OUTPUT_FILE_NAME: "Artifact manifest.",
        REPORT_OUTPUT_FILE_NAME: "Human-readable modeling dataset report.",
    }

    if config.write_future_label_plan:
        artifact_descriptions[
            FUTURE_LABEL_PLAN_FILE_NAME
        ] = "Scaffold plan for future leakage-controlled KU-level labels."

    manifest = build_manifest(output_dir, artifact_descriptions)
    manifest_path = write_dataframe(manifest, output_dir / MANIFEST_OUTPUT_FILE_NAME)

    # Refresh manifest after writing itself, so size/existence is accurate.
    manifest = build_manifest(output_dir, artifact_descriptions)
    manifest.to_csv(manifest_path, index=False)

    return KnowledgeUnitModelingDatasetResult(
        output_dir=output_dir,
        static_features=static_features,
        temporal_sequences=temporal_sequences,
        carrier_observations=carrier_observations,
        summary=summary,
        manifest=manifest,
    )