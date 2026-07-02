"""Temporal aggregation utilities for Knowledge Unit datasets.

This module converts carrier-KU evidence edges into yearly Knowledge Unit
panels and KU-level temporal feature tables.

Expected inputs are produced by:

    scripts/20_build_knowledge_units.py
    scripts/21_audit_knowledge_units.py

Core input tables:

    knowledge_units.parquet
    knowledge_units_with_audit_flags.parquet
    carrier_knowledge_unit_edges.parquet

Important interpretation note:

    paper_count / patent_count / trial_count are observed carrier counts within
    the current carrier graph. In short-window patent-centered graphs, missing
    patent/trial evidence should not be interpreted as absence of historical
    real-world translation.

Trial year handling:

    If trial carrier-KU edges do not have valid year values, they are kept as
    undated KU-level evidence but do not contribute to yearly panel counts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_MODALITIES: tuple[str, ...] = ("paper", "patent", "trial")
TRANSLATION_MODALITIES: tuple[str, ...] = ("patent", "trial")


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects into JSON-serializable Python objects."""

    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    return value


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write JSON with safe conversion."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(data)), handle, indent=2, ensure_ascii=False)


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame based on file extension."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            frame.to_json(path, orient="records", lines=True, force_ascii=False)
        else:
            frame.to_json(path, orient="records", indent=2, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output suffix for DataFrame: {path}")


def markdown_table(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    max_rows: int | None = None,
) -> str:
    """Render a small markdown table."""

    if isinstance(rows, pd.DataFrame):
        data = rows.to_dict("records")
        if columns is None:
            columns = list(rows.columns)
    else:
        data = list(rows)
        if columns is None:
            columns = list(data[0].keys()) if data else []

    if max_rows is not None:
        data = data[:max_rows]

    columns = list(columns)

    if not columns:
        return ""

    def fmt(value: Any) -> str:
        value = json_safe(value)

        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.6g}"

        text = str(value)
        text = text.replace("|", "\\|").replace("\n", " ")
        return text

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"

    body = [
        "| " + " | ".join(fmt(row.get(column)) for column in columns) + " |"
        for row in data
    ]

    return "\n".join([header, sep, *body])


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    table_name: str,
) -> None:
    """Validate required columns are present."""

    missing = sorted(set(columns) - set(frame.columns))

    if missing:
        raise ValueError(
            f"{table_name} missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def normalize_carrier_type(value: Any) -> str:
    """Normalize carrier type strings."""

    text = str(value).strip().lower()

    aliases = {
        "publication": "paper",
        "pubmed": "paper",
        "pmid": "paper",
        "article": "paper",
        "patents": "patent",
        "clinicaltrial": "trial",
        "clinical_trial": "trial",
        "clinical-trial": "trial",
        "nct": "trial",
    }

    return aliases.get(text, text)


def ensure_int_year(series: pd.Series) -> pd.Series:
    """Convert a year-like series to nullable Int64."""

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where((numeric >= 0) & (numeric <= 3000))
    return numeric.astype("Int64")


def load_knowledge_unit_inputs(
    input_dir: str | Path,
    *,
    prefer_audit_flags: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Knowledge Unit and carrier-KU edge tables.

    Parameters
    ----------
    input_dir:
        Directory containing Knowledge Unit output files.
    prefer_audit_flags:
        If True and ``knowledge_units_with_audit_flags.parquet`` exists, load it
        instead of ``knowledge_units.parquet``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(knowledge_units, carrier_knowledge_unit_edges)``.
    """

    input_dir = Path(input_dir)

    ku_flags_path = input_dir / "knowledge_units_with_audit_flags.parquet"
    ku_path = input_dir / "knowledge_units.parquet"
    edges_path = input_dir / "carrier_knowledge_unit_edges.parquet"

    if prefer_audit_flags and ku_flags_path.exists():
        ku_file = ku_flags_path
    else:
        ku_file = ku_path

    if not ku_file.exists():
        raise FileNotFoundError(f"Missing Knowledge Unit table: {ku_file}")

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing carrier-KU edge table: {edges_path}")

    ku = pd.read_parquet(ku_file)
    edges = pd.read_parquet(edges_path)

    validate_knowledge_unit_inputs(ku, edges)

    return ku, edges


def validate_knowledge_unit_inputs(ku: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Validate minimum expected schema."""

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
            "carrier_count",
            "paper_count",
            "patent_count",
            "trial_count",
        ],
        table_name="knowledge_units",
    )

    require_columns(
        edges,
        [
            "ku_id",
            "carrier_type",
            "carrier_id",
            "carrier_key",
            "year",
            "pair_weight",
        ],
        table_name="carrier_knowledge_unit_edges",
    )


def prepare_edges_for_temporal_panel(edges: pd.DataFrame) -> pd.DataFrame:
    """Normalize carrier-KU edges for temporal aggregation."""

    out = edges.copy()

    out["ku_id"] = out["ku_id"].astype(str)
    out["carrier_type"] = out["carrier_type"].map(normalize_carrier_type)
    out["carrier_id"] = out["carrier_id"].astype("string")
    out["carrier_key"] = out["carrier_key"].astype("string")
    out["year"] = ensure_int_year(out["year"])

    if "pair_weight" not in out.columns:
        out["pair_weight"] = 1.0

    out["pair_weight"] = pd.to_numeric(out["pair_weight"], errors="coerce").fillna(0.0)

    return out


def prepare_knowledge_units_for_temporal_panel(ku: pd.DataFrame) -> pd.DataFrame:
    """Normalize KU table for temporal aggregation."""

    out = ku.copy()

    out["ku_id"] = out["ku_id"].astype(str)

    for column in ["carrier_count", "paper_count", "patent_count", "trial_count"]:
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    for column in ["first_year", "last_year"]:
        if column in out.columns:
            out[column] = ensure_int_year(out[column])
        else:
            out[column] = pd.Series([pd.NA] * len(out), dtype="Int64")

    if "total_pair_weight" not in out.columns:
        out["total_pair_weight"] = np.nan

    if "evidence_pattern" not in out.columns:
        has_paper = out["paper_count"] > 0
        has_patent = out["patent_count"] > 0
        has_trial = out["trial_count"] > 0
        out["evidence_pattern"] = (
            np.where(has_paper, "P", "")
            + np.where(has_patent, "A", "")
            + np.where(has_trial, "T", "")
        )
        out["evidence_pattern"] = out["evidence_pattern"].replace("", "none")

    if "modality_count" not in out.columns:
        out["modality_count"] = (
            (out["paper_count"] > 0).astype(int)
            + (out["patent_count"] > 0).astype(int)
            + (out["trial_count"] > 0).astype(int)
        )

    if "has_patent_evidence" not in out.columns:
        out["has_patent_evidence"] = out["patent_count"] > 0

    if "has_trial_evidence" not in out.columns:
        out["has_trial_evidence"] = out["trial_count"] > 0

    if "has_translation_evidence" not in out.columns:
        out["has_translation_evidence"] = (
            out["has_patent_evidence"] | out["has_trial_evidence"]
        )

    if "is_cross_modal" not in out.columns:
        out["is_cross_modal"] = out["modality_count"] >= 2

    return out


def infer_panel_year_range(
    edges: pd.DataFrame,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[int, int]:
    """Infer panel year range from dated carrier-KU edges."""

    dated_years = edges["year"].dropna()

    if start_year is None:
        if dated_years.empty:
            raise ValueError(
                "Unable to infer start_year because no dated carrier-KU edges exist."
            )
        start_year = int(dated_years.min())

    if end_year is None:
        if dated_years.empty:
            raise ValueError(
                "Unable to infer end_year because no dated carrier-KU edges exist."
            )
        end_year = int(dated_years.max())

    if start_year > end_year:
        raise ValueError(f"start_year must be <= end_year, got {start_year}>{end_year}")

    return int(start_year), int(end_year)


def aggregate_edges_by_ku_year(
    edges: pd.DataFrame,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Aggregate dated carrier-KU edges by KU and year.

    Undated edges are excluded from this yearly aggregation.
    """

    modalities = tuple(normalize_carrier_type(m) for m in modalities)

    dated = edges[edges["year"].notna()].copy()

    if dated.empty:
        base_columns = ["ku_id", "year", "carrier_count_year", "pair_weight_year"]

        modality_columns: list[str] = []
        for modality in modalities:
            modality_columns.extend(
                [
                    f"{modality}_count_year",
                    f"{modality}_pair_weight_year",
                ]
            )

        return pd.DataFrame(columns=base_columns + modality_columns)

    dated["year"] = dated["year"].astype(int)

    total = (
        dated.groupby(["ku_id", "year"], as_index=False)
        .agg(
            carrier_count_year=("carrier_key", "nunique"),
            carrier_ku_edge_count_year=("ku_id", "size"),
            pair_weight_year=("pair_weight", "sum"),
        )
    )

    modality_frames = [total]

    for modality in modalities:
        sub = dated[dated["carrier_type"] == modality]

        if sub.empty:
            continue

        agg = (
            sub.groupby(["ku_id", "year"], as_index=False)
            .agg(
                **{
                    f"{modality}_count_year": ("carrier_key", "nunique"),
                    f"{modality}_edge_count_year": ("ku_id", "size"),
                    f"{modality}_pair_weight_year": ("pair_weight", "sum"),
                }
            )
        )

        modality_frames.append(agg)

    out = modality_frames[0]

    for frame in modality_frames[1:]:
        out = out.merge(frame, on=["ku_id", "year"], how="outer")

    count_columns = [
        column
        for column in out.columns
        if column.endswith("_count_year") or column.endswith("_edge_count_year")
    ]
    weight_columns = [column for column in out.columns if column.endswith("_weight_year")]

    for column in count_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    for column in weight_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    for modality in modalities:
        for column, default in [
            (f"{modality}_count_year", 0),
            (f"{modality}_edge_count_year", 0),
            (f"{modality}_pair_weight_year", 0.0),
        ]:
            if column not in out.columns:
                out[column] = default

    out = out.sort_values(["ku_id", "year"]).reset_index(drop=True)

    return out


def build_dense_ku_year_index(
    ku_ids: Sequence[str],
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Build dense KU × year index."""

    years = list(range(int(start_year), int(end_year) + 1))
    ku_ids = [str(ku_id) for ku_id in ku_ids]

    index = pd.MultiIndex.from_product(
        [ku_ids, years],
        names=["ku_id", "year"],
    )

    return index.to_frame(index=False)


def add_cumulative_columns(
    panel: pd.DataFrame,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Add cumulative counts and weights to a KU-year panel."""

    out = panel.sort_values(["ku_id", "year"]).copy()

    year_count_columns = [
        "carrier_count_year",
        "carrier_ku_edge_count_year",
        "pair_weight_year",
    ]

    for modality in modalities:
        year_count_columns.extend(
            [
                f"{modality}_count_year",
                f"{modality}_edge_count_year",
                f"{modality}_pair_weight_year",
            ]
        )

    for column in year_count_columns:
        if column not in out.columns:
            out[column] = 0

    cumulative_map = {
        "carrier_count_year": "carrier_count_cum",
        "carrier_ku_edge_count_year": "carrier_ku_edge_count_cum",
        "pair_weight_year": "pair_weight_cum",
    }

    for modality in modalities:
        cumulative_map[f"{modality}_count_year"] = f"{modality}_count_cum"
        cumulative_map[f"{modality}_edge_count_year"] = f"{modality}_edge_count_cum"
        cumulative_map[f"{modality}_pair_weight_year"] = f"{modality}_pair_weight_cum"

    for source, target in cumulative_map.items():
        out[target] = out.groupby("ku_id", sort=False)[source].cumsum()

    for modality in modalities:
        out[f"has_{modality}_by_year"] = out[f"{modality}_count_cum"] > 0

    out["has_translation_by_year"] = (
        out.get("patent_count_cum", 0) > 0
    ) | (out.get("trial_count_cum", 0) > 0)

    out["has_any_evidence_by_year"] = out["carrier_count_cum"] > 0

    return out


def attach_ku_metadata_to_panel(
    panel: pd.DataFrame,
    ku: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected KU metadata columns to a year panel."""

    metadata_columns = [
        "ku_id",
        "entity_a_id",
        "entity_a_name",
        "entity_a_type",
        "entity_b_id",
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
        "evidence_pattern",
        "modality_count",
        "has_patent_evidence",
        "has_trial_evidence",
        "has_translation_evidence",
        "is_cross_modal",
    ]

    existing = [column for column in metadata_columns if column in ku.columns]
    metadata = ku[existing].drop_duplicates(subset=["ku_id"])

    out = panel.merge(metadata, on="ku_id", how="left")

    front = [
        "ku_id",
        "year",
        "pair_type",
        "entity_a_name",
        "entity_a_type",
        "entity_b_name",
        "entity_b_type",
    ]

    front = [column for column in front if column in out.columns]
    rest = [column for column in out.columns if column not in front]

    return out[front + rest]


def build_knowledge_unit_year_panel(
    ku: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    dense: bool = True,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Build KU × year temporal panel.

    Parameters
    ----------
    ku:
        Knowledge Unit table.
    edges:
        Carrier-KU edge table.
    start_year, end_year:
        Optional inclusive year range. If omitted, inferred from dated edges.
    dense:
        If True, output every KU × year combination between start_year and
        end_year. If False, output only KU-year combinations with dated evidence.
    modalities:
        Carrier modalities to aggregate.

    Returns
    -------
    pd.DataFrame
        KU-year panel with yearly and cumulative evidence counts.
    """

    ku = prepare_knowledge_units_for_temporal_panel(ku)
    edges = prepare_edges_for_temporal_panel(edges)

    start_year, end_year = infer_panel_year_range(
        edges,
        start_year=start_year,
        end_year=end_year,
    )

    yearly = aggregate_edges_by_ku_year(edges, modalities=modalities)

    if dense:
        base = build_dense_ku_year_index(
            ku["ku_id"].astype(str).tolist(),
            start_year=start_year,
            end_year=end_year,
        )
        panel = base.merge(yearly, on=["ku_id", "year"], how="left")
    else:
        panel = yearly.copy()

    numeric_defaults: dict[str, int | float] = {
        "carrier_count_year": 0,
        "carrier_ku_edge_count_year": 0,
        "pair_weight_year": 0.0,
    }

    for modality in modalities:
        numeric_defaults[f"{modality}_count_year"] = 0
        numeric_defaults[f"{modality}_edge_count_year"] = 0
        numeric_defaults[f"{modality}_pair_weight_year"] = 0.0

    for column, default in numeric_defaults.items():
        if column not in panel.columns:
            panel[column] = default
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(default)

        if isinstance(default, int):
            panel[column] = panel[column].astype(int)

    panel = add_cumulative_columns(panel, modalities=modalities)
    panel = attach_ku_metadata_to_panel(panel, ku)

    return panel.sort_values(["ku_id", "year"]).reset_index(drop=True)


def first_year_by_modality(
    edges: pd.DataFrame,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Compute first observed year by KU and carrier modality."""

    edges = prepare_edges_for_temporal_panel(edges)
    modalities = tuple(normalize_carrier_type(m) for m in modalities)

    dated = edges[edges["year"].notna()].copy()

    columns = [
        "ku_id",
        "modality",
        "first_year",
        "count_at_first_year",
        "pair_weight_at_first_year",
    ]

    if dated.empty:
        return pd.DataFrame(columns=columns)

    dated["year"] = dated["year"].astype(int)
    dated = dated[dated["carrier_type"].isin(modalities)].copy()

    if dated.empty:
        return pd.DataFrame(columns=columns)

    yearly = (
        dated.groupby(["ku_id", "carrier_type", "year"], as_index=False)
        .agg(
            count_at_year=("carrier_key", "nunique"),
            pair_weight_at_year=("pair_weight", "sum"),
        )
    )

    first = (
        yearly.sort_values(["ku_id", "carrier_type", "year"])
        .groupby(["ku_id", "carrier_type"], as_index=False)
        .first()
        .rename(
            columns={
                "carrier_type": "modality",
                "year": "first_year",
                "count_at_year": "count_at_first_year",
                "pair_weight_at_year": "pair_weight_at_first_year",
            }
        )
    )

    return first[columns].sort_values(["ku_id", "modality"]).reset_index(drop=True)


def pivot_first_years(
    first_events: pd.DataFrame,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Pivot first-observed event table to one row per KU."""

    if first_events.empty:
        return pd.DataFrame(columns=["ku_id"])

    first_year = first_events.pivot(
        index="ku_id",
        columns="modality",
        values="first_year",
    )

    first_count = first_events.pivot(
        index="ku_id",
        columns="modality",
        values="count_at_first_year",
    )

    first_weight = first_events.pivot(
        index="ku_id",
        columns="modality",
        values="pair_weight_at_first_year",
    )

    frames = []

    for modality in modalities:
        frame = pd.DataFrame(index=first_year.index)

        if modality in first_year.columns:
            frame[f"first_{modality}_year"] = first_year[modality].astype("Int64")
        else:
            frame[f"first_{modality}_year"] = pd.Series(
                [pd.NA] * len(first_year),
                index=first_year.index,
                dtype="Int64",
            )

        if modality in first_count.columns:
            frame[f"{modality}_count_at_first_year"] = first_count[modality]
        else:
            frame[f"{modality}_count_at_first_year"] = 0

        if modality in first_weight.columns:
            frame[f"{modality}_pair_weight_at_first_year"] = first_weight[modality]
        else:
            frame[f"{modality}_pair_weight_at_first_year"] = 0.0

        frames.append(frame)

    out = pd.concat(frames, axis=1)
    out = out.reset_index()

    return out


def build_temporal_feature_table(
    ku: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Build one-row-per-KU temporal feature table."""

    ku = prepare_knowledge_units_for_temporal_panel(ku)
    edges = prepare_edges_for_temporal_panel(edges)

    first_events = first_year_by_modality(edges, modalities=modalities)
    first_wide = pivot_first_years(first_events, modalities=modalities)

    out = ku.merge(first_wide, on="ku_id", how="left")

    for modality in modalities:
        year_col = f"first_{modality}_year"
        count_col = f"{modality}_count_at_first_year"
        weight_col = f"{modality}_pair_weight_at_first_year"

        if year_col not in out.columns:
            out[year_col] = pd.Series([pd.NA] * len(out), dtype="Int64")
        else:
            out[year_col] = ensure_int_year(out[year_col])

        if count_col not in out.columns:
            out[count_col] = 0
        out[count_col] = pd.to_numeric(out[count_col], errors="coerce").fillna(0).astype(int)

        if weight_col not in out.columns:
            out[weight_col] = 0.0
        out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)

    year_cols = [f"first_{modality}_year" for modality in modalities]

    out["first_observed_year"] = out[year_cols].min(axis=1).astype("Int64")

    translation_year_cols = [
        f"first_{modality}_year"
        for modality in TRANSLATION_MODALITIES
        if f"first_{modality}_year" in out.columns
    ]

    if translation_year_cols:
        out["first_translation_year"] = out[translation_year_cols].min(axis=1).astype(
            "Int64"
        )
    else:
        out["first_translation_year"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    out["has_dated_paper_evidence"] = out["first_paper_year"].notna()
    out["has_dated_patent_evidence"] = out["first_patent_year"].notna()
    out["has_dated_trial_evidence"] = out["first_trial_year"].notna()
    out["has_dated_translation_evidence"] = out["first_translation_year"].notna()

    out["paper_to_patent_lag"] = (
        out["first_patent_year"] - out["first_paper_year"]
    ).astype("Int64")
    out.loc[
        out["first_patent_year"].isna() | out["first_paper_year"].isna(),
        "paper_to_patent_lag",
    ] = pd.NA

    out["paper_to_trial_lag"] = (
        out["first_trial_year"] - out["first_paper_year"]
    ).astype("Int64")
    out.loc[
        out["first_trial_year"].isna() | out["first_paper_year"].isna(),
        "paper_to_trial_lag",
    ] = pd.NA

    out["patent_to_trial_lag"] = (
        out["first_trial_year"] - out["first_patent_year"]
    ).astype("Int64")
    out.loc[
        out["first_trial_year"].isna() | out["first_patent_year"].isna(),
        "patent_to_trial_lag",
    ] = pd.NA

    out["paper_before_patent"] = (
        out["first_paper_year"].notna()
        & out["first_patent_year"].notna()
        & (out["first_paper_year"] <= out["first_patent_year"])
    )

    out["paper_before_trial"] = (
        out["first_paper_year"].notna()
        & out["first_trial_year"].notna()
        & (out["first_paper_year"] <= out["first_trial_year"])
    )

    out["patent_before_trial"] = (
        out["first_patent_year"].notna()
        & out["first_trial_year"].notna()
        & (out["first_patent_year"] <= out["first_trial_year"])
    )

    undated_summary = summarize_undated_edges(edges)

    out = out.merge(undated_summary, on="ku_id", how="left")

    for modality in modalities:
        column = f"{modality}_undated_count"
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    out["undated_carrier_count"] = pd.to_numeric(
        out.get("undated_carrier_count", 0),
        errors="coerce",
    ).fillna(0).astype(int)

    out["has_undated_trial_evidence"] = out.get("trial_undated_count", 0) > 0

    front = [
        "ku_id",
        "pair_type",
        "entity_a_name",
        "entity_a_type",
        "entity_b_name",
        "entity_b_type",
        "carrier_count",
        "paper_count",
        "patent_count",
        "trial_count",
        "evidence_pattern",
        "modality_count",
        "first_observed_year",
        "first_paper_year",
        "first_patent_year",
        "first_trial_year",
        "first_translation_year",
        "paper_to_patent_lag",
        "paper_to_trial_lag",
        "patent_to_trial_lag",
        "has_dated_paper_evidence",
        "has_dated_patent_evidence",
        "has_dated_trial_evidence",
        "has_dated_translation_evidence",
        "has_undated_trial_evidence",
        "has_translation_evidence",
        "is_cross_modal",
    ]

    front = [column for column in front if column in out.columns]
    rest = [column for column in out.columns if column not in front]

    return out[front + rest]


def summarize_undated_edges(
    edges: pd.DataFrame,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Summarize undated carrier-KU evidence by KU."""

    edges = prepare_edges_for_temporal_panel(edges)
    modalities = tuple(normalize_carrier_type(m) for m in modalities)

    undated = edges[edges["year"].isna()].copy()

    base_columns = ["ku_id", "undated_carrier_count", "undated_pair_weight"]

    modality_columns: list[str] = []
    for modality in modalities:
        modality_columns.extend(
            [
                f"{modality}_undated_count",
                f"{modality}_undated_pair_weight",
            ]
        )

    if undated.empty:
        return pd.DataFrame(columns=base_columns + modality_columns)

    total = (
        undated.groupby("ku_id", as_index=False)
        .agg(
            undated_carrier_count=("carrier_key", "nunique"),
            undated_pair_weight=("pair_weight", "sum"),
        )
    )

    out = total

    for modality in modalities:
        sub = undated[undated["carrier_type"] == modality]

        if sub.empty:
            continue

        agg = (
            sub.groupby("ku_id", as_index=False)
            .agg(
                **{
                    f"{modality}_undated_count": ("carrier_key", "nunique"),
                    f"{modality}_undated_pair_weight": ("pair_weight", "sum"),
                }
            )
        )

        out = out.merge(agg, on="ku_id", how="left")

    for column in modality_columns:
        if column not in out.columns:
            out[column] = 0

    count_columns = [
        column
        for column in out.columns
        if column.endswith("_undated_count") or column == "undated_carrier_count"
    ]
    weight_columns = [
        column
        for column in out.columns
        if column.endswith("_undated_pair_weight") or column == "undated_pair_weight"
    ]

    for column in count_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    for column in weight_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    return out


def build_first_observed_events(
    ku: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> pd.DataFrame:
    """Build first-observed event table for dated and undated evidence."""

    ku = prepare_knowledge_units_for_temporal_panel(ku)
    edges = prepare_edges_for_temporal_panel(edges)

    dated_events = first_year_by_modality(edges, modalities=modalities)
    dated_events["is_dated"] = True

    undated = summarize_undated_edges(edges, modalities=modalities)

    undated_rows: list[dict[str, Any]] = []

    for _, row in undated.iterrows():
        ku_id = row["ku_id"]

        for modality in modalities:
            count = int(row.get(f"{modality}_undated_count", 0) or 0)
            weight = float(row.get(f"{modality}_undated_pair_weight", 0.0) or 0.0)

            if count <= 0:
                continue

            undated_rows.append(
                {
                    "ku_id": ku_id,
                    "modality": modality,
                    "first_year": pd.NA,
                    "count_at_first_year": count,
                    "pair_weight_at_first_year": weight,
                    "is_dated": False,
                }
            )

    undated_events = pd.DataFrame(undated_rows)

    if undated_events.empty:
        events = dated_events
    elif dated_events.empty:
        events = undated_events
    else:
        events = pd.concat([dated_events, undated_events], ignore_index=True)

    if events.empty:
        return pd.DataFrame(
            columns=[
                "ku_id",
                "modality",
                "first_year",
                "count_at_first_year",
                "pair_weight_at_first_year",
                "is_dated",
                "pair_type",
                "entity_a_name",
                "entity_a_type",
                "entity_b_name",
                "entity_b_type",
            ]
        )

    metadata_cols = [
        "ku_id",
        "pair_type",
        "entity_a_name",
        "entity_a_type",
        "entity_b_name",
        "entity_b_type",
    ]
    metadata_cols = [column for column in metadata_cols if column in ku.columns]

    events = events.merge(ku[metadata_cols], on="ku_id", how="left")

    return events.sort_values(["ku_id", "modality", "is_dated"], ascending=[True, True, False]).reset_index(drop=True)


def summarize_temporal_panel(
    ku: pd.DataFrame,
    edges: pd.DataFrame,
    panel: pd.DataFrame,
    temporal_features: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    """Build JSON summary for temporal panel outputs."""

    edges = prepare_edges_for_temporal_panel(edges)

    dated_edges = edges[edges["year"].notna()]
    undated_edges = edges[edges["year"].isna()]

    modality_edge_summary: list[dict[str, Any]] = []

    for carrier_type, group in edges.groupby("carrier_type"):
        dated = group[group["year"].notna()]
        modality_edge_summary.append(
            {
                "carrier_type": carrier_type,
                "edge_count": int(len(group)),
                "ku_count": int(group["ku_id"].nunique()),
                "carrier_count": int(group["carrier_key"].nunique()),
                "dated_edge_count": int(len(dated)),
                "undated_edge_count": int(group["year"].isna().sum()),
                "year_min": json_safe(dated["year"].min()) if not dated.empty else None,
                "year_max": json_safe(dated["year"].max()) if not dated.empty else None,
            }
        )

    lag_summary = {}

    for column in ["paper_to_patent_lag", "paper_to_trial_lag", "patent_to_trial_lag"]:
        if column in temporal_features.columns:
            values = pd.to_numeric(temporal_features[column], errors="coerce").dropna()

            lag_summary[column] = {
                "count": int(len(values)),
                "min": json_safe(values.min()) if len(values) else None,
                "q25": json_safe(values.quantile(0.25)) if len(values) else None,
                "median": json_safe(values.median()) if len(values) else None,
                "q75": json_safe(values.quantile(0.75)) if len(values) else None,
                "max": json_safe(values.max()) if len(values) else None,
            }

    summary = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "year_count": int(end_year - start_year + 1),
        "knowledge_unit_count": int(len(ku)),
        "panel_row_count": int(len(panel)),
        "temporal_feature_row_count": int(len(temporal_features)),
        "carrier_ku_edge_count": int(len(edges)),
        "dated_edge_count": int(len(dated_edges)),
        "undated_edge_count": int(len(undated_edges)),
        "dated_edge_fraction": float(len(dated_edges) / len(edges)) if len(edges) else None,
        "ku_with_dated_paper_evidence": int(
            temporal_features.get("has_dated_paper_evidence", pd.Series(dtype=bool)).sum()
        ),
        "ku_with_dated_patent_evidence": int(
            temporal_features.get("has_dated_patent_evidence", pd.Series(dtype=bool)).sum()
        ),
        "ku_with_dated_trial_evidence": int(
            temporal_features.get("has_dated_trial_evidence", pd.Series(dtype=bool)).sum()
        ),
        "ku_with_undated_trial_evidence": int(
            temporal_features.get("has_undated_trial_evidence", pd.Series(dtype=bool)).sum()
        ),
        "ku_with_dated_translation_evidence": int(
            temporal_features.get(
                "has_dated_translation_evidence", pd.Series(dtype=bool)
            ).sum()
        ),
        "modality_edge_summary": modality_edge_summary,
        "lag_summary": lag_summary,
        "interpretation_note": (
            "Yearly panel counts include only carrier-KU edges with valid years. "
            "Undated trial evidence is retained in temporal feature columns such as "
            "trial_undated_count and has_undated_trial_evidence."
        ),
    }

    return json_safe(summary)


def summarize_panel_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize yearly panel across KUs."""

    if panel.empty:
        return pd.DataFrame()

    count_cols = [
        "carrier_count_year",
        "paper_count_year",
        "patent_count_year",
        "trial_count_year",
        "carrier_count_cum",
        "paper_count_cum",
        "patent_count_cum",
        "trial_count_cum",
    ]

    available_count_cols = [column for column in count_cols if column in panel.columns]

    aggregations: dict[str, tuple[str, str]] = {
        "active_ku_count": ("carrier_count_year", lambda s: int((s > 0).sum())),
        "cumulative_active_ku_count": ("carrier_count_cum", lambda s: int((s > 0).sum())),
    }

    for column in available_count_cols:
        aggregations[f"{column}_sum"] = (column, "sum")

    out = panel.groupby("year", as_index=False).agg(**aggregations)

    return out.sort_values("year").reset_index(drop=True)


def build_temporal_report(
    *,
    summary: Mapping[str, Any],
    panel_year_summary: pd.DataFrame,
    temporal_features: pd.DataFrame,
    first_events: pd.DataFrame,
    top_n: int = 25,
) -> str:
    """Build Markdown report for temporal panel."""

    summary_rows = [
        {"metric": "Knowledge Units", "value": summary.get("knowledge_unit_count")},
        {"metric": "Panel rows", "value": summary.get("panel_row_count")},
        {"metric": "Temporal feature rows", "value": summary.get("temporal_feature_row_count")},
        {"metric": "Start year", "value": summary.get("start_year")},
        {"metric": "End year", "value": summary.get("end_year")},
        {"metric": "Carrier-KU edges", "value": summary.get("carrier_ku_edge_count")},
        {"metric": "Dated edges", "value": summary.get("dated_edge_count")},
        {"metric": "Undated edges", "value": summary.get("undated_edge_count")},
        {"metric": "Dated edge fraction", "value": summary.get("dated_edge_fraction")},
        {
            "metric": "KUs with dated paper evidence",
            "value": summary.get("ku_with_dated_paper_evidence"),
        },
        {
            "metric": "KUs with dated patent evidence",
            "value": summary.get("ku_with_dated_patent_evidence"),
        },
        {
            "metric": "KUs with dated trial evidence",
            "value": summary.get("ku_with_dated_trial_evidence"),
        },
        {
            "metric": "KUs with undated trial evidence",
            "value": summary.get("ku_with_undated_trial_evidence"),
        },
        {
            "metric": "KUs with dated translation evidence",
            "value": summary.get("ku_with_dated_translation_evidence"),
        },
    ]

    modality_edge_summary = summary.get("modality_edge_summary", [])
    lag_summary = summary.get("lag_summary", {})

    lag_rows = []
    for lag_name, values in lag_summary.items():
        row = {"lag": lag_name}
        row.update(values)
        lag_rows.append(row)

    top_patent = temporal_features[
        temporal_features.get("first_patent_year", pd.Series(dtype="Int64")).notna()
    ].copy()

    if not top_patent.empty:
        top_patent = top_patent.sort_values(
            ["paper_to_patent_lag", "carrier_count"],
            ascending=[False, False],
        ).head(top_n)

    top_patent_cols = [
        "entity_a_name",
        "entity_a_type",
        "entity_b_name",
        "entity_b_type",
        "pair_type",
        "carrier_count",
        "paper_count",
        "patent_count",
        "trial_count",
        "first_paper_year",
        "first_patent_year",
        "paper_to_patent_lag",
        "evidence_pattern",
    ]
    top_patent_cols = [column for column in top_patent_cols if column in top_patent.columns]

    first_event_summary = (
        first_events.groupby(["modality", "is_dated"], as_index=False)
        .agg(
            event_count=("ku_id", "nunique"),
            first_year_min=("first_year", "min"),
            first_year_max=("first_year", "max"),
        )
        if not first_events.empty
        else pd.DataFrame()
    )

    return "\n".join(
        [
            "# Knowledge Unit Temporal Panel Report",
            "",
            "## 1. Summary",
            "",
            markdown_table(summary_rows, ["metric", "value"]),
            "",
            "## 2. Interpretation note",
            "",
            str(summary.get("interpretation_note", "")),
            "",
            "The yearly panel includes only dated carrier-KU evidence. Undated trial evidence is preserved as KU-level evidence but does not contribute to yearly counts until trial years are patched.",
            "",
            "## 3. Edge dating by carrier type",
            "",
            markdown_table(
                modality_edge_summary,
                [
                    "carrier_type",
                    "edge_count",
                    "ku_count",
                    "carrier_count",
                    "dated_edge_count",
                    "undated_edge_count",
                    "year_min",
                    "year_max",
                ],
            ),
            "",
            "## 4. Year summary",
            "",
            markdown_table(panel_year_summary.head(100), max_rows=100),
            "",
            "## 5. First event summary",
            "",
            markdown_table(first_event_summary),
            "",
            "## 6. Lag summary",
            "",
            markdown_table(
                lag_rows,
                ["lag", "count", "min", "q25", "median", "q75", "max"],
            ),
            "",
            "## 7. Top long paper-to-patent lag KUs",
            "",
            markdown_table(top_patent, top_patent_cols, max_rows=top_n),
            "",
        ]
    )


def artifact_row(
    artifact: str,
    path: str | Path,
    description: str,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Create manifest row."""

    path = Path(path)

    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }


def build_temporal_outputs(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    start_year: int | None = None,
    end_year: int | None = None,
    dense: bool = True,
    prefer_audit_flags: bool = True,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> dict[str, Any]:
    """Build all temporal panel output tables in memory.

    This function does not write files. It returns tables and summary objects so
    CLI scripts can decide where and how to persist them.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    ku, edges = load_knowledge_unit_inputs(
        input_dir,
        prefer_audit_flags=prefer_audit_flags,
    )

    ku = prepare_knowledge_units_for_temporal_panel(ku)
    edges = prepare_edges_for_temporal_panel(edges)

    inferred_start_year, inferred_end_year = infer_panel_year_range(
        edges,
        start_year=start_year,
        end_year=end_year,
    )

    panel = build_knowledge_unit_year_panel(
        ku,
        edges,
        start_year=inferred_start_year,
        end_year=inferred_end_year,
        dense=dense,
        modalities=modalities,
    )

    temporal_features = build_temporal_feature_table(
        ku,
        edges,
        modalities=modalities,
    )

    first_events = build_first_observed_events(
        ku,
        edges,
        modalities=modalities,
    )

    panel_year_summary = summarize_panel_by_year(panel)

    temporal_summary = summarize_temporal_panel(
        ku,
        edges,
        panel,
        temporal_features,
        start_year=inferred_start_year,
        end_year=inferred_end_year,
    )

    report = build_temporal_report(
        summary=temporal_summary,
        panel_year_summary=panel_year_summary,
        temporal_features=temporal_features,
        first_events=first_events,
    )

    return {
        "knowledge_units": ku,
        "carrier_knowledge_unit_edges": edges,
        "knowledge_unit_year_panel": panel,
        "knowledge_unit_temporal_features": temporal_features,
        "knowledge_unit_first_observed_events": first_events,
        "knowledge_unit_year_summary": panel_year_summary,
        "knowledge_unit_temporal_summary": temporal_summary,
        "knowledge_unit_temporal_report": report,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "start_year": inferred_start_year,
        "end_year": inferred_end_year,
        "dense": dense,
    }