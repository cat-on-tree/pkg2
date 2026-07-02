#!/usr/bin/env python
"""Audit an entity-pair Knowledge Unit dataset.

This script audits outputs produced by:

    scripts/20_build_knowledge_units.py

Expected input directory:

    knowledge_units.parquet
    carrier_knowledge_unit_edges.parquet
    knowledge_unit_summary.json
    run_config.json

Main audit outputs:

    knowledge_units_with_audit_flags.parquet
    knowledge_unit_modality_summary.csv
    knowledge_unit_pair_type_summary.csv
    knowledge_unit_pair_type_by_modality.csv
    knowledge_unit_entity_hub_summary.csv
    carrier_knowledge_unit_edge_summary.csv
    carrier_knowledge_unit_year_summary.csv
    top_knowledge_units.csv
    top_paper_only_knowledge_units.csv
    top_patent_supported_knowledge_units.csv
    top_trial_supported_knowledge_units.csv
    top_cross_modal_knowledge_units.csv
    knowledge_unit_audit_summary.json
    knowledge_unit_audit_report.md
    knowledge_unit_audit_manifest.csv

Interpretation note:

    paper_count / patent_count / trial_count are observed carrier counts within
    the current carrier graph. For a short-window patent-centered graph, lack of
    patent or trial evidence should not be interpreted as absence of historical
    translation in the real world.

Example:

    python scripts/20_audit_knowledge_units.py \
      --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --top-n 100
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.knowledge_units import (  # noqa: E402
    json_safe,
    markdown_table,
    write_dataframe,
    write_json,
)


CORE_KU_COLUMNS = [
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
    "evidence_pattern",
    "modality_count",
    "observed_status",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Audit an entity-pair Knowledge Unit dataset."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing knowledge_units.parquet and carrier_knowledge_unit_edges.parquet.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for audit outputs. Defaults to --input-dir. "
            "Use a subdirectory if you want to keep audit files separate."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top KU rows to write for each top table. Default: 100.",
    )

    parser.add_argument(
        "--hub-top-n",
        type=int,
        default=500,
        help="Number of top entity hub rows to include in report snippets. Default: 500.",
    )

    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Sequence[str], *, table_name: str) -> None:
    """Validate required columns."""

    missing = set(columns) - set(frame.columns)

    if missing:
        raise ValueError(
            f"{table_name} missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """Read JSON if it exists, else return empty dict."""

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_modality_flags(ku: pd.DataFrame) -> pd.DataFrame:
    """Add modality and observed-status flags to Knowledge Unit table."""

    out = ku.copy()

    for column in ["paper_count", "patent_count", "trial_count", "carrier_count"]:
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    has_paper = out["paper_count"] > 0
    has_patent = out["patent_count"] > 0
    has_trial = out["trial_count"] > 0

    out["has_paper_evidence"] = has_paper
    out["has_patent_evidence"] = has_patent
    out["has_trial_evidence"] = has_trial
    out["has_translation_evidence"] = has_patent | has_trial

    out["modality_count"] = (
        has_paper.astype(int) + has_patent.astype(int) + has_trial.astype(int)
    )

    out["evidence_pattern"] = (
        np.where(has_paper, "P", "")
        + np.where(has_patent, "A", "")
        + np.where(has_trial, "T", "")
    )
    out["evidence_pattern"] = out["evidence_pattern"].replace("", "none")

    out["is_paper_only"] = has_paper & ~has_patent & ~has_trial
    out["is_patent_only"] = ~has_paper & has_patent & ~has_trial
    out["is_trial_only"] = ~has_paper & ~has_patent & has_trial
    out["is_paper_patent"] = has_paper & has_patent & ~has_trial
    out["is_paper_trial"] = has_paper & ~has_patent & has_trial
    out["is_patent_trial"] = ~has_paper & has_patent & has_trial
    out["is_all_modalities"] = has_paper & has_patent & has_trial
    out["is_cross_modal"] = out["modality_count"] >= 2

    conditions = [
        out["is_paper_only"],
        out["is_patent_only"],
        out["is_trial_only"],
        out["is_paper_patent"],
        out["is_paper_trial"],
        out["is_patent_trial"],
        out["is_all_modalities"],
    ]

    choices = [
        "paper_only",
        "patent_only",
        "trial_only",
        "paper_patent",
        "paper_trial",
        "patent_trial",
        "paper_patent_trial",
    ]

    out["observed_status"] = np.select(conditions, choices, default="none")

    return out


def summarize_modality(ku: pd.DataFrame) -> pd.DataFrame:
    """Summarize KU evidence modality patterns."""

    if ku.empty:
        return pd.DataFrame(
            columns=[
                "evidence_pattern",
                "observed_status",
                "modality_count",
                "ku_count",
                "ku_fraction",
                "carrier_count_sum",
                "median_carrier_count",
                "max_carrier_count",
                "paper_count_sum",
                "patent_count_sum",
                "trial_count_sum",
                "first_year_min",
                "last_year_max",
            ]
        )

    total = len(ku)

    summary = (
        ku.groupby(["evidence_pattern", "observed_status", "modality_count"], as_index=False)
        .agg(
            ku_count=("ku_id", "nunique"),
            carrier_count_sum=("carrier_count", "sum"),
            median_carrier_count=("carrier_count", "median"),
            max_carrier_count=("carrier_count", "max"),
            paper_count_sum=("paper_count", "sum"),
            patent_count_sum=("patent_count", "sum"),
            trial_count_sum=("trial_count", "sum"),
            first_year_min=("first_year", "min"),
            last_year_max=("last_year", "max"),
        )
        .sort_values(["modality_count", "ku_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    summary["ku_fraction"] = summary["ku_count"] / total

    ordered = [
        "evidence_pattern",
        "observed_status",
        "modality_count",
        "ku_count",
        "ku_fraction",
        "carrier_count_sum",
        "median_carrier_count",
        "max_carrier_count",
        "paper_count_sum",
        "patent_count_sum",
        "trial_count_sum",
        "first_year_min",
        "last_year_max",
    ]

    return summary[ordered]


def summarize_pair_types(ku: pd.DataFrame) -> pd.DataFrame:
    """Summarize KUs by pair type."""

    if ku.empty:
        return pd.DataFrame()

    summary = (
        ku.groupby("pair_type", as_index=False)
        .agg(
            ku_count=("ku_id", "nunique"),
            carrier_count_sum=("carrier_count", "sum"),
            median_carrier_count=("carrier_count", "median"),
            max_carrier_count=("carrier_count", "max"),
            paper_count_sum=("paper_count", "sum"),
            patent_count_sum=("patent_count", "sum"),
            trial_count_sum=("trial_count", "sum"),
            paper_only_ku=("is_paper_only", "sum"),
            patent_supported_ku=("has_patent_evidence", "sum"),
            trial_supported_ku=("has_trial_evidence", "sum"),
            cross_modal_ku=("is_cross_modal", "sum"),
        )
        .sort_values(["ku_count", "carrier_count_sum"], ascending=[False, False])
        .reset_index(drop=True)
    )

    summary["paper_only_fraction"] = summary["paper_only_ku"] / summary["ku_count"]
    summary["patent_supported_fraction"] = summary["patent_supported_ku"] / summary["ku_count"]
    summary["trial_supported_fraction"] = summary["trial_supported_ku"] / summary["ku_count"]
    summary["cross_modal_fraction"] = summary["cross_modal_ku"] / summary["ku_count"]

    return summary


def summarize_pair_type_by_modality(ku: pd.DataFrame) -> pd.DataFrame:
    """Summarize pair type distribution by evidence modality."""

    if ku.empty:
        return pd.DataFrame()

    summary = (
        ku.groupby(["evidence_pattern", "observed_status", "pair_type"], as_index=False)
        .agg(
            ku_count=("ku_id", "nunique"),
            carrier_count_sum=("carrier_count", "sum"),
            median_carrier_count=("carrier_count", "median"),
            max_carrier_count=("carrier_count", "max"),
            paper_count_sum=("paper_count", "sum"),
            patent_count_sum=("patent_count", "sum"),
            trial_count_sum=("trial_count", "sum"),
        )
        .sort_values(
            ["evidence_pattern", "ku_count", "carrier_count_sum"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )

    totals = summary.groupby("evidence_pattern")["ku_count"].transform("sum")
    summary["fraction_within_evidence_pattern"] = summary["ku_count"] / totals

    return summary


def make_entity_long_table(ku: pd.DataFrame) -> pd.DataFrame:
    """Convert KU table into one row per participating entity per KU."""

    common_cols = [
        "ku_id",
        "pair_type",
        "carrier_count",
        "paper_count",
        "patent_count",
        "trial_count",
        "total_pair_weight",
        "evidence_pattern",
        "observed_status",
        "modality_count",
        "is_cross_modal",
        "has_translation_evidence",
    ]

    left = ku[
        common_cols
        + [
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
        common_cols
        + [
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

    entities = pd.concat([left, right], ignore_index=True)

    return entities


def summarize_entity_hubs(ku: pd.DataFrame) -> pd.DataFrame:
    """Summarize high-degree entities in the KU graph."""

    if ku.empty:
        return pd.DataFrame()

    entities = make_entity_long_table(ku)

    summary = (
        entities.groupby(["entity_id", "entity_name", "entity_type"], as_index=False)
        .agg(
            ku_degree=("ku_id", "nunique"),
            carrier_count_sum=("carrier_count", "sum"),
            median_carrier_count=("carrier_count", "median"),
            max_carrier_count=("carrier_count", "max"),
            paper_count_sum=("paper_count", "sum"),
            patent_count_sum=("patent_count", "sum"),
            trial_count_sum=("trial_count", "sum"),
            cross_modal_ku=("is_cross_modal", "sum"),
            translation_supported_ku=("has_translation_evidence", "sum"),
        )
        .sort_values(["ku_degree", "carrier_count_sum"], ascending=[False, False])
        .reset_index(drop=True)
    )

    total_ku = len(ku)
    summary["ku_degree_fraction"] = summary["ku_degree"] / total_ku if total_ku else 0.0

    return summary


def summarize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize carrier-KU edges by carrier type."""

    if edges.empty:
        return pd.DataFrame()

    out = (
        edges.groupby("carrier_type", as_index=False)
        .agg(
            carrier_ku_edge_count=("ku_id", "size"),
            ku_count=("ku_id", "nunique"),
            carrier_count=("carrier_key", "nunique"),
            pair_weight_sum=("pair_weight", "sum"),
            pair_weight_mean=("pair_weight", "mean"),
            year_non_null=("year", lambda s: int(s.notna().sum())),
            missing_year_count=("year", lambda s: int(s.isna().sum())),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .sort_values("carrier_ku_edge_count", ascending=False)
        .reset_index(drop=True)
    )

    return out


def summarize_edges_by_year(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize carrier-KU edges by carrier type and year."""

    if edges.empty or "year" not in edges.columns:
        return pd.DataFrame()

    dated = edges[edges["year"].notna()].copy()

    if dated.empty:
        return pd.DataFrame(
            columns=[
                "carrier_type",
                "year",
                "carrier_ku_edge_count",
                "ku_count",
                "carrier_count",
                "pair_weight_sum",
            ]
        )

    dated["year"] = dated["year"].astype(int)

    out = (
        dated.groupby(["carrier_type", "year"], as_index=False)
        .agg(
            carrier_ku_edge_count=("ku_id", "size"),
            ku_count=("ku_id", "nunique"),
            carrier_count=("carrier_key", "nunique"),
            pair_weight_sum=("pair_weight", "sum"),
        )
        .sort_values(["carrier_type", "year"])
        .reset_index(drop=True)
    )

    return out


def top_ku_table(ku: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Return top KUs by support."""

    existing_cols = [column for column in CORE_KU_COLUMNS if column in ku.columns]

    return (
        ku.sort_values(
            ["carrier_count", "total_pair_weight", "ku_id"],
            ascending=[False, False, True],
        )
        .head(top_n)[existing_cols]
        .reset_index(drop=True)
    )


def select_top_subsets(ku: pd.DataFrame, *, top_n: int) -> dict[str, pd.DataFrame]:
    """Build top-KU subset tables."""

    subsets = {
        "top_knowledge_units": ku,
        "top_paper_only_knowledge_units": ku[ku["is_paper_only"]],
        "top_patent_supported_knowledge_units": ku[ku["has_patent_evidence"]],
        "top_trial_supported_knowledge_units": ku[ku["has_trial_evidence"]],
        "top_cross_modal_knowledge_units": ku[ku["is_cross_modal"]],
        "top_translation_supported_knowledge_units": ku[ku["has_translation_evidence"]],
    }

    return {name: top_ku_table(frame, top_n=top_n) for name, frame in subsets.items()}


def artifact_row(
    artifact: str,
    path: Path,
    description: str,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Create manifest row."""

    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_audit_summary(
    *,
    ku: pd.DataFrame,
    edges: pd.DataFrame,
    modality_summary: pd.DataFrame,
    pair_type_summary: pd.DataFrame,
    edge_summary: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    top_n: int,
    run_metadata: Mapping[str, Any],
    build_summary: Mapping[str, Any],
    build_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build JSON-safe audit summary."""

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "top_n": int(top_n),
        "knowledge_unit_count": int(len(ku)),
        "carrier_ku_edge_count": int(len(edges)),
        "pair_type_count": int(ku["pair_type"].nunique()) if "pair_type" in ku.columns else 0,
        "carrier_count": int(edges["carrier_key"].nunique()) if "carrier_key" in edges.columns else None,
        "paper_only_ku_count": int(ku["is_paper_only"].sum()),
        "patent_supported_ku_count": int(ku["has_patent_evidence"].sum()),
        "trial_supported_ku_count": int(ku["has_trial_evidence"].sum()),
        "translation_supported_ku_count": int(ku["has_translation_evidence"].sum()),
        "cross_modal_ku_count": int(ku["is_cross_modal"].sum()),
        "all_modalities_ku_count": int(ku["is_all_modalities"].sum()),
        "modality_summary": modality_summary.to_dict("records"),
        "pair_type_summary": pair_type_summary.to_dict("records"),
        "edge_summary": edge_summary.to_dict("records"),
        "run_metadata": dict(run_metadata),
        "build_summary": dict(build_summary),
        "build_config": dict(build_config),
        "interpretation_note": (
            "paper_count, patent_count, and trial_count are observed support counts "
            "within the current carrier graph, not all-history real-world counts."
        ),
    }

    if "first_year" in ku.columns and "last_year" in ku.columns:
        summary["first_year_min"] = json_safe(ku["first_year"].min())
        summary["last_year_max"] = json_safe(ku["last_year"].max())

    return json_safe(summary)


def build_report(
    *,
    audit_summary: Mapping[str, Any],
    modality_summary: pd.DataFrame,
    pair_type_summary: pd.DataFrame,
    pair_type_by_modality: pd.DataFrame,
    entity_hub_summary: pd.DataFrame,
    edge_summary: pd.DataFrame,
    top_tables: Mapping[str, pd.DataFrame],
    top_n_report: int = 25,
) -> str:
    """Build Markdown audit report."""

    summary_rows = [
        {"metric": "Knowledge Units", "value": audit_summary.get("knowledge_unit_count")},
        {"metric": "Carrier-KU edges", "value": audit_summary.get("carrier_ku_edge_count")},
        {"metric": "Pair types", "value": audit_summary.get("pair_type_count")},
        {"metric": "Carrier count", "value": audit_summary.get("carrier_count")},
        {"metric": "Paper-only KUs", "value": audit_summary.get("paper_only_ku_count")},
        {"metric": "Patent-supported KUs", "value": audit_summary.get("patent_supported_ku_count")},
        {"metric": "Trial-supported KUs", "value": audit_summary.get("trial_supported_ku_count")},
        {"metric": "Translation-supported KUs", "value": audit_summary.get("translation_supported_ku_count")},
        {"metric": "Cross-modal KUs", "value": audit_summary.get("cross_modal_ku_count")},
        {"metric": "All-modality KUs", "value": audit_summary.get("all_modalities_ku_count")},
        {"metric": "First year min", "value": audit_summary.get("first_year_min")},
        {"metric": "Last year max", "value": audit_summary.get("last_year_max")},
    ]

    top_cols = [column for column in CORE_KU_COLUMNS if column not in {"ku_id"}]

    hub_cols = [
        "entity_name",
        "entity_type",
        "ku_degree",
        "ku_degree_fraction",
        "carrier_count_sum",
        "paper_count_sum",
        "patent_count_sum",
        "trial_count_sum",
        "cross_modal_ku",
        "translation_supported_ku",
    ]

    pair_type_modality_cols = [
        "evidence_pattern",
        "observed_status",
        "pair_type",
        "ku_count",
        "fraction_within_evidence_pattern",
        "carrier_count_sum",
        "paper_count_sum",
        "patent_count_sum",
        "trial_count_sum",
    ]

    return "\n".join(
        [
            "# Knowledge Unit Audit Report",
            "",
            "## 1. Summary",
            "",
            markdown_table(summary_rows, ["metric", "value"]),
            "",
            "## 2. Interpretation note",
            "",
            (
                "`paper_count`, `patent_count`, and `trial_count` are observed "
                "carrier counts within the current carrier graph. In a short-window "
                "patent-centered graph, `patent_count=0` or `trial_count=0` should "
                "not be interpreted as absence of historical real-world translation."
            ),
            "",
            "Evidence pattern notation:",
            "",
            "- `P`: paper evidence observed",
            "- `A`: patent/application evidence observed",
            "- `T`: trial evidence observed",
            "",
            "## 3. Evidence modality summary",
            "",
            markdown_table(
                modality_summary.to_dict("records"),
                [
                    "evidence_pattern",
                    "observed_status",
                    "modality_count",
                    "ku_count",
                    "ku_fraction",
                    "carrier_count_sum",
                    "median_carrier_count",
                    "max_carrier_count",
                    "paper_count_sum",
                    "patent_count_sum",
                    "trial_count_sum",
                ],
            ),
            "",
            "## 4. Pair type summary",
            "",
            markdown_table(
                pair_type_summary.to_dict("records"),
                [
                    "pair_type",
                    "ku_count",
                    "carrier_count_sum",
                    "median_carrier_count",
                    "max_carrier_count",
                    "paper_only_ku",
                    "patent_supported_ku",
                    "trial_supported_ku",
                    "cross_modal_ku",
                ],
            ),
            "",
            "## 5. Pair type by modality",
            "",
            markdown_table(
                pair_type_by_modality.head(50).to_dict("records"),
                pair_type_modality_cols,
            ),
            "",
            "## 6. Carrier-KU edge summary",
            "",
            markdown_table(
                edge_summary.to_dict("records"),
                [
                    "carrier_type",
                    "carrier_ku_edge_count",
                    "ku_count",
                    "carrier_count",
                    "pair_weight_sum",
                    "pair_weight_mean",
                    "year_non_null",
                    "missing_year_count",
                    "year_min",
                    "year_max",
                ],
            ),
            "",
            "## 7. Top entity hubs",
            "",
            markdown_table(
                entity_hub_summary.head(top_n_report).to_dict("records"),
                hub_cols,
            ),
            "",
            "## 8. Top Knowledge Units",
            "",
            markdown_table(
                top_tables["top_knowledge_units"].head(top_n_report).to_dict("records"),
                top_cols,
            ),
            "",
            "## 9. Top paper-only Knowledge Units",
            "",
            markdown_table(
                top_tables["top_paper_only_knowledge_units"]
                .head(top_n_report)
                .to_dict("records"),
                top_cols,
            ),
            "",
            "## 10. Top patent-supported Knowledge Units",
            "",
            markdown_table(
                top_tables["top_patent_supported_knowledge_units"]
                .head(top_n_report)
                .to_dict("records"),
                top_cols,
            ),
            "",
            "## 11. Top trial-supported Knowledge Units",
            "",
            markdown_table(
                top_tables["top_trial_supported_knowledge_units"]
                .head(top_n_report)
                .to_dict("records"),
                top_cols,
            ),
            "",
            "## 12. Top cross-modal Knowledge Units",
            "",
            markdown_table(
                top_tables["top_cross_modal_knowledge_units"]
                .head(top_n_report)
                .to_dict("records"),
                top_cols,
            ),
            "",
        ]
    )


def main() -> None:
    """Entry point."""

    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    top_n = int(args.top_n)

    if top_n <= 0:
        raise ValueError("--top-n must be positive.")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ku_path = input_dir / "knowledge_units.parquet"
    edges_path = input_dir / "carrier_knowledge_unit_edges.parquet"

    if not ku_path.exists():
        raise FileNotFoundError(f"Missing input file: {ku_path}")

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing input file: {edges_path}")

    print()
    print("Auditing Knowledge Unit dataset...")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    ku = pd.read_parquet(ku_path)
    edges = pd.read_parquet(edges_path)

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
        table_name="knowledge_units.parquet",
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
        table_name="carrier_knowledge_unit_edges.parquet",
    )

    ku = add_modality_flags(ku)

    modality_summary = summarize_modality(ku)
    pair_type_summary = summarize_pair_types(ku)
    pair_type_by_modality = summarize_pair_type_by_modality(ku)
    entity_hub_summary = summarize_entity_hubs(ku)
    edge_summary = summarize_edges(edges)
    edge_year_summary = summarize_edges_by_year(edges)
    top_tables = select_top_subsets(ku, top_n=top_n)

    build_summary = read_json_if_exists(input_dir / "knowledge_unit_summary.json")
    build_config = read_json_if_exists(input_dir / "run_config.json")

    run_metadata = {
        "script": "scripts/20_audit_knowledge_units.py",
        "command": " ".join(sys.argv),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "top_n": top_n,
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_unix": time.time(),
    }

    audit_summary = build_audit_summary(
        ku=ku,
        edges=edges,
        modality_summary=modality_summary,
        pair_type_summary=pair_type_summary,
        edge_summary=edge_summary,
        input_dir=input_dir,
        output_dir=output_dir,
        top_n=top_n,
        run_metadata=run_metadata,
        build_summary=build_summary,
        build_config=build_config,
    )

    artifacts: list[dict[str, Any]] = []

    def write_artifact(
        artifact: str,
        frame_or_data: pd.DataFrame | Mapping[str, Any] | str,
        filename: str,
        description: str,
    ) -> None:
        path = output_dir / filename

        if isinstance(frame_or_data, pd.DataFrame):
            write_dataframe(frame_or_data, path)
            row_count = len(frame_or_data)
        elif isinstance(frame_or_data, str):
            write_text(path, frame_or_data)
            row_count = None
        else:
            write_json(path, frame_or_data)
            row_count = None

        artifacts.append(
            artifact_row(
                artifact,
                path,
                description,
                row_count=row_count,
            )
        )

    write_artifact(
        "knowledge_units_with_audit_flags",
        ku,
        "knowledge_units_with_audit_flags.parquet",
        "Knowledge Unit table augmented with evidence modality flags.",
    )

    write_artifact(
        "knowledge_unit_modality_summary",
        modality_summary,
        "knowledge_unit_modality_summary.csv",
        "Knowledge Unit summary by observed evidence modality pattern.",
    )

    write_artifact(
        "knowledge_unit_pair_type_summary",
        pair_type_summary,
        "knowledge_unit_pair_type_summary.csv",
        "Knowledge Unit summary by pair type.",
    )

    write_artifact(
        "knowledge_unit_pair_type_by_modality",
        pair_type_by_modality,
        "knowledge_unit_pair_type_by_modality.csv",
        "Knowledge Unit pair type distribution by evidence modality.",
    )

    write_artifact(
        "knowledge_unit_entity_hub_summary",
        entity_hub_summary,
        "knowledge_unit_entity_hub_summary.csv",
        "Entity-level hub summary over the KU graph.",
    )

    write_artifact(
        "carrier_knowledge_unit_edge_summary",
        edge_summary,
        "carrier_knowledge_unit_edge_summary.csv",
        "Carrier-KU edge summary by carrier type.",
    )

    write_artifact(
        "carrier_knowledge_unit_year_summary",
        edge_year_summary,
        "carrier_knowledge_unit_year_summary.csv",
        "Dated carrier-KU edge summary by carrier type and year.",
    )

    for artifact, frame in top_tables.items():
        write_artifact(
            artifact,
            frame,
            f"{artifact}.csv",
            f"Top {top_n} rows for {artifact.replace('_', ' ')}.",
        )

    write_artifact(
        "knowledge_unit_audit_summary",
        audit_summary,
        "knowledge_unit_audit_summary.json",
        "JSON summary for Knowledge Unit audit.",
    )

    report = build_report(
        audit_summary=audit_summary,
        modality_summary=modality_summary,
        pair_type_summary=pair_type_summary,
        pair_type_by_modality=pair_type_by_modality,
        entity_hub_summary=entity_hub_summary,
        edge_summary=edge_summary,
        top_tables=top_tables,
        top_n_report=min(25, top_n),
    )

    write_artifact(
        "knowledge_unit_audit_report",
        report,
        "knowledge_unit_audit_report.md",
        "Markdown Knowledge Unit audit report.",
    )

    manifest_path = output_dir / "knowledge_unit_audit_manifest.csv"
    write_dataframe(pd.DataFrame(artifacts), manifest_path)

    artifacts.append(
        artifact_row(
            "knowledge_unit_audit_manifest",
            manifest_path,
            "Manifest of Knowledge Unit audit artifacts.",
            row_count=len(artifacts),
        )
    )

    write_dataframe(pd.DataFrame(artifacts), manifest_path)

    print("Knowledge Unit audit complete.")
    print()
    print(f"Knowledge Units:              {audit_summary['knowledge_unit_count']}")
    print(f"Carrier-KU edges:             {audit_summary['carrier_ku_edge_count']}")
    print(f"Paper-only KUs:               {audit_summary['paper_only_ku_count']}")
    print(f"Patent-supported KUs:         {audit_summary['patent_supported_ku_count']}")
    print(f"Trial-supported KUs:          {audit_summary['trial_supported_ku_count']}")
    print(f"Translation-supported KUs:    {audit_summary['translation_supported_ku_count']}")
    print(f"Cross-modal KUs:              {audit_summary['cross_modal_ku_count']}")
    print()
    print("Main outputs:")
    print(f"  {output_dir / 'knowledge_unit_audit_report.md'}")
    print(f"  {output_dir / 'knowledge_unit_modality_summary.csv'}")
    print(f"  {output_dir / 'knowledge_unit_entity_hub_summary.csv'}")
    print(f"  {output_dir / 'knowledge_unit_audit_summary.json'}")
    print(f"  {manifest_path}")
    print()


if __name__ == "__main__":
    main()