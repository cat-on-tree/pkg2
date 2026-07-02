#!/usr/bin/env python
"""Sanity checks for Knowledge Unit datasets.

This script validates a Knowledge Unit dataset after:

    scripts/20_build_knowledge_units.py
    scripts/21_audit_knowledge_units.py
    scripts/22_build_knowledge_unit_temporal_panel.py

It is intended for quick post-build checks, especially after changing
pair-type policies or canonicalization logic.

Recommended use:

    python scripts/check_knowledge_unit_sanity.py \
      --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
      --temporal-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel \
      --expected-pair-types chemical-disease gene-disease chemical-gene \
      --disallowed-pair-types disease-disease \
      --disallowed-entity-types species \
      --expect-no-duplicates \
      --expect-dense-temporal \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_KU_COLUMNS = [
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
]

REQUIRED_EDGE_COLUMNS = [
    "ku_id",
    "entity_a_id",
    "entity_a_name",
    "entity_a_type",
    "entity_b_id",
    "entity_b_name",
    "entity_b_type",
    "pair_type",
    "carrier_type",
    "carrier_id",
    "carrier_key",
    "year",
    "pair_weight",
]

DEFAULT_EXPECTED_PAIR_TYPES = [
    "chemical-disease",
    "gene-disease",
    "chemical-gene",
]

DEFAULT_DISALLOWED_PAIR_TYPES = [
    "disease-disease",
]

DEFAULT_DISALLOWED_ENTITY_TYPES = [
    "species",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sanity checks on a Knowledge Unit dataset."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Knowledge Unit dataset directory.",
    )

    parser.add_argument(
        "--temporal-dir",
        type=Path,
        default=None,
        help="Optional KU temporal panel output directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for sanity check artifacts. "
            "Defaults to <input-dir>/sanity_check."
        ),
    )

    parser.add_argument(
        "--expected-pair-types",
        nargs="*",
        default=DEFAULT_EXPECTED_PAIR_TYPES,
        help="Allowed pair types expected in knowledge_units.parquet.",
    )

    parser.add_argument(
        "--disallowed-pair-types",
        nargs="*",
        default=DEFAULT_DISALLOWED_PAIR_TYPES,
        help="Pair types that must not appear.",
    )

    parser.add_argument(
        "--disallowed-entity-types",
        nargs="*",
        default=DEFAULT_DISALLOWED_ENTITY_TYPES,
        help="Entity types that must not appear in either side of a KU.",
    )

    parser.add_argument(
        "--expect-no-duplicates",
        action="store_true",
        help="Fail sanity status if reversed / unordered duplicate KUs exist.",
    )

    parser.add_argument(
        "--expect-dense-temporal",
        action="store_true",
        help=(
            "If --temporal-dir is provided, check that panel rows equal "
            "ku_count * year_count."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite sanity check output directory if it exists.",
    )

    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if value is pd.NA:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    return value


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(dict(data)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output path: {path}")


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def require_columns(frame: pd.DataFrame, columns: list[str], table_name: str) -> list[str]:
    missing = [column for column in columns if column not in frame.columns]
    return missing


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not columns:
        return ""

    def fmt(value: Any) -> str:
        value = json_safe(value)

        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.6g}"

        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"

    body = [
        "| " + " | ".join(fmt(row.get(column)) for column in columns) + " |"
        for row in rows
    ]

    return "\n".join([header, sep, *body])


def build_unordered_pair_key(frame: pd.DataFrame) -> pd.Series:
    """Build an unordered entity-pair key independent of entity orientation."""

    a = frame["entity_a_id"].astype(str)
    b = frame["entity_b_id"].astype(str)

    left = np.where(a <= b, a, b)
    right = np.where(a <= b, b, a)

    return frame["pair_type"].astype(str) + "::" + left.astype(str) + "::" + right.astype(str)


def audit_reversed_duplicates(ku: pd.DataFrame) -> pd.DataFrame:
    work = ku.copy()
    work["unordered_pair_key"] = build_unordered_pair_key(work)

    dup = (
        work.groupby("unordered_pair_key", as_index=False)
        .agg(
            n_rows=("ku_id", "size"),
            pair_type=("pair_type", "first"),
            carrier_count_sum=("carrier_count", "sum"),
            max_carrier_count=("carrier_count", "max"),
            ku_ids=("ku_id", lambda values: "|".join(sorted(map(str, values)))),
            examples=(
                "ku_id",
                lambda values: "|".join(sorted(map(str, values))[:5]),
            ),
        )
    )

    dup = dup[dup["n_rows"] > 1].sort_values(
        ["n_rows", "carrier_count_sum"],
        ascending=[False, False],
    )

    return dup.reset_index(drop=True)


def audit_invalid_edges(ku: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    valid_ids = set(ku["ku_id"].astype(str))
    invalid = edges[~edges["ku_id"].astype(str).isin(valid_ids)].copy()
    return invalid.reset_index(drop=True)


def aggregate_counts_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    counts = (
        edges.groupby("ku_id", as_index=False)
        .agg(
            carrier_count_from_edges=("carrier_key", "nunique"),
            total_pair_weight_from_edges=("pair_weight", "sum"),
            first_year_from_edges=("year", "min"),
            last_year_from_edges=("year", "max"),
        )
    )

    modality_counts = (
        edges.groupby(["ku_id", "carrier_type"])["carrier_key"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    for modality in ["paper", "patent", "trial"]:
        if modality not in modality_counts.columns:
            modality_counts[modality] = 0

    modality_counts = modality_counts.rename(
        columns={
            "paper": "paper_count_from_edges",
            "patent": "patent_count_from_edges",
            "trial": "trial_count_from_edges",
        }
    )

    out = counts.merge(
        modality_counts[
            [
                "ku_id",
                "paper_count_from_edges",
                "patent_count_from_edges",
                "trial_count_from_edges",
            ]
        ],
        on="ku_id",
        how="left",
    )

    return out


def audit_ku_edge_count_consistency(ku: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_counts_from_edges(edges)

    cols = [
        "ku_id",
        "carrier_count",
        "paper_count",
        "patent_count",
        "trial_count",
        "first_year",
        "last_year",
    ]
    existing_cols = [column for column in cols if column in ku.columns]

    merged = ku[existing_cols].merge(agg, on="ku_id", how="left")

    for column in [
        "carrier_count_from_edges",
        "paper_count_from_edges",
        "patent_count_from_edges",
        "trial_count_from_edges",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0).astype(int)

    mismatch_mask = pd.Series(False, index=merged.index)

    compare_pairs = [
        ("carrier_count", "carrier_count_from_edges"),
        ("paper_count", "paper_count_from_edges"),
        ("patent_count", "patent_count_from_edges"),
        ("trial_count", "trial_count_from_edges"),
    ]

    for left, right in compare_pairs:
        if left in merged.columns and right in merged.columns:
            mismatch_mask |= (
                pd.to_numeric(merged[left], errors="coerce").fillna(-1).astype(int)
                != pd.to_numeric(merged[right], errors="coerce").fillna(-2).astype(int)
            )

    mismatches = merged[mismatch_mask].copy()
    return mismatches.reset_index(drop=True)


def summarize_pair_types(ku: pd.DataFrame) -> pd.DataFrame:
    return (
        ku.groupby("pair_type", as_index=False)
        .agg(
            ku_count=("ku_id", "nunique"),
            carrier_count_sum=("carrier_count", "sum"),
            paper_count_sum=("paper_count", "sum"),
            patent_count_sum=("patent_count", "sum"),
            trial_count_sum=("trial_count", "sum"),
            median_carrier_count=("carrier_count", "median"),
            max_carrier_count=("carrier_count", "max"),
        )
        .sort_values(["ku_count", "carrier_count_sum"], ascending=[False, False])
        .reset_index(drop=True)
    )


def summarize_modalities(ku: pd.DataFrame) -> pd.DataFrame:
    work = ku.copy()

    work["evidence_pattern"] = (
        np.where(work["paper_count"] > 0, "P", "")
        + np.where(work["patent_count"] > 0, "A", "")
        + np.where(work["trial_count"] > 0, "T", "")
    )

    work["evidence_pattern"] = work["evidence_pattern"].replace("", "none")

    summary = (
        work.groupby("evidence_pattern", as_index=False)
        .agg(
            ku_count=("ku_id", "nunique"),
            ku_fraction=("ku_id", lambda s: len(s) / len(work) if len(work) else 0),
            carrier_count_sum=("carrier_count", "sum"),
            paper_count_sum=("paper_count", "sum"),
            patent_count_sum=("patent_count", "sum"),
            trial_count_sum=("trial_count", "sum"),
            median_carrier_count=("carrier_count", "median"),
            max_carrier_count=("carrier_count", "max"),
        )
    )

    order = {"P": 1, "A": 2, "T": 3, "PA": 4, "PT": 5, "AT": 6, "PAT": 7, "none": 99}
    summary["_order"] = summary["evidence_pattern"].map(order).fillna(98)
    summary = summary.sort_values(["_order", "evidence_pattern"]).drop(columns=["_order"])

    return summary.reset_index(drop=True)


def check_temporal_outputs(
    *,
    ku: pd.DataFrame,
    temporal_dir: Path,
    expect_dense: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    temporal_summary_path = temporal_dir / "knowledge_unit_temporal_summary.json"
    temporal_features_path = temporal_dir / "knowledge_unit_temporal_features.parquet"
    year_summary_path = temporal_dir / "knowledge_unit_year_summary.csv"
    panel_path = temporal_dir / "knowledge_unit_year_panel.parquet"

    temporal_summary = load_json_if_exists(temporal_summary_path)

    result: dict[str, Any] = {
        "temporal_dir": str(temporal_dir),
        "has_temporal_summary": temporal_summary is not None,
        "has_temporal_features": temporal_features_path.exists(),
        "has_year_summary": year_summary_path.exists(),
        "has_year_panel": panel_path.exists(),
    }

    required_files = [
        temporal_summary_path,
        temporal_features_path,
        year_summary_path,
        panel_path,
    ]

    for path in required_files:
        checks.append(
            {
                "check": f"temporal_file_exists:{path.name}",
                "status": "PASS" if path.exists() else "FAIL",
                "detail": str(path),
            }
        )

    if temporal_summary is not None:
        result.update(
            {
                "temporal_knowledge_unit_count": temporal_summary.get("knowledge_unit_count"),
                "panel_row_count": temporal_summary.get("panel_row_count"),
                "temporal_feature_row_count": temporal_summary.get("temporal_feature_row_count"),
                "start_year": temporal_summary.get("start_year"),
                "end_year": temporal_summary.get("end_year"),
                "year_count": temporal_summary.get("year_count"),
                "dated_edge_count": temporal_summary.get("dated_edge_count"),
                "undated_edge_count": temporal_summary.get("undated_edge_count"),
                "dated_edge_fraction": temporal_summary.get("dated_edge_fraction"),
                "ku_with_dated_trial_evidence": temporal_summary.get(
                    "ku_with_dated_trial_evidence"
                ),
                "ku_with_undated_trial_evidence": temporal_summary.get(
                    "ku_with_undated_trial_evidence"
                ),
            }
        )

        checks.append(
            {
                "check": "temporal_ku_count_matches",
                "status": "PASS"
                if int(temporal_summary.get("knowledge_unit_count", -1)) == len(ku)
                else "FAIL",
                "detail": (
                    f"temporal={temporal_summary.get('knowledge_unit_count')}, "
                    f"ku={len(ku)}"
                ),
            }
        )

        if expect_dense:
            expected_rows = int(temporal_summary.get("knowledge_unit_count", 0)) * int(
                temporal_summary.get("year_count", 0)
            )
            actual_rows = int(temporal_summary.get("panel_row_count", -1))

            checks.append(
                {
                    "check": "dense_panel_row_count",
                    "status": "PASS" if actual_rows == expected_rows else "FAIL",
                    "detail": f"actual={actual_rows}, expected={expected_rows}",
                }
            )

    if temporal_features_path.exists():
        temporal_features = pd.read_parquet(temporal_features_path)
        result["temporal_features_rows_loaded"] = len(temporal_features)

        checks.append(
            {
                "check": "temporal_features_row_count_matches",
                "status": "PASS" if len(temporal_features) == len(ku) else "FAIL",
                "detail": f"temporal_features={len(temporal_features)}, ku={len(ku)}",
            }
        )

    return result, checks


def build_report(summary: dict[str, Any], checks: list[dict[str, Any]]) -> str:
    failed = [row for row in checks if row["status"] == "FAIL"]
    warn = [row for row in checks if row["status"] == "WARN"]

    overview_rows = [
        {"metric": "Overall status", "value": summary["overall_status"]},
        {"metric": "Knowledge Units", "value": summary["ku_count"]},
        {"metric": "Carrier-KU edges", "value": summary["edge_count"]},
        {"metric": "Pair types", "value": ", ".join(summary["observed_pair_types"])},
        {"metric": "Duplicate unordered pairs", "value": summary["duplicate_unordered_pair_count"]},
        {"metric": "Invalid edge rows", "value": summary["invalid_edge_count"]},
        {"metric": "Count mismatch rows", "value": summary["count_mismatch_count"]},
        {"metric": "Paper-only KUs", "value": summary["paper_only_ku_count"]},
        {"metric": "Patent-supported KUs", "value": summary["patent_supported_ku_count"]},
        {"metric": "Trial-supported KUs", "value": summary["trial_supported_ku_count"]},
        {"metric": "Translation-supported KUs", "value": summary["translation_supported_ku_count"]},
        {"metric": "Cross-modal KUs", "value": summary["cross_modal_ku_count"]},
    ]

    check_rows = checks

    failed_rows = failed[:50]
    warn_rows = warn[:50]

    lines = [
        "# Knowledge Unit Sanity Check Report",
        "",
        "## 1. Overview",
        "",
        markdown_table(overview_rows, ["metric", "value"]),
        "",
        "## 2. Check results",
        "",
        markdown_table(check_rows, ["check", "status", "detail"]),
        "",
    ]

    if failed_rows:
        lines.extend(
            [
                "## 3. Failed checks",
                "",
                markdown_table(failed_rows, ["check", "status", "detail"]),
                "",
            ]
        )

    if warn_rows:
        lines.extend(
            [
                "## 4. Warnings",
                "",
                markdown_table(warn_rows, ["check", "status", "detail"]),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "sanity_check")).resolve()

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    ku_path = input_dir / "knowledge_units.parquet"
    edges_path = input_dir / "carrier_knowledge_unit_edges.parquet"
    modality_summary_path = input_dir / "knowledge_unit_modality_summary.csv"
    build_summary_path = input_dir / "knowledge_unit_summary.json"

    if not ku_path.exists():
        raise FileNotFoundError(f"Missing KU table: {ku_path}")

    if not edges_path.exists():
        raise FileNotFoundError(f"Missing carrier-KU edge table: {edges_path}")

    ku = pd.read_parquet(ku_path)
    edges = pd.read_parquet(edges_path)

    checks: list[dict[str, Any]] = []

    missing_ku_cols = require_columns(ku, REQUIRED_KU_COLUMNS, "knowledge_units")
    missing_edge_cols = require_columns(edges, REQUIRED_EDGE_COLUMNS, "carrier_knowledge_unit_edges")

    checks.append(
        {
            "check": "knowledge_units_required_columns",
            "status": "PASS" if not missing_ku_cols else "FAIL",
            "detail": ",".join(missing_ku_cols) if missing_ku_cols else "all required columns present",
        }
    )

    checks.append(
        {
            "check": "carrier_knowledge_unit_edges_required_columns",
            "status": "PASS" if not missing_edge_cols else "FAIL",
            "detail": ",".join(missing_edge_cols) if missing_edge_cols else "all required columns present",
        }
    )

    if missing_ku_cols or missing_edge_cols:
        raise ValueError("Required columns are missing; cannot continue robust sanity checks.")

    expected_pair_types = set(args.expected_pair_types)
    disallowed_pair_types = set(args.disallowed_pair_types)
    disallowed_entity_types = set(args.disallowed_entity_types)

    observed_pair_types = set(ku["pair_type"].dropna().astype(str))
    unexpected_pair_types = sorted(observed_pair_types - expected_pair_types)
    missing_expected_pair_types = sorted(expected_pair_types - observed_pair_types)
    observed_disallowed_pair_types = sorted(observed_pair_types & disallowed_pair_types)

    checks.append(
        {
            "check": "only_expected_pair_types",
            "status": "PASS" if not unexpected_pair_types else "FAIL",
            "detail": f"unexpected={unexpected_pair_types}",
        }
    )

    checks.append(
        {
            "check": "expected_pair_types_present_or_absent_ok",
            "status": "PASS",
            "detail": f"missing_expected={missing_expected_pair_types}",
        }
    )

    checks.append(
        {
            "check": "no_disallowed_pair_types",
            "status": "PASS" if not observed_disallowed_pair_types else "FAIL",
            "detail": f"observed_disallowed={observed_disallowed_pair_types}",
        }
    )

    observed_entity_types = set(ku["entity_a_type"].dropna().astype(str)) | set(
        ku["entity_b_type"].dropna().astype(str)
    )
    observed_disallowed_entity_types = sorted(observed_entity_types & disallowed_entity_types)

    checks.append(
        {
            "check": "no_disallowed_entity_types",
            "status": "PASS" if not observed_disallowed_entity_types else "FAIL",
            "detail": f"observed_disallowed={observed_disallowed_entity_types}",
        }
    )

    duplicate_ku_id_count = int(ku["ku_id"].duplicated().sum())
    checks.append(
        {
            "check": "unique_ku_id",
            "status": "PASS" if duplicate_ku_id_count == 0 else "FAIL",
            "detail": f"duplicate_ku_id_count={duplicate_ku_id_count}",
        }
    )

    self_pair_count = int(
        (ku["entity_a_id"].astype(str) == ku["entity_b_id"].astype(str)).sum()
    )
    checks.append(
        {
            "check": "no_self_pairs",
            "status": "PASS" if self_pair_count == 0 else "FAIL",
            "detail": f"self_pair_count={self_pair_count}",
        }
    )

    duplicates = audit_reversed_duplicates(ku)
    duplicate_count = len(duplicates)
    duplicate_rows_total = int(duplicates["n_rows"].sum()) if duplicate_count else 0

    duplicate_status = "PASS" if duplicate_count == 0 else (
        "FAIL" if args.expect_no_duplicates else "WARN"
    )
    checks.append(
        {
            "check": "no_reversed_or_unordered_duplicate_pairs",
            "status": duplicate_status,
            "detail": (
                f"duplicate_unordered_pairs={duplicate_count}, "
                f"duplicate_rows_total={duplicate_rows_total}"
            ),
        }
    )

    invalid_edges = audit_invalid_edges(ku, edges)
    checks.append(
        {
            "check": "all_edges_reference_existing_ku",
            "status": "PASS" if len(invalid_edges) == 0 else "FAIL",
            "detail": f"invalid_edge_rows={len(invalid_edges)}",
        }
    )

    count_mismatches = audit_ku_edge_count_consistency(ku, edges)
    checks.append(
        {
            "check": "ku_counts_match_edges",
            "status": "PASS" if len(count_mismatches) == 0 else "FAIL",
            "detail": f"mismatch_rows={len(count_mismatches)}",
        }
    )

    pair_type_summary = summarize_pair_types(ku)
    modality_summary = summarize_modalities(ku)

    paper_only_ku_count = int(
        ((ku["paper_count"] > 0) & (ku["patent_count"] == 0) & (ku["trial_count"] == 0)).sum()
    )
    patent_supported_ku_count = int((ku["patent_count"] > 0).sum())
    trial_supported_ku_count = int((ku["trial_count"] > 0).sum())
    translation_supported_ku_count = int(
        ((ku["patent_count"] > 0) | (ku["trial_count"] > 0)).sum()
    )
    modality_count = (
        (ku["paper_count"] > 0).astype(int)
        + (ku["patent_count"] > 0).astype(int)
        + (ku["trial_count"] > 0).astype(int)
    )
    cross_modal_ku_count = int((modality_count >= 2).sum())

    build_summary = load_json_if_exists(build_summary_path)

    if build_summary is not None:
        checks.append(
            {
                "check": "summary_ku_count_matches",
                "status": "PASS"
                if int(build_summary.get("knowledge_unit_count", -1)) == len(ku)
                else "FAIL",
                "detail": (
                    f"summary={build_summary.get('knowledge_unit_count')}, "
                    f"actual={len(ku)}"
                ),
            }
        )

        checks.append(
            {
                "check": "summary_edge_count_matches",
                "status": "PASS"
                if int(build_summary.get("filtered_carrier_ku_edge_count", -1)) == len(edges)
                else "FAIL",
                "detail": (
                    f"summary={build_summary.get('filtered_carrier_ku_edge_count')}, "
                    f"actual={len(edges)}"
                ),
            }
        )

    if modality_summary_path.exists():
        external_modality_summary = pd.read_csv(modality_summary_path)
        checks.append(
            {
                "check": "audit_modality_summary_exists",
                "status": "PASS",
                "detail": str(modality_summary_path),
            }
        )

        if "ku_count" in external_modality_summary.columns:
            external_total = int(external_modality_summary["ku_count"].sum())
            checks.append(
                {
                    "check": "audit_modality_summary_total_matches",
                    "status": "PASS" if external_total == len(ku) else "FAIL",
                    "detail": f"audit_total={external_total}, actual={len(ku)}",
                }
            )
    else:
        checks.append(
            {
                "check": "audit_modality_summary_exists",
                "status": "WARN",
                "detail": f"missing={modality_summary_path}",
            }
        )

    temporal_result: dict[str, Any] | None = None
    if args.temporal_dir is not None:
        temporal_result, temporal_checks = check_temporal_outputs(
            ku=ku,
            temporal_dir=args.temporal_dir.resolve(),
            expect_dense=args.expect_dense_temporal,
        )
        checks.extend(temporal_checks)

    failed_checks = [row for row in checks if row["status"] == "FAIL"]
    warn_checks = [row for row in checks if row["status"] == "WARN"]

    summary: dict[str, Any] = {
        "overall_status": "PASS" if not failed_checks else "FAIL",
        "input_dir": str(input_dir),
        "temporal_dir": str(args.temporal_dir.resolve()) if args.temporal_dir else None,
        "output_dir": str(output_dir),
        "ku_count": int(len(ku)),
        "edge_count": int(len(edges)),
        "observed_pair_types": sorted(observed_pair_types),
        "expected_pair_types": sorted(expected_pair_types),
        "unexpected_pair_types": unexpected_pair_types,
        "missing_expected_pair_types": missing_expected_pair_types,
        "disallowed_pair_types": sorted(disallowed_pair_types),
        "observed_disallowed_pair_types": observed_disallowed_pair_types,
        "observed_entity_types": sorted(observed_entity_types),
        "disallowed_entity_types": sorted(disallowed_entity_types),
        "observed_disallowed_entity_types": observed_disallowed_entity_types,
        "duplicate_ku_id_count": duplicate_ku_id_count,
        "self_pair_count": self_pair_count,
        "duplicate_unordered_pair_count": duplicate_count,
        "duplicate_unordered_pair_rows_total": duplicate_rows_total,
        "invalid_edge_count": int(len(invalid_edges)),
        "count_mismatch_count": int(len(count_mismatches)),
        "paper_only_ku_count": paper_only_ku_count,
        "patent_supported_ku_count": patent_supported_ku_count,
        "trial_supported_ku_count": trial_supported_ku_count,
        "translation_supported_ku_count": translation_supported_ku_count,
        "cross_modal_ku_count": cross_modal_ku_count,
        "failed_check_count": len(failed_checks),
        "warning_check_count": len(warn_checks),
        "checks": checks,
    }

    if temporal_result is not None:
        summary["temporal"] = temporal_result

    write_json(output_dir / "knowledge_unit_sanity_summary.json", summary)
    write_dataframe(pair_type_summary, output_dir / "pair_type_summary_check.csv")
    write_dataframe(modality_summary, output_dir / "modality_summary_check.csv")
    write_dataframe(duplicates, output_dir / "reversed_pair_duplicates.csv")
    write_dataframe(invalid_edges.head(100_000), output_dir / "invalid_edges.csv")
    write_dataframe(count_mismatches.head(100_000), output_dir / "count_mismatches.csv")

    report = build_report(summary, checks)
    write_text(output_dir / "knowledge_unit_sanity_report.md", report)

    print()
    print("Knowledge Unit sanity check complete.")
    print()
    print(f"Overall status:              {summary['overall_status']}")
    print(f"Knowledge Units:             {summary['ku_count']}")
    print(f"Carrier-KU edges:            {summary['edge_count']}")
    print(f"Observed pair types:         {', '.join(summary['observed_pair_types'])}")
    print(f"Unexpected pair types:       {summary['unexpected_pair_types']}")
    print(f"Disallowed pair types:       {summary['observed_disallowed_pair_types']}")
    print(f"Disallowed entity types:     {summary['observed_disallowed_entity_types']}")
    print(f"Duplicate unordered pairs:   {summary['duplicate_unordered_pair_count']}")
    print(f"Invalid edge rows:           {summary['invalid_edge_count']}")
    print(f"Count mismatch rows:         {summary['count_mismatch_count']}")
    print(f"Paper-only KUs:              {summary['paper_only_ku_count']}")
    print(f"Patent-supported KUs:        {summary['patent_supported_ku_count']}")
    print(f"Trial-supported KUs:         {summary['trial_supported_ku_count']}")
    print(f"Translation-supported KUs:   {summary['translation_supported_ku_count']}")
    print(f"Cross-modal KUs:             {summary['cross_modal_ku_count']}")
    print(f"Failed checks:               {summary['failed_check_count']}")
    print(f"Warnings:                    {summary['warning_check_count']}")
    print()
    print("Outputs:")
    print(f"  {output_dir / 'knowledge_unit_sanity_summary.json'}")
    print(f"  {output_dir / 'knowledge_unit_sanity_report.md'}")
    print(f"  {output_dir / 'pair_type_summary_check.csv'}")
    print(f"  {output_dir / 'modality_summary_check.csv'}")
    print(f"  {output_dir / 'reversed_pair_duplicates.csv'}")
    print(f"  {output_dir / 'invalid_edges.csv'}")
    print(f"  {output_dir / 'count_mismatches.csv'}")
    print()

    if failed_checks:
        print("Failed checks:")
        for row in failed_checks:
            print(f"  - {row['check']}: {row['detail']}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()