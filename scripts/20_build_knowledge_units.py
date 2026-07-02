#!/usr/bin/env python
"""Build entity-pair Knowledge Units from a carrier graph.

This script implements Stage 04A:

    Knowledge Unit construction MVP

Current KU definition:

    Knowledge Unit = typed BioEntity pair

Main KU layer:

    The main translational KU layer focuses on typed BioEntity pairs that are
    more directly aligned with mechanisms, interventions, and indications:

        chemical-disease
        gene-disease
        chemical-gene

    Disease-disease pairs are intentionally excluded from the main layer because
    disease co-mentions can conflate comorbidity, complications, subtype
    hierarchy, near-synonymy, and background co-mention. Disease-disease pairs
    can be built later as a separate exploratory comorbidity layer.

Input:

    data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/

Output example:

    data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/

Core outputs:

    knowledge_units.parquet
    carrier_knowledge_unit_edges.parquet
    knowledge_unit_type_summary.csv
    knowledge_unit_summary.json
    knowledge_unit_report.md
    run_config.json
    dataset_manifest.csv

Important interpretation note:

    Carrier co-mention evidence does not imply an asserted biomedical relation.
    V1/V2 Knowledge Units are typed entity-pair co-mention units, not
    relation-level claims.

Typical run:

    python scripts/20_build_knowledge_units.py \
      --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated \
      --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
      --exclude-bioentity-types species \
      --pair-types chemical-disease gene-disease chemical-gene \
      --min-carrier-count 3 \
      --max-entities-per-carrier 0 \
      --overwrite

Dry run:

    python scripts/20_build_knowledge_units.py \
      --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated \
      --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_probe \
      --exclude-bioentity-types species \
      --pair-types chemical-disease gene-disease chemical-gene \
      --min-carrier-count 3 \
      --max-entities-per-carrier 0 \
      --dry-run \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.knowledge_units import (  # noqa: E402
    DEFAULT_EXCLUDED_BIOENTITY_TYPES,
    DEFAULT_PAIR_TYPES,
    KnowledgeUnitBuildConfig,
    build_knowledge_unit_report,
    build_knowledge_units_from_graph_dir,
    filter_mentions_for_knowledge_units,
    allowed_entity_types_from_pair_types,
    json_safe,
    load_standardized_carrier_bioentity_mentions,
    markdown_table,
    write_dataframe,
    write_json,
    write_knowledge_unit_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build typed entity-pair Knowledge Units from carrier-BioEntity "
            "mention edges."
        )
    )

    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Processed carrier graph directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output Knowledge Unit dataset directory.",
    )

    parser.add_argument(
        "--carrier-types",
        nargs="*",
        default=["paper", "patent", "trial"],
        help="Carrier types to use. Default: paper patent trial.",
    )

    parser.add_argument(
        "--exclude-bioentity-types",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_BIOENTITY_TYPES),
        help="BioEntity types to exclude. Default: species.",
    )

    parser.add_argument(
        "--pair-types",
        nargs="+",
        default=list(DEFAULT_PAIR_TYPES),
        help=(
            "Allowed typed BioEntity pair types. "
            "Default is the main translational KU layer from "
            "pkg2.knowledge_units.DEFAULT_PAIR_TYPES, typically: "
            "chemical-disease gene-disease chemical-gene. "
            "Disease-disease is intentionally excluded from the main layer; "
            "build it separately if doing exploratory comorbidity analysis."
        ),
    )

    parser.add_argument(
        "--min-carrier-count",
        type=int,
        default=3,
        help="Minimum number of distinct carriers supporting a KU. Default: 3.",
    )

    parser.add_argument(
        "--min-paper-count",
        type=int,
        default=0,
        help="Minimum number of paper carriers supporting a KU. Default: 0.",
    )

    parser.add_argument(
        "--min-patent-count",
        type=int,
        default=0,
        help="Minimum number of patent carriers supporting a KU. Default: 0.",
    )

    parser.add_argument(
        "--min-trial-count",
        type=int,
        default=0,
        help="Minimum number of trial carriers supporting a KU. Default: 0.",
    )

    parser.add_argument(
        "--pair-weight-method",
        default="sqrt_product",
        choices=["sqrt_product", "min", "product", "sum", "mean"],
        help="Pair evidence weight function. Default: sqrt_product.",
    )

    parser.add_argument(
        "--max-entities-per-carrier",
        type=int,
        default=0,
        help=(
            "Maximum number of BioEntities per carrier used for pair generation, "
            "after sorting by mention_count. Use <=0 to disable. Default: 0, "
            "meaning no cap."
        ),
    )

    parser.add_argument(
        "--drop-unyearled-edges",
        action="store_true",
        help="Drop mention edges whose year cannot be inferred.",
    )

    parser.add_argument(
        "--ku-id-prefix",
        default="KU",
        help="Prefix for stable Knowledge Unit IDs. Default: KU.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and inspect inputs, but do not build final KU tables.",
    )

    parser.add_argument(
        "--write-probe-tables",
        action="store_true",
        help="During dry-run, write mention summary probe tables.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top KUs shown in console/report summaries. Default: 25.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    return parser.parse_args()


def prepare_output_dir(output_dir: Path, *, overwrite: bool = False) -> Path:
    """Prepare output directory."""

    output_dir = Path(output_dir)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def local_artifact_row(
    artifact: str,
    path: Path,
    description: str,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Create a manifest artifact row."""

    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }


def normalize_optional_max_entities(value: int | None) -> int | None:
    """Normalize max entities argument."""

    if value is None:
        return None

    if value <= 0:
        return None

    return int(value)


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Build reproducibility metadata."""

    return {
        "script": "scripts/20_build_knowledge_units.py",
        "command": " ".join(sys.argv),
        "graph_dir": str(args.graph_dir),
        "output_dir": str(args.output_dir),
        "carrier_types": list(args.carrier_types),
        "exclude_bioentity_types": list(args.exclude_bioentity_types),
        "pair_types": list(args.pair_types),
        "min_carrier_count": int(args.min_carrier_count),
        "min_paper_count": int(args.min_paper_count),
        "min_patent_count": int(args.min_patent_count),
        "min_trial_count": int(args.min_trial_count),
        "pair_weight_method": args.pair_weight_method,
        "max_entities_per_carrier": normalize_optional_max_entities(
            args.max_entities_per_carrier
        ),
        "keep_unyearled_edges": not args.drop_unyearled_edges,
        "ku_id_prefix": args.ku_id_prefix,
        "dry_run": bool(args.dry_run),
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_unix": time.time(),
        "ku_definition": "typed_entity_pair",
        "ku_layer": "main_translational_entity_pair_layer",
        "relation_interpretation": (
            "Carrier co-mention evidence does not imply an asserted biomedical "
            "relation. KUs are typed entity-pair co-mention units, not "
            "relation-level claims."
        ),
        "disease_disease_policy": (
            "Disease-disease pairs are excluded from the main translational KU layer "
            "because they conflate comorbidity, complications, subtype hierarchy, "
            "near-synonymy, and background co-mention. They should be built as a "
            "separate exploratory layer if needed."
        ),
        "primekg_status": (
            "PrimeKG is not used in this run. Future PrimeKG integration should "
            "start with schema inspection and ID/name mapping audit."
        ),
    }


def make_build_config(args: argparse.Namespace) -> KnowledgeUnitBuildConfig:
    """Build KnowledgeUnitBuildConfig from CLI args."""

    return KnowledgeUnitBuildConfig(
        graph_dir=str(args.graph_dir),
        output_dir=str(args.output_dir),
        carrier_types=tuple(args.carrier_types),
        exclude_bioentity_types=tuple(args.exclude_bioentity_types),
        pair_types=tuple(args.pair_types),
        min_carrier_count=int(args.min_carrier_count),
        min_paper_count=int(args.min_paper_count),
        min_patent_count=int(args.min_patent_count),
        min_trial_count=int(args.min_trial_count),
        pair_weight_method=args.pair_weight_method,
        max_entities_per_carrier=normalize_optional_max_entities(
            args.max_entities_per_carrier
        ),
        keep_unyearled_edges=not args.drop_unyearled_edges,
        ku_id_prefix=args.ku_id_prefix,
    )


def summarize_mentions(mentions: pd.DataFrame) -> dict[str, Any]:
    """Build a compact mention-edge summary."""

    summary: dict[str, Any] = {
        "mention_edge_count": int(len(mentions)),
        "carrier_count": int(mentions["carrier_key"].nunique())
        if "carrier_key" in mentions.columns
        else None,
        "bioentity_count": int(mentions["bioentity_id"].nunique())
        if "bioentity_id" in mentions.columns
        else None,
        "carrier_types": sorted(mentions["carrier_type"].dropna().astype(str).unique())
        if "carrier_type" in mentions.columns
        else [],
        "bioentity_types": sorted(
            mentions["bioentity_type"].dropna().astype(str).unique()
        )
        if "bioentity_type" in mentions.columns
        else [],
    }

    if "year" in mentions.columns:
        years = pd.to_numeric(mentions["year"], errors="coerce")
        summary["year_min"] = int(years.min()) if years.notna().any() else None
        summary["year_max"] = int(years.max()) if years.notna().any() else None
        summary["missing_year_count"] = int(years.isna().sum())

    if "mention_count" in mentions.columns:
        mention_count = pd.to_numeric(mentions["mention_count"], errors="coerce")
        summary["mention_count_sum"] = float(mention_count.fillna(0).sum())
        summary["mention_count_mean"] = float(mention_count.mean())
        summary["mention_count_max"] = float(mention_count.max())

    return json_safe(summary)


def summarize_mentions_by_type(mentions: pd.DataFrame) -> pd.DataFrame:
    """Summarize mention edges by carrier type and BioEntity type."""

    if mentions.empty:
        return pd.DataFrame(
            columns=[
                "carrier_type",
                "bioentity_type",
                "mention_edge_count",
                "carrier_count",
                "bioentity_count",
                "mention_count_sum",
            ]
        )

    return (
        mentions.groupby(["carrier_type", "bioentity_type"], as_index=False)
        .agg(
            mention_edge_count=("bioentity_id", "size"),
            carrier_count=("carrier_key", "nunique"),
            bioentity_count=("bioentity_id", "nunique"),
            mention_count_sum=("mention_count", "sum"),
        )
        .sort_values(
            ["mention_edge_count", "carrier_count"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def summarize_entities_per_carrier(mentions: pd.DataFrame) -> pd.DataFrame:
    """Summarize number of mentioned BioEntities per carrier."""

    if mentions.empty:
        return pd.DataFrame(
            columns=[
                "carrier_type",
                "carrier_count",
                "entities_per_carrier_min",
                "entities_per_carrier_q25",
                "entities_per_carrier_median",
                "entities_per_carrier_q75",
                "entities_per_carrier_max",
                "entities_per_carrier_mean",
            ]
        )

    per_carrier = (
        mentions.groupby(["carrier_type", "carrier_key"], as_index=False)
        .agg(entity_count=("bioentity_id", "nunique"))
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []

    for carrier_type, group in per_carrier.groupby("carrier_type", sort=True):
        values = group["entity_count"]

        rows.append(
            {
                "carrier_type": carrier_type,
                "carrier_count": int(len(group)),
                "entities_per_carrier_min": int(values.min()),
                "entities_per_carrier_q25": float(values.quantile(0.25)),
                "entities_per_carrier_median": float(values.median()),
                "entities_per_carrier_q75": float(values.quantile(0.75)),
                "entities_per_carrier_max": int(values.max()),
                "entities_per_carrier_mean": float(values.mean()),
            }
        )

    return pd.DataFrame(rows)


def build_dry_run_report(
    *,
    raw_summary: dict[str, Any],
    filtered_summary: dict[str, Any],
    raw_type_summary: pd.DataFrame,
    filtered_type_summary: pd.DataFrame,
    entities_per_carrier_summary: pd.DataFrame,
    run_metadata: dict[str, Any],
) -> str:
    """Build Markdown dry-run report."""

    raw_summary_rows = [
        {"metric": key, "value": json.dumps(value, ensure_ascii=False)}
        for key, value in raw_summary.items()
    ]

    filtered_summary_rows = [
        {"metric": key, "value": json.dumps(value, ensure_ascii=False)}
        for key, value in filtered_summary.items()
    ]

    return "\n".join(
        [
            "# Knowledge Unit Construction Dry Run Report",
            "",
            "## 1. Run metadata",
            "",
            "```json",
            json.dumps(json_safe(run_metadata), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 2. Raw mention summary",
            "",
            markdown_table(raw_summary_rows, ["metric", "value"]),
            "",
            "## 3. Filtered mention summary",
            "",
            markdown_table(filtered_summary_rows, ["metric", "value"]),
            "",
            "## 4. Raw mentions by type",
            "",
            markdown_table(
                raw_type_summary.to_dict("records"),
                [
                    "carrier_type",
                    "bioentity_type",
                    "mention_edge_count",
                    "carrier_count",
                    "bioentity_count",
                    "mention_count_sum",
                ],
            ),
            "",
            "## 5. Filtered mentions by type",
            "",
            markdown_table(
                filtered_type_summary.to_dict("records"),
                [
                    "carrier_type",
                    "bioentity_type",
                    "mention_edge_count",
                    "carrier_count",
                    "bioentity_count",
                    "mention_count_sum",
                ],
            ),
            "",
            "## 6. Entities per carrier after filtering",
            "",
            markdown_table(
                entities_per_carrier_summary.to_dict("records"),
                [
                    "carrier_type",
                    "carrier_count",
                    "entities_per_carrier_min",
                    "entities_per_carrier_q25",
                    "entities_per_carrier_median",
                    "entities_per_carrier_q75",
                    "entities_per_carrier_max",
                    "entities_per_carrier_mean",
                ],
            ),
            "",
            "## 7. Interpretation",
            "",
            "This dry run only inspects carrier-BioEntity mention edges and filtering effects.",
            "",
            "It does not build final Knowledge Unit tables.",
            "",
            "V1 Knowledge Units will be typed entity-pair co-mention units.",
            "",
        ]
    )


def run_dry_run(args: argparse.Namespace, output_dir: Path, run_metadata: dict[str, Any]) -> None:
    """Run dry-run inspection."""

    print()
    print("Running Knowledge Unit construction dry run...")
    print(f"Graph directory:  {args.graph_dir}")
    print(f"Output directory: {output_dir}")
    print()

    mentions = load_standardized_carrier_bioentity_mentions(
        args.graph_dir,
        carrier_types=args.carrier_types,
    )

    allowed_entity_types = allowed_entity_types_from_pair_types(args.pair_types)

    filtered_mentions = filter_mentions_for_knowledge_units(
        mentions,
        exclude_bioentity_types=args.exclude_bioentity_types,
        allowed_entity_types=allowed_entity_types,
        keep_unyearled_edges=not args.drop_unyearled_edges,
    )

    raw_summary = summarize_mentions(mentions)
    filtered_summary = summarize_mentions(filtered_mentions)

    raw_type_summary = summarize_mentions_by_type(mentions)
    filtered_type_summary = summarize_mentions_by_type(filtered_mentions)
    entities_per_carrier_summary = summarize_entities_per_carrier(filtered_mentions)

    dry_run_summary = {
        "raw_summary": raw_summary,
        "filtered_summary": filtered_summary,
        "allowed_entity_types_from_pair_types": sorted(allowed_entity_types),
        "excluded_bioentity_types": list(args.exclude_bioentity_types),
        "pair_types": list(args.pair_types),
        "run_metadata": run_metadata,
    }

    dry_run_summary_path = output_dir / "dry_run_summary.json"
    write_json(dry_run_summary_path, dry_run_summary)

    raw_type_summary_path = output_dir / "raw_mentions_by_type.csv"
    filtered_type_summary_path = output_dir / "filtered_mentions_by_type.csv"
    entities_per_carrier_path = output_dir / "filtered_entities_per_carrier_summary.csv"

    write_dataframe(raw_type_summary, raw_type_summary_path)
    write_dataframe(filtered_type_summary, filtered_type_summary_path)
    write_dataframe(entities_per_carrier_summary, entities_per_carrier_path)

    if args.write_probe_tables:
        probe_mentions_path = output_dir / "filtered_mentions_probe.parquet"
        write_dataframe(filtered_mentions.head(100_000), probe_mentions_path)

    report = build_dry_run_report(
        raw_summary=raw_summary,
        filtered_summary=filtered_summary,
        raw_type_summary=raw_type_summary,
        filtered_type_summary=filtered_type_summary,
        entities_per_carrier_summary=entities_per_carrier_summary,
        run_metadata=run_metadata,
    )

    report_path = output_dir / "dry_run_report.md"
    write_text(report_path, report)

    manifest_rows = [
        local_artifact_row(
            "dry_run_summary",
            dry_run_summary_path,
            "Dry-run JSON summary for carrier-BioEntity mention filtering.",
        ),
        local_artifact_row(
            "raw_mentions_by_type",
            raw_type_summary_path,
            "Raw carrier-BioEntity mention summary by carrier and BioEntity type.",
            row_count=len(raw_type_summary),
        ),
        local_artifact_row(
            "filtered_mentions_by_type",
            filtered_type_summary_path,
            "Filtered carrier-BioEntity mention summary by carrier and BioEntity type.",
            row_count=len(filtered_type_summary),
        ),
        local_artifact_row(
            "filtered_entities_per_carrier_summary",
            entities_per_carrier_path,
            "Distribution summary of entity counts per carrier after filtering.",
            row_count=len(entities_per_carrier_summary),
        ),
        local_artifact_row(
            "dry_run_report",
            report_path,
            "Markdown dry-run report.",
        ),
    ]

    if args.write_probe_tables:
        manifest_rows.append(
            local_artifact_row(
                "filtered_mentions_probe",
                output_dir / "filtered_mentions_probe.parquet",
                "First 100,000 filtered mention rows for manual inspection.",
                row_count=min(len(filtered_mentions), 100_000),
            )
        )

    manifest_path = output_dir / "dataset_manifest.csv"
    manifest_rows.append(
        local_artifact_row(
            "dataset_manifest",
            manifest_path,
            "Dry-run artifact manifest.",
            row_count=len(manifest_rows),
        )
    )
    write_dataframe(pd.DataFrame(manifest_rows), manifest_path)

    print("Dry run complete.")
    print()
    print("Raw mention summary:")
    for key, value in raw_summary.items():
        print(f"  {key}: {value}")
    print()
    print("Filtered mention summary:")
    for key, value in filtered_summary.items():
        print(f"  {key}: {value}")
    print()
    print(f"Dry-run report: {report_path}")
    print(f"Dry-run summary: {dry_run_summary_path}")
    print()


def run_build(args: argparse.Namespace, output_dir: Path, run_metadata: dict[str, Any]) -> None:
    """Run full Knowledge Unit construction."""

    print()
    print("Building entity-pair Knowledge Units...")
    print(f"Graph directory:  {args.graph_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("Configuration:")
    print(f"  carrier_types: {args.carrier_types}")
    print(f"  exclude_bioentity_types: {args.exclude_bioentity_types}")
    print(f"  pair_types: {args.pair_types}")
    print(f"  min_carrier_count: {args.min_carrier_count}")
    print(f"  pair_weight_method: {args.pair_weight_method}")
    print(f"  max_entities_per_carrier: {normalize_optional_max_entities(args.max_entities_per_carrier)}")
    if "disease-disease" in {str(value).lower() for value in args.pair_types}:
        print()
        print(
            "  WARNING: disease-disease is included. This is not recommended for "
            "the main translational KU layer; use it only for exploratory "
            "comorbidity / disease co-context analysis."
        )
    print()

    config = make_build_config(args)

    try:
        result = build_knowledge_units_from_graph_dir(
            args.graph_dir,
            config=config,
        )
    except Exception:
        print()
        print("Knowledge Unit construction failed.")
        traceback.print_exc()
        print()
        raise

    result.run_config.update(run_metadata)

    artifacts = write_knowledge_unit_outputs(result, output_dir)

    report = build_knowledge_unit_report(
        result=result,
        output_dir=output_dir,
        title="Knowledge Unit Construction Report",
    )

    report_path = output_dir / "knowledge_unit_report.md"
    write_text(report_path, report)

    artifacts.append(
        local_artifact_row(
            "knowledge_unit_report",
            report_path,
            "Markdown report for Knowledge Unit construction.",
        )
    )

    manifest_path = output_dir / "dataset_manifest.csv"
    # Refresh dataset manifest including report artifact.
    write_dataframe(pd.DataFrame(artifacts), manifest_path)

    summary = result.summary

    print()
    print("Knowledge Unit construction complete.")
    print()
    print(f"Knowledge Units:          {summary.get('knowledge_unit_count')}")
    print(f"Carrier-KU edges:         {summary.get('filtered_carrier_ku_edge_count')}")
    print(f"Pair types:               {summary.get('pair_type_count')}")
    print(f"Carrier count:            {summary.get('carrier_count')}")
    print(f"Paper carriers:           {summary.get('paper_carrier_count')}")
    print(f"Patent carriers:          {summary.get('patent_carrier_count')}")
    print(f"Trial carriers:           {summary.get('trial_carrier_count')}")
    print()
    print("Output files:")
    print(f"  {output_dir / 'knowledge_units.parquet'}")
    print(f"  {output_dir / 'carrier_knowledge_unit_edges.parquet'}")
    print(f"  {output_dir / 'knowledge_unit_type_summary.csv'}")
    print(f"  {output_dir / 'knowledge_unit_summary.json'}")
    print(f"  {report_path}")
    print(f"  {manifest_path}")
    print()

    top_units = summary.get("top_knowledge_units", [])[: min(args.top_n, 10)]

    if top_units:
        print("Top Knowledge Units:")
        for idx, row in enumerate(top_units, start=1):
            print(
                f"  {idx}. "
                f"{row.get('entity_a_name')} ({row.get('entity_a_type')}) - "
                f"{row.get('entity_b_name')} ({row.get('entity_b_type')}) "
                f"[{row.get('pair_type')}], "
                f"carriers={row.get('carrier_count')}, "
                f"papers={row.get('paper_count')}, "
                f"patents={row.get('patent_count')}, "
                f"trials={row.get('trial_count')}"
            )
        print()


def main() -> None:
    """Entry point."""

    args = parse_args()

    if not args.graph_dir.exists():
        raise FileNotFoundError(f"Graph directory does not exist: {args.graph_dir}")

    output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    run_metadata = build_run_metadata(args)
    write_json(output_dir / "run_config.json", run_metadata)

    if args.dry_run:
        run_dry_run(args, output_dir, run_metadata)
    else:
        run_build(args, output_dir, run_metadata)


if __name__ == "__main__":
    main()