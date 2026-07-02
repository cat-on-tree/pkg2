#!/usr/bin/env python
"""Build a temporal panel for an entity-pair Knowledge Unit dataset.

This script consumes outputs from:

    scripts/20_build_knowledge_units.py
    scripts/21_audit_knowledge_units.py

Expected input files:

    knowledge_units.parquet
    carrier_knowledge_unit_edges.parquet

Optional preferred input file:

    knowledge_units_with_audit_flags.parquet

Main outputs:

    knowledge_unit_year_panel.parquet
    knowledge_unit_temporal_features.parquet
    knowledge_unit_first_observed_events.parquet
    knowledge_unit_year_summary.csv
    knowledge_unit_temporal_summary.json
    knowledge_unit_temporal_report.md
    knowledge_unit_temporal_manifest.csv

Important interpretation note:

    The yearly panel includes only carrier-KU evidence with valid years.
    In the current 2018-2019 patent-centered graph, trial carrier years are
    unavailable, so trial evidence is preserved as undated KU-level evidence
    in knowledge_unit_temporal_features.parquet but does not contribute to
    yearly counts until trial years are patched.

Examples:

    # Sparse panel, safer default
    python scripts/22_build_knowledge_unit_temporal_panel.py \
      --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --overwrite

    # Dense KU x year panel
    python scripts/22_build_knowledge_unit_temporal_panel.py \
      --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --dense \
      --overwrite

    # Explicit year range
    python scripts/22_build_knowledge_unit_temporal_panel.py \
      --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v1 \
      --start-year 1874 \
      --end-year 2021 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.knowledge_unit_temporal import (  # noqa: E402
    artifact_row,
    build_temporal_outputs,
    json_safe,
    write_dataframe,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build a KU-year temporal panel from carrier-KU evidence edges."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing knowledge_units.parquet and "
            "carrier_knowledge_unit_edges.parquet."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --input-dir.",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help=(
            "Inclusive panel start year. If omitted, inferred from dated "
            "carrier-KU edges."
        ),
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=(
            "Inclusive panel end year. If omitted, inferred from dated "
            "carrier-KU edges."
        ),
    )

    parser.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Write a dense KU x year panel. By default, the script writes a "
            "sparse panel containing only KU-year rows with dated evidence. "
            "Dense mode can be large."
        ),
    )

    parser.add_argument(
        "--no-audit-flags",
        action="store_true",
        help=(
            "Do not prefer knowledge_units_with_audit_flags.parquet. "
            "Use knowledge_units.parquet even if audit flags are available."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing temporal output files.",
    )

    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_can_write(paths: list[Path], *, overwrite: bool) -> None:
    """Refuse to overwrite existing outputs unless --overwrite is set."""

    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Temporal output files already exist. Re-run with --overwrite to replace them:\n"
            f"  {joined}"
        )


def add_run_metadata(
    summary: Mapping[str, Any],
    *,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Attach script run metadata to temporal summary."""

    out = dict(summary)

    out["run_metadata"] = {
        "script": "scripts/22_build_knowledge_unit_temporal_panel.py",
        "command": " ".join(sys.argv),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "start_year_arg": args.start_year,
        "end_year_arg": args.end_year,
        "dense": bool(args.dense),
        "prefer_audit_flags": not bool(args.no_audit_flags),
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_unix": time.time(),
    }

    return json_safe(out)


def main() -> None:
    """Entry point."""

    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or args.input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "knowledge_unit_year_panel": output_dir / "knowledge_unit_year_panel.parquet",
        "knowledge_unit_temporal_features": output_dir
        / "knowledge_unit_temporal_features.parquet",
        "knowledge_unit_first_observed_events": output_dir
        / "knowledge_unit_first_observed_events.parquet",
        "knowledge_unit_year_summary": output_dir / "knowledge_unit_year_summary.csv",
        "knowledge_unit_temporal_summary": output_dir
        / "knowledge_unit_temporal_summary.json",
        "knowledge_unit_temporal_report": output_dir
        / "knowledge_unit_temporal_report.md",
        "knowledge_unit_temporal_manifest": output_dir
        / "knowledge_unit_temporal_manifest.csv",
    }

    ensure_can_write(list(output_paths.values()), overwrite=args.overwrite)

    print()
    print("Building Knowledge Unit temporal panel...")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("Configuration:")
    print(f"  start_year:          {args.start_year}")
    print(f"  end_year:            {args.end_year}")
    print(f"  dense:               {args.dense}")
    print(f"  prefer_audit_flags:  {not args.no_audit_flags}")
    print()

    outputs = build_temporal_outputs(
        input_dir=input_dir,
        output_dir=output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        dense=bool(args.dense),
        prefer_audit_flags=not bool(args.no_audit_flags),
    )

    temporal_summary = add_run_metadata(
        outputs["knowledge_unit_temporal_summary"],
        input_dir=input_dir,
        output_dir=output_dir,
        args=args,
    )

    artifacts: list[dict[str, Any]] = []

    def write_frame_artifact(
        artifact: str,
        frame: pd.DataFrame,
        path: Path,
        description: str,
    ) -> None:
        write_dataframe(frame, path)
        artifacts.append(
            artifact_row(
                artifact,
                path,
                description,
                row_count=len(frame),
            )
        )

    def write_json_artifact(
        artifact: str,
        data: Mapping[str, Any],
        path: Path,
        description: str,
    ) -> None:
        write_json(path, data)
        artifacts.append(
            artifact_row(
                artifact,
                path,
                description,
                row_count=None,
            )
        )

    def write_text_artifact(
        artifact: str,
        text: str,
        path: Path,
        description: str,
    ) -> None:
        write_text(path, text)
        artifacts.append(
            artifact_row(
                artifact,
                path,
                description,
                row_count=None,
            )
        )

    write_frame_artifact(
        "knowledge_unit_year_panel",
        outputs["knowledge_unit_year_panel"],
        output_paths["knowledge_unit_year_panel"],
        (
            "KU-year temporal panel. Sparse by default; dense only if --dense "
            "was used. Yearly counts include only dated carrier-KU evidence."
        ),
    )

    write_frame_artifact(
        "knowledge_unit_temporal_features",
        outputs["knowledge_unit_temporal_features"],
        output_paths["knowledge_unit_temporal_features"],
        (
            "One-row-per-KU temporal feature table, including first observed "
            "paper/patent/trial years where available and undated evidence counts."
        ),
    )

    write_frame_artifact(
        "knowledge_unit_first_observed_events",
        outputs["knowledge_unit_first_observed_events"],
        output_paths["knowledge_unit_first_observed_events"],
        "First observed dated and undated evidence events by KU and modality.",
    )

    write_frame_artifact(
        "knowledge_unit_year_summary",
        outputs["knowledge_unit_year_summary"],
        output_paths["knowledge_unit_year_summary"],
        "Aggregate yearly summary over the KU temporal panel.",
    )

    write_json_artifact(
        "knowledge_unit_temporal_summary",
        temporal_summary,
        output_paths["knowledge_unit_temporal_summary"],
        "JSON summary for KU temporal panel construction.",
    )

    write_text_artifact(
        "knowledge_unit_temporal_report",
        outputs["knowledge_unit_temporal_report"],
        output_paths["knowledge_unit_temporal_report"],
        "Markdown report for KU temporal panel construction.",
    )

    manifest_path = output_paths["knowledge_unit_temporal_manifest"]

    # First write manifest without manifest row to compute file size after creation.
    write_dataframe(pd.DataFrame(artifacts), manifest_path)

    artifacts.append(
        artifact_row(
            "knowledge_unit_temporal_manifest",
            manifest_path,
            "Manifest of temporal panel output artifacts.",
            row_count=len(artifacts),
        )
    )

    write_dataframe(pd.DataFrame(artifacts), manifest_path)

    print("Knowledge Unit temporal panel complete.")
    print()
    print(f"Knowledge Units:                  {temporal_summary['knowledge_unit_count']}")
    print(f"Panel rows:                       {temporal_summary['panel_row_count']}")
    print(f"Temporal feature rows:            {temporal_summary['temporal_feature_row_count']}")
    print(f"Year range:                       {temporal_summary['start_year']} - {temporal_summary['end_year']}")
    print(f"Carrier-KU edges:                 {temporal_summary['carrier_ku_edge_count']}")
    print(f"Dated edges:                      {temporal_summary['dated_edge_count']}")
    print(f"Undated edges:                    {temporal_summary['undated_edge_count']}")
    print(f"Dated edge fraction:              {temporal_summary['dated_edge_fraction']}")
    print(f"KUs with dated paper evidence:    {temporal_summary['ku_with_dated_paper_evidence']}")
    print(f"KUs with dated patent evidence:   {temporal_summary['ku_with_dated_patent_evidence']}")
    print(f"KUs with dated trial evidence:    {temporal_summary['ku_with_dated_trial_evidence']}")
    print(f"KUs with undated trial evidence:  {temporal_summary['ku_with_undated_trial_evidence']}")
    print(f"KUs with dated translation ev.:   {temporal_summary['ku_with_dated_translation_evidence']}")
    print()
    print("Main outputs:")
    print(f"  {output_paths['knowledge_unit_year_panel']}")
    print(f"  {output_paths['knowledge_unit_temporal_features']}")
    print(f"  {output_paths['knowledge_unit_first_observed_events']}")
    print(f"  {output_paths['knowledge_unit_year_summary']}")
    print(f"  {output_paths['knowledge_unit_temporal_summary']}")
    print(f"  {output_paths['knowledge_unit_temporal_report']}")
    print(f"  {manifest_path}")
    print()


if __name__ == "__main__":
    main()