#!/usr/bin/env python
"""Build a KU temporal modeling dataset from Stage 04 outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import typer
from rich.console import Console
from rich.table import Table

from pkg2.knowledge_unit_modeling_dataset import (
    KnowledgeUnitModelingDatasetConfig,
    build_knowledge_unit_modeling_dataset,
)


app = typer.Typer(
    add_completion=False,
    help="Build KU-level temporal modeling dataset from Stage 04 outputs.",
)

console = Console()


@app.command()
def main(
    knowledge_unit_dir: Path = typer.Option(
        ...,
        "--knowledge-unit-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Stage 04 Knowledge Unit dataset directory.",
    ),
    temporal_dir: Path = typer.Option(
        ...,
        "--temporal-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Stage 04 KU temporal panel directory.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Output directory for the Stage 04F modeling dataset.",
    ),
    start_year: Optional[int] = typer.Option(
        None,
        "--start-year",
        help="Optional start year for temporal sequence clipping.",
    ),
    end_year: Optional[int] = typer.Option(
        None,
        "--end-year",
        help="Optional end year for temporal sequence clipping.",
    ),
    keep_zero_year_rows: bool = typer.Option(
        True,
        "--keep-zero-year-rows/--drop-zero-year-rows",
        help="Keep dense KU-year rows with no annual observations.",
    ),
    add_entity_hub_features: bool = typer.Option(
        True,
        "--add-entity-hub-features/--no-entity-hub-features",
        help="Add entity KU-degree covariates to static feature table.",
    ),
    add_modality_flags: bool = typer.Option(
        True,
        "--add-modality-flags/--no-modality-flags",
        help="Add observed modality flags to static feature table.",
    ),
    add_lag_features: bool = typer.Option(
        True,
        "--add-lag-features/--no-lag-features",
        help="Add first-year lag covariates such as paper_to_trial_lag.",
    ),
    ensure_default_sequence_columns: bool = typer.Option(
        True,
        "--ensure-default-sequence-columns/--no-ensure-default-sequence-columns",
        help="Ensure standard temporal sequence columns exist.",
    ),
    write_future_label_plan: bool = typer.Option(
        True,
        "--write-future-label-plan/--no-future-label-plan",
        help="Write future_label_plan.json scaffold.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite output directory if it already exists.",
    ),
) -> None:
    """Build KU temporal modeling dataset."""

    if start_year is not None and end_year is not None and start_year > end_year:
        raise typer.BadParameter("--start-year must be <= --end-year")

    config = KnowledgeUnitModelingDatasetConfig(
        start_year=start_year,
        end_year=end_year,
        keep_zero_year_rows=keep_zero_year_rows,
        add_entity_hub_features=add_entity_hub_features,
        add_modality_flags=add_modality_flags,
        add_lag_features=add_lag_features,
        ensure_default_sequence_columns=ensure_default_sequence_columns,
        write_future_label_plan=write_future_label_plan,
    )

    console.rule("[bold cyan]Stage 04F: KU temporal modeling dataset preparation")

    console.print("[bold]Input Knowledge Unit directory:[/bold]", str(knowledge_unit_dir))
    console.print("[bold]Input temporal panel directory:[/bold]", str(temporal_dir))
    console.print("[bold]Output directory:[/bold]", str(output_dir))
    console.print("[bold]Overwrite:[/bold]", overwrite)

    result = build_knowledge_unit_modeling_dataset(
        knowledge_unit_dir=knowledge_unit_dir,
        temporal_dir=temporal_dir,
        output_dir=output_dir,
        config=config,
        overwrite=overwrite,
    )

    summary = result.summary

    table = Table(title="KU temporal modeling dataset summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    for key in [
        "ku_count",
        "static_feature_rows",
        "temporal_sequence_rows",
        "carrier_observation_rows",
        "year_min",
        "year_max",
        "year_count",
        "trains_model",
        "defines_rule_based_state_scores",
        "has_future_labels",
    ]:
        table.add_row(key, str(summary.get(key)))

    console.print(table)

    console.print("\n[bold green]Output artifacts:[/bold green]")
    for _, row in result.manifest.iterrows():
        console.print(
            f"- [bold]{row['artifact']}[/bold]: {row['path']} "
            f"({row['file_size_bytes']} bytes)"
        )

    console.print("\n[bold green]Stage 04F complete.[/bold green]")
    console.print(
        "This dataset is algorithm-ready input for later long-window temporal "
        "prediction dataset construction and model ablations. It does not train "
        "an encoder or define rule-based Knowledge State scores."
    )


if __name__ == "__main__":
    app()