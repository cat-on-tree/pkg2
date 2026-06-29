#!/usr/bin/env python
"""Build a Patent-Paper link prediction dataset.

This script constructs a carrier-level link prediction dataset from the
aggregated PKG2 diabetes Knowledge Carrier Graph prototype.

The current task is Patent-Paper link prediction. This is a carrier-level proxy
task used to validate graph signal and establish reusable dataset construction
infrastructure. It is not the final knowledge-unit-level translation prediction
task.

Example full-context dataset:

    python scripts/11_build_link_prediction_dataset.py \
      --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated \
      --output-dir data/processed/link_prediction_patent_paper_diabetes_2018_2019_v1_full \
      --filter-isolated-nodes \
      --overwrite

Example no-species dataset:

    python scripts/11_build_link_prediction_dataset.py \
      --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated \
      --output-dir data/processed/link_prediction_patent_paper_diabetes_2018_2019_v1_no_species \
      --filter-isolated-nodes \
      --exclude-bioentity-types species \
      --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.graph_schema import (  # noqa: E402
    DEFAULT_EXCLUDABLE_BIOENTITY_IDS,
    DEFAULT_TARGET_EDGE_TABLE,
)
from pkg2.io import connect_duckdb  # noqa: E402
from pkg2.link_prediction import build_patent_paper_link_prediction_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a Patent-Paper link prediction dataset from an aggregated "
            "PKG2 Knowledge Carrier Graph directory."
        )
    )

    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Input aggregated graph directory containing node and edge Parquet files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dataset directory.",
    )

    parser.add_argument(
        "--target-edge-table",
        default=DEFAULT_TARGET_EDGE_TABLE,
        help=f"Target edge table for link prediction. Default: {DEFAULT_TARGET_EDGE_TABLE}",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio for positive target edges. Default: 0.8",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for positive target edges. Default: 0.1",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio for positive target edges. Default: 0.1",
    )

    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Number of negative examples per positive example. Default: 1.0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and negative sampling. Default: 42",
    )

    parser.add_argument(
        "--filter-isolated-nodes",
        action="store_true",
        help="Remove isolated nodes before dataset construction.",
    )

    parser.add_argument(
        "--keep-isolated-nodes",
        action="store_true",
        help="Keep isolated nodes. Overrides --filter-isolated-nodes if both are provided.",
    )

    parser.add_argument(
        "--exclude-bioentity-types",
        nargs="*",
        default=None,
        help="BioEntity types to exclude, for example: --exclude-bioentity-types species",
    )

    parser.add_argument(
        "--exclude-bioentity-ids",
        nargs="*",
        default=None,
        help="Specific BioEntity node IDs to exclude.",
    )

    parser.add_argument(
        "--exclude-default-species-ids",
        action="store_true",
        help=(
            "Exclude default major species BioEntity IDs: "
            + ", ".join(DEFAULT_EXCLUDABLE_BIOENTITY_IDS)
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB thread count. Default: 1",
    )

    parser.add_argument(
        "--memory-limit",
        default="20GB",
        help="DuckDB memory limit. Default: 20GB",
    )

    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/interim/duckdb_tmp"),
        help="DuckDB temporary directory. Default: data/interim/duckdb_tmp",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )

    return parser.parse_args()


def _normalize_optional_list(values: list[str] | None) -> list[str]:
    """Normalize argparse optional list values."""

    if not values:
        return []

    return [str(value).strip() for value in values if str(value).strip()]


def _print_summary(summary: dict[str, Any]) -> None:
    """Print a compact dataset construction summary."""

    print()
    print("Link prediction dataset construction complete.")
    print()
    print(f"Graph directory:   {summary['graph_dir']}")
    print(f"Output directory:  {summary['output_dir']}")
    print(f"Target edge table: {summary['target_edge_table']}")
    print()

    filtering = summary.get("filtering_summary", {})
    print("Filtering:")
    print(f"  Source nodes:     {filtering.get('source_node_count')}")
    print(f"  Filtered nodes:   {filtering.get('filtered_node_count')}")
    print(f"  Removed nodes:    {filtering.get('removed_node_count')}")
    print(f"  Source edges:     {filtering.get('source_edge_count')}")
    print(f"  Filtered edges:   {filtering.get('filtered_edge_count')}")
    print(f"  Removed edges:    {filtering.get('removed_edge_count')}")
    print()

    split_counts = summary.get("split_counts", {})
    print("Positive splits:")
    print(f"  Train: {split_counts.get('positive_edges_train')}")
    print(f"  Val:   {split_counts.get('positive_edges_val')}")
    print(f"  Test:  {split_counts.get('positive_edges_test')}")
    print()

    negative_counts = summary.get("negative_sampling_summary", {}).get("negative_split_counts", {})
    print("Negative splits:")
    print(f"  Train: {negative_counts.get('negative_edges_train')}")
    print(f"  Val:   {negative_counts.get('negative_edges_val')}")
    print(f"  Test:  {negative_counts.get('negative_edges_test')}")
    print()

    labeled_counts = summary.get("labeled_counts", {})
    print("Labeled edge tables:")
    print(f"  Train: {labeled_counts.get('labeled_edges_train')}")
    print(f"  Val:   {labeled_counts.get('labeled_edges_val')}")
    print(f"  Test:  {labeled_counts.get('labeled_edges_test')}")
    print()

    print(f"Dataset manifest: {summary.get('dataset_manifest')}")
    print(f"Dataset report:   {summary.get('dataset_report')}")
    print()


def main() -> None:
    """Run dataset construction."""

    args = parse_args()

    filter_isolated_nodes = args.filter_isolated_nodes
    if args.keep_isolated_nodes:
        filter_isolated_nodes = False

    exclude_bioentity_types = _normalize_optional_list(args.exclude_bioentity_types)
    exclude_bioentity_ids = _normalize_optional_list(args.exclude_bioentity_ids)

    if args.exclude_default_species_ids:
        exclude_bioentity_ids.extend(DEFAULT_EXCLUDABLE_BIOENTITY_IDS)

    connection = connect_duckdb(
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
    )

    summary = build_patent_paper_link_prediction_dataset(
        connection,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        target_edge_table=args.target_edge_table,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        filter_isolated_nodes=filter_isolated_nodes,
        exclude_bioentity_types=exclude_bioentity_types,
        exclude_bioentity_ids=exclude_bioentity_ids,
        overwrite=args.overwrite,
    )

    _print_summary(summary)


if __name__ == "__main__":
    main()