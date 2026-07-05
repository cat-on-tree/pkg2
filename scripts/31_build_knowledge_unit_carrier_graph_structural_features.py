#!/usr/bin/env python
"""Build carrier-graph structural readout features for Stage 06F.

Stage 06F goal
--------------

Compare the Knowledge Unit representation against a same-task representation
derived directly from the original carrier graph.

The Stage 05F / Stage 06 prediction unit is:

    KU x cutoff_year

The original carrier graph nodes are carriers and context nodes:

    paper, patent, trial, project, bioentity, ...

Therefore, even for a direct carrier-graph baseline, we need a readout step:

    original carrier graph
      -> cutoff-safe carrier node structural features
      -> pool structural features over carriers supporting a KU before cutoff
      -> KU x cutoff_year feature vector

This script implements a lightweight, cutoff-safe carrier graph baseline:

    1. Normalize carrier-KU observations.
    2. Normalize original carrier graph edge files into directed carrier-centric
       structural edge events.
    3. For each cutoff year, compute carrier node structural features using only
       graph edges with edge_year <= cutoff_year.
    4. Pool those carrier structural features over carriers supporting each KU
       before cutoff_year.
    5. Write script-30-compatible feature-set directories.

This is not a GNN and does not train graph embeddings. It is a direct structural
readout baseline from the original carrier graph.

Inputs
------

Stage 05F prediction dataset:

    data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/
      prediction_feature_table.parquet
      feature_columns.json
      target_columns.json

Carrier-KU observations:

    data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset/
      knowledge_unit_carrier_observations.parquet

or:

    data/datasets/knowledge_units/diabetes_2000_2024_v1_translational/
      carrier_knowledge_unit_edges.parquet

Aggregated carrier graph directory:

    data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated/

Outputs
-------

    data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/

Main files:

    prediction_feature_table_with_carrier_graph_structural.parquet
    carrier_graph_structural_feature_columns.json
    ku_plus_carrier_graph_structural_feature_columns.json
    carrier_graph_structural_summary.json
    carrier_graph_structural_manifest.csv

Feature-set directories usable by script 30:

    feature_sets/carrier_graph_structural_only/
      prediction_feature_table.parquet
      feature_columns.json
      target_columns.json

    feature_sets/ku_plus_carrier_graph_structural/
      prediction_feature_table.parquet
      feature_columns.json
      target_columns.json

Example
-------

Smoke test:

    python scripts/31_build_knowledge_unit_carrier_graph_structural_features.py \\
      --prediction-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction \\
      --knowledge-unit-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational \\
      --modeling-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset \\
      --graph-dir data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated \\
      --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural_smoke \\
      --cutoffs 2019 \\
      --max-edge-rows 500000 \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite

Full feature construction:

    python scripts/31_build_knowledge_unit_carrier_graph_structural_features.py \\
      --prediction-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction \\
      --knowledge-unit-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational \\
      --modeling-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset \\
      --graph-dir data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated \\
      --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_PREDICTION_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction"
)

DEFAULT_KNOWLEDGE_UNIT_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_translational"
)

DEFAULT_MODELING_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset"
)

DEFAULT_GRAPH_DIR = Path(
    "data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build carrier-graph structural readout features for Stage 06F."
    )

    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help="Stage 05F temporal prediction dataset directory.",
    )

    parser.add_argument(
        "--knowledge-unit-dir",
        type=Path,
        default=DEFAULT_KNOWLEDGE_UNIT_DIR,
        help="Long-window KU dataset directory.",
    )

    parser.add_argument(
        "--modeling-dir",
        type=Path,
        default=DEFAULT_MODELING_DIR,
        help="KU modeling dataset directory.",
    )

    parser.add_argument(
        "--carrier-observations-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit carrier-KU observation parquet. "
            "If omitted, the script searches modeling-dir and knowledge-unit-dir."
        ),
    )

    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=DEFAULT_GRAPH_DIR,
        help="Aggregated carrier graph directory.",
    )

    parser.add_argument(
        "--edge-files",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Optional explicit graph edge parquet files. "
            "If omitted, all parquet files under graph-dir are considered."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--history-start-year",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--history-end-year",
        type=int,
        default=2024,
    )

    parser.add_argument(
        "--cutoffs",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional cutoff years to build features for. "
            "If omitted, uses all cutoff_year values in the prediction table."
        ),
    )

    parser.add_argument(
        "--max-edge-rows",
        type=int,
        default=0,
        help=(
            "Optional cap on directed carrier-centric graph edge rows after normalization. "
            "0 means no cap. Useful for smoke tests only."
        ),
    )

    parser.add_argument(
        "--allow-undated-edges",
        action="store_true",
        help=(
            "Allow graph edge files without a year column by assigning them "
            "history_start_year. Use with caution because undated aggregated edges "
            "may include future information."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--memory-limit",
        type=str,
        default="24GB",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress logging.",
    )

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.prediction_dir = args.prediction_dir.resolve()
    args.knowledge_unit_dir = args.knowledge_unit_dir.resolve()
    args.modeling_dir = args.modeling_dir.resolve()
    args.graph_dir = args.graph_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.carrier_observations_path is not None:
        args.carrier_observations_path = args.carrier_observations_path.resolve()

    if args.edge_files:
        args.edge_files = [p.resolve() for p in args.edge_files]

    if args.history_start_year > args.history_end_year:
        raise ValueError("--history-start-year cannot exceed --history-end-year")

    return args


def log(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_sets").mkdir(parents=True, exist_ok=True)
    (output_dir / "intermediate").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def duck_path(path: Path) -> str:
    text = str(path.resolve()).replace("\\", "/")
    return text.replace("'", "''")


def configure_duckdb(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(args.threads)}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    return con


def get_parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{duck_path(path)}')"
    ).fetchall()
    return [str(row[0]) for row in rows]


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def link_or_copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


# ---------------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------------


def find_column(
    columns: list[str],
    candidates: list[str],
    *,
    contains_all: list[str] | None = None,
    required: bool = True,
    label: str = "column",
) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    if contains_all:
        tokens = [t.lower() for t in contains_all]
        for col in columns:
            low = col.lower()
            if all(token in low for token in tokens):
                return col

    if required:
        raise ValueError(
            f"Could not infer {label}. Candidates={candidates}, contains_all={contains_all}, "
            f"columns={columns}"
        )

    return None


def infer_year_column(columns: list[str]) -> str | None:
    return find_column(
        columns,
        [
            "edge_year",
            "year",
            "carrier_year",
            "evidence_year",
            "publication_year",
            "pub_year",
            "filing_year",
            "grant_year",
            "start_year",
            "first_year",
            "min_year",
        ],
        contains_all=["year"],
        required=False,
        label="year column",
    )


def infer_weight_column(columns: list[str]) -> str | None:
    return find_column(
        columns,
        ["weight", "edge_weight", "score", "count", "mention_count"],
        required=False,
        label="weight column",
    )


def infer_endpoint_kinds_from_filename(filename: str) -> tuple[str, str]:
    name = filename.lower()

    known = ["paper", "patent", "trial", "project", "bioentity", "entity"]
    found = [kind for kind in known if kind in name]

    # Normalize entity to bioentity.
    found = ["bioentity" if kind == "entity" else kind for kind in found]
    found = ordered_unique(found)

    if len(found) >= 2:
        return found[0], found[1]

    return "unknown", "unknown"


def infer_endpoint_columns_from_filename(
    columns: list[str],
    filename: str,
) -> tuple[str | None, str | None]:
    name = filename.lower()

    endpoint_candidates = {
        "paper": [
            "paper_id",
            "pmid",
            "pubmed_id",
            "publication_id",
            "source_paper_id",
            "target_paper_id",
        ],
        "patent": [
            "patent_id",
            "patent_publication_id",
            "publication_number",
            "source_patent_id",
            "target_patent_id",
        ],
        "trial": [
            "trial_id",
            "clinicaltrial_id",
            "clinical_trial_id",
            "nct_id",
            "source_trial_id",
            "target_trial_id",
        ],
        "bioentity": [
            "bioentity_id",
            "bio_entity_id",
            "entity_id",
            "source_bioentity_id",
            "target_bioentity_id",
        ],
        "project": [
            "project_id",
            "project_number",
            "core_project_num",
            "source_project_id",
            "target_project_id",
        ],
    }

    mentioned = [key for key in endpoint_candidates if key in name]
    if len(mentioned) < 2:
        return None, None

    found: list[str] = []
    for key in mentioned[:2]:
        col = find_column(
            columns,
            endpoint_candidates[key],
            contains_all=[key, "id"],
            required=False,
            label=f"{key} endpoint",
        )
        if col is not None:
            found.append(col)

    if len(found) >= 2 and found[0] != found[1]:
        return found[0], found[1]

    return None, None


def infer_endpoint_columns(
    columns: list[str],
    filename: str,
) -> tuple[str | None, str | None]:
    pair_candidates = [
        ("source_id", "target_id"),
        ("src_id", "dst_id"),
        ("source", "target"),
        ("src", "dst"),
        ("node1", "node2"),
        ("node_1", "node_2"),
        ("head", "tail"),
        ("from_id", "to_id"),
        ("from", "to"),
    ]

    lower_to_original = {c.lower(): c for c in columns}

    for left, right in pair_candidates:
        if left.lower() in lower_to_original and right.lower() in lower_to_original:
            return lower_to_original[left.lower()], lower_to_original[right.lower()]

    left, right = infer_endpoint_columns_from_filename(columns, filename)
    if left is not None and right is not None:
        return left, right

    id_like = []
    for col in columns:
        low = col.lower()
        if (
            "id" in low
            or "pmid" in low
            or "nct" in low
            or "number" in low
            or "num" in low
        ):
            if "year" not in low and "count" not in low and "weight" not in low:
                id_like.append(col)

    id_like = ordered_unique(id_like)
    if len(id_like) >= 2:
        return id_like[0], id_like[1]

    return None, None


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        args.prediction_dir / "prediction_feature_table.parquet",
        args.prediction_dir / "feature_columns.json",
        args.prediction_dir / "target_columns.json",
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required prediction inputs:\n" + "\n".join(str(p) for p in missing)
        )

    if not args.graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {args.graph_dir}")


def find_carrier_observations_path(args: argparse.Namespace) -> Path:
    if args.carrier_observations_path is not None:
        if not args.carrier_observations_path.exists():
            raise FileNotFoundError(
                f"Carrier observation file not found: {args.carrier_observations_path}"
            )
        return args.carrier_observations_path

    candidates = [
        args.modeling_dir / "knowledge_unit_carrier_observations.parquet",
        args.knowledge_unit_dir / "carrier_knowledge_unit_edges.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find carrier-KU observations. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def discover_edge_files(args: argparse.Namespace) -> list[Path]:
    if args.edge_files:
        files = []
        for path in args.edge_files:
            if not path.exists():
                raise FileNotFoundError(f"Explicit edge file not found: {path}")
            files.append(path)
        return sorted(files)

    files = sorted(args.graph_dir.rglob("*.parquet"))

    files = [
        path
        for path in files
        if "prediction_feature_table" not in path.name.lower()
        and "embedding" not in path.name.lower()
        and "manifest" not in path.name.lower()
        and "structural" not in path.name.lower()
    ]

    if not files:
        raise FileNotFoundError(f"No parquet edge files found under {args.graph_dir}")

    return files


# ---------------------------------------------------------------------------
# Feature specs
# ---------------------------------------------------------------------------


def feature_spec_numeric_columns(feature_spec: dict[str, Any]) -> list[str]:
    for key in [
        "numeric_feature_columns",
        "numeric_columns",
        "numerical_feature_columns",
    ]:
        value = feature_spec.get(key)
        if value:
            return [str(col) for col in value]
    return []


def feature_spec_categorical_columns(feature_spec: dict[str, Any]) -> list[str]:
    for key in [
        "categorical_feature_columns",
        "categorical_columns",
        "category_feature_columns",
    ]:
        value = feature_spec.get(key)
        if value:
            return [str(col) for col in value]
    return []


def make_feature_spec(
    *,
    base_spec: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    description: str,
) -> dict[str, Any]:
    spec = dict(base_spec)
    spec["description"] = description
    spec["numeric_feature_columns"] = ordered_unique(numeric_columns)
    spec["categorical_feature_columns"] = ordered_unique(categorical_columns)
    spec["feature_count"] = len(spec["numeric_feature_columns"]) + len(
        spec["categorical_feature_columns"]
    )
    return spec


def carrier_graph_structural_feature_columns() -> list[str]:
    core_metrics = [
        "degree",
        "weighted_degree",
        "unique_neighbor",
        "edge_type_count",
        "paper_neighbor",
        "patent_neighbor",
        "trial_neighbor",
        "project_neighbor",
        "bioentity_neighbor",
        "unknown_neighbor",
    ]

    cols = [
        "cgs_support_count",
        "cgs_coverage_count",
        "cgs_coverage_fraction",
        "cgs_support_paper_count",
        "cgs_support_patent_count",
        "cgs_support_trial_count",
        "cgs_support_other_count",
    ]

    for metric in core_metrics:
        for stat in ["mean", "max", "sum", "std"]:
            cols.append(f"cgs_{metric}_{stat}")

    return cols


# ---------------------------------------------------------------------------
# Prediction and observation normalization
# ---------------------------------------------------------------------------


def build_prediction_examples(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = args.prediction_dir / "prediction_feature_table.parquet"
    columns = get_parquet_columns(con, path)

    ku_col = find_column(
        columns,
        ["ku_id", "knowledge_unit_id"],
        contains_all=["ku", "id"],
        required=True,
        label="prediction ku id",
    )

    cutoff_col = find_column(
        columns,
        ["cutoff_year"],
        contains_all=["cutoff"],
        required=True,
        label="cutoff year",
    )

    con.execute(
        f"""
        CREATE TEMP TABLE prediction_examples AS
        SELECT
          CAST("{ku_col}" AS VARCHAR) AS ku_id_norm,
          TRY_CAST("{cutoff_col}" AS INTEGER) AS cutoff_year_norm,
          *
        FROM read_parquet('{duck_path(path)}')
        """
    )

    if args.cutoffs:
        cutoffs = sorted(set(int(c) for c in args.cutoffs))
    else:
        cutoffs = [
            int(row[0])
            for row in con.execute(
                """
                SELECT DISTINCT cutoff_year_norm
                FROM prediction_examples
                WHERE cutoff_year_norm IS NOT NULL
                ORDER BY 1
                """
            ).fetchall()
        ]

    cutoff_df = pd.DataFrame({"cutoff_year": cutoffs})
    con.register("cutoff_years_df", cutoff_df)
    con.execute("CREATE TEMP TABLE cutoff_years AS SELECT * FROM cutoff_years_df")
    con.unregister("cutoff_years_df")

    rows = con.execute("SELECT COUNT(*) FROM prediction_examples").fetchone()[0]
    ku_count = con.execute("SELECT COUNT(DISTINCT ku_id_norm) FROM prediction_examples").fetchone()[0]

    return {
        "prediction_table": str(path),
        "ku_col": ku_col,
        "cutoff_col": cutoff_col,
        "rows": int(rows),
        "ku_count": int(ku_count),
        "cutoffs": cutoffs,
    }


def normalize_carrier_type_expr(column_expr: str) -> str:
    return f"""
    CASE
      WHEN LOWER({column_expr}) IN ('paper', 'publication', 'pubmed', 'pmid') THEN 'paper'
      WHEN LOWER({column_expr}) IN ('patent', 'patent_publication') THEN 'patent'
      WHEN LOWER({column_expr}) IN ('trial', 'clinicaltrial', 'clinical_trial', 'clinical trial', 'nct') THEN 'trial'
      ELSE LOWER({column_expr})
    END
    """


def build_observations(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    observations_path: Path,
) -> dict[str, Any]:
    columns = get_parquet_columns(con, observations_path)

    ku_col = find_column(
        columns,
        ["ku_id", "knowledge_unit_id", "knowledge_unit"],
        contains_all=["ku", "id"],
        required=True,
        label="observation ku id",
    )

    carrier_col = find_column(
        columns,
        ["carrier_id", "source_carrier_id", "carrier", "node_id", "source_id"],
        contains_all=["carrier", "id"],
        required=True,
        label="carrier id",
    )

    carrier_type_col = find_column(
        columns,
        [
            "carrier_type",
            "source_type",
            "carrier_node_type",
            "node_type",
            "modality",
            "evidence_type",
        ],
        required=False,
        label="carrier type",
    )

    year_col = find_column(
        columns,
        [
            "carrier_year",
            "evidence_year",
            "year",
            "publication_year",
            "filing_year",
            "start_year",
            "first_year",
        ],
        contains_all=["year"],
        required=True,
        label="carrier year",
    )

    weight_col = find_column(
        columns,
        ["pair_weight", "edge_weight", "weight", "mention_count", "carrier_weight"],
        required=False,
        label="observation weight",
    )

    carrier_type_expr = (
        normalize_carrier_type_expr(f'CAST("{carrier_type_col}" AS VARCHAR)')
        if carrier_type_col is not None
        else "NULL::VARCHAR"
    )

    weight_expr = (
        f"COALESCE(TRY_CAST(\"{weight_col}\" AS DOUBLE), 1.0)"
        if weight_col is not None
        else "1.0"
    )

    con.execute(
        f"""
        CREATE TEMP TABLE obs_norm AS
        SELECT
          CAST("{ku_col}" AS VARCHAR) AS ku_id,
          CAST("{carrier_col}" AS VARCHAR) AS carrier_id,
          {carrier_type_expr} AS carrier_type,
          TRY_CAST("{year_col}" AS INTEGER) AS carrier_year,
          SUM({weight_expr})::DOUBLE AS pair_weight
        FROM read_parquet('{duck_path(observations_path)}')
        WHERE "{ku_col}" IS NOT NULL
          AND "{carrier_col}" IS NOT NULL
          AND TRY_CAST("{year_col}" AS INTEGER) BETWEEN {args.history_start_year} AND {args.history_end_year}
        GROUP BY 1, 2, 3, 4
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE carrier_nodes AS
        SELECT DISTINCT carrier_id
        FROM obs_norm
        WHERE carrier_id IS NOT NULL
        """
    )

    stats = con.execute(
        """
        SELECT
          COUNT(*) AS rows,
          COUNT(DISTINCT ku_id) AS ku_count,
          COUNT(DISTINCT carrier_id) AS carrier_count,
          MIN(carrier_year) AS min_year,
          MAX(carrier_year) AS max_year
        FROM obs_norm
        """
    ).fetchone()

    return {
        "observations_path": str(observations_path),
        "columns": columns,
        "ku_col": ku_col,
        "carrier_col": carrier_col,
        "carrier_type_col": carrier_type_col,
        "year_col": year_col,
        "weight_col": weight_col,
        "rows": int(stats[0]),
        "ku_count": int(stats[1]),
        "carrier_count": int(stats[2]),
        "min_year": int(stats[3]) if stats[3] is not None else None,
        "max_year": int(stats[4]) if stats[4] is not None else None,
    }


# ---------------------------------------------------------------------------
# Graph structural edge normalization
# ---------------------------------------------------------------------------


def build_directed_carrier_edges(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    edge_files: list[Path],
) -> dict[str, Any]:
    output_path = args.output_dir / "intermediate" / "carrier_graph_directed_carrier_edges.parquet"

    select_parts: list[str] = []
    file_summaries: list[dict[str, Any]] = []

    for path in edge_files:
        columns = get_parquet_columns(con, path)
        src_col, dst_col = infer_endpoint_columns(columns, path.name)
        year_col = infer_year_column(columns)
        weight_col = infer_weight_column(columns)
        src_kind, dst_kind = infer_endpoint_kinds_from_filename(path.name)

        if src_col is None or dst_col is None:
            file_summaries.append(
                {
                    "path": str(path),
                    "status": "skipped_no_endpoints",
                    "columns": columns,
                }
            )
            continue

        if year_col is None and not args.allow_undated_edges:
            file_summaries.append(
                {
                    "path": str(path),
                    "status": "skipped_no_year",
                    "src_col": src_col,
                    "dst_col": dst_col,
                    "columns": columns,
                }
            )
            continue

        year_expr = (
            f"TRY_CAST(\"{year_col}\" AS INTEGER)"
            if year_col is not None
            else f"{int(args.history_start_year)}"
        )

        weight_expr = (
            f"COALESCE(TRY_CAST(\"{weight_col}\" AS DOUBLE), 1.0)"
            if weight_col is not None
            else "1.0"
        )

        edge_type = path.stem.replace("'", "").replace('"', "")

        # Direction: src as carrier node, dst as neighbor.
        select_parts.append(
            f"""
            SELECT
              CAST("{src_col}" AS VARCHAR) AS node_id,
              CAST("{dst_col}" AS VARCHAR) AS neighbor_id,
              '{dst_kind}' AS neighbor_kind,
              TRY_CAST({year_expr} AS INTEGER) AS edge_year,
              {weight_expr}::DOUBLE AS weight,
              '{edge_type}' AS edge_type
            FROM read_parquet('{duck_path(path)}')
            WHERE "{src_col}" IS NOT NULL
              AND "{dst_col}" IS NOT NULL
              AND CAST("{src_col}" AS VARCHAR) <> CAST("{dst_col}" AS VARCHAR)
              AND TRY_CAST({year_expr} AS INTEGER) BETWEEN {args.history_start_year} AND {args.history_end_year}
              AND CAST("{src_col}" AS VARCHAR) IN (SELECT carrier_id FROM carrier_nodes)
            """
        )

        # Reverse direction: dst as carrier node, src as neighbor.
        select_parts.append(
            f"""
            SELECT
              CAST("{dst_col}" AS VARCHAR) AS node_id,
              CAST("{src_col}" AS VARCHAR) AS neighbor_id,
              '{src_kind}' AS neighbor_kind,
              TRY_CAST({year_expr} AS INTEGER) AS edge_year,
              {weight_expr}::DOUBLE AS weight,
              '{edge_type}' AS edge_type
            FROM read_parquet('{duck_path(path)}')
            WHERE "{src_col}" IS NOT NULL
              AND "{dst_col}" IS NOT NULL
              AND CAST("{src_col}" AS VARCHAR) <> CAST("{dst_col}" AS VARCHAR)
              AND TRY_CAST({year_expr} AS INTEGER) BETWEEN {args.history_start_year} AND {args.history_end_year}
              AND CAST("{dst_col}" AS VARCHAR) IN (SELECT carrier_id FROM carrier_nodes)
            """
        )

        file_summaries.append(
            {
                "path": str(path),
                "status": "used",
                "src_col": src_col,
                "dst_col": dst_col,
                "year_col": year_col,
                "weight_col": weight_col,
                "src_kind": src_kind,
                "dst_kind": dst_kind,
                "edge_type": edge_type,
            }
        )

    if not select_parts:
        raise RuntimeError(
            "No graph edge files could be used. "
            "If graph files have no year column, rerun with --allow-undated-edges "
            "only if this does not introduce future leakage."
        )

    union_sql = "\nUNION ALL\n".join(select_parts)

    limit_sql = ""
    if args.max_edge_rows and args.max_edge_rows > 0:
        limit_sql = f"LIMIT {int(args.max_edge_rows)}"

    con.execute(
        f"""
        COPY (
          SELECT
            node_id,
            neighbor_id,
            CASE
              WHEN neighbor_kind IN ('paper', 'patent', 'trial', 'project', 'bioentity') THEN neighbor_kind
              ELSE 'unknown'
            END AS neighbor_kind,
            edge_year,
            weight,
            edge_type
          FROM ({union_sql})
          WHERE node_id IS NOT NULL
            AND neighbor_id IS NOT NULL
            AND edge_year IS NOT NULL
          {limit_sql}
        )
        TO '{duck_path(output_path)}'
        (FORMAT PARQUET)
        """
    )

    stats = con.execute(
        f"""
        SELECT
          COUNT(*) AS rows,
          COUNT(DISTINCT node_id) AS carrier_nodes_with_edges,
          COUNT(DISTINCT neighbor_id) AS neighbor_nodes,
          MIN(edge_year) AS min_year,
          MAX(edge_year) AS max_year
        FROM read_parquet('{duck_path(output_path)}')
        """
    ).fetchone()

    return {
        "directed_carrier_edges_path": str(output_path),
        "edge_files_total": len(edge_files),
        "edge_files_used": sum(1 for row in file_summaries if row["status"] == "used"),
        "edge_file_summaries": file_summaries,
        "directed_edge_rows": int(stats[0]),
        "carrier_nodes_with_edges": int(stats[1]),
        "neighbor_nodes": int(stats[2]),
        "min_year": int(stats[3]) if stats[3] is not None else None,
        "max_year": int(stats[4]) if stats[4] is not None else None,
        "max_edge_rows": args.max_edge_rows,
    }


# ---------------------------------------------------------------------------
# Structural features and pooling
# ---------------------------------------------------------------------------


def build_structural_features(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    *,
    directed_edges_path: Path,
    feature_columns: list[str],
) -> tuple[Path, dict[str, Any]]:
    t0 = time.time()

    log(args, "Computing carrier node structural features by cutoff ...")

    con.execute(
        f"""
        CREATE TEMP TABLE carrier_node_structural AS
        SELECT
          e.node_id AS carrier_id,
          c.cutoff_year,

          COUNT(*)::DOUBLE AS node_degree,
          SUM(COALESCE(e.weight, 1.0))::DOUBLE AS node_weighted_degree,
          COUNT(DISTINCT e.neighbor_id)::DOUBLE AS node_unique_neighbor,
          COUNT(DISTINCT e.edge_type)::DOUBLE AS node_edge_type_count,

          SUM(CASE WHEN e.neighbor_kind = 'paper' THEN 1 ELSE 0 END)::DOUBLE AS node_paper_neighbor,
          SUM(CASE WHEN e.neighbor_kind = 'patent' THEN 1 ELSE 0 END)::DOUBLE AS node_patent_neighbor,
          SUM(CASE WHEN e.neighbor_kind = 'trial' THEN 1 ELSE 0 END)::DOUBLE AS node_trial_neighbor,
          SUM(CASE WHEN e.neighbor_kind = 'project' THEN 1 ELSE 0 END)::DOUBLE AS node_project_neighbor,
          SUM(CASE WHEN e.neighbor_kind = 'bioentity' THEN 1 ELSE 0 END)::DOUBLE AS node_bioentity_neighbor,
          SUM(CASE WHEN e.neighbor_kind = 'unknown' THEN 1 ELSE 0 END)::DOUBLE AS node_unknown_neighbor

        FROM read_parquet('{duck_path(directed_edges_path)}') e
        JOIN cutoff_years c
          ON e.edge_year <= c.cutoff_year
        GROUP BY 1, 2
        """
    )

    node_struct_rows = con.execute(
        "SELECT COUNT(*) FROM carrier_node_structural"
    ).fetchone()[0]

    log(args, f"  carrier_node_structural rows: {node_struct_rows:,}")

    log(args, "Building KU-support carrier table by cutoff ...")

    con.execute(
        """
        CREATE TEMP TABLE obs_support_cutoff AS
        SELECT
          o.ku_id,
          o.carrier_id,
          COALESCE(
            CASE
              WHEN ANY_VALUE(o.carrier_type) IN ('paper', 'patent', 'trial') THEN ANY_VALUE(o.carrier_type)
              ELSE 'other'
            END,
            'other'
          ) AS carrier_type,
          c.cutoff_year
        FROM obs_norm o
        JOIN cutoff_years c
          ON o.carrier_year <= c.cutoff_year
        GROUP BY 1, 2, 4
        """
    )

    support_rows = con.execute(
        "SELECT COUNT(*) FROM obs_support_cutoff"
    ).fetchone()[0]

    log(args, f"  obs_support_cutoff rows: {support_rows:,}")

    log(args, "Pooling carrier structural features to KU x cutoff ...")

    con.execute(
        """
        CREATE TEMP TABLE carrier_graph_structural_pooled AS
        SELECT
          o.ku_id AS ku_id_norm,
          o.cutoff_year AS cutoff_year_norm,

          COUNT(*)::DOUBLE AS cgs_support_count,
          SUM(CASE WHEN s.carrier_id IS NOT NULL THEN 1 ELSE 0 END)::DOUBLE AS cgs_coverage_count,
          CASE
            WHEN COUNT(*) > 0
            THEN SUM(CASE WHEN s.carrier_id IS NOT NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)::DOUBLE
            ELSE 0.0
          END AS cgs_coverage_fraction,

          SUM(CASE WHEN o.carrier_type = 'paper' THEN 1 ELSE 0 END)::DOUBLE AS cgs_support_paper_count,
          SUM(CASE WHEN o.carrier_type = 'patent' THEN 1 ELSE 0 END)::DOUBLE AS cgs_support_patent_count,
          SUM(CASE WHEN o.carrier_type = 'trial' THEN 1 ELSE 0 END)::DOUBLE AS cgs_support_trial_count,
          SUM(CASE WHEN o.carrier_type = 'other' THEN 1 ELSE 0 END)::DOUBLE AS cgs_support_other_count,

          AVG(COALESCE(s.node_degree, 0.0))::DOUBLE AS cgs_degree_mean,
          MAX(COALESCE(s.node_degree, 0.0))::DOUBLE AS cgs_degree_max,
          SUM(COALESCE(s.node_degree, 0.0))::DOUBLE AS cgs_degree_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_degree, 0.0)), 0.0)::DOUBLE AS cgs_degree_std,

          AVG(COALESCE(s.node_weighted_degree, 0.0))::DOUBLE AS cgs_weighted_degree_mean,
          MAX(COALESCE(s.node_weighted_degree, 0.0))::DOUBLE AS cgs_weighted_degree_max,
          SUM(COALESCE(s.node_weighted_degree, 0.0))::DOUBLE AS cgs_weighted_degree_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_weighted_degree, 0.0)), 0.0)::DOUBLE AS cgs_weighted_degree_std,

          AVG(COALESCE(s.node_unique_neighbor, 0.0))::DOUBLE AS cgs_unique_neighbor_mean,
          MAX(COALESCE(s.node_unique_neighbor, 0.0))::DOUBLE AS cgs_unique_neighbor_max,
          SUM(COALESCE(s.node_unique_neighbor, 0.0))::DOUBLE AS cgs_unique_neighbor_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_unique_neighbor, 0.0)), 0.0)::DOUBLE AS cgs_unique_neighbor_std,

          AVG(COALESCE(s.node_edge_type_count, 0.0))::DOUBLE AS cgs_edge_type_count_mean,
          MAX(COALESCE(s.node_edge_type_count, 0.0))::DOUBLE AS cgs_edge_type_count_max,
          SUM(COALESCE(s.node_edge_type_count, 0.0))::DOUBLE AS cgs_edge_type_count_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_edge_type_count, 0.0)), 0.0)::DOUBLE AS cgs_edge_type_count_std,

          AVG(COALESCE(s.node_paper_neighbor, 0.0))::DOUBLE AS cgs_paper_neighbor_mean,
          MAX(COALESCE(s.node_paper_neighbor, 0.0))::DOUBLE AS cgs_paper_neighbor_max,
          SUM(COALESCE(s.node_paper_neighbor, 0.0))::DOUBLE AS cgs_paper_neighbor_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_paper_neighbor, 0.0)), 0.0)::DOUBLE AS cgs_paper_neighbor_std,

          AVG(COALESCE(s.node_patent_neighbor, 0.0))::DOUBLE AS cgs_patent_neighbor_mean,
          MAX(COALESCE(s.node_patent_neighbor, 0.0))::DOUBLE AS cgs_patent_neighbor_max,
          SUM(COALESCE(s.node_patent_neighbor, 0.0))::DOUBLE AS cgs_patent_neighbor_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_patent_neighbor, 0.0)), 0.0)::DOUBLE AS cgs_patent_neighbor_std,

          AVG(COALESCE(s.node_trial_neighbor, 0.0))::DOUBLE AS cgs_trial_neighbor_mean,
          MAX(COALESCE(s.node_trial_neighbor, 0.0))::DOUBLE AS cgs_trial_neighbor_max,
          SUM(COALESCE(s.node_trial_neighbor, 0.0))::DOUBLE AS cgs_trial_neighbor_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_trial_neighbor, 0.0)), 0.0)::DOUBLE AS cgs_trial_neighbor_std,

          AVG(COALESCE(s.node_project_neighbor, 0.0))::DOUBLE AS cgs_project_neighbor_mean,
          MAX(COALESCE(s.node_project_neighbor, 0.0))::DOUBLE AS cgs_project_neighbor_max,
          SUM(COALESCE(s.node_project_neighbor, 0.0))::DOUBLE AS cgs_project_neighbor_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_project_neighbor, 0.0)), 0.0)::DOUBLE AS cgs_project_neighbor_std,

          AVG(COALESCE(s.node_bioentity_neighbor, 0.0))::DOUBLE AS cgs_bioentity_neighbor_mean,
          MAX(COALESCE(s.node_bioentity_neighbor, 0.0))::DOUBLE AS cgs_bioentity_neighbor_max,
          SUM(COALESCE(s.node_bioentity_neighbor, 0.0))::DOUBLE AS cgs_bioentity_neighbor_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_bioentity_neighbor, 0.0)), 0.0)::DOUBLE AS cgs_bioentity_neighbor_std,

          AVG(COALESCE(s.node_unknown_neighbor, 0.0))::DOUBLE AS cgs_unknown_neighbor_mean,
          MAX(COALESCE(s.node_unknown_neighbor, 0.0))::DOUBLE AS cgs_unknown_neighbor_max,
          SUM(COALESCE(s.node_unknown_neighbor, 0.0))::DOUBLE AS cgs_unknown_neighbor_sum,
          COALESCE(STDDEV_POP(COALESCE(s.node_unknown_neighbor, 0.0)), 0.0)::DOUBLE AS cgs_unknown_neighbor_std

        FROM obs_support_cutoff o
        LEFT JOIN carrier_node_structural s
          ON o.carrier_id = s.carrier_id
         AND o.cutoff_year = s.cutoff_year
        GROUP BY 1, 2
        """
    )

    pooled_rows = con.execute(
        "SELECT COUNT(*) FROM carrier_graph_structural_pooled"
    ).fetchone()[0]

    log(args, f"  carrier_graph_structural_pooled rows: {pooled_rows:,}")

    output_path = args.output_dir / "prediction_feature_table_with_carrier_graph_structural.parquet"

    feature_exprs = ",\n          ".join(
        [f"COALESCE(f.{col}, 0.0)::DOUBLE AS {col}" for col in feature_columns]
    )

    log(args, "Joining pooled structural features to prediction table ...")

    con.execute(
        f"""
        COPY (
          SELECT
            p.*,
            {feature_exprs}
          FROM prediction_examples p
          LEFT JOIN carrier_graph_structural_pooled f
            ON p.ku_id_norm = f.ku_id_norm
           AND p.cutoff_year_norm = f.cutoff_year_norm
        )
        TO '{duck_path(output_path)}'
        (FORMAT PARQUET)
        """
    )

    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{duck_path(output_path)}')"
    ).fetchone()[0]

    log(args, "Writing feature diagnostics ...")

    diagnostics = []
    for col in feature_columns:
        nonzero, mean_value, max_value = con.execute(
            f"""
            SELECT
              SUM(CASE WHEN {col} <> 0 THEN 1 ELSE 0 END),
              AVG({col}),
              MAX({col})
            FROM read_parquet('{duck_path(output_path)}')
            """
        ).fetchone()
        diagnostics.append(
            {
                "feature": col,
                "nonzero_rows": int(nonzero or 0),
                "nonzero_fraction": float((nonzero or 0) / row_count) if row_count else 0.0,
                "mean": float(mean_value or 0.0),
                "max": float(max_value or 0.0),
            }
        )

    diagnostics_path = args.output_dir / "carrier_graph_structural_feature_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)

    elapsed = time.time() - t0

    summary = {
        "augmented_prediction_table": str(output_path),
        "row_count": int(row_count),
        "feature_count": len(feature_columns),
        "node_structural_rows": int(node_struct_rows),
        "support_rows": int(support_rows),
        "pooled_rows": int(pooled_rows),
        "feature_diagnostics_path": str(diagnostics_path),
        "runtime_seconds": elapsed,
    }

    return output_path, summary


# ---------------------------------------------------------------------------
# Feature-set outputs
# ---------------------------------------------------------------------------


def create_feature_set_dir(
    *,
    feature_set_dir: Path,
    augmented_table_path: Path,
    target_spec_path: Path,
    feature_spec: dict[str, Any],
) -> dict[str, Any]:
    feature_set_dir.mkdir(parents=True, exist_ok=True)

    mode = link_or_copy_file(
        augmented_table_path,
        feature_set_dir / "prediction_feature_table.parquet",
    )

    shutil.copy2(target_spec_path, feature_set_dir / "target_columns.json")
    write_json(feature_set_dir / "feature_columns.json", feature_spec)

    return {
        "feature_set_dir": str(feature_set_dir),
        "table_mode": mode,
        "feature_count": feature_spec.get("feature_count"),
        "numeric_feature_count": len(feature_spec.get("numeric_feature_columns", [])),
        "categorical_feature_count": len(feature_spec.get("categorical_feature_columns", [])),
    }


def write_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "artifact": path.name,
                    "relative_path": str(path.relative_to(output_dir)),
                    "path": str(path),
                    "file_size_bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "carrier_graph_structural_manifest.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = normalize_args(parse_args())
    validate_inputs(args)

    start = time.time()
    prepare_output_dir(args.output_dir, args.overwrite)

    write_json(args.output_dir / "run_args.json", vars(args))

    con = configure_duckdb(args)

    log(args, "Loading base feature specs ...")
    base_feature_spec = read_json(args.prediction_dir / "feature_columns.json")
    target_spec_path = args.prediction_dir / "target_columns.json"

    base_numeric = feature_spec_numeric_columns(base_feature_spec)
    base_categorical = feature_spec_categorical_columns(base_feature_spec)

    log(args, "Building prediction examples table ...")
    prediction_summary = build_prediction_examples(con, args)

    log(args, f"Using cutoffs: {prediction_summary['cutoffs']}")

    log(args, "Finding carrier-KU observations ...")
    observations_path = find_carrier_observations_path(args)
    log(args, f"Carrier observations: {observations_path}")

    log(args, "Normalizing carrier-KU observations ...")
    observation_summary = build_observations(con, args, observations_path)

    log(args, "Discovering carrier graph edge files ...")
    edge_files = discover_edge_files(args)
    log(args, f"Candidate edge files: {len(edge_files)}")

    log(args, "Building directed carrier-centric graph edges ...")
    graph_edge_summary = build_directed_carrier_edges(con, args, edge_files)
    directed_edges_path = Path(graph_edge_summary["directed_carrier_edges_path"])

    log(args, f"Directed carrier edge rows: {graph_edge_summary['directed_edge_rows']:,}")
    log(args, f"Directed carrier edges: {directed_edges_path}")

    structural_feature_columns = carrier_graph_structural_feature_columns()

    log(args, "Building carrier graph structural readout features ...")
    augmented_table_path, structural_summary = build_structural_features(
        con,
        args,
        directed_edges_path=directed_edges_path,
        feature_columns=structural_feature_columns,
    )

    carrier_graph_structural_only_spec = make_feature_spec(
        base_spec=base_feature_spec,
        numeric_columns=structural_feature_columns,
        categorical_columns=[],
        description=(
            "Carrier graph structural readout only representation for Stage 06F. "
            "Features are cutoff-safe structural statistics of original carrier graph "
            "nodes, pooled over carriers supporting each KU x cutoff example."
        ),
    )

    ku_plus_carrier_graph_structural_spec = make_feature_spec(
        base_spec=base_feature_spec,
        numeric_columns=base_numeric + structural_feature_columns,
        categorical_columns=base_categorical,
        description=(
            "Full KU representation plus cutoff-safe carrier graph structural "
            "readout features for Stage 06F."
        ),
    )

    write_json(
        args.output_dir / "carrier_graph_structural_feature_columns.json",
        carrier_graph_structural_only_spec,
    )

    write_json(
        args.output_dir / "ku_plus_carrier_graph_structural_feature_columns.json",
        ku_plus_carrier_graph_structural_spec,
    )

    feature_sets_dir = args.output_dir / "feature_sets"

    log(args, "Creating script-30-compatible feature set directories ...")

    carrier_only_set = create_feature_set_dir(
        feature_set_dir=feature_sets_dir / "carrier_graph_structural_only",
        augmented_table_path=augmented_table_path,
        target_spec_path=target_spec_path,
        feature_spec=carrier_graph_structural_only_spec,
    )

    ku_plus_set = create_feature_set_dir(
        feature_set_dir=feature_sets_dir / "ku_plus_carrier_graph_structural",
        augmented_table_path=augmented_table_path,
        target_spec_path=target_spec_path,
        feature_spec=ku_plus_carrier_graph_structural_spec,
    )

    runtime_seconds = time.time() - start

    summary = {
        "script": "scripts/31_build_knowledge_unit_carrier_graph_structural_features.py",
        "status": "ok",
        "prediction_summary": prediction_summary,
        "observation_summary": observation_summary,
        "graph_edge_summary": graph_edge_summary,
        "structural_summary": structural_summary,
        "structural_feature_count": len(structural_feature_columns),
        "base_numeric_feature_count": len(base_numeric),
        "base_categorical_feature_count": len(base_categorical),
        "carrier_graph_structural_only_feature_set": carrier_only_set,
        "ku_plus_carrier_graph_structural_feature_set": ku_plus_set,
        "allow_undated_edges": bool(args.allow_undated_edges),
        "runtime_seconds": runtime_seconds,
    }

    write_json(args.output_dir / "carrier_graph_structural_summary.json", summary)
    write_manifest(args.output_dir)

    log(args, "")
    log(args, "Carrier graph structural feature construction complete.")
    log(args, f"Output directory: {args.output_dir}")
    log(args, f"Augmented table: {augmented_table_path}")
    log(args, f"Carrier structural only feature set: {feature_sets_dir / 'carrier_graph_structural_only'}")
    log(args, f"KU + carrier structural feature set: {feature_sets_dir / 'ku_plus_carrier_graph_structural'}")
    log(args, f"Runtime seconds: {runtime_seconds:.1f}")


if __name__ == "__main__":
    main()