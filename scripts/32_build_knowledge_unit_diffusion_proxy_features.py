#!/usr/bin/env python
"""Build KU-KU diffusion proxy features for Stage 06G.

Stage 06G goal
--------------

Build a simple, interpretable, cutoff-safe KU-neighborhood diffusion proxy
baseline.

Previous stages:

    Stage 06E:
      KU representation ablation.

    Stage 06F:
      Carrier graph structural readout baseline.

This script adds a KU-field diffusion proxy:

    KU-KU shared-entity graph
      -> neighbor KU temporal-state aggregation at each cutoff
      -> diffusion proxy features for each KU x cutoff_year

This is not a GNN. It is a simple graph-neighborhood aggregation baseline.

Prediction unit:

    KU x cutoff_year

Core idea:

    If neighboring KUs are already active, growing, patented, trialed, or
    translated before a cutoff year, that local neighborhood state may predict
    future emergence of the focal KU.

Outputs
-------

Default output directory:

    data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy/

Main files:

    prediction_feature_table_with_diffusion_proxy.parquet
    diffusion_proxy_feature_columns.json
    ku_plus_diffusion_proxy_feature_columns.json
    ku_plus_carrier_structural_plus_diffusion_proxy_feature_columns.json
    diffusion_proxy_summary.json
    diffusion_proxy_manifest.csv

Feature-set directories compatible with script 30:

    feature_sets/diffusion_proxy_only/
      prediction_feature_table.parquet
      feature_columns.json
      target_columns.json

    feature_sets/ku_plus_diffusion_proxy/
      prediction_feature_table.parquet
      feature_columns.json
      target_columns.json

    feature_sets/ku_plus_carrier_structural_plus_diffusion_proxy/
      prediction_feature_table.parquet
      feature_columns.json
      target_columns.json

Example
-------

Smoke test:

    python scripts/32_build_knowledge_unit_diffusion_proxy_features.py \\
      --prediction-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction \\
      --carrier-structural-feature-set-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/feature_sets/ku_plus_carrier_graph_structural \\
      --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy_smoke \\
      --cutoffs 2019 \\
      --top-neighbors 50 \\
      --max-entity-degree 1000 \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite

Full construction:

    python scripts/32_build_knowledge_unit_diffusion_proxy_features.py \\
      --prediction-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction \\
      --carrier-structural-feature-set-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/feature_sets/ku_plus_carrier_graph_structural \\
      --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy \\
      --top-neighbors 100 \\
      --max-entity-degree 2000 \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_PREDICTION_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction"
)

DEFAULT_CARRIER_STRUCTURAL_FEATURE_SET_DIR = Path(
    "data/datasets/knowledge_units/"
    "diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/"
    "feature_sets/ku_plus_carrier_graph_structural"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/datasets/knowledge_units/"
    "diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KU-KU diffusion proxy features for Stage 06G."
    )

    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help="Base Stage 05F KU temporal prediction dataset directory.",
    )

    parser.add_argument(
        "--carrier-structural-feature-set-dir",
        type=Path,
        default=DEFAULT_CARRIER_STRUCTURAL_FEATURE_SET_DIR,
        help=(
            "Optional Stage 06F KU+carrier-structural feature set directory. "
            "If present, this script also creates "
            "ku_plus_carrier_structural_plus_diffusion_proxy."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--cutoffs",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional cutoff years. If omitted, uses all cutoff_year values "
            "in the prediction table."
        ),
    )

    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=100,
        help="Keep at most this many highest-weight KU neighbors per source KU.",
    )

    parser.add_argument(
        "--max-entity-degree",
        type=int,
        default=2000,
        help=(
            "Ignore shared entities appearing in more than this many KUs. "
            "This prevents very high-degree hub entities from making the KU-KU "
            "graph too dense."
        ),
    )

    parser.add_argument(
        "--min-entity-degree",
        type=int,
        default=2,
        help="Minimum KU degree for a shared entity to induce KU-KU edges.",
    )

    parser.add_argument(
        "--include-inactive-neighbors",
        action="store_true",
        help=(
            "If set, include neighbors with zero historical activity at cutoff. "
            "Default excludes inactive neighbor states to reduce future-universe leakage."
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
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.prediction_dir = args.prediction_dir.resolve()
    args.carrier_structural_feature_set_dir = (
        args.carrier_structural_feature_set_dir.resolve()
        if args.carrier_structural_feature_set_dir
        else None
    )
    args.output_dir = args.output_dir.resolve()

    if args.top_neighbors <= 0:
        raise ValueError("--top-neighbors must be positive.")

    if args.min_entity_degree < 2:
        raise ValueError("--min-entity-degree must be >= 2.")

    if args.max_entity_degree < args.min_entity_degree:
        raise ValueError("--max-entity-degree must be >= --min-entity-degree.")

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


def validate_prediction_dir(prediction_dir: Path) -> None:
    required = [
        prediction_dir / "prediction_feature_table.parquet",
        prediction_dir / "feature_columns.json",
        prediction_dir / "target_columns.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(str(p) for p in missing)
        )


# ---------------------------------------------------------------------------
# Column helpers
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
            f"Could not infer {label}. Candidates={candidates}, "
            f"contains_all={contains_all}, columns={columns}"
        )

    return None


def q(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def numeric_expr(
    columns: list[str],
    candidates: list[str],
    *,
    default: str = "0.0",
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return f"COALESCE(TRY_CAST({q(candidate)} AS DOUBLE), {default})"

    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            col = lower_to_original[candidate.lower()]
            return f"COALESCE(TRY_CAST({q(col)} AS DOUBLE), {default})"

    return default


def numeric_expr_or_none(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """Return numeric SQL expression if any candidate column exists, else None."""

    for candidate in candidates:
        if candidate in columns:
            return f"COALESCE(TRY_CAST({q(candidate)} AS DOUBLE), 0.0)"

    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            col = lower_to_original[candidate.lower()]
            return f"COALESCE(TRY_CAST({q(col)} AS DOUBLE), 0.0)"

    return None


def sum_expr(exprs: list[str]) -> str:
    if not exprs:
        return "0.0"
    return "(" + " + ".join(exprs) + ")"


def feature_spec_numeric_columns(feature_spec: dict[str, Any]) -> list[str]:
    for key in ["numeric_feature_columns", "numeric_columns", "numerical_feature_columns"]:
        value = feature_spec.get(key)
        if value:
            return [str(col) for col in value]
    return []


def feature_spec_categorical_columns(feature_spec: dict[str, Any]) -> list[str]:
    for key in ["categorical_feature_columns", "categorical_columns", "category_feature_columns"]:
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


# ---------------------------------------------------------------------------
# Diffusion feature list
# ---------------------------------------------------------------------------


def diffusion_proxy_feature_columns() -> list[str]:
    metrics = [
        "recent_3yr_paper",
        "recent_3yr_patent",
        "recent_3yr_trial",
        "recent_3yr_translation",
        "recent_3yr_total",
        "history_patent",
        "history_trial",
        "history_translation",
        "history_total",
        "growth_total",
        "growth_paper",
    ]

    cols = [
        "kud_neighbor_count",
        "kud_active_neighbor_count",
        "kud_neighbor_weight_sum",
        "kud_shared_entity_mean",
        "kud_shared_entity_max",
        "kud_history_patent_any_fraction",
        "kud_history_trial_any_fraction",
        "kud_history_translation_any_fraction",
        "kud_top10_recent_translation_mean",
        "kud_top10_recent_patent_mean",
        "kud_top10_recent_trial_mean",
    ]

    for metric in metrics:
        for stat in ["mean", "max", "sum", "weighted_mean"]:
            cols.append(f"kud_{metric}_{stat}")

    return cols


# ---------------------------------------------------------------------------
# Build prediction examples and KU-KU graph
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
        label="KU id",
    )
    cutoff_col = find_column(
        columns,
        ["cutoff_year"],
        contains_all=["cutoff"],
        label="cutoff year",
    )

    con.execute(
        f"""
        CREATE TEMP TABLE prediction_examples AS
        SELECT
          CAST({q(ku_col)} AS VARCHAR) AS ku_id_norm,
          TRY_CAST({q(cutoff_col)} AS INTEGER) AS cutoff_year_norm,
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
    ku_count = con.execute(
        "SELECT COUNT(DISTINCT ku_id_norm) FROM prediction_examples"
    ).fetchone()[0]

    return {
        "prediction_table": str(path),
        "ku_col": ku_col,
        "cutoff_col": cutoff_col,
        "rows": int(rows),
        "ku_count": int(ku_count),
        "cutoffs": cutoffs,
        "columns": columns,
    }


def entity_key_expr(
    *,
    columns: list[str],
    side: str,
) -> str:
    """Build an entity key expression for entity_a or entity_b."""

    id_col = find_column(
        columns,
        [
            f"entity_{side}_id",
            f"{side}_entity_id",
            f"entity_{side}_curie",
            f"entity_{side}_identifier",
        ],
        required=False,
        label=f"entity_{side}_id",
    )
    name_col = find_column(
        columns,
        [
            f"entity_{side}_name",
            f"{side}_entity_name",
            f"entity_{side}_label",
            f"{side}_name",
        ],
        required=False,
        label=f"entity_{side}_name",
    )
    type_col = find_column(
        columns,
        [
            f"entity_{side}_type",
            f"{side}_entity_type",
            f"{side}_type",
        ],
        required=False,
        label=f"entity_{side}_type",
    )

    if id_col is None and name_col is None:
        raise ValueError(
            f"Could not infer entity_{side} id/name columns from prediction table."
        )

    identity_parts = []
    if id_col is not None:
        identity_parts.append(f"NULLIF(LOWER(CAST({q(id_col)} AS VARCHAR)), '')")
    if name_col is not None:
        identity_parts.append(f"NULLIF(LOWER(CAST({q(name_col)} AS VARCHAR)), '')")

    identity_expr = f"COALESCE({', '.join(identity_parts)})"

    if type_col is not None:
        type_expr = f"COALESCE(NULLIF(LOWER(CAST({q(type_col)} AS VARCHAR)), ''), 'unknown')"
    else:
        type_expr = "'unknown'"

    return f"({type_expr} || ':' || {identity_expr})"


def build_ku_ku_graph(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    *,
    prediction_summary: dict[str, Any],
) -> dict[str, Any]:
    log(args, "Building KU-entity map ...")

    columns = prediction_summary["columns"]

    entity_a_expr = entity_key_expr(columns=columns, side="a")
    entity_b_expr = entity_key_expr(columns=columns, side="b")

    con.execute(
        f"""
        CREATE TEMP TABLE ku_entity_map AS
        SELECT DISTINCT ku_id_norm, entity_key
        FROM (
          SELECT ku_id_norm, {entity_a_expr} AS entity_key
          FROM prediction_examples
          UNION ALL
          SELECT ku_id_norm, {entity_b_expr} AS entity_key
          FROM prediction_examples
        )
        WHERE entity_key IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE entity_degree AS
        SELECT
          entity_key,
          COUNT(DISTINCT ku_id_norm) AS entity_degree
        FROM ku_entity_map
        GROUP BY 1
        """
    )

    entity_stats = con.execute(
        """
        SELECT
          COUNT(*) AS entity_count,
          MIN(entity_degree) AS min_degree,
          MAX(entity_degree) AS max_degree,
          AVG(entity_degree) AS mean_degree
        FROM entity_degree
        """
    ).fetchone()

    log(args, "Building shared-entity KU-KU graph ...")

    con.execute(
        f"""
        CREATE TEMP TABLE ku_ku_edges_raw AS
        SELECT
          a.ku_id_norm AS src_ku_id,
          b.ku_id_norm AS dst_ku_id,
          SUM(1.0 / LN(1.0 + d.entity_degree))::DOUBLE AS edge_weight,
          COUNT(*)::DOUBLE AS shared_entity_count
        FROM ku_entity_map a
        JOIN ku_entity_map b
          ON a.entity_key = b.entity_key
         AND a.ku_id_norm <> b.ku_id_norm
        JOIN entity_degree d
          ON a.entity_key = d.entity_key
        WHERE d.entity_degree BETWEEN {int(args.min_entity_degree)} AND {int(args.max_entity_degree)}
        GROUP BY 1, 2
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE ku_ku_edges AS
        SELECT
          src_ku_id,
          dst_ku_id,
          edge_weight,
          shared_entity_count
        FROM (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY src_ku_id
              ORDER BY edge_weight DESC, shared_entity_count DESC, dst_ku_id
            ) AS neighbor_rank
          FROM ku_ku_edges_raw
        )
        WHERE neighbor_rank <= {int(args.top_neighbors)}
        """
    )

    raw_edges = con.execute("SELECT COUNT(*) FROM ku_ku_edges_raw").fetchone()[0]
    kept_edges = con.execute("SELECT COUNT(*) FROM ku_ku_edges").fetchone()[0]
    src_nodes = con.execute("SELECT COUNT(DISTINCT src_ku_id) FROM ku_ku_edges").fetchone()[0]

    edge_path = args.output_dir / "intermediate" / "ku_ku_diffusion_edges.parquet"
    con.execute(
        f"""
        COPY (
          SELECT * FROM ku_ku_edges
        )
        TO '{duck_path(edge_path)}'
        (FORMAT PARQUET)
        """
    )

    return {
        "entity_count": int(entity_stats[0] or 0),
        "entity_degree_min": int(entity_stats[1] or 0),
        "entity_degree_max": int(entity_stats[2] or 0),
        "entity_degree_mean": float(entity_stats[3] or 0.0),
        "min_entity_degree": int(args.min_entity_degree),
        "max_entity_degree": int(args.max_entity_degree),
        "top_neighbors": int(args.top_neighbors),
        "raw_edge_count": int(raw_edges),
        "kept_edge_count": int(kept_edges),
        "source_ku_count_with_neighbors": int(src_nodes),
        "edge_path": str(edge_path),
    }


# ---------------------------------------------------------------------------
# KU cutoff state and diffusion features
# ---------------------------------------------------------------------------


def build_ku_cutoff_state(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    *,
    prediction_summary: dict[str, Any],
) -> dict[str, Any]:
    log(args, "Building cutoff-safe KU temporal state table ...")

    columns = prediction_summary["columns"]

    recent_paper = numeric_expr(
        columns,
        ["recent_3yr_paper_count", "recent_3yr_publication_count"],
    )
    recent_patent = numeric_expr(
        columns,
        ["recent_3yr_patent_count"],
    )
    recent_trial = numeric_expr(
        columns,
        ["recent_3yr_trial_count"],
    )

    recent_translation_direct = numeric_expr_or_none(
        columns,
        ["recent_3yr_translation_count"],
    )
    if recent_translation_direct is None:
        recent_translation = sum_expr([recent_patent, recent_trial])
    else:
        recent_translation = recent_translation_direct

    recent_total_direct = numeric_expr_or_none(
        columns,
        ["recent_3yr_total_count"],
    )
    if recent_total_direct is None:
        recent_total = sum_expr([recent_paper, recent_patent, recent_trial])
    else:
        recent_total = recent_total_direct

    history_patent = numeric_expr(
        columns,
        ["history_patent_count", "history_patent_heat"],
    )
    history_trial = numeric_expr(
        columns,
        ["history_trial_count", "history_trial_heat"],
    )

    history_translation_direct = numeric_expr_or_none(
        columns,
        ["history_translation_count", "history_translation_heat"],
    )
    if history_translation_direct is None:
        history_translation = sum_expr([history_patent, history_trial])
    else:
        history_translation = history_translation_direct

    history_total_direct = numeric_expr_or_none(
        columns,
        ["history_total_count", "history_total_heat"],
    )
    if history_total_direct is None:
        history_total = sum_expr([recent_total, history_patent, history_trial])
    else:
        history_total = history_total_direct

    growth_total = numeric_expr(
        columns,
        [
            "growth_total_3yr_vs_prev3yr",
            "growth_total_3yr_ratio",
            "growth_total",
        ],
        default="0.0",
    )
    growth_paper = numeric_expr(
        columns,
        [
            "growth_paper_3yr_vs_prev3yr",
            "growth_paper_3yr_ratio",
            "growth_paper",
        ],
        default="0.0",
    )

    con.execute(
        f"""
        CREATE TEMP TABLE ku_cutoff_state AS
        SELECT
          ku_id_norm,
          cutoff_year_norm,

          {recent_paper}::DOUBLE AS d_recent_3yr_paper,
          {recent_patent}::DOUBLE AS d_recent_3yr_patent,
          {recent_trial}::DOUBLE AS d_recent_3yr_trial,
          {recent_translation}::DOUBLE AS d_recent_3yr_translation,
          {recent_total}::DOUBLE AS d_recent_3yr_total,

          {history_patent}::DOUBLE AS d_history_patent,
          {history_trial}::DOUBLE AS d_history_trial,
          {history_translation}::DOUBLE AS d_history_translation,
          {history_total}::DOUBLE AS d_history_total,

          {growth_total}::DOUBLE AS d_growth_total,
          {growth_paper}::DOUBLE AS d_growth_paper

        FROM prediction_examples
        WHERE cutoff_year_norm IN (SELECT cutoff_year FROM cutoff_years)
        """
    )

    rows = con.execute("SELECT COUNT(*) FROM ku_cutoff_state").fetchone()[0]
    return {"ku_cutoff_state_rows": int(rows)}


def build_diffusion_proxy_features(
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    *,
    feature_columns: list[str],
) -> tuple[Path, dict[str, Any]]:
    """Build diffusion proxy features using per-cutoff aggregation.

    Previous implementation created one giant neighbor_state table:

        ku_ku_edges x all cutoff states

    For full 2005-2021 cutoffs this can easily reach hundreds of millions of
    rows. This implementation processes one cutoff at a time:

        for cutoff:
            ku_ku_edges x ku_cutoff_state[cutoff]
            aggregate
            write cutoff parquet

    Then it concatenates the cutoff parquet files into the final diffusion
    feature table. This substantially reduces peak memory pressure.
    """

    cutoff_rows = con.execute(
        """
        SELECT DISTINCT cutoff_year
        FROM cutoff_years
        ORDER BY cutoff_year
        """
    ).fetchall()
    cutoffs = [int(row[0]) for row in cutoff_rows]

    if not cutoffs:
        raise RuntimeError("No cutoff years available in cutoff_years table.")

    cutoff_output_dir = args.output_dir / "intermediate" / "diffusion_proxy_by_cutoff"
    cutoff_output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale per-cutoff files when output dir is reused.
    for old_file in cutoff_output_dir.glob("diffusion_proxy_cutoff_*.parquet"):
        old_file.unlink()

    top10_feature_columns = {
        "kud_top10_recent_translation_mean",
        "kud_top10_recent_patent_mean",
        "kud_top10_recent_trial_mean",
    }

    feature_expr_parts = []
    for col in feature_columns:
        if col in top10_feature_columns:
            feature_expr_parts.append(
                f"COALESCE(t.{col}, 0.0)::DOUBLE AS {col}"
            )
        else:
            feature_expr_parts.append(
                f"COALESCE(a.{col}, 0.0)::DOUBLE AS {col}"
            )

    feature_exprs = ",\n            ".join(feature_expr_parts)

    metric_names = [
        "recent_3yr_paper",
        "recent_3yr_patent",
        "recent_3yr_trial",
        "recent_3yr_translation",
        "recent_3yr_total",
        "history_patent",
        "history_trial",
        "history_translation",
        "history_total",
        "growth_total",
        "growth_paper",
    ]

    aggregate_exprs = []
    for metric in metric_names:
        source = f"d_{metric}"
        aggregate_exprs.extend(
            [
                f"AVG({source})::DOUBLE AS kud_{metric}_mean",
                f"MAX({source})::DOUBLE AS kud_{metric}_max",
                f"SUM({source})::DOUBLE AS kud_{metric}_sum",
                (
                    f"(SUM(edge_weight * {source}) / "
                    f"NULLIF(SUM(edge_weight), 0.0))::DOUBLE "
                    f"AS kud_{metric}_weighted_mean"
                ),
            ]
        )

    aggregate_expr_sql = ",\n            ".join(aggregate_exprs)

    inactive_filter = ""
    if not args.include_inactive_neighbors:
        inactive_filter = """
        AND (
          s.d_history_total > 0
          OR s.d_recent_3yr_total > 0
          OR s.d_history_patent > 0
          OR s.d_history_trial > 0
          OR s.d_history_translation > 0
        )
        """

    active_neighbor_condition = """
        (
          d_history_total > 0
          OR d_recent_3yr_total > 0
          OR d_history_patent > 0
          OR d_history_trial > 0
          OR d_history_translation > 0
        )
    """

    per_cutoff_summaries: list[dict[str, Any]] = []
    cutoff_paths: list[Path] = []

    total_neighbor_state_rows = 0
    total_feature_rows = 0
    total_nonzero_neighbor_rows = 0

    log(args, "Building diffusion proxy features by cutoff ...")

    for cutoff in cutoffs:
        log(args, f"  cutoff {cutoff}: joining KU-KU graph to neighbor states ...")

        con.execute("DROP TABLE IF EXISTS neighbor_state_cutoff")
        con.execute("DROP TABLE IF EXISTS diffusion_agg_cutoff")
        con.execute("DROP TABLE IF EXISTS diffusion_top10_cutoff")

        con.execute(
            f"""
            CREATE TEMP TABLE neighbor_state_cutoff AS
            SELECT
              e.src_ku_id AS ku_id_norm,
              s.cutoff_year_norm,
              e.dst_ku_id AS neighbor_ku_id,
              e.edge_weight,
              e.shared_entity_count,

              s.d_recent_3yr_paper,
              s.d_recent_3yr_patent,
              s.d_recent_3yr_trial,
              s.d_recent_3yr_translation,
              s.d_recent_3yr_total,
              s.d_history_patent,
              s.d_history_trial,
              s.d_history_translation,
              s.d_history_total,
              s.d_growth_total,
              s.d_growth_paper

            FROM ku_ku_edges e
            JOIN ku_cutoff_state s
              ON e.dst_ku_id = s.ku_id_norm
             AND s.cutoff_year_norm = {int(cutoff)}
            WHERE 1 = 1
            {inactive_filter}
            """
        )

        neighbor_rows = con.execute(
            "SELECT COUNT(*) FROM neighbor_state_cutoff"
        ).fetchone()[0]

        total_neighbor_state_rows += int(neighbor_rows)

        log(args, f"    neighbor_state rows: {int(neighbor_rows):,}")

        log(args, f"  cutoff {cutoff}: aggregating neighbor diffusion states ...")

        con.execute(
            f"""
            CREATE TEMP TABLE diffusion_agg_cutoff AS
            SELECT
              ku_id_norm,
              cutoff_year_norm,

              COUNT(*)::DOUBLE AS kud_neighbor_count,

              SUM(
                CASE WHEN {active_neighbor_condition}
                THEN 1.0 ELSE 0.0 END
              )::DOUBLE AS kud_active_neighbor_count,

              SUM(edge_weight)::DOUBLE AS kud_neighbor_weight_sum,
              AVG(shared_entity_count)::DOUBLE AS kud_shared_entity_mean,
              MAX(shared_entity_count)::DOUBLE AS kud_shared_entity_max,

              AVG(CASE WHEN d_history_patent > 0 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS kud_history_patent_any_fraction,
              AVG(CASE WHEN d_history_trial > 0 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS kud_history_trial_any_fraction,
              AVG(CASE WHEN d_history_translation > 0 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS kud_history_translation_any_fraction,

              {aggregate_expr_sql}

            FROM neighbor_state_cutoff
            GROUP BY 1, 2
            """
        )

        con.execute(
            """
            CREATE TEMP TABLE diffusion_top10_cutoff AS
            WITH ranked AS (
              SELECT
                ku_id_norm,
                cutoff_year_norm,
                d_recent_3yr_translation,
                d_recent_3yr_patent,
                d_recent_3yr_trial,

                ROW_NUMBER() OVER (
                  PARTITION BY ku_id_norm, cutoff_year_norm
                  ORDER BY d_recent_3yr_translation DESC, edge_weight DESC, neighbor_ku_id
                ) AS rn_translation,

                ROW_NUMBER() OVER (
                  PARTITION BY ku_id_norm, cutoff_year_norm
                  ORDER BY d_recent_3yr_patent DESC, edge_weight DESC, neighbor_ku_id
                ) AS rn_patent,

                ROW_NUMBER() OVER (
                  PARTITION BY ku_id_norm, cutoff_year_norm
                  ORDER BY d_recent_3yr_trial DESC, edge_weight DESC, neighbor_ku_id
                ) AS rn_trial

              FROM neighbor_state_cutoff
            )
            SELECT
              ku_id_norm,
              cutoff_year_norm,

              AVG(
                CASE WHEN rn_translation <= 10
                THEN d_recent_3yr_translation ELSE NULL END
              )::DOUBLE AS kud_top10_recent_translation_mean,

              AVG(
                CASE WHEN rn_patent <= 10
                THEN d_recent_3yr_patent ELSE NULL END
              )::DOUBLE AS kud_top10_recent_patent_mean,

              AVG(
                CASE WHEN rn_trial <= 10
                THEN d_recent_3yr_trial ELSE NULL END
              )::DOUBLE AS kud_top10_recent_trial_mean

            FROM ranked
            GROUP BY 1, 2
            """
        )

        cutoff_path = cutoff_output_dir / f"diffusion_proxy_cutoff_{cutoff}.parquet"
        cutoff_paths.append(cutoff_path)

        log(args, f"  cutoff {cutoff}: writing diffusion features ...")

        con.execute(
            f"""
            COPY (
              SELECT
                p.ku_id_norm,
                p.cutoff_year_norm,
                {feature_exprs}

              FROM (
                SELECT DISTINCT ku_id_norm, cutoff_year_norm
                FROM prediction_examples
                WHERE cutoff_year_norm = {int(cutoff)}
              ) p

              LEFT JOIN diffusion_agg_cutoff a
                ON p.ku_id_norm = a.ku_id_norm
               AND p.cutoff_year_norm = a.cutoff_year_norm

              LEFT JOIN diffusion_top10_cutoff t
                ON p.ku_id_norm = t.ku_id_norm
               AND p.cutoff_year_norm = t.cutoff_year_norm
            )
            TO '{duck_path(cutoff_path)}'
            (FORMAT PARQUET)
            """
        )

        feature_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{duck_path(cutoff_path)}')"
        ).fetchone()[0]

        nonzero_neighbor_rows = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{duck_path(cutoff_path)}')
            WHERE kud_neighbor_count > 0
            """
        ).fetchone()[0]

        total_feature_rows += int(feature_rows)
        total_nonzero_neighbor_rows += int(nonzero_neighbor_rows)

        per_cutoff_summaries.append(
            {
                "cutoff_year": int(cutoff),
                "neighbor_state_rows": int(neighbor_rows),
                "feature_rows": int(feature_rows),
                "rows_with_nonzero_neighbors": int(nonzero_neighbor_rows),
                "rows_with_nonzero_neighbors_fraction": (
                    float(nonzero_neighbor_rows / feature_rows)
                    if feature_rows
                    else 0.0
                ),
                "path": str(cutoff_path),
            }
        )

        log(
            args,
            (
                f"    cutoff {cutoff} done: "
                f"feature_rows={int(feature_rows):,}, "
                f"nonzero_neighbor_rows={int(nonzero_neighbor_rows):,}"
            ),
        )

        # Drop large temporary tables before the next cutoff.
        con.execute("DROP TABLE IF EXISTS neighbor_state_cutoff")
        con.execute("DROP TABLE IF EXISTS diffusion_agg_cutoff")
        con.execute("DROP TABLE IF EXISTS diffusion_top10_cutoff")

    diffusion_path = args.output_dir / "intermediate" / "diffusion_proxy_features.parquet"

    log(args, "Concatenating per-cutoff diffusion feature files ...")

    glob_path = str(cutoff_output_dir / "diffusion_proxy_cutoff_*.parquet")
    glob_path = glob_path.replace("\\", "/").replace("'", "''")

    con.execute(
        f"""
        COPY (
          SELECT *
          FROM read_parquet('{glob_path}')
        )
        TO '{duck_path(diffusion_path)}'
        (FORMAT PARQUET)
        """
    )

    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{duck_path(diffusion_path)}')"
    ).fetchone()[0]

    nonzero_neighbor_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{duck_path(diffusion_path)}')
        WHERE kud_neighbor_count > 0
        """
    ).fetchone()[0]

    log(args, "Writing diffusion feature diagnostics ...")

    diagnostics = []
    for col in feature_columns:
        nonzero, mean_value, max_value = con.execute(
            f"""
            SELECT
              SUM(CASE WHEN {col} <> 0 THEN 1 ELSE 0 END),
              AVG({col}),
              MAX({col})
            FROM read_parquet('{duck_path(diffusion_path)}')
            """
        ).fetchone()

        diagnostics.append(
            {
                "feature": col,
                "nonzero_rows": int(nonzero or 0),
                "nonzero_fraction": float((nonzero or 0) / rows) if rows else 0.0,
                "mean": float(mean_value or 0.0),
                "max": float(max_value or 0.0),
            }
        )

    diagnostics_path = args.output_dir / "diffusion_proxy_feature_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)

    per_cutoff_summary_path = args.output_dir / "diffusion_proxy_per_cutoff_summary.csv"
    pd.DataFrame(per_cutoff_summaries).to_csv(per_cutoff_summary_path, index=False)

    return diffusion_path, {
        "diffusion_feature_rows": int(rows),
        "neighbor_state_rows": int(total_neighbor_state_rows),
        "rows_with_nonzero_neighbors": int(nonzero_neighbor_rows),
        "rows_with_nonzero_neighbors_fraction": (
            float(nonzero_neighbor_rows / rows) if rows else 0.0
        ),
        "per_cutoff_summary_path": str(per_cutoff_summary_path),
        "per_cutoff_summaries": per_cutoff_summaries,
        "diffusion_feature_path": str(diffusion_path),
        "feature_diagnostics_path": str(diagnostics_path),
        "per_cutoff_output_dir": str(cutoff_output_dir),
    }


# ---------------------------------------------------------------------------
# Augment feature tables
# ---------------------------------------------------------------------------


def augment_prediction_table(
    con: duckdb.DuckDBPyConnection,
    *,
    input_table_path: Path,
    output_table_path: Path,
    diffusion_feature_path: Path,
    feature_columns: list[str],
) -> dict[str, Any]:
    input_columns = get_parquet_columns(con, input_table_path)

    ku_col = find_column(
        input_columns,
        ["ku_id", "knowledge_unit_id", "ku_id_norm"],
        contains_all=["ku", "id"],
        label=f"KU id in {input_table_path}",
    )
    cutoff_col = find_column(
        input_columns,
        ["cutoff_year", "cutoff_year_norm"],
        contains_all=["cutoff"],
        label=f"cutoff year in {input_table_path}",
    )

    feature_exprs = ",\n          ".join(
        [f"COALESCE(f.{col}, 0.0)::DOUBLE AS {col}" for col in feature_columns]
    )

    con.execute(
        f"""
        COPY (
          SELECT
            p.*,
            {feature_exprs}
          FROM read_parquet('{duck_path(input_table_path)}') p
          LEFT JOIN read_parquet('{duck_path(diffusion_feature_path)}') f
            ON CAST(p.{q(ku_col)} AS VARCHAR) = f.ku_id_norm
           AND TRY_CAST(p.{q(cutoff_col)} AS INTEGER) = f.cutoff_year_norm
        )
        TO '{duck_path(output_table_path)}'
        (FORMAT PARQUET)
        """
    )

    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{duck_path(output_table_path)}')"
    ).fetchone()[0]

    return {
        "input_table": str(input_table_path),
        "output_table": str(output_table_path),
        "ku_col": ku_col,
        "cutoff_col": cutoff_col,
        "rows": int(rows),
    }


def create_feature_set_dir(
    *,
    feature_set_dir: Path,
    table_path: Path,
    target_spec_path: Path,
    feature_spec: dict[str, Any],
) -> dict[str, Any]:
    feature_set_dir.mkdir(parents=True, exist_ok=True)

    mode = link_or_copy_file(
        table_path,
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
    pd.DataFrame(rows).to_csv(output_dir / "diffusion_proxy_manifest.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = normalize_args(parse_args())

    validate_prediction_dir(args.prediction_dir)

    start = time.time()
    prepare_output_dir(args.output_dir, args.overwrite)

    write_json(args.output_dir / "run_args.json", vars(args))

    con = configure_duckdb(args)

    base_table_path = args.prediction_dir / "prediction_feature_table.parquet"
    base_feature_spec_path = args.prediction_dir / "feature_columns.json"
    target_spec_path = args.prediction_dir / "target_columns.json"

    base_feature_spec = read_json(base_feature_spec_path)
    base_numeric = feature_spec_numeric_columns(base_feature_spec)
    base_categorical = feature_spec_categorical_columns(base_feature_spec)

    diffusion_columns = diffusion_proxy_feature_columns()

    log(args, "Building prediction examples ...")
    prediction_summary = build_prediction_examples(con, args)
    log(args, f"Using cutoffs: {prediction_summary['cutoffs']}")

    log(args, "Building KU-KU shared-entity graph ...")
    graph_summary = build_ku_ku_graph(
        con,
        args,
        prediction_summary=prediction_summary,
    )
    log(args, f"KU-KU kept edges: {graph_summary['kept_edge_count']:,}")

    state_summary = build_ku_cutoff_state(
        con,
        args,
        prediction_summary=prediction_summary,
    )

    diffusion_feature_path, diffusion_summary = build_diffusion_proxy_features(
        con,
        args,
        feature_columns=diffusion_columns,
    )

    log(args, "Augmenting base KU prediction table with diffusion features ...")
    ku_plus_diffusion_table = (
        args.output_dir / "prediction_feature_table_with_diffusion_proxy.parquet"
    )
    ku_plus_diffusion_summary = augment_prediction_table(
        con,
        input_table_path=base_table_path,
        output_table_path=ku_plus_diffusion_table,
        diffusion_feature_path=diffusion_feature_path,
        feature_columns=diffusion_columns,
    )

    diffusion_only_spec = make_feature_spec(
        base_spec=base_feature_spec,
        numeric_columns=diffusion_columns,
        categorical_columns=[],
        description=(
            "Stage 06G diffusion_proxy_only representation. "
            "Features are cutoff-safe KU-KU shared-entity neighborhood "
            "temporal-state aggregation features."
        ),
    )

    ku_plus_diffusion_spec = make_feature_spec(
        base_spec=base_feature_spec,
        numeric_columns=base_numeric + diffusion_columns,
        categorical_columns=base_categorical,
        description=(
            "Stage 06G KU full representation plus cutoff-safe KU-KU diffusion "
            "proxy features."
        ),
    )

    write_json(
        args.output_dir / "diffusion_proxy_feature_columns.json",
        diffusion_only_spec,
    )
    write_json(
        args.output_dir / "ku_plus_diffusion_proxy_feature_columns.json",
        ku_plus_diffusion_spec,
    )

    feature_sets_dir = args.output_dir / "feature_sets"

    diffusion_only_set = create_feature_set_dir(
        feature_set_dir=feature_sets_dir / "diffusion_proxy_only",
        table_path=ku_plus_diffusion_table,
        target_spec_path=target_spec_path,
        feature_spec=diffusion_only_spec,
    )

    ku_plus_diffusion_set = create_feature_set_dir(
        feature_set_dir=feature_sets_dir / "ku_plus_diffusion_proxy",
        table_path=ku_plus_diffusion_table,
        target_spec_path=target_spec_path,
        feature_spec=ku_plus_diffusion_spec,
    )

    carrier_plus_diffusion_set = None
    carrier_plus_diffusion_summary = None

    carrier_dir = args.carrier_structural_feature_set_dir
    if carrier_dir and (carrier_dir / "prediction_feature_table.parquet").exists():
        log(args, "Augmenting KU+carrier-structural feature table with diffusion features ...")

        carrier_table_path = carrier_dir / "prediction_feature_table.parquet"
        carrier_feature_spec_path = carrier_dir / "feature_columns.json"
        carrier_target_spec_path = carrier_dir / "target_columns.json"

        carrier_feature_spec = read_json(carrier_feature_spec_path)
        carrier_numeric = feature_spec_numeric_columns(carrier_feature_spec)
        carrier_categorical = feature_spec_categorical_columns(carrier_feature_spec)

        carrier_plus_diffusion_table = (
            args.output_dir
            / "prediction_feature_table_with_carrier_structural_and_diffusion_proxy.parquet"
        )

        carrier_plus_diffusion_summary = augment_prediction_table(
            con,
            input_table_path=carrier_table_path,
            output_table_path=carrier_plus_diffusion_table,
            diffusion_feature_path=diffusion_feature_path,
            feature_columns=diffusion_columns,
        )

        carrier_plus_diffusion_spec = make_feature_spec(
            base_spec=carrier_feature_spec,
            numeric_columns=carrier_numeric + diffusion_columns,
            categorical_columns=carrier_categorical,
            description=(
                "Stage 06G KU full plus carrier graph structural readout plus "
                "cutoff-safe KU-KU diffusion proxy features."
            ),
        )

        write_json(
            args.output_dir
            / "ku_plus_carrier_structural_plus_diffusion_proxy_feature_columns.json",
            carrier_plus_diffusion_spec,
        )

        carrier_plus_diffusion_set = create_feature_set_dir(
            feature_set_dir=feature_sets_dir / "ku_plus_carrier_structural_plus_diffusion_proxy",
            table_path=carrier_plus_diffusion_table,
            target_spec_path=carrier_target_spec_path,
            feature_spec=carrier_plus_diffusion_spec,
        )
    else:
        log(
            args,
            "Carrier structural feature set not found; skipping "
            "ku_plus_carrier_structural_plus_diffusion_proxy.",
        )

    runtime_seconds = time.time() - start

    summary = {
        "script": "scripts/32_build_knowledge_unit_diffusion_proxy_features.py",
        "status": "ok",
        "prediction_summary": prediction_summary,
        "graph_summary": graph_summary,
        "state_summary": state_summary,
        "diffusion_summary": diffusion_summary,
        "ku_plus_diffusion_summary": ku_plus_diffusion_summary,
        "carrier_plus_diffusion_summary": carrier_plus_diffusion_summary,
        "diffusion_feature_count": len(diffusion_columns),
        "base_numeric_feature_count": len(base_numeric),
        "base_categorical_feature_count": len(base_categorical),
        "diffusion_proxy_only_feature_set": diffusion_only_set,
        "ku_plus_diffusion_proxy_feature_set": ku_plus_diffusion_set,
        "ku_plus_carrier_structural_plus_diffusion_proxy_feature_set": carrier_plus_diffusion_set,
        "include_inactive_neighbors": bool(args.include_inactive_neighbors),
        "runtime_seconds": runtime_seconds,
    }

    write_json(args.output_dir / "diffusion_proxy_summary.json", summary)
    write_manifest(args.output_dir)

    log(args, "")
    log(args, "KU-KU diffusion proxy feature construction complete.")
    log(args, f"Output directory: {args.output_dir}")
    log(args, f"Diffusion-only feature set: {feature_sets_dir / 'diffusion_proxy_only'}")
    log(args, f"KU + diffusion feature set: {feature_sets_dir / 'ku_plus_diffusion_proxy'}")
    if carrier_plus_diffusion_set:
        log(
            args,
            "KU + carrier structural + diffusion feature set: "
            f"{feature_sets_dir / 'ku_plus_carrier_structural_plus_diffusion_proxy'}",
        )
    log(args, f"Runtime seconds: {runtime_seconds:.1f}")


if __name__ == "__main__":
    main()