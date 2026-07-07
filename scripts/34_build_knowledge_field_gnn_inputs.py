#!/usr/bin/env python
"""Build algorithm-ready Temporal Knowledge Field GNN inputs.

Purpose
-------

This script converts a latest-window KU prediction-style dataset into
algorithm-ready GNN inputs.

Input is a prediction-style dataset root, for example:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural_latest_split/

or:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy_latest_split/

Expected input files:

    prediction_feature_table.parquet
    feature_columns.json
    target_columns.json
    latest_split_summary.json

It also needs the KU evidence dataset root:

    data/datasets/knowledge_units/diabetes_2000_2024_v1_translational/

Expected KU file:

    knowledge_units.parquet

Output is a direct GNN input directory, for example:

    data/datasets/knowledge_field/
      diabetes_2000_2024_v1_gnn_local_source_latest_split/

Main outputs:

    graph/
      ku_node_index.parquet
      ku_ku_edges.parquet
      edge_index.npy
      edge_weight.npy
      graph_summary.json

    snapshots/
      train/
        cutoff_2005/
          x_numeric.npy
          x_categorical.npy
          y_translation_heat.npy
          y_translation_label.npy
          mask_translation_eligible.npy
          ...
      validation/
        cutoff_2019/
          ...
      test/
        cutoff_2021/
          ...

    feature_schema.json
    categorical_vocab.json
    split_manifest.json
    gnn_input_summary.json
    gnn_input_manifest.csv

Design
------

Rows in prediction_feature_table are KU x cutoff_year.

GNN training usually needs:

    node_idx
    edge_index / edge_weight
    X_t for each cutoff snapshot
    y_t and eligible masks for each task

This script materializes those directly as numpy arrays.

Recommended first Stage 07 GNN input
------------------------------------

Use local + carrier source/exposure features, without handcrafted kud_* proxy,
so that the GNN learns diffusion from the KU-KU graph:

    --feature-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural_latest_split

Proxy-augmented GNN input
-------------------------

To also feed handcrafted diffusion proxy features:

    --feature-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy_latest_split

Graph reuse
-----------

If you build multiple feature versions, you do not need to rebuild the KU-KU
graph every time. Use:

    --graph-source-dir <previous_gnn_input_dir>/graph

Example
-------

    python scripts/34_build_knowledge_field_gnn_inputs.py \\
      --feature-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural_latest_split \\
      --ku-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational \\
      --output-dir data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_latest_split \\
      --max-entity-degree 500 \\
      --top-neighbors 30 \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


DEFAULT_FEATURE_DIR = Path(
    "data/datasets/knowledge_units/"
    "diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural_latest_split"
)

DEFAULT_KU_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_translational"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/datasets/knowledge_field/"
    "diabetes_2000_2024_v1_gnn_local_source_latest_split"
)

TASKS = ["translation", "patent", "trial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build algorithm-ready Temporal Knowledge Field GNN inputs."
    )

    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help=(
            "Latest-split prediction-style dataset root. Must contain "
            "prediction_feature_table.parquet, feature_columns.json, "
            "and target_columns.json."
        ),
    )

    parser.add_argument(
        "--ku-dir",
        type=Path,
        default=DEFAULT_KU_DIR,
        help="KU evidence dataset root containing knowledge_units.parquet.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output GNN input dataset directory.",
    )

    parser.add_argument(
        "--graph-source-dir",
        type=Path,
        default=None,
        help=(
            "Optional existing graph directory to reuse. If provided, should "
            "contain ku_node_index.parquet, ku_ku_edges.parquet, edge_index.npy, "
            "edge_weight.npy. This avoids rebuilding KU-KU graph."
        ),
    )

    parser.add_argument(
        "--max-entity-degree",
        type=int,
        default=500,
        help=(
            "Maximum number of KUs incident to an entity when constructing "
            "KU-KU shared-entity edges. Hub entities above this are ignored."
        ),
    )

    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=30,
        help="Keep top-K outgoing neighbors per KU by edge weight.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Rows per DuckDB Arrow batch when materializing numpy arrays.",
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
        "--temp-dir",
        type=Path,
        default=Path("data/tmp/duckdb_stage07_gnn_inputs"),
        help="DuckDB temp/spill directory.",
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
    args.feature_dir = args.feature_dir.resolve()
    args.ku_dir = args.ku_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()

    if args.graph_source_dir is not None:
        args.graph_source_dir = args.graph_source_dir.resolve()

    if args.max_entity_degree <= 1:
        raise ValueError("--max-entity-degree must be > 1.")

    if args.top_neighbors <= 0:
        raise ValueError("--top-neighbors must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    return args


def log(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, flush=True)


def duck_path(path: Path) -> str:
    text = str(path.resolve()).replace("\\", "/")
    return text.replace("'", "''")


def q(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def configure_duckdb(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(args.threads)}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{duck_path(args.temp_dir)}'")
    return con


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        args.feature_dir / "prediction_feature_table.parquet",
        args.feature_dir / "feature_columns.json",
        args.feature_dir / "target_columns.json",
        args.ku_dir / "knowledge_units.parquet",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(str(p) for p in missing)
        )

    if args.graph_source_dir is not None:
        required_graph = [
            args.graph_source_dir / "ku_node_index.parquet",
            args.graph_source_dir / "ku_ku_edges.parquet",
            args.graph_source_dir / "edge_index.npy",
            args.graph_source_dir / "edge_weight.npy",
        ]
        missing_graph = [p for p in required_graph if not p.exists()]
        if missing_graph:
            raise FileNotFoundError(
                "Missing required graph-source files:\n"
                + "\n".join(str(p) for p in missing_graph)
            )


def get_parquet_columns(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
) -> dict[str, str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{duck_path(parquet_path)}')"
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def infer_column(columns: dict[str, str], candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in columns}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    if required:
        raise ValueError(f"Could not infer required column from candidates={candidates}")

    return None


def is_numeric_duckdb_type(dtype: str) -> bool:
    d = dtype.upper()
    numeric_tokens = [
        "INT",
        "BIGINT",
        "SMALLINT",
        "TINYINT",
        "UBIGINT",
        "UINTEGER",
        "USMALLINT",
        "UTINYINT",
        "DOUBLE",
        "FLOAT",
        "REAL",
        "DECIMAL",
        "HUGEINT",
        "BOOLEAN",
    ]
    return any(tok in d for tok in numeric_tokens)


def extract_feature_columns_from_spec(
    feature_spec: dict[str, Any],
    table_columns: dict[str, str],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return numeric and categorical feature columns.

    This function supports several possible feature_columns.json layouts.
    """

    def list_from_keys(keys: list[str]) -> list[str]:
        out: list[str] = []
        for key in keys:
            value = feature_spec.get(key)
            if isinstance(value, list):
                out.extend(str(x) for x in value)
        return out

    numeric = list_from_keys(
        [
            "numeric_features",
            "numeric_feature_columns",
            "numeric_columns",
            "numerical_features",
            "numerical_columns",
        ]
    )

    categorical = list_from_keys(
        [
            "categorical_features",
            "categorical_feature_columns",
            "categorical_columns",
            "category_features",
        ]
    )

    explicit_all = list_from_keys(
        [
            "feature_columns",
            "features",
            "all_features",
            "all_feature_columns",
            "model_features",
        ]
    )

    if not numeric and not categorical and explicit_all:
        for col in explicit_all:
            if col not in table_columns:
                continue
            if is_numeric_duckdb_type(table_columns[col]):
                numeric.append(col)
            else:
                categorical.append(col)

    # Remove columns not present and deduplicate while preserving order.
    def clean(cols: list[str]) -> list[str]:
        seen = set()
        out = []
        for col in cols:
            if col not in table_columns:
                continue
            if col in seen:
                continue
            seen.add(col)
            out.append(col)
        return out

    numeric = clean(numeric)
    categorical = clean(categorical)

    # Avoid accidental overlap.
    cat_set = set(categorical)
    numeric = [col for col in numeric if col not in cat_set]

    forbidden_prefixes = (
        "target_future_",
        "label_future_",
        "eligible_",
    )

    forbidden_exact = {
        "split",
        "is_train",
        "is_validation",
        "is_val",
        "is_test",
        "train_mask",
        "validation_mask",
        "val_mask",
        "test_mask",
    }

    def allowed(col: str) -> bool:
        if col in forbidden_exact:
            return False
        return not any(col.startswith(p) for p in forbidden_prefixes)

    numeric = [c for c in numeric if allowed(c)]
    categorical = [c for c in categorical if allowed(c)]

    if not numeric and not categorical:
        raise ValueError(
            "Could not identify any feature columns from feature_columns.json. "
            "Expected keys such as numeric_features/categorical_features or "
            "feature_columns."
        )

    schema_info = {
        "feature_spec_keys": sorted(feature_spec.keys()),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "numeric_features": numeric,
        "categorical_features": categorical,
    }

    return numeric, categorical, schema_info


def infer_split_cutoffs(
    con: duckdb.DuckDBPyConnection,
    feature_table: Path,
) -> dict[str, list[int]]:
    rows = con.execute(
        f"""
        SELECT
          split,
          TRY_CAST(cutoff_year AS INTEGER) AS cutoff_year,
          COUNT(*) AS row_count
        FROM read_parquet('{duck_path(feature_table)}')
        GROUP BY 1, 2
        ORDER BY 2, 1
        """
    ).fetchdf()

    out: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for split_name in out:
        vals = rows.loc[rows["split"] == split_name, "cutoff_year"].tolist()
        out[split_name] = [int(v) for v in vals]

    return out


def build_or_copy_graph(
    *,
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    graph_dir: Path,
) -> dict[str, Any]:
    graph_dir.mkdir(parents=True, exist_ok=True)

    if args.graph_source_dir is not None:
        log(args, f"Reusing graph artifacts from: {args.graph_source_dir}")

        for filename in [
            "ku_node_index.parquet",
            "ku_ku_edges.parquet",
            "edge_index.npy",
            "edge_weight.npy",
            "graph_summary.json",
        ]:
            src = args.graph_source_dir / filename
            if src.exists():
                shutil.copy2(src, graph_dir / filename)

        summary_path = graph_dir / "graph_summary.json"
        if summary_path.exists():
            return read_json(summary_path)

        # Fallback summary.
        edge_count = int(np.load(graph_dir / "edge_weight.npy", mmap_mode="r").shape[0])
        node_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{duck_path(graph_dir / "ku_node_index.parquet")}')
            """
        ).fetchone()[0]

        summary = {
            "status": "ok",
            "reused_from": str(args.graph_source_dir),
            "node_count": int(node_count),
            "edge_count": edge_count,
        }
        write_json(summary_path, summary)
        return summary

    log(args, "Building KU node index and KU-KU shared-entity graph ...")

    feature_table = args.feature_dir / "prediction_feature_table.parquet"
    ku_table = args.ku_dir / "knowledge_units.parquet"

    feature_columns = get_parquet_columns(con, feature_table)
    ku_columns = get_parquet_columns(con, ku_table)

    feature_ku_col = infer_column(feature_columns, ["ku_id", "knowledge_unit_id"])
    ku_ku_col = infer_column(ku_columns, ["ku_id", "knowledge_unit_id"])
    entity_a_col = infer_column(ku_columns, ["entity_a_id", "entity1_id", "source_entity_id"])
    entity_b_col = infer_column(ku_columns, ["entity_b_id", "entity2_id", "target_entity_id"])

    pair_type_col = infer_column(ku_columns, ["pair_type"], required=False)
    entity_a_type_col = infer_column(ku_columns, ["entity_a_type"], required=False)
    entity_b_type_col = infer_column(ku_columns, ["entity_b_type"], required=False)
    entity_a_name_col = infer_column(ku_columns, ["entity_a_name"], required=False)
    entity_b_name_col = infer_column(ku_columns, ["entity_b_name"], required=False)

    optional_selects = []
    for col, alias in [
        (pair_type_col, "pair_type"),
        (entity_a_type_col, "entity_a_type"),
        (entity_b_type_col, "entity_b_type"),
        (entity_a_name_col, "entity_a_name"),
        (entity_b_name_col, "entity_b_name"),
    ]:
        if col is not None:
            optional_selects.append(f"k.{q(col)} AS {q(alias)}")
        else:
            optional_selects.append(f"NULL AS {q(alias)}")

    node_index_path = graph_dir / "ku_node_index.parquet"

    log(args, "  Writing ku_node_index.parquet ...")
    con.execute(
        f"""
        COPY (
          SELECT
            CAST(ROW_NUMBER() OVER (ORDER BY f.{q(feature_ku_col)}) - 1 AS BIGINT)
              AS node_idx,
            f.{q(feature_ku_col)} AS ku_id,
            k.{q(entity_a_col)} AS entity_a_id,
            k.{q(entity_b_col)} AS entity_b_id,
            {", ".join(optional_selects)}
          FROM (
            SELECT DISTINCT {q(feature_ku_col)}
            FROM read_parquet('{duck_path(feature_table)}')
          ) AS f
          LEFT JOIN read_parquet('{duck_path(ku_table)}') AS k
            ON f.{q(feature_ku_col)} = k.{q(ku_ku_col)}
          ORDER BY node_idx
        )
        TO '{duck_path(node_index_path)}'
        (FORMAT PARQUET)
        """
    )

    node_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{duck_path(node_index_path)}')"
    ).fetchone()[0]

    null_entity_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{duck_path(node_index_path)}')
        WHERE entity_a_id IS NULL OR entity_b_id IS NULL
        """
    ).fetchone()[0]

    if int(null_entity_count) > 0:
        raise ValueError(
            f"Found {null_entity_count} nodes with missing entity_a_id/entity_b_id. "
            "Cannot construct KU-KU graph."
        )

    edges_path = graph_dir / "ku_ku_edges.parquet"

    log(args, "  Constructing shared-entity edges ...")
    log(args, f"    max_entity_degree = {args.max_entity_degree}")
    log(args, f"    top_neighbors     = {args.top_neighbors}")

    con.execute(
        f"""
        COPY (
          WITH node_entities AS (
            SELECT DISTINCT
              node_idx,
              ku_id,
              CAST(entity_a_id AS VARCHAR) AS entity_id
            FROM read_parquet('{duck_path(node_index_path)}')
            WHERE entity_a_id IS NOT NULL

            UNION

            SELECT DISTINCT
              node_idx,
              ku_id,
              CAST(entity_b_id AS VARCHAR) AS entity_id
            FROM read_parquet('{duck_path(node_index_path)}')
            WHERE entity_b_id IS NOT NULL
          ),
          entity_degrees AS (
            SELECT
              entity_id,
              COUNT(*) AS entity_degree
            FROM node_entities
            GROUP BY entity_id
            HAVING COUNT(*) BETWEEN 2 AND {int(args.max_entity_degree)}
          ),
          filtered_node_entities AS (
            SELECT
              ne.node_idx,
              ne.ku_id,
              ne.entity_id,
              ed.entity_degree
            FROM node_entities AS ne
            INNER JOIN entity_degrees AS ed
              ON ne.entity_id = ed.entity_id
          ),
          candidate_edges AS (
            SELECT
              a.node_idx AS src_node_idx,
              b.node_idx AS dst_node_idx,
              CAST(SUM(1.0 / LN(1.0 + a.entity_degree)) AS FLOAT) AS edge_weight,
              COUNT(*) AS shared_entity_count
            FROM filtered_node_entities AS a
            INNER JOIN filtered_node_entities AS b
              ON a.entity_id = b.entity_id
             AND a.node_idx <> b.node_idx
            GROUP BY 1, 2
          ),
          ranked_edges AS (
            SELECT
              src_node_idx,
              dst_node_idx,
              edge_weight,
              shared_entity_count,
              ROW_NUMBER() OVER (
                PARTITION BY src_node_idx
                ORDER BY edge_weight DESC, shared_entity_count DESC, dst_node_idx
              ) AS rn
            FROM candidate_edges
          )
          SELECT
            src_node_idx,
            dst_node_idx,
            edge_weight,
            shared_entity_count
          FROM ranked_edges
          WHERE rn <= {int(args.top_neighbors)}
          ORDER BY src_node_idx, dst_node_idx
        )
        TO '{duck_path(edges_path)}'
        (FORMAT PARQUET)
        """
    )

    edge_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{duck_path(edges_path)}')"
    ).fetchone()[0]

    log(args, f"  Nodes: {int(node_count):,}")
    log(args, f"  Directed edges: {int(edge_count):,}")

    materialize_edge_numpy(
        con=con,
        args=args,
        edges_path=edges_path,
        graph_dir=graph_dir,
        edge_count=int(edge_count),
    )

    entity_stats = con.execute(
        f"""
        WITH node_entities AS (
          SELECT DISTINCT CAST(entity_a_id AS VARCHAR) AS entity_id
          FROM read_parquet('{duck_path(node_index_path)}')
          WHERE entity_a_id IS NOT NULL
          UNION
          SELECT DISTINCT CAST(entity_b_id AS VARCHAR) AS entity_id
          FROM read_parquet('{duck_path(node_index_path)}')
          WHERE entity_b_id IS NOT NULL
        )
        SELECT COUNT(*) AS unique_entity_count
        FROM node_entities
        """
    ).fetchone()[0]

    graph_summary = {
        "status": "ok",
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "unique_entity_count": int(entity_stats),
        "max_entity_degree": int(args.max_entity_degree),
        "top_neighbors": int(args.top_neighbors),
        "node_index_path": str(node_index_path),
        "edge_parquet_path": str(edges_path),
        "edge_index_path": str(graph_dir / "edge_index.npy"),
        "edge_weight_path": str(graph_dir / "edge_weight.npy"),
    }

    write_json(graph_dir / "graph_summary.json", graph_summary)
    return graph_summary


def materialize_edge_numpy(
    *,
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    edges_path: Path,
    graph_dir: Path,
    edge_count: int,
) -> None:
    edge_index_path = graph_dir / "edge_index.npy"
    edge_weight_path = graph_dir / "edge_weight.npy"

    edge_index = np.lib.format.open_memmap(
        edge_index_path,
        mode="w+",
        dtype=np.int64,
        shape=(2, edge_count),
    )
    edge_weight = np.lib.format.open_memmap(
        edge_weight_path,
        mode="w+",
        dtype=np.float32,
        shape=(edge_count,),
    )

    query = f"""
        SELECT
          src_node_idx,
          dst_node_idx,
          edge_weight
        FROM read_parquet('{duck_path(edges_path)}')
        ORDER BY src_node_idx, dst_node_idx
    """

    reader = con.execute(query).to_arrow_reader(batch_size=args.batch_size)

    offset = 0
    for batch in reader:
        df = batch.to_pandas()
        n = len(df)
        edge_index[0, offset : offset + n] = df["src_node_idx"].to_numpy(np.int64)
        edge_index[1, offset : offset + n] = df["dst_node_idx"].to_numpy(np.int64)
        edge_weight[offset : offset + n] = df["edge_weight"].to_numpy(np.float32)
        offset += n

    if offset != edge_count:
        raise RuntimeError(f"Edge numpy materialization mismatch: {offset} != {edge_count}")

    edge_index.flush()
    edge_weight.flush()


def build_categorical_vocab(
    *,
    con: duckdb.DuckDBPyConnection,
    feature_table: Path,
    categorical_features: list[str],
) -> dict[str, dict[str, int]]:
    vocab: dict[str, dict[str, int]] = {}

    for col in categorical_features:
        df = con.execute(
            f"""
            SELECT DISTINCT CAST({q(col)} AS VARCHAR) AS value
            FROM read_parquet('{duck_path(feature_table)}')
            ORDER BY value
            """
        ).fetchdf()

        values = ["__MISSING__"]
        for value in df["value"].tolist():
            if value is None:
                continue
            text = str(value)
            if text not in values:
                values.append(text)

        vocab[col] = {value: idx for idx, value in enumerate(values)}

    return vocab


def cutoff_to_split_map(
    con: duckdb.DuckDBPyConnection,
    feature_table: Path,
) -> dict[int, str]:
    df = con.execute(
        f"""
        SELECT
          TRY_CAST(cutoff_year AS INTEGER) AS cutoff_year,
          split,
          COUNT(*) AS row_count
        FROM read_parquet('{duck_path(feature_table)}')
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()

    out: dict[int, str] = {}
    for _, row in df.iterrows():
        cutoff = int(row["cutoff_year"])
        split = str(row["split"])
        if cutoff in out and out[cutoff] != split:
            raise ValueError(f"Cutoff {cutoff} has multiple splits.")
        out[cutoff] = split

    return out


def get_node_count(
    con: duckdb.DuckDBPyConnection,
    graph_dir: Path,
) -> int:
    return int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{duck_path(graph_dir / "ku_node_index.parquet")}')
            """
        ).fetchone()[0]
    )


def target_columns_for_task(task: str) -> tuple[str, str, str]:
    return (
        f"target_future_{task}_heat_3yr",
        f"label_future_{task}_emergence_3yr",
        f"eligible_{task}_task",
    )


def ensure_required_task_columns(
    table_columns: dict[str, str],
    tasks: list[str],
) -> None:
    missing = []
    for task in tasks:
        for col in target_columns_for_task(task):
            if col not in table_columns:
                missing.append(col)

    if missing:
        raise ValueError(
            "Missing required task target/label/eligibility columns:\n"
            + "\n".join(missing)
        )


def materialize_snapshots(
    *,
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    graph_dir: Path,
    snapshots_dir: Path,
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_vocab: dict[str, dict[str, int]],
    tasks: list[str],
) -> dict[str, Any]:
    feature_table = args.feature_dir / "prediction_feature_table.parquet"
    node_index_path = graph_dir / "ku_node_index.parquet"

    node_count = get_node_count(con, graph_dir)
    cutoff_split = cutoff_to_split_map(con, feature_table)

    split_manifest: dict[str, Any] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    snapshot_summaries = []

    select_columns = ["n.node_idx"]
    select_columns += [f"p.{q(col)} AS {q(col)}" for col in numeric_features]
    select_columns += [f"p.{q(col)} AS {q(col)}" for col in categorical_features]

    for task in tasks:
        heat_col, label_col, eligible_col = target_columns_for_task(task)
        select_columns.extend(
            [
                f"p.{q(heat_col)} AS {q(heat_col)}",
                f"p.{q(label_col)} AS {q(label_col)}",
                f"p.{q(eligible_col)} AS {q(eligible_col)}",
            ]
        )

    select_sql = ",\n              ".join(select_columns)

    for cutoff, split in sorted(cutoff_split.items()):
        if split not in split_manifest:
            continue

        out_dir = snapshots_dir / split / f"cutoff_{cutoff}"
        out_dir.mkdir(parents=True, exist_ok=True)

        log(args, f"Materializing snapshot: split={split}, cutoff={cutoff}")

        x_numeric_path = out_dir / "x_numeric.npy"
        x_categorical_path = out_dir / "x_categorical.npy"

        x_numeric = np.lib.format.open_memmap(
            x_numeric_path,
            mode="w+",
            dtype=np.float32,
            shape=(node_count, len(numeric_features)),
        )

        x_categorical = np.lib.format.open_memmap(
            x_categorical_path,
            mode="w+",
            dtype=np.int64,
            shape=(node_count, len(categorical_features)),
        )

        label_arrays: dict[str, dict[str, np.memmap]] = {}

        for task in tasks:
            label_arrays[task] = {
                "heat": np.lib.format.open_memmap(
                    out_dir / f"y_{task}_heat.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(node_count,),
                ),
                "label": np.lib.format.open_memmap(
                    out_dir / f"y_{task}_label.npy",
                    mode="w+",
                    dtype=np.int8,
                    shape=(node_count,),
                ),
                "eligible": np.lib.format.open_memmap(
                    out_dir / f"mask_{task}_eligible.npy",
                    mode="w+",
                    dtype=np.bool_,
                    shape=(node_count,),
                ),
            }

        query = f"""
            SELECT
              {select_sql}
            FROM read_parquet('{duck_path(feature_table)}') AS p
            INNER JOIN read_parquet('{duck_path(node_index_path)}') AS n
              ON p.ku_id = n.ku_id
            WHERE TRY_CAST(p.cutoff_year AS INTEGER) = {int(cutoff)}
            ORDER BY n.node_idx
        """

        reader = con.execute(query).to_arrow_reader(batch_size=args.batch_size)

        filled = 0
        for batch in reader:
            df = batch.to_pandas()
            n = len(df)

            if n == 0:
                continue

            node_idx = df["node_idx"].to_numpy(np.int64)

            if len(numeric_features) > 0:
                numeric_block = (
                    df[numeric_features]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                x_numeric[node_idx, :] = numeric_block

            if len(categorical_features) > 0:
                cat_block = np.zeros((n, len(categorical_features)), dtype=np.int64)
                for j, col in enumerate(categorical_features):
                    mapping = categorical_vocab[col]
                    series = df[col].astype("string").fillna("__MISSING__")
                    cat_block[:, j] = (
                        series.map(mapping)
                        .fillna(0)
                        .astype("int64")
                        .to_numpy()
                    )
                x_categorical[node_idx, :] = cat_block

            for task in tasks:
                heat_col, label_col, eligible_col = target_columns_for_task(task)

                y_heat = (
                    pd.to_numeric(df[heat_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                y_label = (
                    pd.to_numeric(df[label_col], errors="coerce")
                    .fillna(0)
                    .astype("int8")
                    .to_numpy()
                )
                mask = (
                    df[eligible_col]
                    .fillna(False)
                    .astype(bool)
                    .to_numpy(dtype=np.bool_)
                )

                label_arrays[task]["heat"][node_idx] = y_heat
                label_arrays[task]["label"][node_idx] = y_label
                label_arrays[task]["eligible"][node_idx] = mask

            filled += n

        if filled != node_count:
            raise RuntimeError(
                f"Snapshot cutoff={cutoff} filled {filled} rows, "
                f"expected node_count={node_count}."
            )

        x_numeric.flush()
        x_categorical.flush()

        task_summary = {}
        for task in tasks:
            for arr in label_arrays[task].values():
                arr.flush()

            eligible_arr = np.load(
                out_dir / f"mask_{task}_eligible.npy",
                mmap_mode="r",
            )
            label_arr = np.load(
                out_dir / f"y_{task}_label.npy",
                mmap_mode="r",
            )

            eligible_count = int(eligible_arr.sum())
            positive_count = int(label_arr[eligible_arr].sum()) if eligible_count > 0 else 0

            task_summary[task] = {
                "eligible_count": eligible_count,
                "positive_count": positive_count,
                "positive_rate": (
                    float(positive_count / eligible_count)
                    if eligible_count > 0
                    else None
                ),
            }

        snapshot_info = {
            "cutoff_year": int(cutoff),
            "split": split,
            "path": str(out_dir),
            "node_count": int(node_count),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "tasks": task_summary,
            "files": {
                "x_numeric": str(x_numeric_path),
                "x_categorical": str(x_categorical_path),
            },
        }

        write_json(out_dir / "snapshot_summary.json", snapshot_info)

        split_manifest[split].append(
            {
                "cutoff_year": int(cutoff),
                "path": str(out_dir),
            }
        )

        snapshot_summaries.append(snapshot_info)

    return {
        "split_manifest": split_manifest,
        "snapshot_summaries": snapshot_summaries,
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

    pd.DataFrame(rows).to_csv(
        output_dir / "gnn_input_manifest.csv",
        index=False,
    )


def main() -> None:
    args = normalize_args(parse_args())
    validate_inputs(args)

    start = time.time()

    prepare_output_dir(args.output_dir, args.overwrite)

    log(args, "Building Temporal Knowledge Field GNN inputs ...")
    log(args, f"Feature dir: {args.feature_dir}")
    log(args, f"KU dir:      {args.ku_dir}")
    log(args, f"Output dir:  {args.output_dir}")
    log(args, "")

    con = configure_duckdb(args)

    feature_table = args.feature_dir / "prediction_feature_table.parquet"
    feature_spec_path = args.feature_dir / "feature_columns.json"
    target_spec_path = args.feature_dir / "target_columns.json"

    table_columns = get_parquet_columns(con, feature_table)
    ensure_required_task_columns(table_columns, TASKS)

    feature_spec = read_json(feature_spec_path)
    target_spec = read_json(target_spec_path)

    numeric_features, categorical_features, feature_schema = extract_feature_columns_from_spec(
        feature_spec,
        table_columns,
    )

    log(args, "Feature schema:")
    log(args, f"  numeric features:     {len(numeric_features)}")
    log(args, f"  categorical features: {len(categorical_features)}")
    log(args, "")

    graph_dir = args.output_dir / "graph"
    graph_summary = build_or_copy_graph(
        con=con,
        args=args,
        graph_dir=graph_dir,
    )

    log(args, "")
    log(args, "Building categorical vocabularies ...")
    categorical_vocab = build_categorical_vocab(
        con=con,
        feature_table=feature_table,
        categorical_features=categorical_features,
    )

    snapshots_dir = args.output_dir / "snapshots"

    log(args, "")
    log(args, "Materializing split/cutoff snapshots as numpy tensors ...")
    snapshot_result = materialize_snapshots(
        con=con,
        args=args,
        graph_dir=graph_dir,
        snapshots_dir=snapshots_dir,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_vocab=categorical_vocab,
        tasks=TASKS,
    )

    runtime_seconds = time.time() - start

    split_manifest = snapshot_result["split_manifest"]
    snapshot_summaries = snapshot_result["snapshot_summaries"]

    split_manifest_path = args.output_dir / "split_manifest.json"
    feature_schema_path = args.output_dir / "feature_schema.json"
    categorical_vocab_path = args.output_dir / "categorical_vocab.json"

    write_json(split_manifest_path, split_manifest)

    feature_schema_out = {
        **feature_schema,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "source_feature_columns_json": str(feature_spec_path),
        "source_target_columns_json": str(target_spec_path),
    }
    write_json(feature_schema_path, feature_schema_out)
    write_json(categorical_vocab_path, categorical_vocab)

    summary = {
        "status": "ok",
        "script": "scripts/34_build_knowledge_field_gnn_inputs.py",
        "feature_dir": str(args.feature_dir),
        "ku_dir": str(args.ku_dir),
        "output_dir": str(args.output_dir),
        "tasks": TASKS,
        "graph": graph_summary,
        "feature_schema": {
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        "splits": {
            split: [item["cutoff_year"] for item in items]
            for split, items in split_manifest.items()
        },
        "snapshot_count": len(snapshot_summaries),
        "snapshots": snapshot_summaries,
        "target_spec_keys": sorted(target_spec.keys()),
        "runtime_seconds": runtime_seconds,
    }

    write_json(args.output_dir / "gnn_input_summary.json", summary)

    # Preserve source specs.
    provenance_dir = args.output_dir / "source_prediction_provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    for src in [
        feature_spec_path,
        target_spec_path,
        args.feature_dir / "latest_split_summary.json",
        args.feature_dir / "latest_split_leakage_note.md",
        args.feature_dir / "latest_split_cutoff_summary.csv",
        args.feature_dir / "latest_split_task_summary.csv",
    ]:
        if src.exists():
            shutil.copy2(src, provenance_dir / src.name)

    write_manifest(args.output_dir)

    log(args, "")
    log(args, "GNN input dataset complete.")
    log(args, f"Output directory: {args.output_dir}")
    log(args, f"Runtime seconds: {runtime_seconds:.1f}")


if __name__ == "__main__":
    main()