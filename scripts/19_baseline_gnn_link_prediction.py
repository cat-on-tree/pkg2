#!/usr/bin/env python
"""Run supervised GNN baselines for Patent-Paper link prediction.

This script trains GraphSAGE or R-GCN on a leakage-controlled carrier context
graph and evaluates Patent-Paper link prediction using the fixed labeled splits.

Supported GNN models:
    - GraphSAGE
    - R-GCN

Important leakage rule:
    Message passing uses context_edges only.
    Target Patent-Paper edges are NOT included in the message-passing graph.

Training signal:
    Unlike KG embedding baselines, GraphSAGE / R-GCN here are supervised
    edge-prediction baselines trained with labeled_edges_train.

Typical dry run:

    python scripts/19_baseline_gnn_link_prediction.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/gnn_graphsage_probe \
      --model graphsage \
      --dry-run \
      --list-relations \
      --overwrite

Typical GraphSAGE run:

    python scripts/19_baseline_gnn_link_prediction.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/gnn_graphsage_e50 \
      --model graphsage \
      --hidden-dim 128 \
      --num-layers 2 \
      --epochs 50 \
      --learning-rate 0.001 \
      --weight-decay 0.00001 \
      --dropout 0.2 \
      --early-stopping-patience 10 \
      --device auto \
      --seed 42 \
      --overwrite

Typical R-GCN run:

    python scripts/19_baseline_gnn_link_prediction.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/gnn_rgcn_e50 \
      --model rgcn \
      --hidden-dim 128 \
      --num-layers 2 \
      --num-bases 8 \
      --epochs 50 \
      --learning-rate 0.001 \
      --weight-decay 0.00001 \
      --dropout 0.2 \
      --early-stopping-patience 10 \
      --device auto \
      --seed 42 \
      --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.embedding_baselines import (  # noqa: E402
    prediction_frames_to_rows,
    summarize_embedding_matrix,
    summarize_prediction_scores,
)
from pkg2.gnn_baselines import (  # noqa: E402
    build_gnn_link_predictor,
    encode_node_types,
    labeled_edges_by_split_to_tensors,
    predict_labeled_edges_by_split,
    resolve_torch_device,
    summarize_context_graph_tensors,
    summarize_edge_splits,
    summarize_model,
    train_supervised_gnn_link_predictor,
    triples_to_edge_tensors,
)
from pkg2.graph_data import (  # noqa: E402
    DEFAULT_SPLITS,
    LABEL_COLUMN,
    add_graph_indices_to_labeled_edges_by_split,
    build_heterogeneous_graph,
    build_node_index,
    load_graph_tables,
)
from pkg2.graph_io import (  # noqa: E402
    append_manifest_artifact,
    prepare_output_dir,
    save_embedding_artifacts,
    save_predictions_by_split,
    write_dataframe,
    write_json,
    write_manifest,
    write_text,
)
from pkg2.io import markdown_table  # noqa: E402
from pkg2.kg_embedding import (  # noqa: E402
    build_kg_triples_from_edge_index_dict,
    build_node_type_to_entity_ids,
)
from pkg2.metrics import (  # noqa: E402
    evaluate_prediction_rows_by_split,
    write_metrics_csv,
    write_metrics_summary_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run supervised GraphSAGE / R-GCN baseline for Patent-Paper link prediction."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Input link prediction dataset directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for GNN baseline artifacts.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["graphsage", "sage", "rgcn", "r-gcn"],
        help="GNN model.",
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate. Default: train val test.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension / trainable node embedding dimension. Default: 128.",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GNN layers. Default: 2.",
    )

    parser.add_argument(
        "--num-bases",
        type=int,
        default=8,
        help="Number of bases for R-GCN basis decomposition. Default: 8.",
    )

    parser.add_argument(
        "--decoder-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for concat MLP decoder. Default: hidden_dim.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs. Default: 50.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="AdamW learning rate. Default: 0.001.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay. Default: 1e-5.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability. Default: 0.2.",
    )

    parser.add_argument(
        "--embedding-std",
        type=float,
        default=0.02,
        help="Standard deviation for trainable node/type embedding initialization.",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience on validation AUPRC. Set <=0 to disable.",
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation AUPRC improvement for early stopping. Default: 0.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic CuDNN behavior where possible.",
    )

    parser.add_argument(
        "--no-reverse-relations",
        action="store_true",
        help="Do not add reverse relations to the context graph.",
    )

    parser.add_argument(
        "--no-deduplicate-triples",
        action="store_true",
        help="Do not deduplicate context graph triples.",
    )

    parser.add_argument(
        "--drop-missing-edges",
        action="store_true",
        help="Drop context/labeled edges whose endpoints are missing from node index.",
    )

    parser.add_argument(
        "--no-node-type-embeddings",
        action="store_true",
        help="Disable node type embeddings.",
    )

    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=[10, 50, 100, 500, 1000],
        help="K values for Precision@K / Recall@K.",
    )

    parser.add_argument(
        "--list-relations",
        action="store_true",
        help="Print relation index and edge counts.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build graph tensors and write metadata, but do not train.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print training progress every N epochs. Default: 1.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    return parser.parse_args()


def format_value(value: Any) -> str:
    """Format values for Markdown tables."""

    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def json_safe(value: Any) -> Any:
    """Convert numpy / pandas / torch objects into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if not isinstance(value, (list, tuple, dict, np.ndarray, torch.Tensor)):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

    return value


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build run configuration for reproducibility."""

    patience = args.early_stopping_patience
    if patience is not None and patience <= 0:
        patience = None

    return {
        "script": "scripts/19_baseline_gnn_link_prediction.py",
        "command": " ".join(sys.argv),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "splits": list(args.splits),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_bases": args.num_bases,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "embedding_std": args.embedding_std,
        "early_stopping_patience": patience,
        "min_delta": args.min_delta,
        "device": args.device,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "add_reverse_relations": not args.no_reverse_relations,
        "deduplicate_triples": not args.no_deduplicate_triples,
        "drop_missing_edges": args.drop_missing_edges,
        "use_node_type_embeddings": not args.no_node_type_embeddings,
        "k_values": list(args.k_values),
        "training_mode": "supervised_edge_prediction",
        "message_passing_edges": "context_edges_only",
        "target_edges_in_message_passing": False,
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_unix": time.time(),
    }


def compute_dataframe_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute deterministic fingerprint for a split DataFrame."""

    if frame.empty:
        return {
            "row_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "hash_columns": [],
        }

    preferred_columns = [
        "source_id",
        "target_id",
        "source_node_id",
        "target_node_id",
        "patent_id",
        "paper_id",
        "graph_source_idx",
        "graph_target_idx",
        "source_graph_node_idx",
        "target_graph_node_idx",
        LABEL_COLUMN,
        "label",
    ]

    columns = [column for column in preferred_columns if column in frame.columns]
    columns = list(dict.fromkeys(columns))

    if not columns:
        columns = sorted(dict.fromkeys(frame.columns))

    data = frame[columns].copy()

    for column in columns:
        data[column] = data[column].astype(str)

    data = data.sort_values(columns, kind="mergesort").reset_index(drop=True)
    payload = data.to_csv(index=False).encode("utf-8")
    sha256 = hashlib.sha256(payload).hexdigest()

    label_col = LABEL_COLUMN if LABEL_COLUMN in frame.columns else "label"
    positive_count = int(frame[label_col].sum()) if label_col in frame.columns else 0

    return {
        "row_count": int(len(frame)),
        "positive_count": positive_count,
        "negative_count": int(len(frame) - positive_count),
        "sha256": sha256,
        "hash_columns": columns,
    }


def compute_split_fingerprints(
    labeled_edges_by_split: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Compute fingerprints for all labeled splits."""

    return {
        split: compute_dataframe_fingerprint(frame)
        for split, frame in labeled_edges_by_split.items()
    }


def summarize_labeled_splits(
    labeled_edges_by_split: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Summarize labeled split DataFrames."""

    summary: dict[str, Any] = {}

    for split, frame in labeled_edges_by_split.items():
        label_col = LABEL_COLUMN if LABEL_COLUMN in frame.columns else "label"
        positive_count = int(frame[label_col].sum()) if label_col in frame.columns else 0

        summary[f"labeled_edges:{split}"] = int(len(frame))
        summary[f"positive_edges:{split}"] = positive_count
        summary[f"negative_edges:{split}"] = int(len(frame) - positive_count)

    return summary


def build_node_type_arrays_from_node_index(
    node_index: Any,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Build node type id array aligned to global node indices.

    This uses ``build_node_type_to_entity_ids`` because that helper already knows
    the project-specific node index structure.
    """

    num_nodes = len(node_index.nodes)
    node_type_to_entity_ids = build_node_type_to_entity_ids(node_index)

    node_type_names = ["unknown"] * num_nodes

    for node_type, entity_ids in node_type_to_entity_ids.items():
        for entity_id in entity_ids:
            entity_id = int(entity_id)
            if 0 <= entity_id < num_nodes:
                node_type_names[entity_id] = str(node_type)

    node_type_ids, node_type_to_id = encode_node_types(node_type_names)

    rows = []
    for node_type, node_type_id in sorted(node_type_to_id.items(), key=lambda item: item[1]):
        rows.append(
            {
                "node_type_id": node_type_id,
                "node_type": node_type,
                "node_count": int((node_type_ids == node_type_id).sum()),
            }
        )

    return node_type_ids, pd.DataFrame(rows), node_type_to_id


def build_metrics_table(metrics_by_split: dict[str, dict[str, Any]], splits: list[str]) -> str:
    """Build Markdown metrics table."""

    selected_metrics = [
        "example_count",
        "positive_count",
        "negative_count",
        "positive_ratio",
        "score_min",
        "score_max",
        "score_mean",
        "auroc",
        "auprc",
        "mrr",
        "precision_at_100",
        "recall_at_100",
        "precision_at_1000",
        "recall_at_1000",
    ]

    rows = []

    for split in splits:
        metrics = metrics_by_split.get(split, {})
        row = {"split": split}

        for metric in selected_metrics:
            row[metric] = format_value(metrics.get(metric, ""))

        rows.append(row)

    return markdown_table(rows, ["split", *selected_metrics])


def build_report(
    *,
    model_name: str,
    dataset_dir: Path,
    output_dir: Path,
    run_config: dict[str, Any],
    graph_summary: dict[str, Any],
    split_summary: dict[str, Any],
    model_summary: dict[str, Any],
    training_metadata: dict[str, Any],
    embedding_summary: dict[str, Any],
    metrics_by_split: dict[str, dict[str, Any]],
    score_distribution: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    splits: list[str],
) -> str:
    """Build Markdown baseline report."""

    parameter_rows = [
        {"parameter": "model", "value": model_name},
        {"parameter": "dataset_dir", "value": str(dataset_dir)},
        {"parameter": "output_dir", "value": str(output_dir)},
        {"parameter": "command", "value": run_config.get("command", "")},
        {"parameter": "seed", "value": run_config.get("seed", "")},
        {"parameter": "hidden_dim", "value": run_config.get("hidden_dim", "")},
        {"parameter": "num_layers", "value": run_config.get("num_layers", "")},
        {"parameter": "dropout", "value": run_config.get("dropout", "")},
        {"parameter": "learning_rate", "value": run_config.get("learning_rate", "")},
        {"parameter": "weight_decay", "value": run_config.get("weight_decay", "")},
        {
            "parameter": "early_stopping_patience",
            "value": run_config.get("early_stopping_patience", ""),
        },
        {
            "parameter": "message_passing_edges",
            "value": run_config.get("message_passing_edges", ""),
        },
        {
            "parameter": "target_edges_in_message_passing",
            "value": run_config.get("target_edges_in_message_passing", ""),
        },
    ]

    graph_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in graph_summary.items()
        if not str(key).startswith("relation_edges:")
    ]

    split_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in split_summary.items()
    ]

    model_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in model_summary.items()
    ]

    training_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in training_metadata.items()
    ]

    embedding_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in embedding_summary.items()
    ]

    score_rows = [
        {key: format_value(value) for key, value in row.items()}
        for row in score_distribution
    ]

    artifact_table_rows = [
        {key: format_value(value) for key, value in row.items()}
        for row in artifact_rows
    ]

    return "\n".join(
        [
            f"# Supervised GNN Link Prediction Baseline: {model_name}",
            "",
            "## 1. Baseline description",
            "",
            "This baseline trains a supervised message-passing GNN for Patent-Paper link prediction.",
            "",
            "Message passing uses `context_edges` only. Target Patent-Paper edges are excluded from the graph used by the GNN encoder.",
            "",
            "The model is trained with the fixed `labeled_edges_train` split and evaluated on the fixed validation/test splits.",
            "",
            "## 2. Parameters",
            "",
            markdown_table(parameter_rows, ["parameter", "value"]),
            "",
            "## 3. Context graph summary",
            "",
            markdown_table(graph_rows, ["name", "value"]),
            "",
            "## 4. Downstream split summary",
            "",
            markdown_table(split_rows, ["name", "value"]),
            "",
            "## 5. Model summary",
            "",
            markdown_table(model_rows, ["name", "value"]),
            "",
            "## 6. Training metadata",
            "",
            markdown_table(training_rows, ["name", "value"]),
            "",
            "## 7. Embedding summary",
            "",
            markdown_table(embedding_rows, ["name", "value"]),
            "",
            "## 8. Metrics",
            "",
            build_metrics_table(metrics_by_split, splits),
            "",
            "## 9. Score distribution",
            "",
            markdown_table(
                score_rows,
                [
                    "split",
                    "label",
                    "example_count",
                    "score_min",
                    "score_max",
                    "score_mean",
                    "score_q25",
                    "score_median",
                    "score_q75",
                ],
            ),
            "",
            "## 10. Output artifacts",
            "",
            markdown_table(
                artifact_table_rows,
                [
                    "artifact",
                    "path",
                    "description",
                    "row_count",
                    "file_size_bytes",
                ],
            ),
            "",
            "## 11. Reproducibility notes",
            "",
            "- `run_config.json` stores the exact command line and key hyperparameters.",
            "- `split_fingerprint.json` stores hashes of the downstream labeled splits.",
            "- `training_history.csv` stores epoch-level supervised GNN training diagnostics.",
            "- Target Patent-Paper edges are excluded from message passing.",
            "",
        ]
    )


@torch.no_grad()
def compute_node_embeddings(
    *,
    model: torch.nn.Module,
    graph_edge_index: torch.Tensor,
    graph_edge_type: torch.Tensor | None,
    device: str | torch.device,
) -> np.ndarray:
    """Compute final node embeddings from the trained model."""

    torch_device = resolve_torch_device(device)
    model = model.to(torch_device)
    model.eval()

    graph_edge_index = graph_edge_index.to(torch_device)
    graph_edge_type = graph_edge_type.to(torch_device) if graph_edge_type is not None else None

    node_embeddings = model.encode(
        edge_index=graph_edge_index,
        edge_type=graph_edge_type,
    )

    return node_embeddings.detach().cpu().numpy()


def main() -> None:
    """Run supervised GNN link prediction baseline."""

    args = parse_args()

    output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    dataset_dir = Path(args.dataset_dir)
    splits = [str(split).strip() for split in args.splits if str(split).strip()]

    if "train" not in splits:
        raise ValueError("The train split is required.")

    if "val" not in splits:
        raise ValueError("The val split is required for early stopping.")

    run_config = build_run_config(args)
    early_stopping_patience = run_config["early_stopping_patience"]

    top_level_artifacts: list[dict[str, Any]] = []

    run_config_path = output_dir / "run_config.json"
    write_json(run_config_path, json_safe(run_config))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="run_config",
        path=run_config_path,
        description="Exact command line and configuration for reproducibility.",
    )

    print()
    print("Loading supervised GNN baseline input data...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"GNN model:         {args.model}")
    print()

    tables = load_graph_tables(dataset_dir, splits=splits)
    node_index = build_node_index(tables.nodes)

    heterogeneous_graph = build_heterogeneous_graph(
        tables.context_edges,
        nodes=tables.nodes,
        node_index=node_index,
        add_reverse_edges=not args.no_reverse_relations,
        drop_missing=args.drop_missing_edges,
    )

    labeled_edges_by_split = add_graph_indices_to_labeled_edges_by_split(
        tables.labeled_edges_by_split,
        node_index=node_index,
        drop_missing=args.drop_missing_edges,
    )

    split_summary = summarize_labeled_splits(labeled_edges_by_split)
    split_summary_path = output_dir / "split_summary.json"
    write_json(split_summary_path, json_safe(split_summary))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="split_summary",
        path=split_summary_path,
        description="Downstream labeled split row and label counts.",
    )

    split_fingerprints = compute_split_fingerprints(labeled_edges_by_split)
    split_fingerprint_path = output_dir / "split_fingerprint.json"
    write_json(split_fingerprint_path, json_safe(split_fingerprints))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="split_fingerprint",
        path=split_fingerprint_path,
        description="SHA256 fingerprints for downstream labeled splits.",
    )

    print("Building context graph triples / tensors...")
    print()

    triples, relation_infos, relation_frame = build_kg_triples_from_edge_index_dict(
        heterogeneous_graph.edge_index_dict,
        node_index=node_index,
        edge_indices_are_global=False,
        deduplicate=not args.no_deduplicate_triples,
    )

    relation_index_path = output_dir / "relation_index.csv"
    write_dataframe(relation_frame, relation_index_path)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="relation_index",
        path=relation_index_path,
        description="Context graph relation index and metadata.",
        row_count=len(relation_frame),
    )

    graph_edge_index, graph_edge_type = triples_to_edge_tensors(triples)
    num_nodes = len(node_index.nodes)

    num_relations = int(graph_edge_type.max().item()) + 1 if graph_edge_type.numel() else 0

    graph_summary = summarize_context_graph_tensors(
        edge_index=graph_edge_index,
        edge_type=graph_edge_type,
        num_nodes=num_nodes,
    )

    graph_summary.update(
        {
            "num_triples": int(graph_edge_index.shape[1]),
            "num_relations_from_relation_index": int(len(relation_frame)),
            "num_nodes_from_node_index": int(num_nodes),
            "add_reverse_relations": bool(not args.no_reverse_relations),
            "deduplicate_triples": bool(not args.no_deduplicate_triples),
        }
    )

    graph_summary_path = output_dir / "graph_summary.json"
    write_json(graph_summary_path, json_safe(graph_summary))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="graph_summary",
        path=graph_summary_path,
        description="Context graph tensor summary.",
    )

    node_type_ids, node_type_frame, node_type_to_id = build_node_type_arrays_from_node_index(
        node_index
    )

    node_type_index_path = output_dir / "node_type_index.csv"
    write_dataframe(node_type_frame, node_type_index_path)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="node_type_index",
        path=node_type_index_path,
        description="Node type ID mapping used for optional node type embeddings.",
        row_count=len(node_type_frame),
    )

    node_type_ids_path = output_dir / "node_type_ids.npy"
    np.save(node_type_ids_path, node_type_ids)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="node_type_ids",
        path=node_type_ids_path,
        description="Node type IDs aligned to global graph node indices.",
        row_count=len(node_type_ids),
    )

    if args.list_relations:
        print("Context graph relations:")
        print()
        for row in relation_frame.itertuples(index=False):
            print(
                f"  {row.relation_idx:02d} "
                f"{row.relation_name} "
                f"source={row.source_type} "
                f"target={row.target_type} "
                f"edges={row.edge_count}"
            )
        print()

    print("Context graph summary:")
    for key, value in graph_summary.items():
        if not str(key).startswith("relation_edges:"):
            print(f"  {key}: {value}")
    print()

    print("Downstream split summary:")
    for key, value in split_summary.items():
        print(f"  {key}: {value}")
    print()

    edge_splits = labeled_edges_by_split_to_tensors(
        labeled_edges_by_split,
        label_column=LABEL_COLUMN if LABEL_COLUMN in next(iter(labeled_edges_by_split.values())).columns else "label",
    )

    tensor_split_summary = summarize_edge_splits(edge_splits)
    tensor_split_summary_path = output_dir / "edge_split_tensor_summary.json"
    write_json(tensor_split_summary_path, json_safe(tensor_split_summary))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="edge_split_tensor_summary",
        path=tensor_split_summary_path,
        description="Summary of labeled edge tensors used by supervised GNN.",
    )

    if args.dry_run:
        manifest_path = write_manifest(output_dir, top_level_artifacts)
        append_manifest_artifact(
            top_level_artifacts,
            artifact="baseline_manifest",
            path=manifest_path,
            description="Top-level dry-run artifact manifest.",
            row_count=len(top_level_artifacts),
        )
        write_manifest(output_dir, top_level_artifacts)

        print("Dry run complete. No GNN training was performed.")
        return

    node_type_ids_for_model = None if args.no_node_type_embeddings else node_type_ids
    num_node_types = None if args.no_node_type_embeddings else len(node_type_to_id)

    print("Building supervised GNN link predictor...")
    print()

    model = build_gnn_link_predictor(
        model_name=args.model,
        num_nodes=num_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        node_type_ids=node_type_ids_for_model,
        num_node_types=num_node_types,
        num_relations=num_relations,
        num_bases=args.num_bases,
        decoder_hidden_dim=args.decoder_hidden_dim,
        embedding_std=args.embedding_std,
    )

    model_summary = summarize_model(model)
    model_summary_path = output_dir / "model_summary.json"
    write_json(model_summary_path, json_safe(model_summary))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="model_summary",
        path=model_summary_path,
        description="GNN model architecture and parameter summary.",
    )

    print("Model summary:")
    for key, value in model_summary.items():
        print(f"  {key}: {value}")
    print()

    print("Training supervised GNN link predictor...")
    print()

    try:
        training_result = train_supervised_gnn_link_predictor(
            model=model,
            graph_edge_index=graph_edge_index,
            graph_edge_type=graph_edge_type,
            edge_splits=edge_splits,
            train_split="train",
            val_split="val",
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            deterministic=args.deterministic,
            early_stopping_patience=early_stopping_patience,
            min_delta=args.min_delta,
            log_every=args.log_every,
        )
    except Exception:
        print()
        print("Supervised GNN training failed.")
        traceback.print_exc()
        print()
        raise

    trained_model = training_result.model
    training_metadata = training_result.metadata

    training_history_path = output_dir / "training_history.csv"
    write_dataframe(training_result.training_history, training_history_path)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="training_history",
        path=training_history_path,
        description="Epoch-level supervised GNN training diagnostics.",
        row_count=len(training_result.training_history),
    )

    training_metadata_path = output_dir / "training_metadata.json"
    write_json(training_metadata_path, json_safe(training_metadata))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="training_metadata",
        path=training_metadata_path,
        description="Supervised GNN training metadata.",
    )

    model_state_path = output_dir / "model_state.pt"
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "best_state_dict": training_result.best_state_dict,
            "run_config": json_safe(run_config),
            "training_metadata": json_safe(training_metadata),
            "model_summary": json_safe(model_summary),
        },
        model_state_path,
    )
    append_manifest_artifact(
        top_level_artifacts,
        artifact="model_state",
        path=model_state_path,
        description="Trained GNN model state dictionary.",
    )

    print()
    print("GNN training complete.")
    print(f"Best epoch:     {training_metadata.get('best_epoch')}")
    print(f"Best val AUPRC: {training_metadata.get('best_val_auprc')}")
    print()

    embeddings = compute_node_embeddings(
        model=trained_model,
        graph_edge_index=graph_edge_index,
        graph_edge_type=graph_edge_type,
        device=args.device,
    )

    embedding_summary = summarize_embedding_matrix(embeddings)
    embedding_summary_path = output_dir / "embedding_summary.json"
    write_json(embedding_summary_path, json_safe(embedding_summary))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="embedding_summary",
        path=embedding_summary_path,
        description="Summary statistics for final GNN node embeddings.",
    )

    embedding_metadata = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "model_name": args.model,
        "run_config": run_config,
        "graph_summary": graph_summary,
        "split_summary": split_summary,
        "split_fingerprints": split_fingerprints,
        "model_summary": model_summary,
        "training_metadata": training_metadata,
        "embedding_summary": embedding_summary,
        "note": (
            "Node embeddings were learned by supervised GNN link prediction. "
            "Message passing used context_edges only; target Patent-Paper edges "
            "were excluded from the GNN graph."
        ),
    }

    top_level_artifacts.extend(
        save_embedding_artifacts(
            output_dir=output_dir,
            embeddings=embeddings,
            node_index=node_index,
            metadata=json_safe(embedding_metadata),
        )
    )

    print("Generating predictions...")
    print()

    predictions_by_split = predict_labeled_edges_by_split(
        trained_model,
        graph_edge_index=graph_edge_index,
        graph_edge_type=graph_edge_type,
        edge_splits=edge_splits,
        device=args.device,
        score_column="score",
    )

    artifact_rows: list[dict[str, Any]] = []

    artifact_rows.extend(
        save_predictions_by_split(
            predictions_by_split,
            output_dir,
            file_prefix="predictions",
            file_suffix=".parquet",
        )
    )

    prediction_rows = prediction_frames_to_rows(
        predictions_by_split,
        score_column="score",
    )

    metrics_by_split = evaluate_prediction_rows_by_split(
        prediction_rows,
        split_column="split",
        label_column="label",
        score_column="score",
        k_values=sorted({int(k) for k in args.k_values if int(k) > 0}),
        threshold=None,
    )

    metrics_csv_path = output_dir / "metrics.csv"
    write_metrics_csv(
        metrics_csv_path,
        metrics_by_split,
        model_name=f"gnn_{args.model}",
    )
    append_manifest_artifact(
        artifact_rows,
        artifact="metrics",
        path=metrics_csv_path,
        description="Long-form evaluation metrics.",
    )

    metrics_summary_path = output_dir / "metrics_summary.csv"
    write_metrics_summary_csv(metrics_summary_path, metrics_by_split)
    append_manifest_artifact(
        artifact_rows,
        artifact="metrics_summary",
        path=metrics_summary_path,
        description="Compact evaluation metrics summary.",
    )

    score_distribution = summarize_prediction_scores(
        predictions_by_split,
        score_column="score",
    )

    score_distribution_path = output_dir / "score_distribution.csv"
    write_dataframe(pd.DataFrame(score_distribution), score_distribution_path)
    append_manifest_artifact(
        artifact_rows,
        artifact="score_distribution",
        path=score_distribution_path,
        description="Prediction score distribution by split and label.",
        row_count=len(score_distribution),
    )

    top_level_artifacts.extend(artifact_rows)

    report_text = build_report(
        model_name=args.model,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        run_config=run_config,
        graph_summary=graph_summary,
        split_summary=split_summary,
        model_summary=model_summary,
        training_metadata=training_metadata,
        embedding_summary=embedding_summary,
        metrics_by_split=metrics_by_split,
        score_distribution=score_distribution,
        artifact_rows=top_level_artifacts,
        splits=splits,
    )

    report_path = output_dir / "baseline_report.md"
    write_text(report_path, report_text)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="baseline_report",
        path=report_path,
        description="Markdown report for supervised GNN baseline.",
    )

    manifest_path = write_manifest(output_dir, top_level_artifacts)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="baseline_manifest",
        path=manifest_path,
        description="Top-level supervised GNN artifact manifest.",
        row_count=len(top_level_artifacts),
    )
    write_manifest(output_dir, top_level_artifacts)

    test_metrics = metrics_by_split.get("test", {})

    print()
    print("Supervised GNN baseline complete.")
    print()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Model:             {args.model}")
    print()
    print(
        "Test metrics: "
        f"AUROC={format_value(test_metrics.get('auroc'))}, "
        f"AUPRC={format_value(test_metrics.get('auprc'))}"
    )
    print()
    print(f"Metrics summary: {metrics_summary_path}")
    print(f"Report:          {report_path}")
    print(f"Run config:      {run_config_path}")
    print()


if __name__ == "__main__":
    main()