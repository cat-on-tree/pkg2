#!/usr/bin/env python
"""Run Node2Vec graph representation baseline for Patent-Paper link prediction.

This script trains Node2Vec embeddings on the leakage-controlled context graph
and evaluates Patent-Paper link prediction using multiple embedding decoders.

Important:
    This script does NOT re-split train/val/test.
    It reuses the labeled edges already stored in the input dataset directory.

Input dataset directory should contain graph tables such as:
    nodes.parquet
    context_edges.parquet
    labeled_edges_train.parquet
    labeled_edges_val.parquet
    labeled_edges_test.parquet

The context graph should already exclude the target Patent-Paper edge table.

Example:

    python scripts/16_baseline_node2vec.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/node2vec_p1_q1 \
      --embedding-dim 128 \
      --walk-length 20 \
      --context-size 10 \
      --walks-per-node 10 \
      --p 1.0 \
      --q 1.0 \
      --epochs 5 \
      --batch-size 256 \
      --scoring-methods cosine dot negative_l2 hadamard_logistic concat_logistic \
      --device auto \
      --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pkg2.embedding_baselines import (  # noqa: E402
    classifier_summary,
    extract_logistic_coefficients,
    list_supported_scoring_methods,
    prediction_frames_to_rows,
    score_embeddings_by_split,
    summarize_embedding_matrix,
    summarize_prediction_scores,
)
from pkg2.graph_data import (  # noqa: E402
    DEFAULT_SPLITS,
    LABEL_COLUMN,
    add_graph_indices_to_labeled_edges_by_split,
    build_homogeneous_graph,
    build_node_index,
    load_graph_tables,
)
from pkg2.graph_io import (  # noqa: E402
    append_manifest_artifact,
    prepare_output_dir,
    save_embedding_artifacts,
    save_predictions_by_split,
    save_training_history,
    write_dataframe,
    write_json,
    write_manifest,
    write_text,
)
from pkg2.io import markdown_table  # noqa: E402
from pkg2.metrics import (  # noqa: E402
    evaluate_prediction_rows_by_split,
    write_metrics_csv,
    write_metrics_summary_csv,
)


DEFAULT_SCORING_METHODS = [
    "cosine",
    "dot",
    "negative_l2",
    "hadamard_logistic",
    "concat_logistic",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Node2Vec baseline for Patent-Paper link prediction."
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
        help=(
            "Output directory. Embeddings are saved here and each scoring method "
            "gets a subdirectory."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate. Default: train val test",
    )

    parser.add_argument(
        "--scoring-methods",
        nargs="*",
        default=list(DEFAULT_SCORING_METHODS),
        choices=list_supported_scoring_methods(),
        help=(
            "Embedding scoring / decoder methods to evaluate. "
            "Default: cosine dot negative_l2 hadamard_logistic concat_logistic"
        ),
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Node2Vec embedding dimension. Default: 128",
    )

    parser.add_argument(
        "--walk-length",
        type=int,
        default=20,
        help="Node2Vec walk length. Default: 20",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=10,
        help="Node2Vec context size. Default: 10",
    )

    parser.add_argument(
        "--walks-per-node",
        type=int,
        default=10,
        help="Number of walks per node. Default: 10",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=1.0,
        help="Node2Vec return parameter p. p=1,q=1 is DeepWalk-like. Default: 1.0",
    )

    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="Node2Vec in-out parameter q. p=1,q=1 is DeepWalk-like. Default: 1.0",
    )

    parser.add_argument(
        "--num-negative-samples",
        type=int,
        default=1,
        help="Negative samples per positive random-walk context. Default: 1",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Node2Vec training epochs. Default: 5",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Node2Vec random-walk loader batch size. Default: 256",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Node2Vec optimizer learning rate. Default: 0.01",
    )

    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse gradients and SparseAdam. Often useful for Node2Vec.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Node2Vec loader workers. On Windows, 0 is safest. Default: 0",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, etc. Default: auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--classifier-max-iter",
        type=int,
        default=1000,
        help="Max iterations for embedding pair Logistic Regression decoders.",
    )

    parser.add_argument(
        "--classifier-c",
        type=float,
        default=1.0,
        help="C for embedding pair Logistic Regression decoders. Default: 1.0",
    )

    parser.add_argument(
        "--classifier-class-weight",
        default=None,
        choices=[None, "balanced"],
        help="Class weight for embedding pair Logistic Regression decoders.",
    )

    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=[10, 50, 100, 500, 1000],
        help="K values for Precision@K / Recall@K. Default: 10 50 100 500 1000",
    )

    parser.add_argument(
        "--no-reverse-edges",
        action="store_true",
        help="Do not add reverse edges to the homogeneous Node2Vec graph.",
    )

    parser.add_argument(
        "--drop-missing-edges",
        action="store_true",
        help="Drop context/labeled edges whose endpoints are missing from node index.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )

    return parser.parse_args()


def import_torch_and_pyg() -> tuple[Any, Any]:
    """Import torch and PyG Node2Vec with useful errors."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Node2Vec. Install torch first."
        ) from exc

    try:
        from torch_geometric.nn import Node2Vec
    except ImportError as exc:
        raise ImportError(
            "torch-geometric is required for PyG Node2Vec. "
            "Install torch-geometric and its Node2Vec random-walk backend."
        ) from exc

    return torch, Node2Vec


def resolve_device(device: str, torch: Any) -> str:
    """Resolve device string."""

    device = str(device).strip()

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device


def set_random_seed(seed: int, torch: Any) -> None:
    """Set numpy and torch random seeds."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_value(value: Any) -> str:
    """Format values for Markdown tables."""

    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def load_node2vec_input_data(
    *,
    dataset_dir: Path,
    splits: list[str],
    add_reverse_edges: bool,
    drop_missing_edges: bool,
) -> dict[str, Any]:
    """Load graph data needed for Node2Vec."""

    tables = load_graph_tables(dataset_dir, splits=splits)

    node_index = build_node_index(tables.nodes)

    homogeneous_graph = build_homogeneous_graph(
        tables.context_edges,
        nodes=tables.nodes,
        node_index=node_index,
        add_reverse_edges=add_reverse_edges,
        drop_missing=drop_missing_edges,
    )

    labeled_edges_by_split = add_graph_indices_to_labeled_edges_by_split(
        tables.labeled_edges_by_split,
        node_index=node_index,
        drop_missing=drop_missing_edges,
    )

    return {
        "tables": tables,
        "node_index": node_index,
        "homogeneous_graph": homogeneous_graph,
        "labeled_edges_by_split": labeled_edges_by_split,
    }


def summarize_node2vec_input_data(data: dict[str, Any]) -> dict[str, Any]:
    """Summarize Node2Vec input data."""

    node_index = data["node_index"]
    homogeneous_graph = data["homogeneous_graph"]
    labeled_edges_by_split = data["labeled_edges_by_split"]

    summary: dict[str, Any] = {
        "num_nodes": int(len(node_index.nodes)),
        "num_node_types": int(len(node_index.num_nodes_by_type)),
        "num_homogeneous_edges": int(homogeneous_graph.edge_index.shape[1]),
    }

    for node_type, count in sorted(node_index.num_nodes_by_type.items()):
        summary[f"num_nodes:{node_type}"] = int(count)

    for split, frame in labeled_edges_by_split.items():
        positive_count = int(frame[LABEL_COLUMN].sum())
        summary[f"labeled_edges:{split}"] = int(len(frame))
        summary[f"positive_edges:{split}"] = positive_count
        summary[f"negative_edges:{split}"] = int(len(frame) - positive_count)

    return summary


def train_node2vec_embeddings(
    *,
    edge_index: np.ndarray,
    num_nodes: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Train PyG Node2Vec and return embedding matrix."""

    torch, Node2Vec = import_torch_and_pyg()

    set_random_seed(args.seed, torch)

    device = resolve_device(args.device, torch)

    edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long)

    model = Node2Vec(
        edge_index=edge_index_tensor,
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        p=args.p,
        q=args.q,
        num_negative_samples=args.num_negative_samples,
        num_nodes=num_nodes,
        sparse=args.sparse,
    ).to(device)

    loader = model.loader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if args.sparse:
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, Any]] = []

    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_batches = 0
        total_examples = 0

        epoch_started_at = time.time()

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()

            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)

            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

            batch_size = int(pos_rw.size(0))
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            total_batches += 1

        epoch_seconds = time.time() - epoch_started_at
        avg_loss = total_loss / max(total_examples, 1)

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "total_batches": total_batches,
            "total_examples": total_examples,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"loss={avg_loss:.6f}, "
            f"batches={total_batches}, "
            f"seconds={epoch_seconds:.2f}"
        )

    model.eval()

    with torch.no_grad():
        embeddings = model.embedding.weight.detach().cpu().numpy().astype(np.float32)

    training_metadata = {
        "device": device,
        "training_seconds": time.time() - started_at,
        "torch_version": getattr(torch, "__version__", ""),
        "num_epochs": args.epochs,
        "final_loss": history[-1]["loss"] if history else "",
    }

    return embeddings, history, training_metadata


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


def build_score_method_report(
    *,
    scoring_method: str,
    dataset_dir: Path,
    output_dir: Path,
    splits: list[str],
    node2vec_parameters: dict[str, Any],
    data_summary: dict[str, Any],
    embedding_summary: dict[str, Any],
    classifier_info: dict[str, Any],
    metrics_by_split: dict[str, dict[str, Any]],
    score_distribution: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> str:
    """Build Markdown report for one scoring method."""

    parameter_rows = [
        {"parameter": "dataset_dir", "value": str(dataset_dir)},
        {"parameter": "output_dir", "value": str(output_dir)},
        {"parameter": "scoring_method", "value": scoring_method},
        {"parameter": "splits", "value": ", ".join(splits)},
    ]

    for key, value in node2vec_parameters.items():
        parameter_rows.append({"parameter": key, "value": format_value(value)})

    data_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in data_summary.items()
    ]

    embedding_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in embedding_summary.items()
    ]

    classifier_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in classifier_info.items()
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
            f"# Node2Vec Baseline Report: {scoring_method}",
            "",
            "## 1. Baseline description",
            "",
            "This baseline trains Node2Vec embeddings on the leakage-controlled context graph and evaluates Patent-Paper link prediction using an embedding decoder.",
            "",
            "The script does not re-split train/validation/test. It uses the labeled edge splits stored in the input dataset directory.",
            "",
            "The Node2Vec training graph is built from `context_edges` only. Target Patent-Paper edges are not added to the embedding training graph.",
            "",
            "## 2. Parameters",
            "",
            markdown_table(parameter_rows, ["parameter", "value"]),
            "",
            "## 3. Input data summary",
            "",
            markdown_table(data_rows, ["name", "value"]),
            "",
            "## 4. Embedding summary",
            "",
            markdown_table(embedding_rows, ["name", "value"]),
            "",
            "## 5. Decoder / classifier summary",
            "",
            markdown_table(classifier_rows, ["name", "value"]),
            "",
            "## 6. Metrics",
            "",
            build_metrics_table(metrics_by_split, splits),
            "",
            "## 7. Score distribution",
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
            "## 8. Output artifacts",
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
            "## 9. Notes",
            "",
            "- `dot`, `cosine`, `negative_l1`, and `negative_l2` are unsupervised embedding similarity decoders.",
            "- `hadamard_logistic`, `l1_logistic`, `l2_logistic`, and `concat_logistic` train a Logistic Regression decoder on the existing train split.",
            "- Larger scores indicate higher predicted likelihood of a Patent-Paper link.",
            "",
        ]
    )


def build_overall_report(
    *,
    dataset_dir: Path,
    output_dir: Path,
    node2vec_parameters: dict[str, Any],
    data_summary: dict[str, Any],
    embedding_summary: dict[str, Any],
    training_metadata: dict[str, Any],
    scoring_rows: list[dict[str, Any]],
) -> str:
    """Build top-level Node2Vec report."""

    parameter_rows = [
        {"parameter": "dataset_dir", "value": str(dataset_dir)},
        {"parameter": "output_dir", "value": str(output_dir)},
    ]

    for key, value in node2vec_parameters.items():
        parameter_rows.append({"parameter": key, "value": format_value(value)})

    data_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in data_summary.items()
    ]

    embedding_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in embedding_summary.items()
    ]

    training_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in training_metadata.items()
    ]

    scoring_table_rows = [
        {key: format_value(value) for key, value in row.items()}
        for row in scoring_rows
    ]

    return "\n".join(
        [
            "# Node2Vec Baseline Summary",
            "",
            "## 1. Description",
            "",
            "This experiment trains one Node2Vec embedding model and evaluates multiple Patent-Paper link decoders.",
            "",
            "No train/validation/test re-splitting is performed.",
            "",
            "## 2. Node2Vec parameters",
            "",
            markdown_table(parameter_rows, ["parameter", "value"]),
            "",
            "## 3. Input data summary",
            "",
            markdown_table(data_rows, ["name", "value"]),
            "",
            "## 4. Embedding summary",
            "",
            markdown_table(embedding_rows, ["name", "value"]),
            "",
            "## 5. Training metadata",
            "",
            markdown_table(training_rows, ["name", "value"]),
            "",
            "## 6. Scoring method results",
            "",
            markdown_table(
                scoring_table_rows,
                [
                    "scoring_method",
                    "status",
                    "test_auroc",
                    "test_auprc",
                    "output_dir",
                ],
            ),
            "",
            "## 7. Notes",
            "",
            "- `p=1, q=1` is a DeepWalk-like Node2Vec setting.",
            "- The embedding training graph is homogeneous and built from context edges only.",
            "- Each scoring method has its own subdirectory with `metrics_summary.csv`, so the baseline collection script can discover it.",
            "",
        ]
    )


def evaluate_one_scoring_method(
    *,
    scoring_method: str,
    embeddings: np.ndarray,
    labeled_edges_by_split: dict[str, pd.DataFrame],
    output_dir: Path,
    dataset_dir: Path,
    splits: list[str],
    node2vec_parameters: dict[str, Any],
    data_summary: dict[str, Any],
    embedding_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate one embedding scoring method."""

    scoring_output_dir = output_dir / scoring_method
    scoring_output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Evaluating scoring method: {scoring_method}")
    print(f"Output directory:          {scoring_output_dir}")

    classifier, predictions_by_split = score_embeddings_by_split(
        embeddings,
        labeled_edges_by_split,
        scoring_method=scoring_method,
        train_split="train",
        max_iter=args.classifier_max_iter,
        c_value=args.classifier_c,
        class_weight=args.classifier_class_weight,
        random_state=args.seed,
        score_column="score",
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

    score_distribution = summarize_prediction_scores(
        predictions_by_split,
        score_column="score",
    )

    artifact_rows: list[dict[str, Any]] = []

    artifact_rows.extend(
        save_predictions_by_split(
            predictions_by_split,
            scoring_output_dir,
            file_prefix="predictions",
            file_suffix=".parquet",
        )
    )

    metrics_csv_path = scoring_output_dir / "metrics.csv"
    write_metrics_csv(
        metrics_csv_path,
        metrics_by_split,
        model_name=f"node2vec:{scoring_method}",
    )
    append_manifest_artifact(
        artifact_rows,
        artifact="metrics",
        path=metrics_csv_path,
        description="Long-form evaluation metrics.",
    )

    metrics_summary_path = scoring_output_dir / "metrics_summary.csv"
    write_metrics_summary_csv(metrics_summary_path, metrics_by_split)
    append_manifest_artifact(
        artifact_rows,
        artifact="metrics_summary",
        path=metrics_summary_path,
        description="Compact evaluation metrics summary.",
    )

    score_distribution_path = scoring_output_dir / "score_distribution.csv"
    write_dataframe(pd.DataFrame(score_distribution), score_distribution_path)
    append_manifest_artifact(
        artifact_rows,
        artifact="score_distribution",
        path=score_distribution_path,
        description="Prediction score distribution by split and label.",
        row_count=len(score_distribution),
    )

    classifier_info = classifier_summary(classifier)

    classifier_summary_path = scoring_output_dir / "classifier_summary.json"
    write_json(classifier_summary_path, classifier_info)
    append_manifest_artifact(
        artifact_rows,
        artifact="classifier_summary",
        path=classifier_summary_path,
        description="Embedding decoder classifier summary.",
    )

    coefficient_frame = extract_logistic_coefficients(classifier)

    if not coefficient_frame.empty:
        coefficients_path = scoring_output_dir / "classifier_coefficients.csv"
        write_dataframe(coefficient_frame, coefficients_path)
        append_manifest_artifact(
            artifact_rows,
            artifact="classifier_coefficients",
            path=coefficients_path,
            description="Logistic Regression decoder coefficients.",
            row_count=len(coefficient_frame),
        )

    manifest_path = write_manifest(scoring_output_dir, artifact_rows)
    append_manifest_artifact(
        artifact_rows,
        artifact="baseline_manifest",
        path=manifest_path,
        description="Manifest of scoring method artifacts.",
        row_count=len(artifact_rows),
    )

    report_text = build_score_method_report(
        scoring_method=scoring_method,
        dataset_dir=dataset_dir,
        output_dir=scoring_output_dir,
        splits=splits,
        node2vec_parameters=node2vec_parameters,
        data_summary=data_summary,
        embedding_summary=embedding_summary,
        classifier_info=classifier_info,
        metrics_by_split=metrics_by_split,
        score_distribution=score_distribution,
        artifact_rows=artifact_rows,
    )

    report_path = scoring_output_dir / "baseline_report.md"
    write_text(report_path, report_text)
    append_manifest_artifact(
        artifact_rows,
        artifact="baseline_report",
        path=report_path,
        description="Markdown report for this Node2Vec scoring method.",
    )

    write_manifest(scoring_output_dir, artifact_rows)

    test_metrics = metrics_by_split.get("test", {})

    print(
        f"{scoring_method} complete: "
        f"test AUROC={test_metrics.get('auroc'):.6f}, "
        f"test AUPRC={test_metrics.get('auprc'):.6f}"
    )

    return {
        "scoring_method": scoring_method,
        "status": "completed",
        "test_auroc": test_metrics.get("auroc", ""),
        "test_auprc": test_metrics.get("auprc", ""),
        "output_dir": str(scoring_output_dir),
    }


def main() -> None:
    """Run Node2Vec baseline."""

    args = parse_args()

    output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    dataset_dir = Path(args.dataset_dir)
    splits = [str(split).strip() for split in args.splits if str(split).strip()]

    if "train" not in splits:
        raise ValueError("The train split is required for classifier-based decoders.")

    print()
    print("Loading Node2Vec input graph data...")
    print(f"Dataset directory: {dataset_dir}")
    print()

    data = load_node2vec_input_data(
        dataset_dir=dataset_dir,
        splits=splits,
        add_reverse_edges=not args.no_reverse_edges,
        drop_missing_edges=args.drop_missing_edges,
    )

    data_summary = summarize_node2vec_input_data(data)

    print("Input graph summary:")
    for key, value in data_summary.items():
        print(f"  {key}: {value}")
    print()

    node2vec_parameters = {
        "embedding_dim": args.embedding_dim,
        "walk_length": args.walk_length,
        "context_size": args.context_size,
        "walks_per_node": args.walks_per_node,
        "p": args.p,
        "q": args.q,
        "num_negative_samples": args.num_negative_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "sparse": args.sparse,
        "num_workers": args.num_workers,
        "device": args.device,
        "seed": args.seed,
        "add_reverse_edges": not args.no_reverse_edges,
    }

    print("Training Node2Vec embeddings...")
    print()

    try:
        embeddings, training_history, training_metadata = train_node2vec_embeddings(
            edge_index=data["homogeneous_graph"].edge_index,
            num_nodes=data["homogeneous_graph"].num_nodes,
            args=args,
        )
    except Exception:
        print()
        print("Node2Vec training failed.")
        print(
            "If the error mentions random_walk, pyg-lib, or torch-cluster, "
            "your PyG Node2Vec backend may not be installed for this environment."
        )
        print()
        raise

    embedding_summary = summarize_embedding_matrix(embeddings)

    artifact_rows: list[dict[str, Any]] = []

    embedding_metadata = {
        "model": "Node2Vec",
        "dataset_dir": str(dataset_dir),
        "node2vec_parameters": node2vec_parameters,
        "data_summary": data_summary,
        "embedding_summary": embedding_summary,
        "training_metadata": training_metadata,
        "note": (
            "Embeddings were trained on leakage-controlled context_edges only. "
            "No train/val/test re-splitting was performed."
        ),
    }

    artifact_rows.extend(
        save_embedding_artifacts(
            output_dir=output_dir,
            embeddings=embeddings,
            node_index=data["node_index"],
            metadata=embedding_metadata,
        )
    )

    history_path = output_dir / "node2vec_training_history.csv"
    save_training_history(training_history, history_path)
    append_manifest_artifact(
        artifact_rows,
        artifact="node2vec_training_history",
        path=history_path,
        description="Node2Vec training loss history.",
        row_count=len(training_history),
    )

    data_summary_path = output_dir / "node2vec_data_summary.json"
    write_json(data_summary_path, data_summary)
    append_manifest_artifact(
        artifact_rows,
        artifact="node2vec_data_summary",
        path=data_summary_path,
        description="Input graph and labeled split summary.",
    )

    scoring_rows: list[dict[str, Any]] = []

    for scoring_method in args.scoring_methods:
        try:
            row = evaluate_one_scoring_method(
                scoring_method=scoring_method,
                embeddings=embeddings,
                labeled_edges_by_split=data["labeled_edges_by_split"],
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                splits=splits,
                node2vec_parameters=node2vec_parameters,
                data_summary=data_summary,
                embedding_summary=embedding_summary,
                args=args,
            )
            scoring_rows.append(row)

        except Exception:
            print()
            print(f"Scoring method failed: {scoring_method}")
            traceback.print_exc()
            print()

            scoring_rows.append(
                {
                    "scoring_method": scoring_method,
                    "status": "failed",
                    "test_auroc": "",
                    "test_auprc": "",
                    "output_dir": str(output_dir / scoring_method),
                }
            )

    completed = [row for row in scoring_rows if row["status"] == "completed"]

    if not completed:
        raise RuntimeError("No Node2Vec scoring method completed successfully.")

    summary_csv_path = output_dir / "node2vec_scoring_summary.csv"
    write_dataframe(pd.DataFrame(scoring_rows), summary_csv_path)
    append_manifest_artifact(
        artifact_rows,
        artifact="node2vec_scoring_summary",
        path=summary_csv_path,
        description="Summary of all Node2Vec scoring method results.",
        row_count=len(scoring_rows),
    )

    overall_report_text = build_overall_report(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        node2vec_parameters=node2vec_parameters,
        data_summary=data_summary,
        embedding_summary=embedding_summary,
        training_metadata=training_metadata,
        scoring_rows=scoring_rows,
    )

    overall_report_path = output_dir / "node2vec_summary_report.md"
    write_text(overall_report_path, overall_report_text)
    append_manifest_artifact(
        artifact_rows,
        artifact="node2vec_summary_report",
        path=overall_report_path,
        description="Top-level Node2Vec baseline summary report.",
    )

    manifest_path = write_manifest(output_dir, artifact_rows)
    append_manifest_artifact(
        artifact_rows,
        artifact="baseline_manifest",
        path=manifest_path,
        description="Top-level Node2Vec artifact manifest.",
        row_count=len(artifact_rows),
    )

    write_manifest(output_dir, artifact_rows)

    print()
    print("Node2Vec baseline complete.")
    print()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print()
    print("Completed scoring methods:")

    for row in completed:
        print(
            f"  {row['scoring_method']}: "
            f"test AUROC={format_value(row.get('test_auroc'))}, "
            f"test AUPRC={format_value(row.get('test_auprc'))}"
        )

    print()
    print(f"Summary CSV:    {summary_csv_path}")
    print(f"Summary report: {overall_report_path}")
    print()


if __name__ == "__main__":
    main()