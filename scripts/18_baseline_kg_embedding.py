#!/usr/bin/env python
"""Run KG embedding baselines for Patent-Paper link prediction.

This script trains relation-aware KG embedding models on leakage-controlled
context graph triples, then evaluates Patent-Paper link prediction using the
shared embedding decoders.

Supported KG models:
    - DistMult
    - TransE

Important leakage rule:
    KG training uses context_edges only.
    Target Patent-Paper edges are NOT used as KG training triples.

Downstream evaluation:
    This script does NOT re-split train/val/test.
    It reuses labeled_edges_train/val/test from the dataset directory.

Typical dry run:

    python scripts/18_baseline_kg_embedding.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/kg_distmult_probe \
      --model distmult \
      --dry-run \
      --list-relations \
      --overwrite

Typical DistMult run:

    python scripts/18_baseline_kg_embedding.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/kg_distmult_e50 \
      --model distmult \
      --embedding-dim 128 \
      --epochs 50 \
      --batch-size 4096 \
      --num-negative-samples 5 \
      --learning-rate 0.001 \
      --scoring-methods cosine dot negative_l2 hadamard_logistic concat_logistic \
      --device auto \
      --seed 42 \
      --overwrite

Typical TransE run:

    python scripts/18_baseline_kg_embedding.py \
      --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
      --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/kg_transe_e50 \
      --model transe \
      --embedding-dim 128 \
      --epochs 50 \
      --batch-size 4096 \
      --num-negative-samples 5 \
      --learning-rate 0.001 \
      --scoring-methods cosine dot negative_l2 hadamard_logistic concat_logistic \
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
    build_kg_run_metadata,
    build_kg_triples_from_edge_index_dict,
    build_node_type_to_entity_ids,
    relation_infos_to_frame,
    summarize_triples,
    train_kg_embedding_model,
)
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
        description="Run KG embedding baseline for Patent-Paper link prediction."
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
        help="Output directory for KG embedding baseline artifacts.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["distmult", "transe"],
        help="KG embedding model.",
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate. Default: train val test.",
    )

    parser.add_argument(
        "--scoring-methods",
        nargs="*",
        default=list(DEFAULT_SCORING_METHODS),
        choices=list_supported_scoring_methods(),
        help="Embedding decoders for downstream Patent-Paper evaluation.",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Entity and relation embedding dimension. Default: 128.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="KG training epochs. Default: 50.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Positive KG triples per training batch. Default: 4096.",
    )

    parser.add_argument(
        "--num-negative-samples",
        type=int,
        default=5,
        help="Negative KG triples per positive triple. Default: 5.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate. Default: 0.001.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay. Default: 0.",
    )

    parser.add_argument(
        "--transe-p-norm",
        type=int,
        default=1,
        choices=[1, 2],
        help="TransE distance norm. Default: 1.",
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
        help="Do not add reverse relations to the KG training graph.",
    )

    parser.add_argument(
        "--unconstrained-negative-sampling",
        action="store_true",
        help=(
            "Use uniform entity corruption over all node types. "
            "Default is type-constrained negative sampling."
        ),
    )

    parser.add_argument(
        "--no-deduplicate-triples",
        action="store_true",
        help="Do not deduplicate KG triples.",
    )

    parser.add_argument(
        "--drop-missing-edges",
        action="store_true",
        help="Drop context/labeled edges whose endpoints are missing from node index.",
    )

    parser.add_argument(
        "--classifier-max-iter",
        type=int,
        default=1000,
        help="Max iterations for Logistic Regression embedding decoders.",
    )

    parser.add_argument(
        "--classifier-c",
        type=float,
        default=1.0,
        help="C value for Logistic Regression embedding decoders.",
    )

    parser.add_argument(
        "--classifier-class-weight",
        default=None,
        choices=["balanced"],
        help="Class weight for Logistic Regression decoders.",
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
        help="Print KG relation index and triple counts.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build triples and write metadata, but do not train.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print KG training progress every N epochs. Default: 1.",
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
    """Convert common numpy / pandas objects into JSON-safe values."""

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

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None

    return value


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build run configuration for reproducibility."""

    return {
        "script": "scripts/18_baseline_kg_embedding.py",
        "command": " ".join(sys.argv),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "splits": list(args.splits),
        "scoring_methods": list(args.scoring_methods),
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_negative_samples": args.num_negative_samples,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "transe_p_norm": args.transe_p_norm,
        "device": args.device,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "add_reverse_relations": not args.no_reverse_relations,
        "type_constrained_negative_sampling": not args.unconstrained_negative_sampling,
        "deduplicate_triples": not args.no_deduplicate_triples,
        "drop_missing_edges": args.drop_missing_edges,
        "classifier_max_iter": args.classifier_max_iter,
        "classifier_c": args.classifier_c,
        "classifier_class_weight": args.classifier_class_weight,
        "k_values": list(args.k_values),
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_unix": time.time(),
    }


def compute_dataframe_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute a deterministic fingerprint for a split DataFrame.

    The hash is intended as a reproducibility guard. It is not used for model
    training.
    """

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
    """Summarize downstream labeled splits."""

    summary: dict[str, Any] = {}

    for split, frame in labeled_edges_by_split.items():
        label_col = LABEL_COLUMN if LABEL_COLUMN in frame.columns else "label"
        positive_count = int(frame[label_col].sum()) if label_col in frame.columns else 0

        summary[f"labeled_edges:{split}"] = int(len(frame))
        summary[f"positive_edges:{split}"] = positive_count
        summary[f"negative_edges:{split}"] = int(len(frame) - positive_count)

    return summary


def build_metrics_table(metrics_by_split: dict[str, dict[str, Any]], splits: list[str]) -> str:
    """Build a Markdown metrics table."""

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
    model_name: str,
    dataset_dir: Path,
    output_dir: Path,
    splits: list[str],
    run_config: dict[str, Any],
    triples_summary: dict[str, Any],
    embedding_summary: dict[str, Any],
    classifier_info: dict[str, Any],
    metrics_by_split: dict[str, dict[str, Any]],
    score_distribution: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> str:
    """Build Markdown report for one downstream decoder."""

    parameter_rows = [
        {"parameter": "kg_model", "value": model_name},
        {"parameter": "dataset_dir", "value": str(dataset_dir)},
        {"parameter": "output_dir", "value": str(output_dir)},
        {"parameter": "scoring_method", "value": scoring_method},
        {"parameter": "splits", "value": ", ".join(splits)},
        {"parameter": "command", "value": run_config.get("command", "")},
        {"parameter": "seed", "value": run_config.get("seed", "")},
        {
            "parameter": "negative_sampling",
            "value": (
                "type_constrained_uniform"
                if run_config.get("type_constrained_negative_sampling")
                else "uniform_all_entities"
            ),
        },
    ]

    triples_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in triples_summary.items()
        if not str(key).startswith("relation_triples:")
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
            f"# KG Embedding Baseline Report: {model_name} / {scoring_method}",
            "",
            "## 1. Baseline description",
            "",
            "This baseline trains a relation-aware KG embedding model on leakage-controlled context graph triples, then evaluates Patent-Paper link prediction using an embedding decoder.",
            "",
            "The KG training triples are built from `context_edges` only. Target Patent-Paper edges are not used during KG embedding training.",
            "",
            "The script does not re-split train/validation/test. It reuses the labeled split files stored in the dataset directory.",
            "",
            "## 2. Parameters",
            "",
            markdown_table(parameter_rows, ["parameter", "value"]),
            "",
            "## 3. KG triples summary",
            "",
            markdown_table(triples_rows, ["name", "value"]),
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
            "## 9. Reproducibility notes",
            "",
            "- `run_config.json` stores the exact command line and key hyperparameters.",
            "- `split_fingerprint.json` stores hashes of the downstream labeled splits.",
            "- KG negative sampling is used only for KG embedding training and is distinct from Patent-Paper downstream negatives.",
            "- Target Patent-Paper edges are excluded from KG training triples.",
            "",
        ]
    )


def build_overall_report(
    *,
    model_name: str,
    dataset_dir: Path,
    output_dir: Path,
    run_config: dict[str, Any],
    triples_summary: dict[str, Any],
    split_summary: dict[str, Any],
    embedding_summary: dict[str, Any],
    training_metadata: dict[str, Any],
    scoring_rows: list[dict[str, Any]],
) -> str:
    """Build top-level KG embedding baseline report."""

    parameter_rows = [
        {"parameter": "kg_model", "value": model_name},
        {"parameter": "dataset_dir", "value": str(dataset_dir)},
        {"parameter": "output_dir", "value": str(output_dir)},
        {"parameter": "command", "value": run_config.get("command", "")},
        {"parameter": "seed", "value": run_config.get("seed", "")},
        {"parameter": "embedding_dim", "value": run_config.get("embedding_dim", "")},
        {"parameter": "epochs", "value": run_config.get("epochs", "")},
        {"parameter": "batch_size", "value": run_config.get("batch_size", "")},
        {
            "parameter": "num_negative_samples",
            "value": run_config.get("num_negative_samples", ""),
        },
        {
            "parameter": "add_reverse_relations",
            "value": run_config.get("add_reverse_relations", ""),
        },
        {
            "parameter": "type_constrained_negative_sampling",
            "value": run_config.get("type_constrained_negative_sampling", ""),
        },
    ]

    triples_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in triples_summary.items()
        if not str(key).startswith("relation_triples:")
    ]

    split_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in split_summary.items()
    ]

    embedding_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in embedding_summary.items()
    ]

    training_rows = [
        {"name": key, "value": format_value(value)}
        for key, value in training_metadata.items()
        if key not in {"note"}
    ]

    scoring_table_rows = [
        {key: format_value(value) for key, value in row.items()}
        for row in scoring_rows
    ]

    return "\n".join(
        [
            f"# KG Embedding Baseline Summary: {model_name}",
            "",
            "## 1. Description",
            "",
            "This experiment trains a KG embedding model on context graph triples and evaluates multiple Patent-Paper embedding decoders.",
            "",
            "No train/validation/test re-splitting is performed.",
            "",
            "## 2. Parameters",
            "",
            markdown_table(parameter_rows, ["parameter", "value"]),
            "",
            "## 3. KG triples summary",
            "",
            markdown_table(triples_rows, ["name", "value"]),
            "",
            "## 4. Downstream split summary",
            "",
            markdown_table(split_rows, ["name", "value"]),
            "",
            "## 5. Embedding summary",
            "",
            markdown_table(embedding_rows, ["name", "value"]),
            "",
            "## 6. Training metadata",
            "",
            markdown_table(training_rows, ["name", "value"]),
            "",
            "## 7. Scoring method results",
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
            "## 8. Reproducibility",
            "",
            "The exact command line and configuration are saved in:",
            "",
            "```text",
            "run_config.json",
            "```",
            "",
            "Downstream split hashes are saved in:",
            "",
            "```text",
            "split_fingerprint.json",
            "```",
            "",
        ]
    )


def evaluate_one_scoring_method(
    *,
    scoring_method: str,
    model_name: str,
    embeddings: np.ndarray,
    labeled_edges_by_split: dict[str, pd.DataFrame],
    output_dir: Path,
    dataset_dir: Path,
    splits: list[str],
    run_config: dict[str, Any],
    triples_summary: dict[str, Any],
    embedding_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate one downstream embedding decoder."""

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
        model_name=f"kg_{model_name}:{scoring_method}",
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

    report_text = build_score_method_report(
        scoring_method=scoring_method,
        model_name=model_name,
        dataset_dir=dataset_dir,
        output_dir=scoring_output_dir,
        splits=splits,
        run_config=run_config,
        triples_summary=triples_summary,
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
        description="Markdown report for this KG embedding scoring method.",
    )

    manifest_path = write_manifest(scoring_output_dir, artifact_rows)
    append_manifest_artifact(
        artifact_rows,
        artifact="baseline_manifest",
        path=manifest_path,
        description="Manifest of scoring method artifacts.",
        row_count=len(artifact_rows),
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
    """Run KG embedding baseline."""

    args = parse_args()

    output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    dataset_dir = Path(args.dataset_dir)
    splits = [str(split).strip() for split in args.splits if str(split).strip()]

    if "train" not in splits:
        raise ValueError("The train split is required for classifier-based decoders.")

    run_config = build_run_config(args)

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
    print("Loading KG embedding input data...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"KG model:          {args.model}")
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

    print("Building KG triples from context graph...")
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
        description="KG relation index and relation metadata.",
        row_count=len(relation_frame),
    )

    triples_summary = summarize_triples(triples, relation_infos)
    triples_summary_path = output_dir / "kg_triples_summary.json"
    write_json(triples_summary_path, json_safe(triples_summary))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="kg_triples_summary",
        path=triples_summary_path,
        description="Summary of KG training triples.",
    )

    if args.list_relations:
        print("KG relations:")
        print()
        for row in relation_frame.itertuples(index=False):
            print(
                f"  {row.relation_idx:02d} "
                f"{row.relation_name} "
                f"source={row.source_type} "
                f"target={row.target_type} "
                f"triples={row.edge_count}"
            )
        print()

    print("KG triples summary:")
    for key, value in triples_summary.items():
        if not str(key).startswith("relation_triples:"):
            print(f"  {key}: {value}")
    print()

    print("Downstream split summary:")
    for key, value in split_summary.items():
        print(f"  {key}: {value}")
    print()

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

        print("Dry run complete. No KG training was performed.")
        return

    node_type_to_entity_ids = build_node_type_to_entity_ids(node_index)

    print("Training KG embedding model...")
    print()

    kg_result = train_kg_embedding_model(
        triples=triples,
        relation_infos=relation_infos,
        node_type_to_entity_ids=node_type_to_entity_ids,
        num_entities=len(node_index.nodes),
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_negative_samples=args.num_negative_samples,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name="bce",
        type_constrained_negative_sampling=not args.unconstrained_negative_sampling,
        transe_p_norm=args.transe_p_norm,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
        keep_model=False,
        log_every=args.log_every,
    )

    embeddings = kg_result.entity_embeddings
    relation_embeddings = kg_result.relation_embeddings
    embedding_summary = summarize_embedding_matrix(embeddings)

    kg_run_metadata = build_kg_run_metadata(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        model_name=args.model,
        triples_summary=triples_summary,
        relation_infos=relation_infos,
        training_metadata=kg_result.metadata,
        extra={
            "run_config": run_config,
            "split_summary": split_summary,
            "split_fingerprints": split_fingerprints,
            "embedding_summary": embedding_summary,
            "note": (
                "Entity embeddings were trained on leakage-controlled context_edges "
                "only. Target Patent-Paper edges were excluded from KG training. "
                "Downstream Patent-Paper evaluation reuses fixed labeled splits."
            ),
        },
    )

    top_level_artifacts.extend(
        save_embedding_artifacts(
            output_dir=output_dir,
            embeddings=embeddings,
            node_index=node_index,
            metadata=json_safe(kg_run_metadata),
        )
    )

    relation_embeddings_path = output_dir / "relation_embeddings.npy"
    np.save(relation_embeddings_path, relation_embeddings)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="relation_embeddings",
        path=relation_embeddings_path,
        description="KG relation embedding matrix.",
        row_count=int(relation_embeddings.shape[0]),
    )

    training_history_path = output_dir / "kg_training_history.csv"
    write_dataframe(kg_result.training_history, training_history_path)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="kg_training_history",
        path=training_history_path,
        description="KG embedding training loss and diagnostics by epoch.",
        row_count=len(kg_result.training_history),
    )

    training_metadata_path = output_dir / "kg_training_metadata.json"
    write_json(training_metadata_path, json_safe(kg_result.metadata))
    append_manifest_artifact(
        top_level_artifacts,
        artifact="kg_training_metadata",
        path=training_metadata_path,
        description="KG embedding training metadata.",
    )

    print()
    print("KG training complete.")
    print(f"Final loss: {kg_result.metadata.get('final_loss')}")
    print()

    scoring_rows: list[dict[str, Any]] = []

    for scoring_method in args.scoring_methods:
        try:
            row = evaluate_one_scoring_method(
                scoring_method=scoring_method,
                model_name=args.model,
                embeddings=embeddings,
                labeled_edges_by_split=labeled_edges_by_split,
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                splits=splits,
                run_config=run_config,
                triples_summary=triples_summary,
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
        raise RuntimeError("No KG embedding scoring method completed successfully.")

    scoring_summary_path = output_dir / "kg_scoring_summary.csv"
    write_dataframe(pd.DataFrame(scoring_rows), scoring_summary_path)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="kg_scoring_summary",
        path=scoring_summary_path,
        description="Summary of all downstream KG embedding decoder results.",
        row_count=len(scoring_rows),
    )

    overall_report_text = build_overall_report(
        model_name=args.model,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        run_config=run_config,
        triples_summary=triples_summary,
        split_summary=split_summary,
        embedding_summary=embedding_summary,
        training_metadata=kg_result.metadata,
        scoring_rows=scoring_rows,
    )

    overall_report_path = output_dir / "kg_summary_report.md"
    write_text(overall_report_path, overall_report_text)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="kg_summary_report",
        path=overall_report_path,
        description="Top-level KG embedding baseline summary report.",
    )

    manifest_path = write_manifest(output_dir, top_level_artifacts)
    append_manifest_artifact(
        top_level_artifacts,
        artifact="baseline_manifest",
        path=manifest_path,
        description="Top-level KG embedding artifact manifest.",
        row_count=len(top_level_artifacts),
    )
    write_manifest(output_dir, top_level_artifacts)

    print()
    print("KG embedding baseline complete.")
    print()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Model:             {args.model}")
    print()
    print("Completed scoring methods:")

    for row in completed:
        print(
            f"  {row['scoring_method']}: "
            f"test AUROC={format_value(row.get('test_auroc'))}, "
            f"test AUPRC={format_value(row.get('test_auprc'))}"
        )

    print()
    print(f"Summary CSV:    {scoring_summary_path}")
    print(f"Summary report: {overall_report_path}")
    print(f"Run config:     {run_config_path}")
    print()


if __name__ == "__main__":
    main()