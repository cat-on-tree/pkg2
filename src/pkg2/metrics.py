"""Evaluation metrics for link prediction baselines and models.

This module provides shared evaluation utilities for carrier-level and future
knowledge-unit-level link prediction tasks.

The expected prediction format is simple:

    label: binary ground-truth label, 1 for positive edge and 0 for negative edge
    score: model score, where larger means more likely to be a positive edge

The metrics here are intentionally model-agnostic. They can be reused by:
- heuristic baselines
- logistic regression
- random forest / XGBoost
- node embedding baselines
- KG embedding baselines
- GNN models
- temporal models

Primary metrics:
- AUROC
- AUPRC
- Precision@K
- Recall@K
- Hits@K
- MRR
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io import markdown_table, write_csv, write_text


def _import_sklearn_metrics() -> Any:
    """Import sklearn.metrics with a helpful error message."""

    try:
        from sklearn import metrics as sklearn_metrics
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for AUROC and AUPRC evaluation. "
            "Install it with: pip install scikit-learn"
        ) from exc

    return sklearn_metrics


def _as_float_list(values: Iterable[Any]) -> list[float]:
    """Convert an iterable to a list of floats."""

    result = []

    for value in values:
        if value is None:
            raise ValueError("Metric inputs must not contain None values")
        result.append(float(value))

    return result


def _as_int_label_list(values: Iterable[Any]) -> list[int]:
    """Convert an iterable to a list of binary integer labels."""

    result = []

    for value in values:
        if value is None:
            raise ValueError("Labels must not contain None values")

        label = int(value)

        if label not in (0, 1):
            raise ValueError(f"Labels must be binary 0/1 values, got {value!r}")

        result.append(label)

    return result


def _validate_labels_and_scores(labels: Sequence[int], scores: Sequence[float]) -> None:
    """Validate labels and scores."""

    if len(labels) != len(scores):
        raise ValueError(
            f"labels and scores must have the same length, got "
            f"{len(labels)} labels and {len(scores)} scores"
        )

    if not labels:
        raise ValueError("labels and scores must not be empty")

    for score in scores:
        if math.isnan(float(score)):
            raise ValueError("scores must not contain NaN values")


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers."""

    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def count_binary_labels(labels: Iterable[Any]) -> dict[str, int]:
    """Count positive and negative labels."""

    label_list = _as_int_label_list(labels)
    positive_count = sum(label_list)
    negative_count = len(label_list) - positive_count

    return {
        "example_count": len(label_list),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def compute_auroc(labels: Iterable[Any], scores: Iterable[Any]) -> float | None:
    """Compute AUROC.

    Returns None if AUROC is undefined, for example when only one class is
    present in labels.
    """

    label_list = _as_int_label_list(labels)
    score_list = _as_float_list(scores)
    _validate_labels_and_scores(label_list, score_list)

    if len(set(label_list)) < 2:
        return None

    sklearn_metrics = _import_sklearn_metrics()
    return float(sklearn_metrics.roc_auc_score(label_list, score_list))


def compute_auprc(labels: Iterable[Any], scores: Iterable[Any]) -> float | None:
    """Compute area under the precision-recall curve.

    Returns None if labels contain no positive examples.
    """

    label_list = _as_int_label_list(labels)
    score_list = _as_float_list(scores)
    _validate_labels_and_scores(label_list, score_list)

    if sum(label_list) == 0:
        return None

    sklearn_metrics = _import_sklearn_metrics()
    return float(sklearn_metrics.average_precision_score(label_list, score_list))


def compute_threshold_metrics(
    labels: Iterable[Any],
    scores: Iterable[Any],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute threshold-based binary classification metrics.

    These metrics are useful for calibrated probability scores. For heuristic
    scores such as overlap count or Jaccard, ranking metrics are usually more
    informative than threshold metrics.
    """

    label_list = _as_int_label_list(labels)
    score_list = _as_float_list(scores)
    _validate_labels_and_scores(label_list, score_list)

    predictions = [1 if score >= threshold else 0 for score in score_list]

    true_positive = sum(1 for y, y_hat in zip(label_list, predictions) if y == 1 and y_hat == 1)
    false_positive = sum(1 for y, y_hat in zip(label_list, predictions) if y == 0 and y_hat == 1)
    true_negative = sum(1 for y, y_hat in zip(label_list, predictions) if y == 0 and y_hat == 0)
    false_negative = sum(1 for y, y_hat in zip(label_list, predictions) if y == 1 and y_hat == 0)

    accuracy = _safe_divide(true_positive + true_negative, len(label_list))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def rank_examples(
    labels: Iterable[Any],
    scores: Iterable[Any],
    *,
    descending: bool = True,
) -> list[dict[str, int | float]]:
    """Return examples ranked by score.

    Ties are broken by original order to keep ranking deterministic.
    """

    label_list = _as_int_label_list(labels)
    score_list = _as_float_list(scores)
    _validate_labels_and_scores(label_list, score_list)

    ranked = [
        {
            "rank": 0,
            "original_index": index,
            "label": label,
            "score": score,
        }
        for index, (label, score) in enumerate(zip(label_list, score_list))
    ]

    ranked.sort(
        key=lambda row: (
            -float(row["score"]) if descending else float(row["score"]),
            int(row["original_index"]),
        )
    )

    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    return ranked


def compute_top_k_metrics(
    labels: Iterable[Any],
    scores: Iterable[Any],
    *,
    k_values: Iterable[int] = (10, 50, 100, 500, 1000),
) -> dict[str, float | int]:
    """Compute Precision@K, Recall@K, and Hits@K.

    Precision@K:
        Number of positives in top K divided by K.

    Recall@K:
        Number of positives in top K divided by total positives.

    Hits@K:
        1 if at least one positive appears in top K, otherwise 0.
    """

    label_list = _as_int_label_list(labels)
    score_list = _as_float_list(scores)
    _validate_labels_and_scores(label_list, score_list)

    ranked = rank_examples(label_list, score_list)
    total_count = len(ranked)
    positive_count = sum(label_list)

    metrics: dict[str, float | int] = {}

    normalized_k_values = sorted({int(k) for k in k_values if int(k) > 0})

    for k in normalized_k_values:
        actual_k = min(k, total_count)
        top_k = ranked[:actual_k]
        positives_at_k = sum(int(row["label"]) for row in top_k)

        metrics[f"actual_k_at_{k}"] = actual_k
        metrics[f"positives_at_{k}"] = positives_at_k
        metrics[f"precision_at_{k}"] = _safe_divide(positives_at_k, actual_k)
        metrics[f"recall_at_{k}"] = _safe_divide(positives_at_k, positive_count)
        metrics[f"hits_at_{k}"] = 1 if positives_at_k > 0 else 0

    return metrics


def compute_mrr(labels: Iterable[Any], scores: Iterable[Any]) -> float:
    """Compute global mean reciprocal rank.

    For a single ranked list, this is the reciprocal rank of the first positive
    example. If there is no positive example, the result is 0.0.
    """

    label_list = _as_int_label_list(labels)
    score_list = _as_float_list(scores)
    _validate_labels_and_scores(label_list, score_list)

    ranked = rank_examples(label_list, score_list)

    for row in ranked:
        if int(row["label"]) == 1:
            return 1.0 / int(row["rank"])

    return 0.0


def evaluate_binary_scores(
    labels: Iterable[Any],
    scores: Iterable[Any],
    *,
    k_values: Iterable[int] = (10, 50, 100, 500, 1000),
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate binary link prediction scores.

    Parameters
    ----------
    labels:
        Binary labels. Positive edges should be 1, negative edges should be 0.
    scores:
        Prediction scores. Larger means more likely to be positive.
    k_values:
        K values for top-K metrics.
    threshold:
        Optional threshold for classification metrics. If None, threshold-based
        metrics are not computed.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics.
    """

    label_list = _as_int_label_list(labels)
    score_list = _as_float_list(scores)
    _validate_labels_and_scores(label_list, score_list)

    label_counts = count_binary_labels(label_list)

    result: dict[str, Any] = {
        **label_counts,
        "positive_ratio": _safe_divide(
            label_counts["positive_count"],
            label_counts["example_count"],
        ),
        "score_min": min(score_list),
        "score_max": max(score_list),
        "score_mean": sum(score_list) / len(score_list),
        "auroc": compute_auroc(label_list, score_list),
        "auprc": compute_auprc(label_list, score_list),
        "mrr": compute_mrr(label_list, score_list),
    }

    result.update(
        compute_top_k_metrics(
            label_list,
            score_list,
            k_values=k_values,
        )
    )

    if threshold is not None:
        result.update(
            compute_threshold_metrics(
                label_list,
                score_list,
                threshold=threshold,
            )
        )

    return result


def evaluate_prediction_rows(
    rows: Iterable[dict[str, Any]],
    *,
    label_column: str = "label",
    score_column: str = "score",
    k_values: Iterable[int] = (10, 50, 100, 500, 1000),
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate prediction rows containing label and score columns."""

    row_list = list(rows)

    if not row_list:
        raise ValueError("Prediction rows must not be empty")

    labels = [row[label_column] for row in row_list]
    scores = [row[score_column] for row in row_list]

    return evaluate_binary_scores(
        labels,
        scores,
        k_values=k_values,
        threshold=threshold,
    )


def evaluate_prediction_rows_by_split(
    rows: Iterable[dict[str, Any]],
    *,
    split_column: str = "split",
    label_column: str = "label",
    score_column: str = "score",
    k_values: Iterable[int] = (10, 50, 100, 500, 1000),
    threshold: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate prediction rows grouped by split."""

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        split = str(row.get(split_column, ""))
        if not split:
            raise ValueError(f"Prediction row is missing split column: {split_column}")
        grouped_rows[split].append(row)

    result = {}

    for split, split_rows in sorted(grouped_rows.items()):
        result[split] = evaluate_prediction_rows(
            split_rows,
            label_column=label_column,
            score_column=score_column,
            k_values=k_values,
            threshold=threshold,
        )

    return result


def flatten_metrics(
    metrics: dict[str, Any],
    *,
    split: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a metrics dictionary into long-form rows."""

    rows = []

    for metric, value in metrics.items():
        row = {
            "metric": metric,
            "value": value,
        }

        if split is not None:
            row["split"] = split

        if model_name is not None:
            row["model"] = model_name

        rows.append(row)

    preferred_order = ["model", "split", "metric", "value"]
    rows = [
        {key: row.get(key, "") for key in preferred_order if key in row}
        for row in rows
    ]

    return rows


def flatten_metrics_by_split(
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Convert split metrics into long-form rows."""

    rows = []

    for split, split_metrics in metrics_by_split.items():
        rows.extend(
            flatten_metrics(
                split_metrics,
                split=split,
                model_name=model_name,
            )
        )

    return rows


def summarize_selected_metrics(
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    selected_metrics: Iterable[str] = (
        "example_count",
        "positive_count",
        "negative_count",
        "auroc",
        "auprc",
        "mrr",
        "precision_at_100",
        "recall_at_100",
        "precision_at_1000",
        "recall_at_1000",
    ),
) -> list[dict[str, Any]]:
    """Create a compact split-by-metric summary table."""

    rows = []

    for split in ["train", "val", "test"]:
        if split not in metrics_by_split:
            continue

        split_metrics = metrics_by_split[split]
        row = {"split": split}

        for metric in selected_metrics:
            row[metric] = split_metrics.get(metric, "")

        rows.append(row)

    return rows


def write_metrics_csv(
    path: str | Path,
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    model_name: str | None = None,
) -> Path:
    """Write long-form metrics to CSV."""

    path = Path(path)
    rows = flatten_metrics_by_split(metrics_by_split, model_name=model_name)

    columns = []
    if model_name is not None:
        columns.append("model")
    columns.extend(["split", "metric", "value"])

    write_csv(path, rows, columns=columns)
    return path


def write_metrics_summary_csv(
    path: str | Path,
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    selected_metrics: Iterable[str] = (
        "example_count",
        "positive_count",
        "negative_count",
        "auroc",
        "auprc",
        "mrr",
        "precision_at_100",
        "recall_at_100",
        "precision_at_1000",
        "recall_at_1000",
    ),
) -> Path:
    """Write compact metrics summary to CSV."""

    path = Path(path)
    selected_metric_list = list(selected_metrics)
    rows = summarize_selected_metrics(metrics_by_split, selected_metrics=selected_metric_list)

    write_csv(
        path,
        rows,
        columns=["split", *selected_metric_list],
    )

    return path


def write_metrics_report(
    path: str | Path,
    *,
    title: str,
    metrics_by_split: dict[str, dict[str, Any]],
    parameters: dict[str, Any] | None = None,
    selected_metrics: Iterable[str] = (
        "example_count",
        "positive_count",
        "negative_count",
        "auroc",
        "auprc",
        "mrr",
        "precision_at_100",
        "recall_at_100",
        "precision_at_1000",
        "recall_at_1000",
    ),
) -> Path:
    """Write a Markdown metrics report."""

    path = Path(path)

    selected_metric_list = list(selected_metrics)
    summary_rows = summarize_selected_metrics(
        metrics_by_split,
        selected_metrics=selected_metric_list,
    )

    lines = [
        f"# {title}",
        "",
    ]

    if parameters:
        parameter_rows = [
            {
                "parameter": key,
                "value": value,
            }
            for key, value in parameters.items()
        ]

        lines.extend(
            [
                "## Parameters",
                "",
                markdown_table(parameter_rows, ["parameter", "value"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Metrics summary",
            "",
            markdown_table(summary_rows, ["split", *selected_metric_list]),
            "",
            "## Notes",
            "",
            "- AUROC and AUPRC evaluate ranking quality across all labeled examples.",
            "- Precision@K and Recall@K use descending score order.",
            "- Larger score values are treated as more likely positive links.",
            "",
        ]
    )

    write_text(path, "\n".join(lines))
    return path