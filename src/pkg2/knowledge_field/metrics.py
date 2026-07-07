"""Metrics for Temporal Knowledge Field experiments.

This module contains metric utilities shared by MLP, GraphSAGE, and future
Knowledge Field physical GNN models.

The primary prediction task is KU-level future heat / emergence forecasting:

    score_i(t) -> ranks candidate Knowledge Units at cutoff t

For each task, evaluation is restricted to eligible nodes:

    mask_{task}_eligible == True

Supported metrics
-----------------

Regression / ranking against raw heat:

    spearman

Binary emergence metrics:

    auroc
    auprc

Top-K discovery metrics:

    precision_at_K
    recall_at_K
    ndcg_at_K
    enrichment_at_K

Notes
-----

- Scores may be any real-valued model output. Higher means more likely / more
  intense future heat.
- y_heat should be the raw, non-log-transformed heat target.
- y_label should be the binary emergence target.
- For validation/test aggregation across multiple cutoffs, this module defaults
  to macro averaging over cutoffs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TOPK = (100, 1000, 5000)


@dataclass(frozen=True)
class MetricResult:
    """Metric result for one split/cutoff."""

    split: str
    cutoff_year: int
    metrics: dict[str, float | None]

    def to_row(self, *, epoch: int | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "split": self.split,
            "cutoff_year": int(self.cutoff_year),
        }

        if epoch is not None:
            row["epoch"] = int(epoch)

        row.update(self.metrics)
        return row


def as_numpy_1d(x: Any, *, name: str = "array") -> np.ndarray:
    """Convert input to 1D numpy array."""

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x)

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be convertible to a 1D array.")

    return arr


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return a mask selecting rows finite in all arrays."""

    if not arrays:
        raise ValueError("At least one array is required.")

    n = len(arrays[0])
    mask = np.ones(n, dtype=bool)

    for arr in arrays:
        if len(arr) != n:
            raise ValueError("All arrays must have the same length.")
        mask &= np.isfinite(arr)

    return mask


def rank_spearman(
    y_true: Any,
    score: Any,
) -> float | None:
    """Compute Spearman correlation via average ranks.

    Returns None if fewer than two valid observations are available or if the
    correlation is undefined.
    """

    y = as_numpy_1d(y_true, name="y_true").astype(np.float64)
    s = as_numpy_1d(score, name="score").astype(np.float64)

    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_true={len(y)}, score={len(s)}")

    mask = finite_mask(y, s)
    y = y[mask]
    s = s[mask]

    if len(y) < 2:
        return None

    try:
        y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
        s_rank = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)

        corr = np.corrcoef(y_rank, s_rank)[0, 1]
        if np.isnan(corr):
            return None

        return float(corr)
    except Exception:
        return None


def binary_metrics(
    y_label: Any,
    score: Any,
) -> dict[str, float | None]:
    """Compute AUROC and AUPRC for binary labels."""

    y = as_numpy_1d(y_label, name="y_label").astype(np.int8)
    s = as_numpy_1d(score, name="score").astype(np.float64)

    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_label={len(y)}, score={len(s)}")

    mask = finite_mask(s)
    y = y[mask]
    s = s[mask]

    out: dict[str, float | None] = {
        "auroc": None,
        "auprc": None,
    }

    if len(y) == 0 or len(np.unique(y)) < 2:
        return out

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    except Exception:
        pass

    return out


def dcg_at_k(labels_sorted: np.ndarray, k: int) -> float:
    """Compute DCG@K for labels already sorted by predicted score."""

    kk = min(int(k), int(len(labels_sorted)))
    if kk <= 0:
        return 0.0

    gains = labels_sorted[:kk].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, kk + 2, dtype=np.float64))
    return float(np.sum(gains * discounts))


def ndcg_at_k(
    y_label: np.ndarray,
    order: np.ndarray,
    k: int,
) -> float | None:
    """Compute binary-label NDCG@K."""

    if len(y_label) == 0:
        return None

    kk = min(int(k), int(len(y_label)))
    if kk <= 0:
        return None

    labels_sorted = y_label[order].astype(np.float64)
    dcg = dcg_at_k(labels_sorted, kk)

    ideal = np.sort(y_label.astype(np.float64))[::-1]
    ideal_dcg = dcg_at_k(ideal, kk)

    if ideal_dcg <= 0:
        return None

    return float(dcg / ideal_dcg)


def ranking_at_k_metrics(
    y_label: Any,
    score: Any,
    topk: list[int] | tuple[int, ...] = DEFAULT_TOPK,
) -> dict[str, float | None]:
    """Compute top-K discovery metrics.

    Metrics
    -------
    precision_at_K:
        fraction of top-K predictions that are positive.
    recall_at_K:
        fraction of all positives captured in top-K.
    ndcg_at_K:
        normalized discounted cumulative gain using binary labels.
    enrichment_at_K:
        precision_at_K divided by base positive rate.
    """

    y = as_numpy_1d(y_label, name="y_label").astype(np.int8)
    s = as_numpy_1d(score, name="score").astype(np.float64)

    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_label={len(y)}, score={len(s)}")

    mask = finite_mask(s)
    y = y[mask]
    s = s[mask]

    topk_clean = sorted(set(int(k) for k in topk if int(k) > 0))

    out: dict[str, float | None] = {}

    n = int(len(s))
    total_positive = int(np.sum(y > 0))
    base_rate = total_positive / n if n > 0 else 0.0

    if n == 0:
        for k in topk_clean:
            out[f"precision_at_{k}"] = None
            out[f"recall_at_{k}"] = None
            out[f"ndcg_at_{k}"] = None
            out[f"enrichment_at_{k}"] = None
        return out

    # Stable-ish ordering: primary score descending, secondary index ascending.
    # np.lexsort sorts by last key first, so use (-score) as primary.
    order = np.lexsort((np.arange(n), -s))
    labels_sorted = y[order]

    for k in topk_clean:
        kk = min(k, n)

        top_labels = labels_sorted[:kk]
        positive_at_k = float(np.sum(top_labels > 0))

        precision = positive_at_k / kk if kk > 0 else None
        recall = positive_at_k / total_positive if total_positive > 0 else None

        if precision is not None and base_rate > 0:
            enrichment = precision / base_rate
        else:
            enrichment = None

        ndcg = ndcg_at_k(y, order, kk)

        out[f"precision_at_{k}"] = float(precision) if precision is not None else None
        out[f"recall_at_{k}"] = float(recall) if recall is not None else None
        out[f"ndcg_at_{k}"] = float(ndcg) if ndcg is not None else None
        out[f"enrichment_at_{k}"] = (
            float(enrichment) if enrichment is not None else None
        )

    return out


def compute_metrics(
    *,
    y_heat: Any,
    y_label: Any,
    score: Any,
    topk: list[int] | tuple[int, ...] = DEFAULT_TOPK,
) -> dict[str, float | None]:
    """Compute all standard KU forecasting metrics for eligible examples."""

    heat = as_numpy_1d(y_heat, name="y_heat").astype(np.float64)
    label = as_numpy_1d(y_label, name="y_label").astype(np.int8)
    pred = as_numpy_1d(score, name="score").astype(np.float64)

    if not (len(heat) == len(label) == len(pred)):
        raise ValueError(
            "Length mismatch: "
            f"y_heat={len(heat)}, y_label={len(label)}, score={len(pred)}"
        )

    mask = finite_mask(heat, pred)
    heat = heat[mask]
    label = label[mask]
    pred = pred[mask]

    n = int(len(pred))
    positive_count = int(np.sum(label > 0))

    out: dict[str, float | None] = {
        "n": float(n),
        "positive_count": float(positive_count),
        "positive_rate": float(positive_count / n) if n > 0 else None,
        "heat_mean": float(np.mean(heat)) if n > 0 else None,
        "heat_std": float(np.std(heat)) if n > 0 else None,
        "score_mean": float(np.mean(pred)) if n > 0 else None,
        "score_std": float(np.std(pred)) if n > 0 else None,
        "spearman": rank_spearman(heat, pred),
    }

    out.update(binary_metrics(label, pred))
    out.update(ranking_at_k_metrics(label, pred, topk=topk))

    return out


def compute_split_cutoff_metrics(
    *,
    split: str,
    cutoff_year: int,
    y_heat: Any,
    y_label: Any,
    score: Any,
    topk: list[int] | tuple[int, ...] = DEFAULT_TOPK,
) -> MetricResult:
    """Compute metrics and wrap them with split/cutoff metadata."""

    return MetricResult(
        split=str(split),
        cutoff_year=int(cutoff_year),
        metrics=compute_metrics(
            y_heat=y_heat,
            y_label=y_label,
            score=score,
            topk=topk,
        ),
    )


def metrics_to_dataframe(
    rows: list[MetricResult] | list[dict[str, Any]],
    *,
    epoch: int | None = None,
) -> pd.DataFrame:
    """Convert metric rows to a DataFrame."""

    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        if isinstance(row, MetricResult):
            normalized_rows.append(row.to_row(epoch=epoch))
        else:
            normalized_rows.append(dict(row))

    return pd.DataFrame(normalized_rows)


def is_number(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return not math.isnan(float(value))
        except Exception:
            return True

    return False


def aggregate_metric_rows(
    metric_rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    prefix: str,
    skip_keys: set[str] | None = None,
) -> dict[str, float | None]:
    """Macro-average metric rows across cutoffs.

    Parameters
    ----------
    metric_rows:
        Rows containing metric values.
    prefix:
        Prefix added to output names, e.g. "validation" -> "validation_auprc".
    skip_keys:
        Keys that should not be aggregated.
    """

    if isinstance(metric_rows, pd.DataFrame):
        rows = metric_rows.to_dict(orient="records")
    else:
        rows = [dict(x) for x in metric_rows]

    if not rows:
        return {}

    skip = {
        "split",
        "cutoff_year",
        "epoch",
        "monitor_metric",
        "monitor_value",
    }

    if skip_keys:
        skip |= set(skip_keys)

    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in skip:
                continue
            if is_number(value):
                keys.add(key)

    out: dict[str, float | None] = {}

    for key in sorted(keys):
        vals = [
            float(row[key])
            for row in rows
            if key in row and is_number(row[key])
        ]
        out[f"{prefix}_{key}"] = float(np.mean(vals)) if vals else None

    return out


def weighted_aggregate_metric_rows(
    metric_rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    prefix: str,
    weight_key: str = "n",
    skip_keys: set[str] | None = None,
) -> dict[str, float | None]:
    """Weighted-average metric rows across cutoffs.

    This can be useful when cutoffs have different eligible counts. In the
    current KU benchmark each cutoff often has the same node count, but task
    eligibility may vary by cutoff.
    """

    if isinstance(metric_rows, pd.DataFrame):
        rows = metric_rows.to_dict(orient="records")
    else:
        rows = [dict(x) for x in metric_rows]

    if not rows:
        return {}

    skip = {
        "split",
        "cutoff_year",
        "epoch",
        "monitor_metric",
        "monitor_value",
    }

    if skip_keys:
        skip |= set(skip_keys)

    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in skip or key == weight_key:
                continue
            if is_number(value):
                keys.add(key)

    out: dict[str, float | None] = {}

    for key in sorted(keys):
        vals = []
        weights = []

        for row in rows:
            value = row.get(key)
            weight = row.get(weight_key)

            if is_number(value) and is_number(weight) and float(weight) > 0:
                vals.append(float(value))
                weights.append(float(weight))

        if vals and weights:
            out[f"{prefix}_{key}"] = float(np.average(vals, weights=weights))
        else:
            out[f"{prefix}_{key}"] = None

    return out


def get_monitor_value(
    aggregated_metrics: dict[str, float | None],
    *,
    split: str = "validation",
    monitor_metric: str = "auprc",
) -> float:
    """Extract early-stopping monitor value from aggregated metrics.

    Returns -inf if the metric is missing or invalid, so standard "larger is
    better" early stopping works.
    """

    key = f"{split}_{monitor_metric}"
    value = aggregated_metrics.get(key)

    if value is None:
        return -math.inf

    try:
        value_f = float(value)
    except Exception:
        return -math.inf

    if math.isnan(value_f):
        return -math.inf

    return value_f


def summarize_prediction_distribution(
    score: Any,
    *,
    prefix: str = "score",
) -> dict[str, float | None]:
    """Return simple descriptive stats for prediction scores."""

    arr = as_numpy_1d(score, name=prefix).astype(np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_p25": None,
            f"{prefix}_median": None,
            f"{prefix}_p75": None,
            f"{prefix}_max": None,
        }

    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_median": float(np.percentile(arr, 50)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_max": float(np.max(arr)),
    }


def compare_metric_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric_columns: list[str],
    id_columns: list[str] | None = None,
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Compare two metric tables by cutoff/split.

    Useful for comparing local vs local+source vs local+source+proxy runs.
    """

    if id_columns is None:
        id_columns = ["split", "cutoff_year"]

    required_left = set(id_columns + metric_columns)
    required_right = set(id_columns + metric_columns)

    missing_left = required_left - set(left.columns)
    missing_right = required_right - set(right.columns)

    if missing_left:
        raise ValueError(f"Left metric table missing columns: {sorted(missing_left)}")

    if missing_right:
        raise ValueError(f"Right metric table missing columns: {sorted(missing_right)}")

    merged = left[id_columns + metric_columns].merge(
        right[id_columns + metric_columns],
        on=id_columns,
        how="inner",
        suffixes=suffixes,
    )

    for metric in metric_columns:
        left_col = metric + suffixes[0]
        right_col = metric + suffixes[1]
        delta_col = f"{metric}_delta"

        merged[delta_col] = merged[right_col] - merged[left_col]

    return merged