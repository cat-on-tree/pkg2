"""Embedding-based link prediction baseline utilities.

This module provides reusable utilities for evaluating node embeddings on
Patent-Paper link prediction.

It is intended to be shared by graph representation baselines such as:

- Node2Vec
- MetaPath2Vec
- KG embedding models
- GNN models that export node embeddings

The module assumes that each labeled Patent-Paper pair has already been mapped
to graph node indices using `src/pkg2/graph_data.py`.

Core use cases
--------------

1. Direct embedding similarity scoring:

    score = dot(source_embedding, target_embedding)
    score = cosine(source_embedding, target_embedding)

2. Embedding pair operator + Logistic Regression:

    features = source_embedding * target_embedding
    features = abs(source_embedding - target_embedding)
    features = (source_embedding - target_embedding) ** 2
    features = concat(source, target, abs(source-target), source*target)

The resulting prediction rows are compatible with the existing metrics pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pkg2.graph_data import (
    LABEL_COLUMN,
    SOURCE_GRAPH_INDEX_COLUMN,
    SOURCE_ID_COLUMN,
    SOURCE_INDEX_COLUMN,
    SOURCE_TYPE_COLUMN,
    SPLIT_COLUMN,
    TARGET_GRAPH_INDEX_COLUMN,
    TARGET_ID_COLUMN,
    TARGET_INDEX_COLUMN,
    TARGET_TYPE_COLUMN,
)


DIRECT_SCORING_METHODS = {
    "dot",
    "cosine",
    "negative_l2",
    "negative_l1",
}

PAIR_OPERATORS = {
    "hadamard",
    "l1",
    "l2",
    "concat",
}

CLASSIFIER_SCORING_METHODS = {
    "hadamard_logistic",
    "l1_logistic",
    "l2_logistic",
    "concat_logistic",
}

SUPPORTED_SCORING_METHODS = DIRECT_SCORING_METHODS | CLASSIFIER_SCORING_METHODS


@dataclass
class EmbeddingPairClassifier:
    """Trained classifier for embedding-pair features."""

    scoring_method: str
    pair_operator: str
    feature_dim: int
    model: Any


def validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Validate and normalize an embedding matrix.

    Parameters
    ----------
    embeddings:
        Array with shape [num_nodes, embedding_dim].

    Returns
    -------
    np.ndarray
        Float32 embedding matrix.
    """

    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array with shape [num_nodes, embedding_dim]. "
            f"Got shape: {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings must contain at least one node.")

    if embeddings.shape[1] == 0:
        raise ValueError("Embeddings must contain at least one dimension.")

    return embeddings.astype(np.float32, copy=False)


def validate_scoring_method(scoring_method: str) -> str:
    """Validate scoring method name."""

    scoring_method = str(scoring_method).strip()

    if scoring_method not in SUPPORTED_SCORING_METHODS:
        raise ValueError(
            f"Unsupported scoring method: {scoring_method!r}. "
            f"Supported methods: {', '.join(sorted(SUPPORTED_SCORING_METHODS))}"
        )

    return scoring_method


def scoring_method_to_pair_operator(scoring_method: str) -> str:
    """Map classifier scoring method to pair operator."""

    scoring_method = validate_scoring_method(scoring_method)

    if scoring_method == "hadamard_logistic":
        return "hadamard"

    if scoring_method == "l1_logistic":
        return "l1"

    if scoring_method == "l2_logistic":
        return "l2"

    if scoring_method == "concat_logistic":
        return "concat"

    raise ValueError(
        f"Scoring method {scoring_method!r} is not a classifier scoring method."
    )


def validate_labeled_edges(labeled_edges: pd.DataFrame) -> pd.DataFrame:
    """Validate labeled edge table required for embedding scoring."""

    required = {
        SOURCE_GRAPH_INDEX_COLUMN,
        TARGET_GRAPH_INDEX_COLUMN,
        LABEL_COLUMN,
        SPLIT_COLUMN,
    }

    missing = sorted(required - set(labeled_edges.columns))

    if missing:
        raise ValueError(
            "Labeled edge table is missing required columns: "
            + ", ".join(missing)
            + ". Run add_graph_indices_to_labeled_edges first."
        )

    frame = labeled_edges.copy()

    frame[SOURCE_GRAPH_INDEX_COLUMN] = frame[SOURCE_GRAPH_INDEX_COLUMN].astype(np.int64)
    frame[TARGET_GRAPH_INDEX_COLUMN] = frame[TARGET_GRAPH_INDEX_COLUMN].astype(np.int64)
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(int)
    frame[SPLIT_COLUMN] = frame[SPLIT_COLUMN].astype(str)

    return frame


def get_pair_indices(labeled_edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract source and target graph node indices from labeled edges."""

    frame = validate_labeled_edges(labeled_edges)

    source_indices = frame[SOURCE_GRAPH_INDEX_COLUMN].to_numpy(dtype=np.int64)
    target_indices = frame[TARGET_GRAPH_INDEX_COLUMN].to_numpy(dtype=np.int64)

    return source_indices, target_indices


def validate_pair_indices(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    num_nodes: int,
) -> None:
    """Validate pair indices against embedding matrix row count."""

    if len(source_indices) != len(target_indices):
        raise ValueError(
            "Source and target index arrays must have the same length. "
            f"Got {len(source_indices)} and {len(target_indices)}."
        )

    if len(source_indices) == 0:
        return

    min_index = int(min(source_indices.min(), target_indices.min()))
    max_index = int(max(source_indices.max(), target_indices.max()))

    if min_index < 0 or max_index >= num_nodes:
        raise IndexError(
            "Pair indices are out of bounds for embedding matrix. "
            f"Index range: [{min_index}, {max_index}], num_nodes={num_nodes}."
        )


def get_pair_embeddings(
    embeddings: np.ndarray,
    labeled_edges: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source and target embeddings for labeled pairs."""

    embeddings = validate_embeddings(embeddings)
    source_indices, target_indices = get_pair_indices(labeled_edges)

    validate_pair_indices(
        source_indices,
        target_indices,
        num_nodes=embeddings.shape[0],
    )

    return embeddings[source_indices], embeddings[target_indices]


def safe_l2_normalize(
    values: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """L2-normalize rows of a matrix safely."""

    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)

    return values / norms


def score_embedding_pairs(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    *,
    scoring_method: str,
) -> np.ndarray:
    """Score embedding pairs using a direct similarity method.

    Supported direct scoring methods:

    - dot
    - cosine
    - negative_l2
    - negative_l1
    """

    scoring_method = validate_scoring_method(scoring_method)

    if scoring_method not in DIRECT_SCORING_METHODS:
        raise ValueError(
            f"{scoring_method!r} is not a direct scoring method. "
            f"Use one of: {', '.join(sorted(DIRECT_SCORING_METHODS))}"
        )

    source_embeddings = np.asarray(source_embeddings, dtype=np.float32)
    target_embeddings = np.asarray(target_embeddings, dtype=np.float32)

    if source_embeddings.shape != target_embeddings.shape:
        raise ValueError(
            "Source and target embeddings must have the same shape. "
            f"Got {source_embeddings.shape} and {target_embeddings.shape}."
        )

    if scoring_method == "dot":
        return np.sum(source_embeddings * target_embeddings, axis=1).astype(np.float64)

    if scoring_method == "cosine":
        source_normalized = safe_l2_normalize(source_embeddings)
        target_normalized = safe_l2_normalize(target_embeddings)
        return np.sum(source_normalized * target_normalized, axis=1).astype(np.float64)

    if scoring_method == "negative_l2":
        return -np.linalg.norm(source_embeddings - target_embeddings, axis=1).astype(
            np.float64
        )

    if scoring_method == "negative_l1":
        return -np.sum(np.abs(source_embeddings - target_embeddings), axis=1).astype(
            np.float64
        )

    raise ValueError(f"Unhandled scoring method: {scoring_method!r}")


def score_labeled_edges_with_embeddings(
    embeddings: np.ndarray,
    labeled_edges: pd.DataFrame,
    *,
    scoring_method: str,
) -> np.ndarray:
    """Score labeled pairs directly from node embeddings."""

    source_embeddings, target_embeddings = get_pair_embeddings(
        embeddings,
        labeled_edges,
    )

    return score_embedding_pairs(
        source_embeddings,
        target_embeddings,
        scoring_method=scoring_method,
    )


def build_pair_features(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    *,
    pair_operator: str,
) -> np.ndarray:
    """Build pairwise embedding features.

    Supported pair operators:

    - hadamard: source * target
    - l1: abs(source - target)
    - l2: (source - target) ** 2
    - concat: [source, target, abs(source-target), source*target]
    """

    pair_operator = str(pair_operator).strip()

    if pair_operator not in PAIR_OPERATORS:
        raise ValueError(
            f"Unsupported pair operator: {pair_operator!r}. "
            f"Supported operators: {', '.join(sorted(PAIR_OPERATORS))}"
        )

    source_embeddings = np.asarray(source_embeddings, dtype=np.float32)
    target_embeddings = np.asarray(target_embeddings, dtype=np.float32)

    if source_embeddings.shape != target_embeddings.shape:
        raise ValueError(
            "Source and target embeddings must have the same shape. "
            f"Got {source_embeddings.shape} and {target_embeddings.shape}."
        )

    if pair_operator == "hadamard":
        return source_embeddings * target_embeddings

    if pair_operator == "l1":
        return np.abs(source_embeddings - target_embeddings)

    if pair_operator == "l2":
        diff = source_embeddings - target_embeddings
        return diff * diff

    if pair_operator == "concat":
        return np.concatenate(
            [
                source_embeddings,
                target_embeddings,
                np.abs(source_embeddings - target_embeddings),
                source_embeddings * target_embeddings,
            ],
            axis=1,
        )

    raise ValueError(f"Unhandled pair operator: {pair_operator!r}")


def build_pair_features_for_labeled_edges(
    embeddings: np.ndarray,
    labeled_edges: pd.DataFrame,
    *,
    pair_operator: str,
) -> np.ndarray:
    """Build pair features for labeled edges from an embedding matrix."""

    source_embeddings, target_embeddings = get_pair_embeddings(
        embeddings,
        labeled_edges,
    )

    return build_pair_features(
        source_embeddings,
        target_embeddings,
        pair_operator=pair_operator,
    )


def _import_sklearn_classifier_components() -> tuple[Any, Any, Any, Any]:
    """Import sklearn components with a helpful error message."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for embedding pair classifiers. "
            "Install it with: pip install scikit-learn"
        ) from exc

    return SimpleImputer, LogisticRegression, Pipeline, StandardScaler


def train_embedding_pair_classifier(
    embeddings: np.ndarray,
    train_labeled_edges: pd.DataFrame,
    *,
    scoring_method: str = "hadamard_logistic",
    max_iter: int = 1000,
    c_value: float = 1.0,
    class_weight: str | None = None,
    random_state: int = 42,
) -> EmbeddingPairClassifier:
    """Train a Logistic Regression classifier on embedding pair features."""

    scoring_method = validate_scoring_method(scoring_method)

    if scoring_method not in CLASSIFIER_SCORING_METHODS:
        raise ValueError(
            f"{scoring_method!r} is not a classifier scoring method. "
            f"Use one of: {', '.join(sorted(CLASSIFIER_SCORING_METHODS))}"
        )

    pair_operator = scoring_method_to_pair_operator(scoring_method)

    train_frame = validate_labeled_edges(train_labeled_edges)
    x_train = build_pair_features_for_labeled_edges(
        embeddings,
        train_frame,
        pair_operator=pair_operator,
    )
    y_train = train_frame[LABEL_COLUMN].to_numpy(dtype=np.int64)

    SimpleImputer, LogisticRegression, Pipeline, StandardScaler = (
        _import_sklearn_classifier_components()
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    return EmbeddingPairClassifier(
        scoring_method=scoring_method,
        pair_operator=pair_operator,
        feature_dim=int(x_train.shape[1]),
        model=model,
    )


def predict_with_embedding_pair_classifier(
    classifier: EmbeddingPairClassifier,
    embeddings: np.ndarray,
    labeled_edges: pd.DataFrame,
) -> np.ndarray:
    """Predict label=1 probabilities using a trained embedding pair classifier."""

    frame = validate_labeled_edges(labeled_edges)

    features = build_pair_features_for_labeled_edges(
        embeddings,
        frame,
        pair_operator=classifier.pair_operator,
    )

    if hasattr(classifier.model, "predict_proba"):
        return classifier.model.predict_proba(features)[:, 1].astype(np.float64)

    if hasattr(classifier.model, "decision_function"):
        return classifier.model.decision_function(features).astype(np.float64)

    raise AttributeError(
        "Classifier model must expose either predict_proba or decision_function."
    )


def make_prediction_frame(
    labeled_edges: pd.DataFrame,
    scores: np.ndarray,
    *,
    score_column: str = "score",
) -> pd.DataFrame:
    """Create prediction frame compatible with metrics and export pipeline."""

    frame = validate_labeled_edges(labeled_edges)

    scores = np.asarray(scores, dtype=np.float64)

    if len(frame) != len(scores):
        raise ValueError(
            "Number of scores must match number of labeled edges. "
            f"Got {len(scores)} scores for {len(frame)} rows."
        )

    output = pd.DataFrame()

    preferred_columns = [
        SOURCE_INDEX_COLUMN,
        TARGET_INDEX_COLUMN,
        SOURCE_TYPE_COLUMN,
        SOURCE_ID_COLUMN,
        TARGET_TYPE_COLUMN,
        TARGET_ID_COLUMN,
        SOURCE_GRAPH_INDEX_COLUMN,
        TARGET_GRAPH_INDEX_COLUMN,
        SPLIT_COLUMN,
        LABEL_COLUMN,
    ]

    for column in preferred_columns:
        if column in frame.columns:
            output[column] = frame[column].values

    if SOURCE_INDEX_COLUMN not in output.columns:
        output[SOURCE_INDEX_COLUMN] = frame[SOURCE_GRAPH_INDEX_COLUMN].values

    if TARGET_INDEX_COLUMN not in output.columns:
        output[TARGET_INDEX_COLUMN] = frame[TARGET_GRAPH_INDEX_COLUMN].values

    output[score_column] = scores

    return output


def score_labeled_edges(
    embeddings: np.ndarray,
    labeled_edges: pd.DataFrame,
    *,
    scoring_method: str,
    classifier: EmbeddingPairClassifier | None = None,
    score_column: str = "score",
) -> pd.DataFrame:
    """Score one labeled edge table using embeddings.

    For direct methods, `classifier` should be None.

    For classifier methods, provide a trained EmbeddingPairClassifier.
    """

    scoring_method = validate_scoring_method(scoring_method)

    if scoring_method in DIRECT_SCORING_METHODS:
        scores = score_labeled_edges_with_embeddings(
            embeddings,
            labeled_edges,
            scoring_method=scoring_method,
        )

    elif scoring_method in CLASSIFIER_SCORING_METHODS:
        if classifier is None:
            raise ValueError(
                f"A trained classifier is required for {scoring_method!r}."
            )

        if classifier.scoring_method != scoring_method:
            raise ValueError(
                "Classifier scoring method does not match requested scoring method. "
                f"Classifier: {classifier.scoring_method}, requested: {scoring_method}."
            )

        scores = predict_with_embedding_pair_classifier(
            classifier,
            embeddings,
            labeled_edges,
        )

    else:
        raise ValueError(f"Unhandled scoring method: {scoring_method!r}")

    return make_prediction_frame(
        labeled_edges,
        scores,
        score_column=score_column,
    )


def score_labeled_edges_by_split(
    embeddings: np.ndarray,
    labeled_edges_by_split: dict[str, pd.DataFrame],
    *,
    scoring_method: str,
    classifier: EmbeddingPairClassifier | None = None,
    score_column: str = "score",
) -> dict[str, pd.DataFrame]:
    """Score labeled edge tables for multiple splits."""

    return {
        split: score_labeled_edges(
            embeddings,
            frame,
            scoring_method=scoring_method,
            classifier=classifier,
            score_column=score_column,
        )
        for split, frame in labeled_edges_by_split.items()
    }


def train_classifier_and_score_splits(
    embeddings: np.ndarray,
    labeled_edges_by_split: dict[str, pd.DataFrame],
    *,
    scoring_method: str,
    train_split: str = "train",
    max_iter: int = 1000,
    c_value: float = 1.0,
    class_weight: str | None = None,
    random_state: int = 42,
    score_column: str = "score",
) -> tuple[EmbeddingPairClassifier, dict[str, pd.DataFrame]]:
    """Train embedding pair classifier on one split and score all splits."""

    if train_split not in labeled_edges_by_split:
        raise KeyError(
            f"Training split {train_split!r} not found. "
            f"Available splits: {', '.join(sorted(labeled_edges_by_split))}"
        )

    classifier = train_embedding_pair_classifier(
        embeddings,
        labeled_edges_by_split[train_split],
        scoring_method=scoring_method,
        max_iter=max_iter,
        c_value=c_value,
        class_weight=class_weight,
        random_state=random_state,
    )

    predictions_by_split = score_labeled_edges_by_split(
        embeddings,
        labeled_edges_by_split,
        scoring_method=scoring_method,
        classifier=classifier,
        score_column=score_column,
    )

    return classifier, predictions_by_split


def score_embeddings_by_split(
    embeddings: np.ndarray,
    labeled_edges_by_split: dict[str, pd.DataFrame],
    *,
    scoring_method: str,
    train_split: str = "train",
    max_iter: int = 1000,
    c_value: float = 1.0,
    class_weight: str | None = None,
    random_state: int = 42,
    score_column: str = "score",
) -> tuple[EmbeddingPairClassifier | None, dict[str, pd.DataFrame]]:
    """Score embeddings using either direct scoring or classifier scoring.

    Returns
    -------
    classifier:
        None for direct scoring methods. Trained EmbeddingPairClassifier for
        classifier scoring methods.

    predictions_by_split:
        Dict from split name to prediction DataFrame.
    """

    scoring_method = validate_scoring_method(scoring_method)

    if scoring_method in DIRECT_SCORING_METHODS:
        predictions_by_split = score_labeled_edges_by_split(
            embeddings,
            labeled_edges_by_split,
            scoring_method=scoring_method,
            classifier=None,
            score_column=score_column,
        )
        return None, predictions_by_split

    classifier, predictions_by_split = train_classifier_and_score_splits(
        embeddings,
        labeled_edges_by_split,
        scoring_method=scoring_method,
        train_split=train_split,
        max_iter=max_iter,
        c_value=c_value,
        class_weight=class_weight,
        random_state=random_state,
        score_column=score_column,
    )

    return classifier, predictions_by_split


def concatenate_prediction_frames(
    predictions_by_split: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate prediction frames from multiple splits."""

    frames = []

    for split, frame in predictions_by_split.items():
        frame = frame.copy()
        if SPLIT_COLUMN not in frame.columns:
            frame[SPLIT_COLUMN] = split
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def prediction_frames_to_rows(
    predictions_by_split: dict[str, pd.DataFrame],
    *,
    score_column: str = "score",
) -> list[dict[str, Any]]:
    """Convert prediction frames to row dictionaries for metrics pipeline."""

    prediction_frame = concatenate_prediction_frames(predictions_by_split)

    if prediction_frame.empty:
        return []

    required = {SPLIT_COLUMN, LABEL_COLUMN, score_column}
    missing = sorted(required - set(prediction_frame.columns))

    if missing:
        raise ValueError(
            "Prediction frame is missing required columns: "
            + ", ".join(missing)
        )

    return prediction_frame.to_dict(orient="records")


def summarize_prediction_scores(
    predictions_by_split: dict[str, pd.DataFrame],
    *,
    score_column: str = "score",
) -> list[dict[str, Any]]:
    """Summarize prediction score distribution by split and label."""

    rows: list[dict[str, Any]] = []

    for split, frame in predictions_by_split.items():
        if frame.empty:
            continue

        if score_column not in frame.columns:
            raise ValueError(f"Prediction frame is missing {score_column!r} column.")

        for label, group in frame.groupby(LABEL_COLUMN, sort=True):
            scores = group[score_column].astype(float)

            rows.append(
                {
                    "split": split,
                    "label": int(label),
                    "example_count": int(len(group)),
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "score_mean": float(scores.mean()),
                    "score_q25": float(scores.quantile(0.25)),
                    "score_median": float(scores.quantile(0.50)),
                    "score_q75": float(scores.quantile(0.75)),
                }
            )

    return rows


def summarize_embedding_matrix(
    embeddings: np.ndarray,
) -> dict[str, Any]:
    """Summarize an embedding matrix."""

    embeddings = validate_embeddings(embeddings)

    norms = np.linalg.norm(embeddings, axis=1)

    return {
        "num_nodes": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_min": float(np.min(embeddings)),
        "embedding_max": float(np.max(embeddings)),
        "embedding_mean": float(np.mean(embeddings)),
        "embedding_std": float(np.std(embeddings)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
    }


def classifier_summary(
    classifier: EmbeddingPairClassifier | None,
) -> dict[str, Any]:
    """Summarize an embedding pair classifier."""

    if classifier is None:
        return {
            "classifier": "",
            "scoring_method": "",
            "pair_operator": "",
            "feature_dim": "",
        }

    return {
        "classifier": classifier.model.__class__.__name__,
        "scoring_method": classifier.scoring_method,
        "pair_operator": classifier.pair_operator,
        "feature_dim": classifier.feature_dim,
    }


def extract_logistic_coefficients(
    classifier: EmbeddingPairClassifier | None,
) -> pd.DataFrame:
    """Extract Logistic Regression coefficients from a classifier pipeline.

    For high-dimensional embeddings this is mostly diagnostic. The returned
    feature names are generic dimensions of the pair-operator feature space.
    """

    if classifier is None:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    model = classifier.model

    if not hasattr(model, "named_steps") or "model" not in model.named_steps:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    estimator = model.named_steps["model"]

    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    coefficients = estimator.coef_[0]

    rows = []

    if hasattr(estimator, "intercept_"):
        intercept = float(estimator.intercept_[0])
        rows.append(
            {
                "feature": "__intercept__",
                "coefficient": intercept,
                "abs_coefficient": abs(intercept),
            }
        )

    for index, coefficient in enumerate(coefficients):
        coefficient = float(coefficient)
        rows.append(
            {
                "feature": f"{classifier.pair_operator}_dim_{index}",
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
            }
        )

    frame = pd.DataFrame(rows)

    if not frame.empty:
        frame = frame.sort_values(
            "abs_coefficient",
            ascending=False,
        ).reset_index(drop=True)

    return frame


def top_k_prediction_examples(
    predictions_by_split: dict[str, pd.DataFrame],
    *,
    split: str = "test",
    k: int = 100,
    score_column: str = "score",
) -> pd.DataFrame:
    """Return top-k scored prediction examples for one split."""

    if split not in predictions_by_split:
        raise KeyError(
            f"Split {split!r} not found. "
            f"Available splits: {', '.join(sorted(predictions_by_split))}"
        )

    frame = predictions_by_split[split].copy()

    if score_column not in frame.columns:
        raise ValueError(f"Prediction frame is missing {score_column!r} column.")

    return frame.sort_values(
        [score_column, SOURCE_ID_COLUMN, TARGET_ID_COLUMN],
        ascending=[False, True, True],
    ).head(k).reset_index(drop=True)


def list_supported_scoring_methods() -> list[str]:
    """Return sorted supported embedding scoring methods."""

    return sorted(SUPPORTED_SCORING_METHODS)


def list_supported_pair_operators() -> list[str]:
    """Return sorted supported pair operators."""

    return sorted(PAIR_OPERATORS)


def ensure_split_order(
    split_names: Iterable[str],
    *,
    preferred_order: Iterable[str] = ("train", "val", "test"),
) -> list[str]:
    """Return split names in preferred order followed by any remaining names."""

    split_names = [str(split) for split in split_names]
    preferred_order = [str(split) for split in preferred_order]

    ordered = [split for split in preferred_order if split in split_names]
    ordered.extend(split for split in split_names if split not in ordered)

    return ordered