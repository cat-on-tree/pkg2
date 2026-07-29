#!/usr/bin/env python
"""Run KU construction / representation-level ablation.

This script compares pre-KU, minimal pair-evidence, full KU feature, KU-graph,
and KU field-dynamics representations under the same temporal prediction task.

Main comparison table:

    Representation              Model             Uses KU pair history?  Uses KU schema?  Uses KU graph?
    Entity-only pre-KU           HistGB / MLP      No                    No               No
    Raw pair evidence            HistGB / MLP      Minimal               No/Partial       No
    Full KU features             HistGB / MLP      Yes                   Yes              No
    KU graph                     GraphSAGE/GRAND   Yes                   Yes              Yes
    KU field dynamics            DynamicRDS v2     Yes                   Yes              Yes

This script trains only the tabular representation baselines. For KU graph and
DynamicRDS rows, it reads existing run directories containing:

    test_metrics_by_cutoff.csv
    run_summary.json

Typical use
-----------

    python scripts/39_run_ku_construction_ablation.py \
      --feature-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy_latest_split \
      --output-dir data/results/knowledge_field/ku_construction_ablation/stage07i_ku_layer_ablation \
      --task translation \
      --test-cutoff-year 2021 \
      --models histgb \
      --grand-run-dir data/results/knowledge_field/temporal_gnn/local_source_proxy_grand_style_v0_seed42 \
      --dynamic-rds-run-dir data/results/knowledge_field/temporal_gnn/local_source_proxy_dynamic_rds_v2_state8_seed42 \
      --overwrite

Notes
-----

- Entity-only feature detection is heuristic. It uses entity_a/entity_b numeric
  features if present, and entity type categorical features if available.
- Raw pair evidence uses minimal pair-level paper trajectory features and,
  by default, pair_type if available. This is marked as KU schema = Partial.
- Full KU features use feature_columns.json.
- This experiment is about the necessity of the KU representation layer, not
  about proving that the most complex model is always best.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Make src/ importable when running this script directly from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pkg2.knowledge_field.metrics import compute_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KU construction / representation-level ablation."
    )

    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help=(
            "Prediction-style feature dataset directory containing "
            "prediction_feature_table.parquet and feature_columns.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--task",
        choices=["translation", "patent", "trial"],
        default="translation",
    )

    parser.add_argument(
        "--test-cutoff-year",
        type=int,
        default=2021,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["histgb", "mlp"],
        default=["histgb"],
        help="Tabular models to train for representation baselines.",
    )

    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[100, 1000, 5000],
    )

    parser.add_argument(
        "--raw-include-pair-type",
        action="store_true",
        default=True,
        help="Include pair_type in raw pair evidence if available.",
    )

    parser.add_argument(
        "--raw-no-pair-type",
        action="store_true",
        help="Disable pair_type in raw pair evidence.",
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional debug/sample limit for train rows.",
    )

    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional debug/sample limit for test rows.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    # Existing Stage 07 GNN/DynamicRDS runs.
    parser.add_argument(
        "--graphsage-run-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--grand-run-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dynamic-rds-run-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--extra-run",
        nargs="*",
        default=[],
        help=(
            "Optional extra external rows in the form "
            "Representation|Model|UsesPairHistory|UsesSchema|UsesGraph|run_dir"
        ),
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


# ---------------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------------


def log(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output suffix: {path}")


# ---------------------------------------------------------------------------
# Dataset / column helpers
# ---------------------------------------------------------------------------


def task_columns(task: str) -> dict[str, str]:
    return {
        "eligible": f"eligible_{task}_task",
        "heat": f"target_future_{task}_heat_3yr",
        "label": f"label_future_{task}_emergence_3yr",
    }


def load_feature_table(feature_dir: Path) -> pd.DataFrame:
    path = feature_dir / "prediction_feature_table.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction_feature_table.parquet: {path}")
    return pd.read_parquet(path)


def load_feature_spec(feature_dir: Path) -> dict[str, Any]:
    path = feature_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_columns.json: {path}")
    return read_json(path)


def list_from_spec(spec: dict[str, Any], keys: list[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = spec.get(key)
        if isinstance(value, list):
            out.extend(str(x) for x in value)
    return out


def infer_feature_columns(
    frame: pd.DataFrame,
    feature_spec: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Infer numeric/categorical feature columns from feature_columns.json."""

    numeric = list_from_spec(
        feature_spec,
        [
            "numeric_features",
            "numeric_feature_columns",
            "numeric_columns",
            "numerical_features",
            "numerical_columns",
        ],
    )

    categorical = list_from_spec(
        feature_spec,
        [
            "categorical_features",
            "categorical_feature_columns",
            "categorical_columns",
            "category_features",
        ],
    )

    explicit_all = list_from_spec(
        feature_spec,
        [
            "feature_columns",
            "features",
            "all_features",
            "all_feature_columns",
            "model_features",
        ],
    )

    if not numeric and not categorical and explicit_all:
        for col in explicit_all:
            if col not in frame.columns:
                continue
            if pd.api.types.is_numeric_dtype(frame[col]):
                numeric.append(col)
            else:
                categorical.append(col)

    def clean(cols: list[str]) -> list[str]:
        seen = set()
        out = []
        for col in cols:
            if col not in frame.columns:
                continue
            if col in seen:
                continue
            if is_forbidden_feature(col):
                continue
            seen.add(col)
            out.append(col)
        return out

    numeric = clean(numeric)
    categorical = clean(categorical)

    cat_set = set(categorical)
    numeric = [c for c in numeric if c not in cat_set]

    if not numeric and not categorical:
        raise ValueError("Could not infer feature columns from feature_columns.json.")

    return numeric, categorical


def is_forbidden_feature(col: str) -> bool:
    lower = col.lower()

    if lower in {
        "split",
        "is_train",
        "is_validation",
        "is_val",
        "is_test",
        "train_mask",
        "validation_mask",
        "val_mask",
        "test_mask",
    }:
        return True

    forbidden_prefixes = (
        "target_future_",
        "label_future_",
        "eligible_",
    )

    return any(lower.startswith(p) for p in forbidden_prefixes)


def is_numeric_col(frame: pd.DataFrame, col: str) -> bool:
    return col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])


def existing(cols: list[str], frame: pd.DataFrame) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c in frame.columns and c not in seen and not is_forbidden_feature(c):
            out.append(c)
            seen.add(c)
    return out


# ---------------------------------------------------------------------------
# Representation feature selection
# ---------------------------------------------------------------------------


def select_entity_only_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str]]:
    """Pre-KU entity-only features.

    This tries to use entity_a/entity_b marginal features if present. It avoids
    pair-level KU trajectory features such as history_paper_count.
    """

    numeric: list[str] = []

    entity_markers = (
        "entity_a_",
        "entity_b_",
        "entity1_",
        "entity2_",
        "source_entity_",
        "target_entity_",
        "a_entity_",
        "b_entity_",
    )

    exclude_tokens = (
        "name",
        "id",
        "ku",
        "pair",
        "target",
        "label",
        "eligible",
    )

    for col in numeric_features:
        lower = col.lower()

        if not any(m in lower for m in entity_markers):
            continue

        if any(tok in lower for tok in exclude_tokens):
            continue

        numeric.append(col)

    # Entity types are pre-KU entity attributes, not pair-level KU schema.
    categorical_candidates = [
        "entity_a_type",
        "entity_b_type",
        "entity1_type",
        "entity2_type",
        "source_entity_type",
        "target_entity_type",
    ]

    categorical = [
        c for c in categorical_candidates
        if c in frame.columns and c in categorical_features
    ]

    # If entity_a_type/entity_b_type exist in table but not in feature spec,
    # still allow them as lightweight entity-only categorical features.
    for c in categorical_candidates:
        if c in frame.columns and c not in categorical:
            categorical.append(c)

    return existing(numeric, frame), existing(categorical, frame)


def select_raw_pair_evidence_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    include_pair_type: bool,
) -> tuple[list[str], list[str]]:
    """Minimal raw pair evidence features.

    Uses simple pair-level paper trajectory features, avoiding source/proxy/graph
    engineered features. This is a minimal KU-like pair evidence representation.
    """

    numeric: list[str] = []

    include_tokens = (
        "history_paper",
        "recent_3yr_paper",
        "recent_paper",
        "paper_count",
        "paper_age",
        "paper_recency",
        "first_paper",
        "last_paper",
        "paper_growth",
        "growth_paper",
    )

    exclude_tokens = (
        "target",
        "label",
        "eligible",
        "carrier",
        "source",
        "structural",
        "exposure",
        "cgs_",
        "src_",
        "kud_",
        "proxy",
        "diffusion",
        "neighbor",
        "neighbour",
        "nbr_",
        "patent",
        "trial",
    )

    for col in numeric_features:
        lower = col.lower()

        if any(tok in lower for tok in exclude_tokens):
            continue

        if any(tok in lower for tok in include_tokens):
            numeric.append(col)

    categorical: list[str] = []

    if include_pair_type and "pair_type" in frame.columns:
        categorical.append("pair_type")

    return existing(numeric, frame), existing(categorical, frame)


def select_full_ku_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str]]:
    return existing(numeric_features, frame), existing(categorical_features, frame)


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------


def make_xy(
    frame: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    heat_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    cols = numeric_cols + categorical_cols
    if not cols:
        raise ValueError("No feature columns selected.")

    x = frame[cols].copy()
    y = pd.to_numeric(frame[heat_col], errors="coerce").fillna(0.0).to_numpy(float)
    return x, y


def build_histgb_model(
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1e-4,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )


def build_mlp_model(
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Avoid high-cardinality categoricals. The selected categoricals should be
    # pair_type/entity types only.
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=4096,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
        verbose=False,
    )

    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )


def train_and_evaluate_tabular(
    *,
    args: argparse.Namespace,
    representation: str,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    uses_pair_history: str,
    uses_ku_schema: str,
    uses_ku_graph: str,
) -> dict[str, Any]:
    cols = numeric_cols + categorical_cols

    if not cols:
        raise ValueError(f"No features selected for representation={representation}")

    columns_path = (
        args.output_dir
        / "feature_sets"
        / f"{slugify(representation)}_{model_name}_features.json"
    )
    write_json(
        columns_path,
        {
            "representation": representation,
            "model": model_name,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_count": len(cols),
        },
    )

    log(
        args,
        (
            f"Training {representation} / {model_name}: "
            f"numeric={len(numeric_cols)} categorical={len(categorical_cols)}"
        ),
    )

    task_cols = task_columns(args.task)
    heat_col = task_cols["heat"]
    label_col = task_cols["label"]

    x_train, y_train = make_xy(
        train_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        heat_col=heat_col,
    )
    x_test, _ = make_xy(
        test_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        heat_col=heat_col,
    )

    if model_name == "histgb":
        model = build_histgb_model(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            random_state=args.random_state,
        )
    elif model_name == "mlp":
        model = build_mlp_model(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            random_state=args.random_state,
        )
    else:
        raise ValueError(f"Unsupported tabular model: {model_name}")

    start = time.time()
    model.fit(x_train, y_train)
    runtime = time.time() - start

    score = np.asarray(model.predict(x_test), dtype=float)

    metrics = compute_metrics(
        y_heat=pd.to_numeric(test_df[heat_col], errors="coerce").fillna(0.0).to_numpy(float),
        y_label=pd.to_numeric(test_df[label_col], errors="coerce").fillna(0).astype(int).to_numpy(),
        score=score,
        topk=args.topk,
    )

    pred_out = test_df[
        [
            c
            for c in [
                "ku_id",
                "knowledge_unit_id",
                "cutoff_year",
                "split",
                "pair_type",
                "entity_a_type",
                "entity_b_type",
                label_col,
                heat_col,
            ]
            if c in test_df.columns
        ]
    ].copy()
    pred_out["score"] = score
    pred_out_path = (
        args.output_dir
        / "predictions"
        / f"{slugify(representation)}_{model_name}_test_cutoff_{args.test_cutoff_year}.parquet"
    )
    write_dataframe(pred_out, pred_out_path)

    row = {
        "representation": representation,
        "model": model_name,
        "uses_ku_pair_history": uses_pair_history,
        "uses_ku_schema": uses_ku_schema,
        "uses_ku_graph": uses_ku_graph,
        "source": "trained_tabular",
        "feature_count": len(cols),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "runtime_seconds": runtime,
        "prediction_path": str(pred_out_path),
        **standardize_metric_names(metrics),
    }

    return row


# ---------------------------------------------------------------------------
# External run metrics
# ---------------------------------------------------------------------------


def read_external_run_metrics(
    *,
    run_dir: Path,
    representation: str,
    model: str,
    uses_pair_history: str,
    uses_schema: str,
    uses_graph: str,
    test_cutoff_year: int,
) -> dict[str, Any]:
    metrics_path = run_dir / "test_metrics_by_cutoff.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing test_metrics_by_cutoff.csv: {metrics_path}")

    df = pd.read_csv(metrics_path)

    if "cutoff_year" in df.columns:
        sub = df[pd.to_numeric(df["cutoff_year"], errors="coerce") == test_cutoff_year]
        if not sub.empty:
            row0 = sub.iloc[0].to_dict()
        else:
            row0 = df.iloc[0].to_dict()
    else:
        row0 = df.iloc[0].to_dict()

    run_summary_path = run_dir / "run_summary.json"
    run_summary = read_json(run_summary_path) if run_summary_path.exists() else {}

    out = {
        "representation": representation,
        "model": model,
        "uses_ku_pair_history": uses_pair_history,
        "uses_ku_schema": uses_schema,
        "uses_ku_graph": uses_graph,
        "source": "external_run",
        "run_dir": str(run_dir),
        "best_epoch": run_summary.get("best_epoch"),
        "best_val_metric": run_summary.get("best_monitor_metric"),
        "best_val_value": run_summary.get("best_monitor_value"),
    }

    out.update(standardize_external_metric_names(row0))
    return out


def parse_extra_run(text: str) -> tuple[str, str, str, str, str, Path]:
    parts = text.split("|")
    if len(parts) != 6:
        raise ValueError(
            "--extra-run must have 6 pipe-separated fields: "
            "Representation|Model|UsesPairHistory|UsesSchema|UsesGraph|run_dir"
        )

    return (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        parts[4],
        Path(parts[5]),
    )


# ---------------------------------------------------------------------------
# Metrics / report helpers
# ---------------------------------------------------------------------------


def standardize_metric_names(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": metrics.get("n"),
        "positive_count": metrics.get("positive_count"),
        "positive_rate": metrics.get("positive_rate"),
        "auprc": metrics.get("auprc"),
        "auroc": metrics.get("auroc"),
        "spearman": metrics.get("spearman"),
        "precision_at_100": metrics.get("precision_at_100"),
        "precision_at_1000": metrics.get("precision_at_1000"),
        "precision_at_5000": metrics.get("precision_at_5000"),
        "recall_at_1000": metrics.get("recall_at_1000"),
        "ndcg_at_1000": metrics.get("ndcg_at_1000"),
        "enrichment_at_1000": metrics.get("enrichment_at_1000"),
    }


def standardize_external_metric_names(row: dict[str, Any]) -> dict[str, Any]:
    # Stage 07 test_metrics_by_cutoff.csv uses n, positive_count, etc.
    return {
        "n": row.get("n"),
        "positive_count": row.get("positive_count"),
        "positive_rate": row.get("positive_rate"),
        "auprc": row.get("auprc"),
        "auroc": row.get("auroc"),
        "spearman": row.get("spearman"),
        "precision_at_100": row.get("precision_at_100"),
        "precision_at_1000": row.get("precision_at_1000"),
        "precision_at_5000": row.get("precision_at_5000"),
        "recall_at_1000": row.get("recall_at_1000"),
        "ndcg_at_1000": row.get("ndcg_at_1000"),
        "enrichment_at_1000": row.get("enrichment_at_1000"),
    }


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"

    cols = [c for c in columns if c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            if math.isnan(x):
                return ""
            return f"{x:.6g}"
        return str(x)

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    return "\n".join(lines)


def write_report(
    *,
    args: argparse.Namespace,
    rows: pd.DataFrame,
    feature_info: dict[str, Any],
) -> None:
    report = []

    report.append("# KU Construction Ablation Report\n")

    report.append("## Configuration\n")
    report.append("```json")
    report.append(
        json.dumps(
            {
                "feature_dir": str(args.feature_dir),
                "task": args.task,
                "test_cutoff_year": args.test_cutoff_year,
                "models": args.models,
                "topk": args.topk,
                "feature_info": feature_info,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    report.append("```\n")

    report.append("## Main comparison\n")

    cols = [
        "representation",
        "model",
        "uses_ku_pair_history",
        "uses_ku_schema",
        "uses_ku_graph",
        "auprc",
        "auroc",
        "spearman",
        "precision_at_1000",
        "ndcg_at_1000",
        "enrichment_at_1000",
    ]

    report.append(markdown_table(rows, cols))
    report.append("")

    report.append("## Interpretation guide\n")
    report.append(
        """
- `Entity-only pre-KU` tests whether marginal entity-level information is sufficient.
- `Raw pair evidence` tests whether minimal pair-level paper trajectory is sufficient.
- `Full KU features` tests the value of the full KU representation without graph dynamics.
- `KU graph` tests whether adding graph structure on KU nodes improves prediction.
- `KU field dynamics` tests whether explicit DynamicRDS knowledge-field dynamics further improves top-K discovery.
"""
    )

    report.append("## Key logic\n")
    report.append(
        """
If `Entity-only pre-KU` is weaker than `Raw pair evidence`, then pair-level evidence is necessary beyond marginal entity popularity.

If `Raw pair evidence` is weaker than `Full KU features`, then KU construction is not merely count aggregation; KU schema/source/proxy features add value.

If `Full KU features` is weaker than `KU graph` / `KU field dynamics`, then the KU layer is also a useful substrate for graph-based knowledge-field dynamics.
"""
    )

    path = args.output_dir / "ku_construction_ablation_report.md"
    path.write_text("\n".join(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    args.feature_dir = args.feature_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.raw_no_pair_type:
        args.raw_include_pair_type = False

    prepare_output_dir(args.output_dir, args.overwrite)

    log(args, "Running KU construction ablation ...")
    log(args, f"  feature_dir: {args.feature_dir}")
    log(args, f"  output_dir:  {args.output_dir}")
    log(args, f"  task:        {args.task}")
    log(args, "")

    frame = load_feature_table(args.feature_dir)
    spec = load_feature_spec(args.feature_dir)
    numeric_features, categorical_features = infer_feature_columns(frame, spec)

    task_cols = task_columns(args.task)
    required = ["split", "cutoff_year", task_cols["eligible"], task_cols["heat"], task_cols["label"]]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    eligible = frame[task_cols["eligible"]].fillna(False).astype(bool)
    train_mask = eligible & (frame["split"].astype(str) == "train")
    test_mask = (
        eligible
        & (frame["split"].astype(str) == "test")
        & (pd.to_numeric(frame["cutoff_year"], errors="coerce") == args.test_cutoff_year)
    )

    train_df = frame.loc[train_mask].copy()
    test_df = frame.loc[test_mask].copy()

    if args.max_train_rows is not None and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(
            n=args.max_train_rows,
            random_state=args.random_state,
        )

    if args.max_test_rows is not None and len(test_df) > args.max_test_rows:
        test_df = test_df.sample(
            n=args.max_test_rows,
            random_state=args.random_state,
        )

    log(args, f"Rows: train={len(train_df):,} test={len(test_df):,}")
    log(args, f"Feature columns: numeric={len(numeric_features)} categorical={len(categorical_features)}")

    feature_sets: dict[str, dict[str, Any]] = {}

    ent_num, ent_cat = select_entity_only_features(
        frame,
        numeric_features,
        categorical_features,
    )
    feature_sets["Entity-only pre-KU"] = {
        "numeric": ent_num,
        "categorical": ent_cat,
        "uses_pair_history": "No",
        "uses_schema": "No",
        "uses_graph": "No",
    }

    raw_num, raw_cat = select_raw_pair_evidence_features(
        frame,
        numeric_features,
        categorical_features,
        include_pair_type=args.raw_include_pair_type,
    )
    feature_sets["Raw pair evidence"] = {
        "numeric": raw_num,
        "categorical": raw_cat,
        "uses_pair_history": "Minimal",
        "uses_schema": "Partial" if "pair_type" in raw_cat else "No",
        "uses_graph": "No",
    }

    full_num, full_cat = select_full_ku_features(
        frame,
        numeric_features,
        categorical_features,
    )
    feature_sets["Full KU features"] = {
        "numeric": full_num,
        "categorical": full_cat,
        "uses_pair_history": "Yes",
        "uses_schema": "Yes",
        "uses_graph": "No",
    }

    feature_info = {
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "feature_sets": {
            name: {
                "numeric_count": len(cfg["numeric"]),
                "categorical_count": len(cfg["categorical"]),
                "numeric_preview": cfg["numeric"][:30],
                "categorical_preview": cfg["categorical"][:30],
            }
            for name, cfg in feature_sets.items()
        },
    }
    write_json(args.output_dir / "feature_selection_summary.json", feature_info)

    rows: list[dict[str, Any]] = []

    for representation, cfg in feature_sets.items():
        for model_name in args.models:
            if not cfg["numeric"] and not cfg["categorical"]:
                log(args, f"Skipping {representation}/{model_name}: no selected features.")
                rows.append(
                    {
                        "representation": representation,
                        "model": model_name,
                        "uses_ku_pair_history": cfg["uses_pair_history"],
                        "uses_ku_schema": cfg["uses_schema"],
                        "uses_ku_graph": cfg["uses_graph"],
                        "source": "skipped_no_features",
                    }
                )
                continue

            row = train_and_evaluate_tabular(
                args=args,
                representation=representation,
                model_name=model_name,
                train_df=train_df,
                test_df=test_df,
                numeric_cols=cfg["numeric"],
                categorical_cols=cfg["categorical"],
                uses_pair_history=cfg["uses_pair_history"],
                uses_ku_schema=cfg["uses_schema"],
                uses_ku_graph=cfg["uses_graph"],
            )
            rows.append(row)

    # External KU graph / field-dynamics rows.
    if args.graphsage_run_dir is not None:
        rows.append(
            read_external_run_metrics(
                run_dir=args.graphsage_run_dir.resolve(),
                representation="KU graph",
                model="GraphSAGE",
                uses_pair_history="Yes",
                uses_schema="Yes",
                uses_graph="Yes",
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    if args.grand_run_dir is not None:
        rows.append(
            read_external_run_metrics(
                run_dir=args.grand_run_dir.resolve(),
                representation="KU graph",
                model="GRAND-style",
                uses_pair_history="Yes",
                uses_schema="Yes",
                uses_graph="Yes",
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    if args.dynamic_rds_run_dir is not None:
        rows.append(
            read_external_run_metrics(
                run_dir=args.dynamic_rds_run_dir.resolve(),
                representation="KU field dynamics",
                model="DynamicRDS v2 state8",
                uses_pair_history="Yes",
                uses_schema="Yes",
                uses_graph="Yes",
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    for extra in args.extra_run:
        rep, model, uses_pair, uses_schema, uses_graph, run_dir = parse_extra_run(extra)
        rows.append(
            read_external_run_metrics(
                run_dir=run_dir.resolve(),
                representation=rep,
                model=model,
                uses_pair_history=uses_pair,
                uses_schema=uses_schema,
                uses_graph=uses_graph,
                test_cutoff_year=args.test_cutoff_year,
            )
        )

    result = pd.DataFrame(rows)

    # Order rows for readability.
    rep_order = {
        "Entity-only pre-KU": 0,
        "Raw pair evidence": 1,
        "Full KU features": 2,
        "KU graph": 3,
        "KU field dynamics": 4,
    }
    result["_rep_order"] = result["representation"].map(rep_order).fillna(99)
    result = result.sort_values(["_rep_order", "model"]).drop(columns=["_rep_order"])

    out_csv = args.output_dir / "ku_construction_ablation_metrics.csv"
    out_main = args.output_dir / "ku_construction_ablation_main_table.csv"

    write_dataframe(result, out_csv)

    main_cols = [
        "representation",
        "model",
        "uses_ku_pair_history",
        "uses_ku_schema",
        "uses_ku_graph",
        "auprc",
        "auroc",
        "spearman",
        "precision_at_1000",
        "ndcg_at_1000",
        "enrichment_at_1000",
    ]
    write_dataframe(result[[c for c in main_cols if c in result.columns]], out_main)

    write_json(
        args.output_dir / "run_config.json",
        {
            "script": "scripts/39_run_ku_construction_ablation.py",
            "feature_dir": str(args.feature_dir),
            "output_dir": str(args.output_dir),
            "task": args.task,
            "test_cutoff_year": args.test_cutoff_year,
            "models": args.models,
            "topk": args.topk,
            "raw_include_pair_type": args.raw_include_pair_type,
            "max_train_rows": args.max_train_rows,
            "max_test_rows": args.max_test_rows,
            "external_runs": {
                "graphsage_run_dir": str(args.graphsage_run_dir) if args.graphsage_run_dir else None,
                "grand_run_dir": str(args.grand_run_dir) if args.grand_run_dir else None,
                "dynamic_rds_run_dir": str(args.dynamic_rds_run_dir) if args.dynamic_rds_run_dir else None,
                "extra_run": args.extra_run,
            },
        },
    )

    write_report(args=args, rows=result, feature_info=feature_info)

    log(args, "")
    log(args, "KU construction ablation complete.")
    log(args, f"  metrics: {out_csv}")
    log(args, f"  main:    {out_main}")
    log(args, f"  report:  {args.output_dir / 'ku_construction_ablation_report.md'}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()