#!/usr/bin/env python
"""Run stratified evaluation for Temporal Knowledge Field predictions.

This script is a post-hoc analysis tool. It does not train models.

It expects predictions exported by:

    scripts/36_export_knowledge_field_predictions.py

Typical use
-----------

    python scripts/37_run_knowledge_field_stratified_eval.py \
      --gnn-input-dir data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_proxy_latest_split \
      --run-dirs \
        data/results/knowledge_field/temporal_gnn/local_source_proxy_dynamic_rds_v2_state8_seed42 \
        data/results/knowledge_field/temporal_gnn/local_source_proxy_grand_style_v0_seed42 \
        data/results/knowledge_field/temporal_gnn/local_source_proxy_rdgnn_style_v0_seed42 \
      --run-names \
        dynamic_rds_v2_state8 \
        grand_style \
        rdgnn_style \
      --split test \
      --cutoff-year 2021 \
      --task translation \
      --output-dir data/results/knowledge_field/temporal_gnn/stratified_analysis/dynamic_rds_v2_vs_physics_baselines \
      --overwrite

Outputs
-------

    stratified_metrics.csv
    topk_composition.csv
    stratified_report.md
    run_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
        description="Stratified evaluation for Knowledge Field predictions."
    )

    parser.add_argument(
        "--gnn-input-dir",
        type=Path,
        required=True,
        help="GNN input directory used by the runs.",
    )

    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more run directories containing exported predictions.",
    )

    parser.add_argument(
        "--run-names",
        nargs="+",
        type=str,
        default=None,
        help="Optional display names matching --run-dirs.",
    )

    parser.add_argument(
        "--task",
        choices=["translation", "patent", "trial"],
        default="translation",
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
    )

    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=2021,
    )

    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[100, 1000, 5000],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
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
# IO helpers
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


def prediction_path(
    run_dir: Path,
    *,
    task: str,
    split: str,
    cutoff_year: int,
) -> Path:
    pred_dir = run_dir / "predictions"

    parquet = pred_dir / f"predictions_{task}_{split}_cutoff_{cutoff_year}.parquet"
    csv = pred_dir / f"predictions_{task}_{split}_cutoff_{cutoff_year}.csv"

    if parquet.exists():
        return parquet

    if csv.exists():
        return csv

    raise FileNotFoundError(
        "Could not find prediction file. Expected one of:\n"
        f"  {parquet}\n"
        f"  {csv}\n"
        "Run scripts/36_export_knowledge_field_predictions.py first."
    )


def read_prediction_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported prediction file suffix: {path}")


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def load_feature_schema(gnn_input_dir: Path) -> dict[str, Any]:
    path = gnn_input_dir / "feature_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_schema.json: {path}")
    return read_json(path)


def load_node_index(gnn_input_dir: Path) -> pd.DataFrame:
    path = gnn_input_dir / "graph" / "ku_node_index.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing ku_node_index.parquet: {path}")
    return pd.read_parquet(path)


def find_feature(
    numeric_features: list[str],
    candidates: list[str],
    *,
    contains_fallback: list[str] | None = None,
) -> str | None:
    """Find a feature by exact candidate names, then optional contains fallback."""

    feature_set = set(numeric_features)

    for c in candidates:
        if c in feature_set:
            return c

    if contains_fallback:
        lower_features = [(f.lower(), f) for f in numeric_features]
        for token in contains_fallback:
            token_l = token.lower()
            for f_l, f in lower_features:
                if token_l in f_l:
                    return f

    return None


def infer_source_features(numeric_features: list[str]) -> list[str]:
    """Heuristic source/exposure feature group.

    This is intentionally broad but avoids target/label/eligibility columns.
    """

    out: list[str] = []

    forbidden_prefixes = (
        "target_future_",
        "label_future_",
        "eligible_",
    )

    for f in numeric_features:
        fl = f.lower()

        if any(fl.startswith(p) for p in forbidden_prefixes):
            continue

        is_source = (
            fl.startswith("cgs_")
            or "source" in fl
            or "exposure" in fl
            or "carrier" in fl
            or "coverage" in fl
            or "support_" in fl
            or "_support" in fl
            or "project" in fl
        )

        # Avoid accidentally putting proxy features into source.
        is_proxy_like = (
            fl.startswith("kud_")
            or "proxy" in fl
            or "neighbor" in fl
            or "neighbour" in fl
            or "diffusion" in fl
        )

        if is_source and not is_proxy_like:
            out.append(f)

    return sorted(set(out))


def infer_proxy_features(numeric_features: list[str]) -> list[str]:
    """Heuristic diffusion proxy feature group."""

    out: list[str] = []

    forbidden_prefixes = (
        "target_future_",
        "label_future_",
        "eligible_",
    )

    for f in numeric_features:
        fl = f.lower()

        if any(fl.startswith(p) for p in forbidden_prefixes):
            continue

        is_proxy = (
            fl.startswith("kud_")
            or "proxy" in fl
            or "neighbor" in fl
            or "neighbour" in fl
            or "diffusion" in fl
        )

        if is_proxy:
            out.append(f)

    return sorted(set(out))


def zscore_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    """Return row-wise mean z-score over selected columns."""

    if not columns:
        return None

    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return None

    x = frame[cols].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )

    means = x.mean(axis=0)
    stds = x.std(axis=0).replace(0, np.nan)

    z = (x - means) / stds
    score = z.mean(axis=1, skipna=True)

    return score.fillna(0.0)


def load_numeric_feature_subset(
    *,
    gnn_input_dir: Path,
    split: str,
    cutoff_year: int,
    node_idx: np.ndarray,
    columns: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    """Load selected numeric columns for selected node indices."""

    if not columns:
        return pd.DataFrame(index=np.arange(len(node_idx)))

    col_to_idx = {col: i for i, col in enumerate(numeric_features)}
    valid_cols = [c for c in columns if c in col_to_idx]

    if not valid_cols:
        return pd.DataFrame(index=np.arange(len(node_idx)))

    x_path = (
        gnn_input_dir
        / "snapshots"
        / split
        / f"cutoff_{int(cutoff_year)}"
        / "x_numeric.npy"
    )

    if not x_path.exists():
        raise FileNotFoundError(f"Missing x_numeric.npy: {x_path}")

    x = np.load(x_path, mmap_mode="r")
    selected_indices = [col_to_idx[c] for c in valid_cols]

    # np.ix_ materializes only selected rows and columns.
    block = np.asarray(x[np.ix_(node_idx.astype(np.int64), selected_indices)])
    return pd.DataFrame(block, columns=valid_cols)


# ---------------------------------------------------------------------------
# Strata construction
# ---------------------------------------------------------------------------


def quantile_strata(
    values: pd.Series,
    *,
    low_name: str,
    mid_name: str,
    high_name: str,
) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    if v.notna().sum() == 0:
        return pd.Series(["missing"] * len(v), index=v.index, dtype="object")

    q25 = float(v.quantile(0.25))
    q75 = float(v.quantile(0.75))

    def assign(x: float) -> str:
        if pd.isna(x):
            return "missing"
        if x <= q25:
            return low_name
        if x >= q75:
            return high_name
        return mid_name

    return v.map(assign).astype("object")


def build_strata(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Construct all default strata."""

    strata: dict[str, pd.Series] = {}

    # pair_type is from ku_node_index.parquet.
    if "pair_type" in frame.columns:
        strata["pair_type"] = frame["pair_type"].fillna("missing").astype(str)

    # History paper maturity strata.
    if "history_paper_count" in frame.columns:
        h = pd.to_numeric(frame["history_paper_count"], errors="coerce").fillna(0.0)

        def history_stage(x: float) -> str:
            if x <= 0:
                return "no_history_paper"
            if x <= 2:
                return "seed_1_2"
            if x <= 9:
                return "early_3_9"
            if x <= 49:
                return "established_10_49"
            return "mature_50_plus"

        strata["history_paper_stage"] = h.map(history_stage).astype("object")

    # Recent activity strata.
    if "recent_3yr_paper_count" in frame.columns:
        r = pd.to_numeric(frame["recent_3yr_paper_count"], errors="coerce").fillna(0.0)

        def recent_stage(x: float) -> str:
            if x <= 0:
                return "dormant_recent0"
            if x <= 2:
                return "low_recent_1_2"
            if x <= 9:
                return "active_recent_3_9"
            return "very_active_recent_10_plus"

        strata["recent_paper_activity"] = r.map(recent_stage).astype("object")

    # Paper growth strata.
    growth_col = None
    for c in [
        "paper_growth",
        "paper_growth_score",
        "growth_paper_3yr_ratio",
        "history_paper_growth",
        "recent_paper_growth",
    ]:
        if c in frame.columns:
            growth_col = c
            break

    if growth_col is not None:
        strata["paper_growth_quantile"] = quantile_strata(
            frame[growth_col],
            low_name="low_growth_q0_25",
            mid_name="mid_growth_q25_75",
            high_name="high_growth_q75_100",
        )

    if "source_score" in frame.columns:
        strata["source_score_quantile"] = quantile_strata(
            frame["source_score"],
            low_name="low_source_q0_25",
            mid_name="mid_source_q25_75",
            high_name="high_source_q75_100",
        )

    if "proxy_score" in frame.columns:
        strata["proxy_score_quantile"] = quantile_strata(
            frame["proxy_score"],
            low_name="low_proxy_q0_25",
            mid_name="mid_proxy_q25_75",
            high_name="high_proxy_q75_100",
        )

    return strata


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def safe_compute_metrics(
    sub: pd.DataFrame,
    *,
    topk: list[int],
) -> dict[str, float | None]:
    if sub.empty:
        return {}

    return compute_metrics(
        y_heat=sub["y_heat"].to_numpy(),
        y_label=sub["y_label"].to_numpy(),
        score=sub["score"].to_numpy(),
        topk=topk,
    )


def compute_stratified_metrics(
    frame: pd.DataFrame,
    *,
    run_name: str,
    topk: list[int],
) -> pd.DataFrame:
    strata = build_strata(frame)
    rows: list[dict[str, Any]] = []

    # Overall row.
    overall = safe_compute_metrics(frame, topk=topk)
    rows.append(
        {
            "run": run_name,
            "stratification": "overall",
            "stratum": "all",
            **overall,
        }
    )

    for strat_name, labels in strata.items():
        tmp = frame.copy()
        tmp["_stratum"] = labels.fillna("missing").astype(str)

        for stratum, sub in tmp.groupby("_stratum", sort=True):
            metrics = safe_compute_metrics(sub, topk=topk)
            rows.append(
                {
                    "run": run_name,
                    "stratification": strat_name,
                    "stratum": stratum,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def compute_topk_composition(
    frame: pd.DataFrame,
    *,
    run_name: str,
    topk: list[int],
) -> pd.DataFrame:
    strata = build_strata(frame)
    rows: list[dict[str, Any]] = []

    if frame.empty:
        return pd.DataFrame()

    ordered = frame.sort_values(
        ["score", "node_idx"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    for k in sorted(set(int(x) for x in topk if int(x) > 0)):
        kk = min(k, len(ordered))
        top = ordered.iloc[:kk].copy()

        total_top_positive = int(top["y_label"].sum())

        for strat_name, labels in strata.items():
            top["_stratum"] = labels.loc[top.index].fillna("missing").astype(str).to_numpy()

            for stratum, sub in top.groupby("_stratum", sort=True):
                count_in_topk = int(len(sub))
                positive_in_topk = int(sub["y_label"].sum())

                rows.append(
                    {
                        "run": run_name,
                        "topk": int(k),
                        "effective_k": int(kk),
                        "stratification": strat_name,
                        "stratum": str(stratum),
                        "count_in_topk": count_in_topk,
                        "positive_in_topk": positive_in_topk,
                        "precision_within_topk_stratum": (
                            positive_in_topk / count_in_topk
                            if count_in_topk > 0
                            else None
                        ),
                        "share_of_topk": (
                            count_in_topk / kk
                            if kk > 0
                            else None
                        ),
                        "share_of_topk_positives": (
                            positive_in_topk / total_top_positive
                            if total_top_positive > 0
                            else None
                        ),
                        "total_topk_positives": total_top_positive,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def markdown_table(df: pd.DataFrame, *, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"

    use = df[columns].head(max_rows).copy()

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            if math.isnan(x):
                return ""
            return f"{x:.6g}"
        return str(x)

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in use.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")

    return "\n".join(lines)


def write_report(
    *,
    output_dir: Path,
    metrics: pd.DataFrame,
    topk_comp: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    report = []

    report.append("# Knowledge Field Stratified Evaluation Report\n")
    report.append("## Configuration\n")
    report.append("```json")
    report.append(
        json.dumps(
            {
                "gnn_input_dir": str(args.gnn_input_dir),
                "run_dirs": [str(x) for x in args.run_dirs],
                "run_names": args.run_names,
                "task": args.task,
                "split": args.split,
                "cutoff_year": args.cutoff_year,
                "topk": args.topk,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    report.append("```\n")

    report.append("## Overall metrics\n")
    overall = metrics[metrics["stratification"] == "overall"].copy()
    overall_cols = [
        "run",
        "n",
        "positive_count",
        "positive_rate",
        "auprc",
        "auroc",
        "spearman",
        "precision_at_100",
        "precision_at_1000",
        "precision_at_5000",
        "ndcg_at_1000",
        "enrichment_at_1000",
    ]
    overall_cols = [c for c in overall_cols if c in overall.columns]
    report.append(markdown_table(overall, columns=overall_cols, max_rows=50))
    report.append("")

    for strat in [
        "history_paper_stage",
        "recent_paper_activity",
        "paper_growth_quantile",
        "source_score_quantile",
        "proxy_score_quantile",
        "pair_type",
    ]:
        sub = metrics[metrics["stratification"] == strat].copy()
        if sub.empty:
            continue

        report.append(f"## Stratification: `{strat}`\n")
        cols = [
            "run",
            "stratum",
            "n",
            "positive_count",
            "positive_rate",
            "auprc",
            "auroc",
            "spearman",
            "precision_at_1000",
            "ndcg_at_1000",
            "enrichment_at_1000",
        ]
        cols = [c for c in cols if c in sub.columns]
        report.append(markdown_table(sub, columns=cols, max_rows=100))
        report.append("")

    report.append("## Global top-K composition\n")
    if not topk_comp.empty:
        comp = topk_comp[topk_comp["topk"] == 1000].copy()
        cols = [
            "run",
            "stratification",
            "stratum",
            "count_in_topk",
            "positive_in_topk",
            "precision_within_topk_stratum",
            "share_of_topk",
            "share_of_topk_positives",
        ]
        cols = [c for c in cols if c in comp.columns]
        report.append(markdown_table(comp, columns=cols, max_rows=120))
    else:
        report.append("_No top-K composition rows._")

    report.append("")

    path = output_dir / "stratified_report.md"
    path.write_text("\n".join(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def build_base_frame(
    *,
    args: argparse.Namespace,
    first_pred: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gnn_input_dir = args.gnn_input_dir

    schema = load_feature_schema(gnn_input_dir)
    numeric_features = list(schema.get("numeric_features", []))

    if not numeric_features:
        raise ValueError("No numeric_features found in feature_schema.json")

    node_index = load_node_index(gnn_input_dir)

    required_pred_cols = {"node_idx", "split", "cutoff_year", "y_heat", "y_label", "score"}
    missing_pred = required_pred_cols.difference(first_pred.columns)
    if missing_pred:
        raise ValueError(f"Prediction file missing columns: {sorted(missing_pred)}")

    node_idx = first_pred["node_idx"].to_numpy(dtype=np.int64)

    base = first_pred[["node_idx", "split", "cutoff_year", "y_heat", "y_label"]].copy()
    base = base.merge(node_index, on="node_idx", how="left")

    history_col = find_feature(
        numeric_features,
        ["history_paper_count"],
        contains_fallback=["history_paper_count"],
    )
    recent_col = find_feature(
        numeric_features,
        ["recent_3yr_paper_count"],
        contains_fallback=["recent_3yr_paper"],
    )
    growth_col = find_feature(
        numeric_features,
        [
            "paper_growth",
            "paper_growth_score",
            "growth_paper_3yr_ratio",
            "history_paper_growth",
            "recent_paper_growth",
        ],
        contains_fallback=["paper_growth", "growth_paper"],
    )

    source_features = infer_source_features(numeric_features)
    proxy_features = infer_proxy_features(numeric_features)

    columns_to_load: list[str] = []
    for col in [history_col, recent_col, growth_col]:
        if col is not None:
            columns_to_load.append(col)

    columns_to_load.extend(source_features)
    columns_to_load.extend(proxy_features)

    # Deduplicate while preserving order.
    seen = set()
    columns_to_load = [
        c for c in columns_to_load
        if not (c in seen or seen.add(c))
    ]

    feature_frame = load_numeric_feature_subset(
        gnn_input_dir=gnn_input_dir,
        split=args.split,
        cutoff_year=args.cutoff_year,
        node_idx=node_idx,
        columns=columns_to_load,
        numeric_features=numeric_features,
    )

    base = pd.concat(
        [
            base.reset_index(drop=True),
            feature_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    if history_col is not None and history_col != "history_paper_count":
        base["history_paper_count"] = base[history_col]

    if recent_col is not None and recent_col != "recent_3yr_paper_count":
        base["recent_3yr_paper_count"] = base[recent_col]

    if growth_col is not None and growth_col != "paper_growth":
        base["paper_growth"] = base[growth_col]

    source_score = zscore_mean(base, source_features)
    if source_score is not None:
        base["source_score"] = source_score

    proxy_score = zscore_mean(base, proxy_features)
    if proxy_score is not None:
        base["proxy_score"] = proxy_score

    info = {
        "numeric_feature_count": len(numeric_features),
        "history_col": history_col,
        "recent_col": recent_col,
        "growth_col": growth_col,
        "source_feature_count": len(source_features),
        "source_features_preview": source_features[:30],
        "proxy_feature_count": len(proxy_features),
        "proxy_features_preview": proxy_features[:30],
    }

    return base, info


def run_analysis(args: argparse.Namespace) -> None:
    args.gnn_input_dir = args.gnn_input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.run_dirs = [p.resolve() for p in args.run_dirs]

    if args.run_names is not None and len(args.run_names) != len(args.run_dirs):
        raise ValueError("--run-names must have the same length as --run-dirs")

    if args.run_names is None:
        args.run_names = [p.name for p in args.run_dirs]

    prepare_output_dir(args.output_dir, args.overwrite)

    log(args, "Running Knowledge Field stratified evaluation ...")
    log(args, f"  gnn_input_dir: {args.gnn_input_dir}")
    log(args, f"  output_dir:    {args.output_dir}")
    log(args, f"  task:          {args.task}")
    log(args, f"  split:         {args.split}")
    log(args, f"  cutoff_year:   {args.cutoff_year}")
    log(args, "")

    prediction_frames: dict[str, pd.DataFrame] = {}

    for run_name, run_dir in zip(args.run_names, args.run_dirs, strict=True):
        path = prediction_path(
            run_dir,
            task=args.task,
            split=args.split,
            cutoff_year=args.cutoff_year,
        )
        log(args, f"Reading predictions: {run_name} -> {path}")
        pred = read_prediction_file(path)
        prediction_frames[run_name] = pred

    first_run_name = args.run_names[0]
    base, feature_info = build_base_frame(
        args=args,
        first_pred=prediction_frames[first_run_name],
    )

    log(args, "Feature grouping:")
    log(args, f"  history_col: {feature_info['history_col']}")
    log(args, f"  recent_col:  {feature_info['recent_col']}")
    log(args, f"  growth_col:  {feature_info['growth_col']}")
    log(args, f"  source_feature_count: {feature_info['source_feature_count']}")
    log(args, f"  proxy_feature_count:  {feature_info['proxy_feature_count']}")
    log(args, "")

    all_metrics: list[pd.DataFrame] = []
    all_compositions: list[pd.DataFrame] = []

    # Check label alignment across runs and evaluate.
    reference_key = base[["node_idx", "y_heat", "y_label"]].copy()

    for run_name, pred in prediction_frames.items():
        pred_small = pred[["node_idx", "y_heat", "y_label", "score"]].copy()

        merged = base.drop(columns=["score"], errors="ignore").merge(
            pred_small,
            on=["node_idx", "y_heat", "y_label"],
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(reference_key):
            raise ValueError(
                f"Run {run_name} did not align with the reference prediction frame. "
                f"Expected {len(reference_key)} rows, got {len(merged)}."
            )

        log(args, f"Computing stratified metrics for run: {run_name}")

        metrics = compute_stratified_metrics(
            merged,
            run_name=run_name,
            topk=args.topk,
        )
        all_metrics.append(metrics)

        comp = compute_topk_composition(
            merged,
            run_name=run_name,
            topk=args.topk,
        )
        all_compositions.append(comp)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    comp_df = pd.concat(all_compositions, ignore_index=True)

    metrics_path = args.output_dir / "stratified_metrics.csv"
    comp_path = args.output_dir / "topk_composition.csv"

    metrics_df.to_csv(metrics_path, index=False)
    comp_df.to_csv(comp_path, index=False)

    config = {
        "script": "scripts/37_run_knowledge_field_stratified_eval.py",
        "gnn_input_dir": str(args.gnn_input_dir),
        "run_dirs": [str(p) for p in args.run_dirs],
        "run_names": args.run_names,
        "task": args.task,
        "split": args.split,
        "cutoff_year": args.cutoff_year,
        "topk": args.topk,
        "feature_info": feature_info,
        "outputs": {
            "stratified_metrics": str(metrics_path),
            "topk_composition": str(comp_path),
            "stratified_report": str(args.output_dir / "stratified_report.md"),
        },
    }
    write_json(args.output_dir / "run_config.json", config)

    write_report(
        output_dir=args.output_dir,
        metrics=metrics_df,
        topk_comp=comp_df,
        args=args,
    )

    log(args, "")
    log(args, "Stratified evaluation complete.")
    log(args, f"  metrics:     {metrics_path}")
    log(args, f"  composition: {comp_path}")
    log(args, f"  report:      {args.output_dir / 'stratified_report.md'}")


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()