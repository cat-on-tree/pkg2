#!/usr/bin/env python
"""Build Stage 05F KU temporal prediction dataset.

This script constructs a formal expanding-window rolling-origin prediction
dataset from the long-window KU temporal modeling dataset.

Input:
    data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset/

Expected input files:
    knowledge_unit_static_features.parquet
    knowledge_unit_temporal_sequences.parquet
    knowledge_unit_carrier_observations.parquet  # not required for v1 features

Output:
    data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/

Main design:
    example = KU x cutoff_year

    history window:
        history_start_year .. cutoff_year

    future window:
        cutoff_year + 1 .. cutoff_year + horizon_years

Default benchmark:
    history_start_year = 2000
    first_cutoff_year  = 2005
    last_cutoff_year   = 2021
    horizon_years      = 3

    train      = 2005-2016
    validation = 2017-2018
    test       = 2019-2021

Primary targets:
    target_future_patent_heat_3yr
    target_future_trial_heat_3yr
    target_future_translation_heat_3yr

where heat is initially defined as log1p(future count).

Task eligibility:
    patent:
        history_paper_count > 0 and history_patent_count == 0

    trial:
        history_paper_count > 0 and history_trial_count == 0

    translation:
        history_paper_count > 0
        and history_patent_count == 0
        and history_trial_count == 0

This script does not train models. It creates the frozen supervised benchmark
that Stage 06 baselines and Stage 07 models should consume.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODELING_DIR = Path(
    "data/datasets/knowledge_units/"
    "diabetes_2000_2024_v1_translational_modeling_dataset"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/datasets/knowledge_units/"
    "diabetes_2000_2024_v1_temporal_prediction"
)

DEFAULT_TRAIN_CUTOFFS = list(range(2005, 2017))
DEFAULT_VALIDATION_CUTOFFS = [2017, 2018]
DEFAULT_TEST_CUTOFFS = [2019, 2020, 2021]

SAFE_STATIC_COLUMNS = [
    "ku_id",
    "pair_type",
    "entity_a_id",
    "entity_a_name",
    "entity_a_type",
    "entity_b_id",
    "entity_b_name",
    "entity_b_type",
]

FULL_WINDOW_FORBIDDEN_FEATURE_COLUMNS = [
    "carrier_count",
    "paper_count",
    "patent_count",
    "trial_count",
    "total_pair_weight",
    "first_year",
    "last_year",
    "first_observed_year",
    "first_paper_year",
    "first_patent_year",
    "first_trial_year",
    "first_translation_year",
    "has_paper_evidence",
    "has_patent_evidence",
    "has_trial_evidence",
    "has_translation_evidence",
    "has_dated_paper_evidence",
    "has_dated_patent_evidence",
    "has_dated_trial_evidence",
    "has_dated_translation_evidence",
    "evidence_pattern",
    "modality_count",
    "is_cross_modal",
]

IDENTIFIER_COLUMNS = [
    "ku_id",
    "cutoff_year",
    "split",
    "history_start_year",
    "history_end_year",
    "future_start_year",
    "future_end_year",
    "horizon_years",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "pair_type",
    "entity_a_type",
    "entity_b_type",
]

NUMERIC_FEATURE_COLUMNS = [
    "history_paper_count",
    "history_patent_count",
    "history_trial_count",
    "history_translation_count",
    "history_total_count",
    "history_paper_weight_sum",
    "history_patent_weight_sum",
    "history_trial_weight_sum",
    "history_translation_weight_sum",
    "history_total_weight_sum",
    "history_active_year_count",
    "history_paper_active_year_count",
    "history_patent_active_year_count",
    "history_trial_active_year_count",
    "history_translation_active_year_count",
    "history_first_year",
    "history_last_year",
    "history_recency",
    "recent_3yr_paper_count",
    "recent_3yr_patent_count",
    "recent_3yr_trial_count",
    "recent_3yr_translation_count",
    "recent_3yr_total_count",
    "recent_5yr_paper_count",
    "recent_5yr_patent_count",
    "recent_5yr_trial_count",
    "recent_5yr_translation_count",
    "recent_5yr_total_count",
    "prev_3yr_paper_count",
    "prev_3yr_patent_count",
    "prev_3yr_trial_count",
    "prev_3yr_translation_count",
    "prev_3yr_total_count",
    "prev_5yr_paper_count",
    "prev_5yr_patent_count",
    "prev_5yr_trial_count",
    "prev_5yr_translation_count",
    "prev_5yr_total_count",
    "growth_paper_3yr_vs_prev3yr",
    "growth_patent_3yr_vs_prev3yr",
    "growth_trial_3yr_vs_prev3yr",
    "growth_translation_3yr_vs_prev3yr",
    "growth_total_3yr_vs_prev3yr",
    "growth_paper_5yr_vs_prev5yr",
    "growth_patent_5yr_vs_prev5yr",
    "growth_trial_5yr_vs_prev5yr",
    "growth_translation_5yr_vs_prev5yr",
    "growth_total_5yr_vs_prev5yr",
    "growth_paper_3yr_ratio",
    "growth_patent_3yr_ratio",
    "growth_trial_3yr_ratio",
    "growth_translation_3yr_ratio",
    "growth_total_3yr_ratio",
    "growth_paper_5yr_ratio",
    "growth_patent_5yr_ratio",
    "growth_trial_5yr_ratio",
    "growth_translation_5yr_ratio",
    "growth_total_5yr_ratio",
]

ELIGIBILITY_COLUMNS = [
    "has_paper_history",
    "has_prior_patent",
    "has_prior_trial",
    "has_prior_translation",
    "eligible_patent_task",
    "eligible_trial_task",
    "eligible_translation_task",
]

TARGET_COLUMNS = [
    "target_future_patent_count_3yr",
    "target_future_trial_count_3yr",
    "target_future_translation_count_3yr",
    "target_future_total_count_3yr",
    "target_future_patent_weight_sum_3yr",
    "target_future_trial_weight_sum_3yr",
    "target_future_translation_weight_sum_3yr",
    "target_future_total_weight_sum_3yr",
    "target_future_patent_active_years_3yr",
    "target_future_trial_active_years_3yr",
    "target_future_translation_active_years_3yr",
    "target_future_total_active_years_3yr",
    "target_future_patent_heat_3yr",
    "target_future_trial_heat_3yr",
    "target_future_translation_heat_3yr",
    "target_future_total_heat_3yr",
]

LABEL_COLUMNS = [
    "label_future_patent_emergence_3yr",
    "label_future_trial_emergence_3yr",
    "label_future_translation_emergence_3yr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage 05F KU temporal prediction dataset."
    )

    parser.add_argument(
        "--modeling-dir",
        type=Path,
        default=DEFAULT_MODELING_DIR,
        help="Input KU temporal modeling dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output Stage 05F temporal prediction dataset directory.",
    )
    parser.add_argument(
        "--history-start-year",
        type=int,
        default=2000,
        help="Inclusive first year in history windows. Default: 2000.",
    )
    parser.add_argument(
        "--first-cutoff-year",
        type=int,
        default=2005,
        help="First cutoff year to include. Default: 2005.",
    )
    parser.add_argument(
        "--last-cutoff-year",
        type=int,
        default=2021,
        help="Last cutoff year to include. Default: 2021.",
    )
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=3,
        help="Future forecasting horizon in years. Default: 3.",
    )
    parser.add_argument(
        "--train-cutoffs",
        nargs="*",
        type=int,
        default=DEFAULT_TRAIN_CUTOFFS,
        help="Cutoff years assigned to train split.",
    )
    parser.add_argument(
        "--validation-cutoffs",
        nargs="*",
        type=int,
        default=DEFAULT_VALIDATION_CUTOFFS,
        help="Cutoff years assigned to validation split.",
    )
    parser.add_argument(
        "--test-cutoffs",
        nargs="*",
        type=int,
        default=DEFAULT_TEST_CUTOFFS,
        help="Cutoff years assigned to test split.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="24GB")
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/interim/duckdb_tmp"),
        help="DuckDB temp directory.",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def log(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, flush=True)


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def qident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def scan_sql(path: Path) -> str:
    return f"read_parquet({sql_literal(path)})"


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd

        if value is pd.NA:
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)

    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def connect_duckdb(args: argparse.Namespace):
    import duckdb

    args.temp_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET threads={int(args.threads)}")
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute(f"SET temp_directory={sql_literal(args.temp_dir)}")
    return connection


def execute(connection: Any, sql: str) -> None:
    connection.execute(sql)


def fetchone_int(connection: Any, sql: str) -> int:
    value = connection.execute(sql).fetchone()[0]
    return int(value or 0)


def fetchone(connection: Any, sql: str) -> tuple[Any, ...]:
    return connection.execute(sql).fetchone()


def fetch_dicts(connection: Any, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def get_columns(connection: Any, table_or_view: str) -> set[str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM {qident(table_or_view)}"
    ).fetchall()
    return {str(row[0]) for row in rows}


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def require_input_files(modeling_dir: Path) -> dict[str, Path]:
    paths = {
        "static": modeling_dir / "knowledge_unit_static_features.parquet",
        "temporal": modeling_dir / "knowledge_unit_temporal_sequences.parquet",
        "observations": modeling_dir / "knowledge_unit_carrier_observations.parquet",
    }

    required = ["static", "temporal"]
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required modeling dataset files:\n" + "\n".join(missing)
        )

    return paths


def validate_args(args: argparse.Namespace) -> None:
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    if args.horizon_years < 1:
        raise ValueError("--horizon-years must be >= 1")

    if args.first_cutoff_year > args.last_cutoff_year:
        raise ValueError("--first-cutoff-year must be <= --last-cutoff-year")

    all_cutoffs = (
        list(args.train_cutoffs)
        + list(args.validation_cutoffs)
        + list(args.test_cutoffs)
    )

    if len(all_cutoffs) != len(set(all_cutoffs)):
        raise ValueError("Train/validation/test cutoffs must not overlap.")

    expected = set(range(args.first_cutoff_year, args.last_cutoff_year + 1))
    provided = set(all_cutoffs)

    if provided != expected:
        raise ValueError(
            "Provided split cutoffs must exactly cover "
            f"{args.first_cutoff_year}-{args.last_cutoff_year}. "
            f"Missing={sorted(expected - provided)}, extra={sorted(provided - expected)}"
        )

    if max(args.train_cutoffs) >= min(args.validation_cutoffs):
        raise ValueError("Train cutoffs must be earlier than validation cutoffs.")

    if max(args.validation_cutoffs) >= min(args.test_cutoffs):
        raise ValueError("Validation cutoffs must be earlier than test cutoffs.")


def run_step(args: argparse.Namespace, index: int, total: int, label: str, fn) -> Any:
    log(args, f"[{index}/{total}] {label} ...")
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    log(args, f"[{index}/{total}] {label} done in {elapsed:.1f}s")
    return result


def numeric_col_expr(columns: set[str], column: str, default: str = "0.0") -> str:
    if column in columns:
        return f"COALESCE(TRY_CAST({qident(column)} AS DOUBLE), {default})"
    return default


def varchar_col_expr(columns: set[str], column: str) -> str:
    if column in columns:
        return f"CAST({qident(column)} AS VARCHAR)"
    return "CAST(NULL AS VARCHAR)"


def create_input_views(connection: Any, paths: dict[str, Path]) -> None:
    execute(
        connection,
        f"""
        CREATE OR REPLACE TEMP VIEW input_static AS
        SELECT * FROM {scan_sql(paths["static"])}
        """,
    )

    execute(
        connection,
        f"""
        CREATE OR REPLACE TEMP VIEW input_temporal AS
        SELECT * FROM {scan_sql(paths["temporal"])}
        """,
    )


def create_normalized_views(connection: Any) -> None:
    static_cols = get_columns(connection, "input_static")
    temporal_cols = get_columns(connection, "input_temporal")

    missing_static_required = {"ku_id"} - static_cols
    if missing_static_required:
        raise ValueError(
            f"input_static missing required columns: {sorted(missing_static_required)}"
        )

    missing_temporal_required = {"ku_id", "year"} - temporal_cols
    if missing_temporal_required:
        raise ValueError(
            "input_temporal missing required columns: "
            f"{sorted(missing_temporal_required)}"
        )

    static_select_items = []
    for column in SAFE_STATIC_COLUMNS:
        if column == "ku_id":
            static_select_items.append(
                "NULLIF(TRIM(CAST(ku_id AS VARCHAR)), '') AS ku_id"
            )
        else:
            static_select_items.append(
                f"{varchar_col_expr(static_cols, column)} AS {qident(column)}"
            )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE static_safe AS
        SELECT
            {", ".join(static_select_items)}
        FROM input_static
        WHERE NULLIF(TRIM(CAST(ku_id AS VARCHAR)), '') IS NOT NULL
        """
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE temporal_normalized AS
        SELECT
            NULLIF(TRIM(CAST(ku_id AS VARCHAR)), '') AS ku_id,
            TRY_CAST(year AS BIGINT) AS year,

            {numeric_col_expr(temporal_cols, "paper_count_year")} AS paper_count_year,
            {numeric_col_expr(temporal_cols, "patent_count_year")} AS patent_count_year,
            {numeric_col_expr(temporal_cols, "trial_count_year")} AS trial_count_year,
            {numeric_col_expr(temporal_cols, "carrier_count_year")} AS carrier_count_year,

            {numeric_col_expr(temporal_cols, "paper_pair_weight_year")} AS paper_pair_weight_year,
            {numeric_col_expr(temporal_cols, "patent_pair_weight_year")} AS patent_pair_weight_year,
            {numeric_col_expr(temporal_cols, "trial_pair_weight_year")} AS trial_pair_weight_year,
            {numeric_col_expr(temporal_cols, "pair_weight_year")} AS pair_weight_year
        FROM input_temporal
        WHERE NULLIF(TRIM(CAST(ku_id AS VARCHAR)), '') IS NOT NULL
          AND TRY_CAST(year AS BIGINT) IS NOT NULL
        """
    )


def create_cutoff_table(connection: Any, args: argparse.Namespace) -> None:
    rows: list[tuple[int, str, int, int, int, int]] = []

    def add(cutoffs: list[int], split: str) -> None:
        for cutoff in cutoffs:
            rows.append(
                (
                    int(cutoff),
                    split,
                    int(args.history_start_year),
                    int(cutoff),
                    int(cutoff + 1),
                    int(cutoff + args.horizon_years),
                )
            )

    add(list(args.train_cutoffs), "train")
    add(list(args.validation_cutoffs), "validation")
    add(list(args.test_cutoffs), "test")

    values_sql = ",\n".join(
        f"({cutoff}, {sql_literal(split)}, {history_start}, {history_end}, {future_start}, {future_end})"
        for cutoff, split, history_start, history_end, future_start, future_end in rows
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE cutoffs AS
        SELECT *
        FROM (
            VALUES
            {values_sql}
        ) AS t(
            cutoff_year,
            split,
            history_start_year,
            history_end_year,
            future_start_year,
            future_end_year
        )
        """
    )


def validate_temporal_year_range(connection: Any, args: argparse.Namespace) -> dict[str, Any]:
    row = fetchone(
        connection,
        """
        SELECT
            MIN(year) AS year_min,
            MAX(year) AS year_max,
            COUNT(*) AS row_count,
            COUNT(DISTINCT ku_id) AS ku_count
        FROM temporal_normalized
        """
    )

    year_min = int(row[0])
    year_max = int(row[1])
    temporal_row_count = int(row[2])
    temporal_ku_count = int(row[3])

    max_future_end = args.last_cutoff_year + args.horizon_years

    if args.history_start_year < year_min:
        raise ValueError(
            f"--history-start-year {args.history_start_year} is earlier than "
            f"temporal year_min {year_min}"
        )

    if max_future_end > year_max:
        raise ValueError(
            f"Latest future_end_year {max_future_end} exceeds temporal year_max {year_max}"
        )

    return {
        "temporal_year_min": year_min,
        "temporal_year_max": year_max,
        "temporal_row_count": temporal_row_count,
        "temporal_ku_count": temporal_ku_count,
        "max_future_end_year": max_future_end,
    }


def create_prediction_examples(connection: Any, args: argparse.Namespace) -> None:
    h = int(args.horizon_years)

    execute(
        connection,
        f"""
        CREATE TEMP TABLE prediction_examples_raw AS
        SELECT
            s.ku_id,
            c.cutoff_year,
            c.split,
            c.history_start_year,
            c.history_end_year,
            c.future_start_year,
            c.future_end_year,
            {h} AS horizon_years,

            s.pair_type,
            s.entity_a_id,
            s.entity_a_name,
            s.entity_a_type,
            s.entity_b_id,
            s.entity_b_name,
            s.entity_b_type,

            -- History cumulative counts.
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.paper_count_year ELSE 0 END) AS history_paper_count,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.patent_count_year ELSE 0 END) AS history_patent_count,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.trial_count_year ELSE 0 END) AS history_trial_count,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.patent_count_year + y.trial_count_year ELSE 0 END) AS history_translation_count,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.carrier_count_year ELSE 0 END) AS history_total_count,

            -- History cumulative weights.
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.paper_pair_weight_year ELSE 0 END) AS history_paper_weight_sum,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.patent_pair_weight_year ELSE 0 END) AS history_patent_weight_sum,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.trial_pair_weight_year ELSE 0 END) AS history_trial_weight_sum,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.patent_pair_weight_year + y.trial_pair_weight_year ELSE 0 END) AS history_translation_weight_sum,
            SUM(CASE WHEN y.year <= c.cutoff_year THEN y.pair_weight_year ELSE 0 END) AS history_total_weight_sum,

            -- History active years.
            SUM(CASE WHEN y.year <= c.cutoff_year AND y.carrier_count_year > 0 THEN 1 ELSE 0 END) AS history_active_year_count,
            SUM(CASE WHEN y.year <= c.cutoff_year AND y.paper_count_year > 0 THEN 1 ELSE 0 END) AS history_paper_active_year_count,
            SUM(CASE WHEN y.year <= c.cutoff_year AND y.patent_count_year > 0 THEN 1 ELSE 0 END) AS history_patent_active_year_count,
            SUM(CASE WHEN y.year <= c.cutoff_year AND y.trial_count_year > 0 THEN 1 ELSE 0 END) AS history_trial_active_year_count,
            SUM(CASE WHEN y.year <= c.cutoff_year AND y.patent_count_year + y.trial_count_year > 0 THEN 1 ELSE 0 END) AS history_translation_active_year_count,

            MIN(CASE WHEN y.year <= c.cutoff_year AND y.carrier_count_year > 0 THEN y.year ELSE NULL END) AS history_first_year,
            MAX(CASE WHEN y.year <= c.cutoff_year AND y.carrier_count_year > 0 THEN y.year ELSE NULL END) AS history_last_year,

            -- Recent 3-year window: cutoff-2 .. cutoff.
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 2 AND c.cutoff_year THEN y.paper_count_year ELSE 0 END) AS recent_3yr_paper_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 2 AND c.cutoff_year THEN y.patent_count_year ELSE 0 END) AS recent_3yr_patent_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 2 AND c.cutoff_year THEN y.trial_count_year ELSE 0 END) AS recent_3yr_trial_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 2 AND c.cutoff_year THEN y.patent_count_year + y.trial_count_year ELSE 0 END) AS recent_3yr_translation_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 2 AND c.cutoff_year THEN y.carrier_count_year ELSE 0 END) AS recent_3yr_total_count,

            -- Previous 3-year window: cutoff-5 .. cutoff-3.
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 5 AND c.cutoff_year - 3 THEN y.paper_count_year ELSE 0 END) AS prev_3yr_paper_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 5 AND c.cutoff_year - 3 THEN y.patent_count_year ELSE 0 END) AS prev_3yr_patent_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 5 AND c.cutoff_year - 3 THEN y.trial_count_year ELSE 0 END) AS prev_3yr_trial_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 5 AND c.cutoff_year - 3 THEN y.patent_count_year + y.trial_count_year ELSE 0 END) AS prev_3yr_translation_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 5 AND c.cutoff_year - 3 THEN y.carrier_count_year ELSE 0 END) AS prev_3yr_total_count,

            -- Recent 5-year window: cutoff-4 .. cutoff.
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 4 AND c.cutoff_year THEN y.paper_count_year ELSE 0 END) AS recent_5yr_paper_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 4 AND c.cutoff_year THEN y.patent_count_year ELSE 0 END) AS recent_5yr_patent_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 4 AND c.cutoff_year THEN y.trial_count_year ELSE 0 END) AS recent_5yr_trial_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 4 AND c.cutoff_year THEN y.patent_count_year + y.trial_count_year ELSE 0 END) AS recent_5yr_translation_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 4 AND c.cutoff_year THEN y.carrier_count_year ELSE 0 END) AS recent_5yr_total_count,

            -- Previous 5-year window: cutoff-9 .. cutoff-5.
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 9 AND c.cutoff_year - 5 THEN y.paper_count_year ELSE 0 END) AS prev_5yr_paper_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 9 AND c.cutoff_year - 5 THEN y.patent_count_year ELSE 0 END) AS prev_5yr_patent_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 9 AND c.cutoff_year - 5 THEN y.trial_count_year ELSE 0 END) AS prev_5yr_trial_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 9 AND c.cutoff_year - 5 THEN y.patent_count_year + y.trial_count_year ELSE 0 END) AS prev_5yr_translation_count,
            SUM(CASE WHEN y.year BETWEEN c.cutoff_year - 9 AND c.cutoff_year - 5 THEN y.carrier_count_year ELSE 0 END) AS prev_5yr_total_count,

            -- Future targets.
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.paper_count_year ELSE 0 END) AS target_future_paper_count_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.patent_count_year ELSE 0 END) AS target_future_patent_count_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.trial_count_year ELSE 0 END) AS target_future_trial_count_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.patent_count_year + y.trial_count_year ELSE 0 END) AS target_future_translation_count_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.carrier_count_year ELSE 0 END) AS target_future_total_count_3yr,

            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.patent_pair_weight_year ELSE 0 END) AS target_future_patent_weight_sum_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.trial_pair_weight_year ELSE 0 END) AS target_future_trial_weight_sum_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.patent_pair_weight_year + y.trial_pair_weight_year ELSE 0 END) AS target_future_translation_weight_sum_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year THEN y.pair_weight_year ELSE 0 END) AS target_future_total_weight_sum_3yr,

            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year AND y.patent_count_year > 0 THEN 1 ELSE 0 END) AS target_future_patent_active_years_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year AND y.trial_count_year > 0 THEN 1 ELSE 0 END) AS target_future_trial_active_years_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year AND y.patent_count_year + y.trial_count_year > 0 THEN 1 ELSE 0 END) AS target_future_translation_active_years_3yr,
            SUM(CASE WHEN y.year > c.cutoff_year AND y.year <= c.future_end_year AND y.carrier_count_year > 0 THEN 1 ELSE 0 END) AS target_future_total_active_years_3yr

        FROM static_safe s
        CROSS JOIN cutoffs c
        LEFT JOIN temporal_normalized y
            ON s.ku_id = y.ku_id
           AND y.year >= c.history_start_year
           AND y.year <= c.future_end_year
        GROUP BY
            s.ku_id,
            c.cutoff_year,
            c.split,
            c.history_start_year,
            c.history_end_year,
            c.future_start_year,
            c.future_end_year,
            s.pair_type,
            s.entity_a_id,
            s.entity_a_name,
            s.entity_a_type,
            s.entity_b_id,
            s.entity_b_name,
            s.entity_b_type
        """
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_examples AS
        SELECT
            *,
            CASE
                WHEN history_last_year IS NULL THEN NULL
                ELSE cutoff_year - history_last_year
            END AS history_recency,

            recent_3yr_paper_count - prev_3yr_paper_count AS growth_paper_3yr_vs_prev3yr,
            recent_3yr_patent_count - prev_3yr_patent_count AS growth_patent_3yr_vs_prev3yr,
            recent_3yr_trial_count - prev_3yr_trial_count AS growth_trial_3yr_vs_prev3yr,
            recent_3yr_translation_count - prev_3yr_translation_count AS growth_translation_3yr_vs_prev3yr,
            recent_3yr_total_count - prev_3yr_total_count AS growth_total_3yr_vs_prev3yr,

            recent_5yr_paper_count - prev_5yr_paper_count AS growth_paper_5yr_vs_prev5yr,
            recent_5yr_patent_count - prev_5yr_patent_count AS growth_patent_5yr_vs_prev5yr,
            recent_5yr_trial_count - prev_5yr_trial_count AS growth_trial_5yr_vs_prev5yr,
            recent_5yr_translation_count - prev_5yr_translation_count AS growth_translation_5yr_vs_prev5yr,
            recent_5yr_total_count - prev_5yr_total_count AS growth_total_5yr_vs_prev5yr,

            (recent_3yr_paper_count + 1.0) / (prev_3yr_paper_count + 1.0) AS growth_paper_3yr_ratio,
            (recent_3yr_patent_count + 1.0) / (prev_3yr_patent_count + 1.0) AS growth_patent_3yr_ratio,
            (recent_3yr_trial_count + 1.0) / (prev_3yr_trial_count + 1.0) AS growth_trial_3yr_ratio,
            (recent_3yr_translation_count + 1.0) / (prev_3yr_translation_count + 1.0) AS growth_translation_3yr_ratio,
            (recent_3yr_total_count + 1.0) / (prev_3yr_total_count + 1.0) AS growth_total_3yr_ratio,

            (recent_5yr_paper_count + 1.0) / (prev_5yr_paper_count + 1.0) AS growth_paper_5yr_ratio,
            (recent_5yr_patent_count + 1.0) / (prev_5yr_patent_count + 1.0) AS growth_patent_5yr_ratio,
            (recent_5yr_trial_count + 1.0) / (prev_5yr_trial_count + 1.0) AS growth_trial_5yr_ratio,
            (recent_5yr_translation_count + 1.0) / (prev_5yr_translation_count + 1.0) AS growth_translation_5yr_ratio,
            (recent_5yr_total_count + 1.0) / (prev_5yr_total_count + 1.0) AS growth_total_5yr_ratio,

            history_paper_count > 0 AS has_paper_history,
            history_patent_count > 0 AS has_prior_patent,
            history_trial_count > 0 AS has_prior_trial,
            history_patent_count + history_trial_count > 0 AS has_prior_translation,

            history_paper_count > 0 AND history_patent_count = 0 AS eligible_patent_task,
            history_paper_count > 0 AND history_trial_count = 0 AS eligible_trial_task,
            history_paper_count > 0 AND history_patent_count = 0 AND history_trial_count = 0 AS eligible_translation_task,

            LOG(1.0 + target_future_patent_count_3yr) AS target_future_patent_heat_3yr,
            LOG(1.0 + target_future_trial_count_3yr) AS target_future_trial_heat_3yr,
            LOG(1.0 + target_future_translation_count_3yr) AS target_future_translation_heat_3yr,
            LOG(1.0 + target_future_total_count_3yr) AS target_future_total_heat_3yr,

            target_future_patent_count_3yr > 0 AS label_future_patent_emergence_3yr,
            target_future_trial_count_3yr > 0 AS label_future_trial_emergence_3yr,
            target_future_translation_count_3yr > 0 AS label_future_translation_emergence_3yr

        FROM prediction_examples_raw
        """
    )


def create_prediction_feature_table(connection: Any) -> None:
    columns = (
        IDENTIFIER_COLUMNS
        + [
            "pair_type",
            "entity_a_id",
            "entity_a_name",
            "entity_a_type",
            "entity_b_id",
            "entity_b_name",
            "entity_b_type",
        ]
        + NUMERIC_FEATURE_COLUMNS
        + ELIGIBILITY_COLUMNS
        + TARGET_COLUMNS
        + LABEL_COLUMNS
    )

    select_items = ",\n            ".join(qident(column) for column in columns)

    execute(
        connection,
        f"""
        CREATE TEMP TABLE prediction_feature_table AS
        SELECT
            {select_items}
        FROM prediction_examples
        """
    )


def create_task_views(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_examples_patent AS
        SELECT *
        FROM prediction_feature_table
        WHERE eligible_patent_task
        """
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_examples_trial AS
        SELECT *
        FROM prediction_feature_table
        WHERE eligible_trial_task
        """
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_examples_translation AS
        SELECT *
        FROM prediction_feature_table
        WHERE eligible_translation_task
        """
    )


def create_summary_tables(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_label_summary AS
        SELECT *
        FROM (
            SELECT
                'patent' AS task,
                split,
                cutoff_year,
                COUNT(*) FILTER (WHERE eligible_patent_task) AS eligible_count,
                SUM(CASE WHEN eligible_patent_task AND label_future_patent_emergence_3yr THEN 1 ELSE 0 END) AS positive_count,
                AVG(CASE WHEN eligible_patent_task THEN CAST(label_future_patent_emergence_3yr AS DOUBLE) ELSE NULL END) AS positive_rate,
                AVG(CASE WHEN eligible_patent_task THEN target_future_patent_heat_3yr ELSE NULL END) AS target_heat_mean,
                MAX(CASE WHEN eligible_patent_task THEN target_future_patent_heat_3yr ELSE NULL END) AS target_heat_max,
                SUM(CASE WHEN eligible_patent_task THEN target_future_patent_count_3yr ELSE 0 END) AS target_count_sum
            FROM prediction_feature_table
            GROUP BY split, cutoff_year

            UNION ALL

            SELECT
                'trial' AS task,
                split,
                cutoff_year,
                COUNT(*) FILTER (WHERE eligible_trial_task) AS eligible_count,
                SUM(CASE WHEN eligible_trial_task AND label_future_trial_emergence_3yr THEN 1 ELSE 0 END) AS positive_count,
                AVG(CASE WHEN eligible_trial_task THEN CAST(label_future_trial_emergence_3yr AS DOUBLE) ELSE NULL END) AS positive_rate,
                AVG(CASE WHEN eligible_trial_task THEN target_future_trial_heat_3yr ELSE NULL END) AS target_heat_mean,
                MAX(CASE WHEN eligible_trial_task THEN target_future_trial_heat_3yr ELSE NULL END) AS target_heat_max,
                SUM(CASE WHEN eligible_trial_task THEN target_future_trial_count_3yr ELSE 0 END) AS target_count_sum
            FROM prediction_feature_table
            GROUP BY split, cutoff_year

            UNION ALL

            SELECT
                'translation' AS task,
                split,
                cutoff_year,
                COUNT(*) FILTER (WHERE eligible_translation_task) AS eligible_count,
                SUM(CASE WHEN eligible_translation_task AND label_future_translation_emergence_3yr THEN 1 ELSE 0 END) AS positive_count,
                AVG(CASE WHEN eligible_translation_task THEN CAST(label_future_translation_emergence_3yr AS DOUBLE) ELSE NULL END) AS positive_rate,
                AVG(CASE WHEN eligible_translation_task THEN target_future_translation_heat_3yr ELSE NULL END) AS target_heat_mean,
                MAX(CASE WHEN eligible_translation_task THEN target_future_translation_heat_3yr ELSE NULL END) AS target_heat_max,
                SUM(CASE WHEN eligible_translation_task THEN target_future_translation_count_3yr ELSE 0 END) AS target_count_sum
            FROM prediction_feature_table
            GROUP BY split, cutoff_year
        )
        ORDER BY task, cutoff_year
        """
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_split_summary AS
        SELECT
            split,
            MIN(cutoff_year) AS cutoff_min,
            MAX(cutoff_year) AS cutoff_max,
            COUNT(*) AS example_count,
            SUM(CASE WHEN eligible_patent_task THEN 1 ELSE 0 END) AS eligible_patent_count,
            SUM(CASE WHEN eligible_trial_task THEN 1 ELSE 0 END) AS eligible_trial_count,
            SUM(CASE WHEN eligible_translation_task THEN 1 ELSE 0 END) AS eligible_translation_count,
            SUM(CASE WHEN eligible_patent_task AND label_future_patent_emergence_3yr THEN 1 ELSE 0 END) AS patent_positive_count,
            SUM(CASE WHEN eligible_trial_task AND label_future_trial_emergence_3yr THEN 1 ELSE 0 END) AS trial_positive_count,
            SUM(CASE WHEN eligible_translation_task AND label_future_translation_emergence_3yr THEN 1 ELSE 0 END) AS translation_positive_count
        FROM prediction_feature_table
        GROUP BY split
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
                ELSE 99
            END
        """
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_task_summary AS
        SELECT *
        FROM (
            SELECT
                'patent' AS task,
                COUNT(*) FILTER (WHERE eligible_patent_task) AS eligible_count,
                SUM(CASE WHEN eligible_patent_task AND label_future_patent_emergence_3yr THEN 1 ELSE 0 END) AS positive_count,
                AVG(CASE WHEN eligible_patent_task THEN CAST(label_future_patent_emergence_3yr AS DOUBLE) ELSE NULL END) AS positive_rate,
                AVG(CASE WHEN eligible_patent_task THEN target_future_patent_heat_3yr ELSE NULL END) AS target_heat_mean,
                MEDIAN(CASE WHEN eligible_patent_task THEN target_future_patent_heat_3yr ELSE NULL END) AS target_heat_median,
                MAX(CASE WHEN eligible_patent_task THEN target_future_patent_heat_3yr ELSE NULL END) AS target_heat_max,
                SUM(CASE WHEN eligible_patent_task AND split = 'train' THEN 1 ELSE 0 END) AS train_count,
                SUM(CASE WHEN eligible_patent_task AND split = 'validation' THEN 1 ELSE 0 END) AS validation_count,
                SUM(CASE WHEN eligible_patent_task AND split = 'test' THEN 1 ELSE 0 END) AS test_count
            FROM prediction_feature_table

            UNION ALL

            SELECT
                'trial' AS task,
                COUNT(*) FILTER (WHERE eligible_trial_task) AS eligible_count,
                SUM(CASE WHEN eligible_trial_task AND label_future_trial_emergence_3yr THEN 1 ELSE 0 END) AS positive_count,
                AVG(CASE WHEN eligible_trial_task THEN CAST(label_future_trial_emergence_3yr AS DOUBLE) ELSE NULL END) AS positive_rate,
                AVG(CASE WHEN eligible_trial_task THEN target_future_trial_heat_3yr ELSE NULL END) AS target_heat_mean,
                MEDIAN(CASE WHEN eligible_trial_task THEN target_future_trial_heat_3yr ELSE NULL END) AS target_heat_median,
                MAX(CASE WHEN eligible_trial_task THEN target_future_trial_heat_3yr ELSE NULL END) AS target_heat_max,
                SUM(CASE WHEN eligible_trial_task AND split = 'train' THEN 1 ELSE 0 END) AS train_count,
                SUM(CASE WHEN eligible_trial_task AND split = 'validation' THEN 1 ELSE 0 END) AS validation_count,
                SUM(CASE WHEN eligible_trial_task AND split = 'test' THEN 1 ELSE 0 END) AS test_count
            FROM prediction_feature_table

            UNION ALL

            SELECT
                'translation' AS task,
                COUNT(*) FILTER (WHERE eligible_translation_task) AS eligible_count,
                SUM(CASE WHEN eligible_translation_task AND label_future_translation_emergence_3yr THEN 1 ELSE 0 END) AS positive_count,
                AVG(CASE WHEN eligible_translation_task THEN CAST(label_future_translation_emergence_3yr AS DOUBLE) ELSE NULL END) AS positive_rate,
                AVG(CASE WHEN eligible_translation_task THEN target_future_translation_heat_3yr ELSE NULL END) AS target_heat_mean,
                MEDIAN(CASE WHEN eligible_translation_task THEN target_future_translation_heat_3yr ELSE NULL END) AS target_heat_median,
                MAX(CASE WHEN eligible_translation_task THEN target_future_translation_heat_3yr ELSE NULL END) AS target_heat_max,
                SUM(CASE WHEN eligible_translation_task AND split = 'train' THEN 1 ELSE 0 END) AS train_count,
                SUM(CASE WHEN eligible_translation_task AND split = 'validation' THEN 1 ELSE 0 END) AS validation_count,
                SUM(CASE WHEN eligible_translation_task AND split = 'test' THEN 1 ELSE 0 END) AS test_count
            FROM prediction_feature_table
        )
        ORDER BY task
        """
    )

    execute(
        connection,
        """
        CREATE TEMP TABLE prediction_cutoff_summary AS
        SELECT
            cutoff_year,
            split,
            COUNT(*) AS example_count,
            SUM(CASE WHEN eligible_patent_task THEN 1 ELSE 0 END) AS eligible_patent_count,
            SUM(CASE WHEN eligible_trial_task THEN 1 ELSE 0 END) AS eligible_trial_count,
            SUM(CASE WHEN eligible_translation_task THEN 1 ELSE 0 END) AS eligible_translation_count,
            SUM(CASE WHEN eligible_patent_task AND label_future_patent_emergence_3yr THEN 1 ELSE 0 END) AS future_patent_positive_count,
            SUM(CASE WHEN eligible_trial_task AND label_future_trial_emergence_3yr THEN 1 ELSE 0 END) AS future_trial_positive_count,
            SUM(CASE WHEN eligible_translation_task AND label_future_translation_emergence_3yr THEN 1 ELSE 0 END) AS future_translation_positive_count,
            AVG(CASE WHEN eligible_patent_task THEN target_future_patent_heat_3yr ELSE NULL END) AS patent_heat_mean,
            AVG(CASE WHEN eligible_trial_task THEN target_future_trial_heat_3yr ELSE NULL END) AS trial_heat_mean,
            AVG(CASE WHEN eligible_translation_task THEN target_future_translation_heat_3yr ELSE NULL END) AS translation_heat_mean
        FROM prediction_feature_table
        GROUP BY cutoff_year, split
        ORDER BY cutoff_year
        """
    )


def run_leakage_audit(connection: Any, args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    max_future_end = fetchone_int(
        connection,
        "SELECT MAX(future_end_year) FROM prediction_feature_table",
    )

    add(
        "future_end_within_temporal_range",
        max_future_end <= args.last_cutoff_year + args.horizon_years,
        f"max_future_end_year={max_future_end}",
    )

    split_values = [
        row["split"]
        for row in fetch_dicts(
            connection,
            "SELECT DISTINCT split FROM prediction_feature_table ORDER BY split",
        )
    ]
    add(
        "split_values_valid",
        set(split_values) == {"train", "validation", "test"},
        f"split_values={split_values}",
    )

    bad_translation_count = fetchone_int(
        connection,
        """
        SELECT COUNT(*)
        FROM prediction_feature_table
        WHERE target_future_translation_count_3yr
              <> target_future_patent_count_3yr + target_future_trial_count_3yr
        """,
    )
    add(
        "translation_count_equals_patent_plus_trial",
        bad_translation_count == 0,
        f"bad_rows={bad_translation_count}",
    )

    bad_translation_label = fetchone_int(
        connection,
        """
        SELECT COUNT(*)
        FROM prediction_feature_table
        WHERE label_future_translation_emergence_3yr
              <> (label_future_patent_emergence_3yr OR label_future_trial_emergence_3yr)
        """,
    )
    add(
        "translation_label_equals_patent_or_trial",
        bad_translation_label == 0,
        f"bad_rows={bad_translation_label}",
    )

    bad_translation_eligibility = fetchone_int(
        connection,
        """
        SELECT COUNT(*)
        FROM prediction_feature_table
        WHERE eligible_translation_task
          AND NOT (eligible_patent_task AND eligible_trial_task)
        """,
    )
    add(
        "translation_eligibility_implies_patent_and_trial_eligibility",
        bad_translation_eligibility == 0,
        f"bad_rows={bad_translation_eligibility}",
    )

    task_view_bad = {
        "patent": fetchone_int(
            connection,
            """
            SELECT COUNT(*)
            FROM prediction_examples_patent
            WHERE NOT eligible_patent_task
            """,
        ),
        "trial": fetchone_int(
            connection,
            """
            SELECT COUNT(*)
            FROM prediction_examples_trial
            WHERE NOT eligible_trial_task
            """,
        ),
        "translation": fetchone_int(
            connection,
            """
            SELECT COUNT(*)
            FROM prediction_examples_translation
            WHERE NOT eligible_translation_task
            """,
        ),
    }
    add(
        "task_views_only_contain_eligible_rows",
        all(value == 0 for value in task_view_bad.values()),
        f"bad_rows={task_view_bad}",
    )

    feature_columns = CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS
    forbidden_in_features = sorted(
        set(feature_columns)
        & (
            set(TARGET_COLUMNS)
            | set(LABEL_COLUMNS)
            | set(FULL_WINDOW_FORBIDDEN_FEATURE_COLUMNS)
        )
    )
    add(
        "feature_columns_exclude_targets_labels_full_window_summaries",
        len(forbidden_in_features) == 0,
        f"forbidden_columns={forbidden_in_features}",
    )

    train_max = max(args.train_cutoffs)
    validation_min = min(args.validation_cutoffs)
    validation_max = max(args.validation_cutoffs)
    test_min = min(args.test_cutoffs)
    add(
        "temporal_split_order",
        train_max < validation_min and validation_max < test_min,
        (
            f"train_max={train_max}, validation_min={validation_min}, "
            f"validation_max={validation_max}, test_min={test_min}"
        ),
    )

    overall_status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"

    return {
        "overall_status": overall_status,
        "checks": checks,
    }


def write_duckdb_table_parquet(
    connection: Any,
    table_name: str,
    path: Path,
    compression: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    execute(
        connection,
        f"""
        COPY (
            SELECT * FROM {qident(table_name)}
        )
        TO {sql_literal(path)}
        (FORMAT PARQUET, COMPRESSION {compression.upper()})
        """
    )


def write_duckdb_table_csv(connection: Any, table_name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    execute(
        connection,
        f"""
        COPY (
            SELECT * FROM {qident(table_name)}
        )
        TO {sql_literal(path)}
        (HEADER, DELIMITER ',')
        """
    )


def build_dataset_summary(
    connection: Any,
    args: argparse.Namespace,
    temporal_range: dict[str, Any],
    leakage_audit: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    counts = {
        "prediction_example_count": fetchone_int(
            connection, "SELECT COUNT(*) FROM prediction_examples"
        ),
        "prediction_feature_row_count": fetchone_int(
            connection, "SELECT COUNT(*) FROM prediction_feature_table"
        ),
        "patent_task_row_count": fetchone_int(
            connection, "SELECT COUNT(*) FROM prediction_examples_patent"
        ),
        "trial_task_row_count": fetchone_int(
            connection, "SELECT COUNT(*) FROM prediction_examples_trial"
        ),
        "translation_task_row_count": fetchone_int(
            connection, "SELECT COUNT(*) FROM prediction_examples_translation"
        ),
        "ku_count": fetchone_int(
            connection, "SELECT COUNT(DISTINCT ku_id) FROM prediction_feature_table"
        ),
        "cutoff_count": fetchone_int(
            connection, "SELECT COUNT(DISTINCT cutoff_year) FROM prediction_feature_table"
        ),
    }

    split_summary = fetch_dicts(
        connection,
        """
        SELECT *
        FROM prediction_split_summary
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
                ELSE 99
            END
        """,
    )

    task_summary = fetch_dicts(
        connection,
        "SELECT * FROM prediction_task_summary ORDER BY task",
    )

    return {
        "script": "scripts/29_build_knowledge_unit_prediction_dataset.py",
        "command": " ".join(sys.argv),
        "modeling_dir": str(args.modeling_dir),
        "output_dir": str(args.output_dir),
        "history_start_year": args.history_start_year,
        "first_cutoff_year": args.first_cutoff_year,
        "last_cutoff_year": args.last_cutoff_year,
        "horizon_years": args.horizon_years,
        "train_cutoffs": list(args.train_cutoffs),
        "validation_cutoffs": list(args.validation_cutoffs),
        "test_cutoffs": list(args.test_cutoffs),
        "target_definition": {
            "patent_heat": "log1p(sum patent_count_year over future window)",
            "trial_heat": "log1p(sum trial_count_year over future window)",
            "translation_heat": "log1p(sum patent_count_year + trial_count_year over future window)",
            "future_window": "cutoff_year + 1 .. cutoff_year + horizon_years",
        },
        "eligibility_definition": {
            "patent": "history_paper_count > 0 and history_patent_count == 0",
            "trial": "history_paper_count > 0 and history_trial_count == 0",
            "translation": "history_paper_count > 0 and history_patent_count == 0 and history_trial_count == 0",
        },
        "counts": counts,
        "temporal_range": temporal_range,
        "split_summary": split_summary,
        "task_summary": task_summary,
        "feature_columns": {
            "categorical": CATEGORICAL_FEATURE_COLUMNS,
            "numeric": NUMERIC_FEATURE_COLUMNS,
            "all": CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS,
        },
        "target_columns": TARGET_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "leakage_audit_status": leakage_audit["overall_status"],
        "runtime_seconds": runtime_seconds,
        "created_at_unix": time.time(),
    }


def build_report(summary: dict[str, Any], leakage_audit: dict[str, Any]) -> str:
    split_rows = summary.get("split_summary", [])
    task_rows = summary.get("task_summary", [])
    counts = summary.get("counts", {})

    parts = ["# Stage 05F KU Temporal Prediction Dataset Report\n"]

    parts.append("## 1. Dataset summary\n")
    parts.append(
        markdown_table(
            [
                {"metric": "prediction_example_count", "value": counts.get("prediction_example_count")},
                {"metric": "prediction_feature_row_count", "value": counts.get("prediction_feature_row_count")},
                {"metric": "ku_count", "value": counts.get("ku_count")},
                {"metric": "cutoff_count", "value": counts.get("cutoff_count")},
                {"metric": "patent_task_row_count", "value": counts.get("patent_task_row_count")},
                {"metric": "trial_task_row_count", "value": counts.get("trial_task_row_count")},
                {"metric": "translation_task_row_count", "value": counts.get("translation_task_row_count")},
                {"metric": "horizon_years", "value": summary.get("horizon_years")},
                {"metric": "leakage_audit_status", "value": summary.get("leakage_audit_status")},
            ],
            ["metric", "value"],
        )
    )

    parts.append("\n## 2. Forecasting protocol\n")
    parts.append(
        "This dataset uses an expanding-window rolling-origin protocol. "
        "Each example is a `KU × cutoff_year` row. Features are computed from "
        "`history_start_year..cutoff_year`; targets are computed from "
        "`cutoff_year+1..cutoff_year+horizon_years`.\n"
    )

    parts.append("\n## 3. Split summary\n")
    parts.append(markdown_table(split_rows, list(split_rows[0].keys()) if split_rows else []))

    parts.append("\n## 4. Task summary\n")
    parts.append(markdown_table(task_rows, list(task_rows[0].keys()) if task_rows else []))

    parts.append("\n## 5. Leakage audit\n")
    parts.append(
        markdown_table(
            leakage_audit["checks"],
            ["check", "status", "detail"],
        )
    )

    parts.append("\n## 6. Recommended Stage 06 input\n")
    parts.append(
        "Stage 06 baseline scripts should read:\n\n"
        "```text\n"
        "prediction_feature_table.parquet\n"
        "feature_columns.json\n"
        "target_columns.json\n"
        "```\n\n"
        "For the primary overall translation heat task, filter:\n\n"
        "```text\n"
        "eligible_translation_task == true\n"
        "```\n\n"
        "and use:\n\n"
        "```text\n"
        "target_future_translation_heat_3yr\n"
        "```\n"
    )

    return "\n".join(parts)


def artifact_row(
    artifact: str,
    path: Path,
    description: str,
    row_count: int | None = None,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }


def write_outputs(
    connection: Any,
    args: argparse.Namespace,
    summary: dict[str, Any],
    leakage_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    output_dir = args.output_dir

    paths = {
        "prediction_examples": output_dir / "prediction_examples.parquet",
        "prediction_feature_table": output_dir / "prediction_feature_table.parquet",
        "prediction_examples_patent": output_dir / "prediction_examples_patent.parquet",
        "prediction_examples_trial": output_dir / "prediction_examples_trial.parquet",
        "prediction_examples_translation": output_dir / "prediction_examples_translation.parquet",
        "prediction_label_summary": output_dir / "prediction_label_summary.csv",
        "prediction_split_summary": output_dir / "prediction_split_summary.csv",
        "prediction_task_summary": output_dir / "prediction_task_summary.csv",
        "prediction_cutoff_summary": output_dir / "prediction_cutoff_summary.csv",
        "prediction_dataset_summary": output_dir / "prediction_dataset_summary.json",
        "prediction_dataset_report": output_dir / "prediction_dataset_report.md",
        "prediction_leakage_audit_report": output_dir / "prediction_leakage_audit_report.md",
        "feature_columns": output_dir / "feature_columns.json",
        "target_columns": output_dir / "target_columns.json",
        "run_config": output_dir / "run_config.json",
        "prediction_dataset_manifest": output_dir / "prediction_dataset_manifest.csv",
    }

    write_duckdb_table_parquet(
        connection,
        "prediction_examples",
        paths["prediction_examples"],
        args.compression,
    )
    write_duckdb_table_parquet(
        connection,
        "prediction_feature_table",
        paths["prediction_feature_table"],
        args.compression,
    )
    write_duckdb_table_parquet(
        connection,
        "prediction_examples_patent",
        paths["prediction_examples_patent"],
        args.compression,
    )
    write_duckdb_table_parquet(
        connection,
        "prediction_examples_trial",
        paths["prediction_examples_trial"],
        args.compression,
    )
    write_duckdb_table_parquet(
        connection,
        "prediction_examples_translation",
        paths["prediction_examples_translation"],
        args.compression,
    )

    write_duckdb_table_csv(
        connection,
        "prediction_label_summary",
        paths["prediction_label_summary"],
    )
    write_duckdb_table_csv(
        connection,
        "prediction_split_summary",
        paths["prediction_split_summary"],
    )
    write_duckdb_table_csv(
        connection,
        "prediction_task_summary",
        paths["prediction_task_summary"],
    )
    write_duckdb_table_csv(
        connection,
        "prediction_cutoff_summary",
        paths["prediction_cutoff_summary"],
    )

    feature_columns_payload = {
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "feature_columns": CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS,
        "identifier_columns": IDENTIFIER_COLUMNS,
        "eligibility_columns": ELIGIBILITY_COLUMNS,
        "do_not_use_as_features": sorted(
            set(TARGET_COLUMNS)
            | set(LABEL_COLUMNS)
            | set(FULL_WINDOW_FORBIDDEN_FEATURE_COLUMNS)
        ),
    }
    target_columns_payload = {
        "primary_heat_targets": [
            "target_future_patent_heat_3yr",
            "target_future_trial_heat_3yr",
            "target_future_translation_heat_3yr",
        ],
        "count_targets": [
            "target_future_patent_count_3yr",
            "target_future_trial_count_3yr",
            "target_future_translation_count_3yr",
        ],
        "weight_targets": [
            "target_future_patent_weight_sum_3yr",
            "target_future_trial_weight_sum_3yr",
            "target_future_translation_weight_sum_3yr",
        ],
        "binary_emergence_labels": LABEL_COLUMNS,
        "all_target_columns": TARGET_COLUMNS,
        "all_label_columns": LABEL_COLUMNS,
    }

    write_json(paths["feature_columns"], feature_columns_payload)
    write_json(paths["target_columns"], target_columns_payload)
    write_json(paths["prediction_dataset_summary"], summary)

    run_config = {
        "script": "scripts/29_build_knowledge_unit_prediction_dataset.py",
        "command": " ".join(sys.argv),
        "modeling_dir": str(args.modeling_dir),
        "output_dir": str(args.output_dir),
        "history_start_year": args.history_start_year,
        "first_cutoff_year": args.first_cutoff_year,
        "last_cutoff_year": args.last_cutoff_year,
        "horizon_years": args.horizon_years,
        "train_cutoffs": list(args.train_cutoffs),
        "validation_cutoffs": list(args.validation_cutoffs),
        "test_cutoffs": list(args.test_cutoffs),
        "threads": args.threads,
        "memory_limit": args.memory_limit,
        "compression": args.compression,
    }
    write_json(paths["run_config"], run_config)

    report = build_report(summary, leakage_audit)
    write_text(paths["prediction_dataset_report"], report)

    leakage_report = "\n".join(
        [
            "# Stage 05F Prediction Dataset Leakage Audit",
            "",
            f"Overall status: **{leakage_audit['overall_status']}**",
            "",
            markdown_table(leakage_audit["checks"], ["check", "status", "detail"]),
            "",
        ]
    )
    write_text(paths["prediction_leakage_audit_report"], leakage_report)

    artifacts = [
        artifact_row(
            "prediction_examples",
            paths["prediction_examples"],
            "Unified KU × cutoff prediction examples.",
            summary["counts"]["prediction_example_count"],
        ),
        artifact_row(
            "prediction_feature_table",
            paths["prediction_feature_table"],
            "Model-ready Stage 06 feature table with labels and targets.",
            summary["counts"]["prediction_feature_row_count"],
        ),
        artifact_row(
            "prediction_examples_patent",
            paths["prediction_examples_patent"],
            "Patent task eligible examples.",
            summary["counts"]["patent_task_row_count"],
        ),
        artifact_row(
            "prediction_examples_trial",
            paths["prediction_examples_trial"],
            "Trial task eligible examples.",
            summary["counts"]["trial_task_row_count"],
        ),
        artifact_row(
            "prediction_examples_translation",
            paths["prediction_examples_translation"],
            "Translation task eligible examples.",
            summary["counts"]["translation_task_row_count"],
        ),
        artifact_row(
            "prediction_label_summary",
            paths["prediction_label_summary"],
            "Label and target summary by task/split/cutoff.",
        ),
        artifact_row(
            "prediction_split_summary",
            paths["prediction_split_summary"],
            "Prediction example summary by split.",
        ),
        artifact_row(
            "prediction_task_summary",
            paths["prediction_task_summary"],
            "Prediction example summary by task.",
        ),
        artifact_row(
            "prediction_cutoff_summary",
            paths["prediction_cutoff_summary"],
            "Prediction example summary by cutoff year.",
        ),
        artifact_row(
            "prediction_dataset_summary",
            paths["prediction_dataset_summary"],
            "JSON summary for Stage 05F prediction dataset.",
        ),
        artifact_row(
            "prediction_dataset_report",
            paths["prediction_dataset_report"],
            "Human-readable Stage 05F report.",
        ),
        artifact_row(
            "prediction_leakage_audit_report",
            paths["prediction_leakage_audit_report"],
            "Leakage audit report.",
        ),
        artifact_row(
            "feature_columns",
            paths["feature_columns"],
            "Feature column specification for Stage 06.",
        ),
        artifact_row(
            "target_columns",
            paths["target_columns"],
            "Target and label column specification for Stage 06.",
        ),
        artifact_row(
            "run_config",
            paths["run_config"],
            "Run configuration.",
        ),
    ]

    artifacts.append(
        artifact_row(
            "prediction_dataset_manifest",
            paths["prediction_dataset_manifest"],
            "Manifest of Stage 05F artifacts.",
            len(artifacts) + 1,
        )
    )

    write_csv(
        paths["prediction_dataset_manifest"],
        artifacts,
        ["artifact", "path", "description", "row_count", "file_size_bytes"],
    )

    return artifacts


def print_summary(summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    print()
    print("Stage 05F KU temporal prediction dataset complete.")
    print(f"Output directory: {summary['output_dir']}")
    print()
    print(f"Prediction examples:        {counts['prediction_example_count']:,}")
    print(f"Prediction feature rows:    {counts['prediction_feature_row_count']:,}")
    print(f"Patent task rows:           {counts['patent_task_row_count']:,}")
    print(f"Trial task rows:            {counts['trial_task_row_count']:,}")
    print(f"Translation task rows:      {counts['translation_task_row_count']:,}")
    print(f"KU count:                   {counts['ku_count']:,}")
    print(f"Cutoff count:               {counts['cutoff_count']:,}")
    print(f"Leakage audit status:       {summary['leakage_audit_status']}")
    print()
    print("Recommended Stage 06 input:")
    print("  prediction_feature_table.parquet")
    print("  feature_columns.json")
    print("  target_columns.json")
    print()


def main() -> None:
    args = parse_args()
    validate_args(args)

    start_time = time.time()

    args.modeling_dir = args.modeling_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()

    paths = require_input_files(args.modeling_dir)
    prepare_output_dir(args.output_dir, args.overwrite)

    connection = connect_duckdb(args)

    run_step(
        args,
        1,
        9,
        "Creating input views",
        lambda: create_input_views(connection, paths),
    )

    run_step(
        args,
        2,
        9,
        "Creating normalized static and temporal tables",
        lambda: create_normalized_views(connection),
    )

    run_step(
        args,
        3,
        9,
        "Creating cutoff table",
        lambda: create_cutoff_table(connection, args),
    )

    temporal_range = run_step(
        args,
        4,
        9,
        "Validating temporal year range",
        lambda: validate_temporal_year_range(connection, args),
    )

    run_step(
        args,
        5,
        9,
        "Building KU x cutoff prediction examples",
        lambda: create_prediction_examples(connection, args),
    )

    run_step(
        args,
        6,
        9,
        "Creating model-ready prediction feature table and task views",
        lambda: (
            create_prediction_feature_table(connection),
            create_task_views(connection),
        ),
    )

    run_step(
        args,
        7,
        9,
        "Creating summary tables",
        lambda: create_summary_tables(connection),
    )

    leakage_audit = run_step(
        args,
        8,
        9,
        "Running leakage audit",
        lambda: run_leakage_audit(connection, args),
    )

    runtime_seconds = time.time() - start_time

    summary = build_dataset_summary(
        connection,
        args,
        temporal_range,
        leakage_audit,
        runtime_seconds,
    )

    run_step(
        args,
        9,
        9,
        "Writing Stage 05F output artifacts",
        lambda: write_outputs(connection, args, summary, leakage_audit),
    )

    print_summary(summary)


if __name__ == "__main__":
    main()