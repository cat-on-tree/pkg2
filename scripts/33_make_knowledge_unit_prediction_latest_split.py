#!/usr/bin/env python
"""Create latest-window split version of a KU prediction-style dataset.

Stage 07A purpose
-----------------

Stage 06 used an exploratory / diagnostic split:

    train:       2005-2016
    validation:  2017-2018
    test:        2019-2021

That split was useful for baseline diagnostics because it provided multiple
test cutoffs.

For Stage 07 temporal GNN / Knowledge Field model development, we use a
latest-window split:

    train:       2005-2018
    validation:  2019-2020
    test:        2021

Because the prediction horizon is 3 years and the source data ends at 2024,
the latest valid cutoff year is 2021:

    cutoff 2021 -> target window 2022-2024

Important:

    2022-2024 are target years, not cutoff years.

This script does not rebuild features or targets. It only rewrites the split
assignment of an existing leakage-audited prediction-style dataset.

Supported inputs
----------------

Any directory containing:

    prediction_feature_table.parquet
    feature_columns.json
    target_columns.json

Examples:

1. Local KU features only:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction/

2. Local KU + carrier source/exposure features:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/
        feature_sets/
          ku_plus_carrier_graph_structural/

3. Local KU + carrier source/exposure + handcrafted diffusion proxy features:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy/
        feature_sets/
          ku_plus_carrier_structural_plus_diffusion_proxy/

Recommended flat outputs
------------------------

1. Local KU features only:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction_latest_split/

2. Local KU + carrier source/exposure:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural_latest_split/

3. Local KU + carrier source/exposure + handcrafted diffusion proxy:

    data/datasets/knowledge_units/
      diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy_latest_split/

Main outputs
------------

    prediction_feature_table.parquet
    prediction_examples.parquet
    prediction_examples_patent.parquet
    prediction_examples_trial.parquet
    prediction_examples_translation.parquet

    feature_columns.json
    target_columns.json

    latest_split_summary.json
    latest_split_cutoff_summary.csv
    latest_split_task_summary.csv
    latest_split_leakage_note.md
    prediction_dataset_manifest.csv

Examples
--------

Base local latest split:

    python scripts/33_make_knowledge_unit_prediction_latest_split.py \\
      --prediction-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction \\
      --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_latest_split \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite

Local + carrier source latest split:

    python scripts/33_make_knowledge_unit_prediction_latest_split.py \\
      --prediction-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/feature_sets/ku_plus_carrier_graph_structural \\
      --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural_latest_split \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite

Local + carrier source + diffusion proxy latest split:

    python scripts/33_make_knowledge_unit_prediction_latest_split.py \\
      --prediction-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy/feature_sets/ku_plus_carrier_structural_plus_diffusion_proxy \\
      --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_ku_diffusion_proxy_latest_split \\
      --threads 4 \\
      --memory-limit 24GB \\
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_PREDICTION_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_latest_split"
)

DEFAULT_TRAIN_CUTOFFS = list(range(2005, 2019))
DEFAULT_VALIDATION_CUTOFFS = [2019, 2020]
DEFAULT_TEST_CUTOFFS = [2021]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create latest-window split version of a KU prediction-style dataset."
        )
    )

    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help=(
            "Input prediction-style dataset directory. Must contain "
            "prediction_feature_table.parquet, feature_columns.json, "
            "and target_columns.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output latest-window split dataset root directory.",
    )

    parser.add_argument(
        "--train-cutoffs",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_CUTOFFS,
        help="Cutoff years assigned to train split.",
    )

    parser.add_argument(
        "--validation-cutoffs",
        nargs="+",
        type=int,
        default=DEFAULT_VALIDATION_CUTOFFS,
        help="Cutoff years assigned to validation split.",
    )

    parser.add_argument(
        "--test-cutoffs",
        nargs="+",
        type=int,
        default=DEFAULT_TEST_CUTOFFS,
        help="Cutoff years assigned to test split.",
    )

    parser.add_argument(
        "--horizon-years",
        type=int,
        default=3,
        help="Forecast horizon in years. Used for validation and reporting.",
    )

    parser.add_argument(
        "--dataset-end-year",
        type=int,
        default=2024,
        help="Last observed data year. Used for validation and reporting.",
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
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.prediction_dir = args.prediction_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    args.train_cutoffs = sorted(set(int(x) for x in args.train_cutoffs))
    args.validation_cutoffs = sorted(set(int(x) for x in args.validation_cutoffs))
    args.test_cutoffs = sorted(set(int(x) for x in args.test_cutoffs))

    all_cutoffs = args.train_cutoffs + args.validation_cutoffs + args.test_cutoffs

    if not all_cutoffs:
        raise ValueError("At least one cutoff year must be provided.")

    if len(all_cutoffs) != len(set(all_cutoffs)):
        raise ValueError(
            "Train / validation / test cutoffs overlap. "
            f"train={args.train_cutoffs}, "
            f"validation={args.validation_cutoffs}, "
            f"test={args.test_cutoffs}"
        )

    if args.horizon_years <= 0:
        raise ValueError("--horizon-years must be positive.")

    max_cutoff = max(all_cutoffs)
    max_required_target_year = max_cutoff + args.horizon_years

    if max_required_target_year > args.dataset_end_year:
        raise ValueError(
            "Latest cutoff exceeds available target window. "
            f"max_cutoff={max_cutoff}, "
            f"horizon={args.horizon_years}, "
            f"requires target year {max_required_target_year}, "
            f"but dataset_end_year={args.dataset_end_year}."
        )

    return args


def log(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, flush=True)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def duck_path(path: Path) -> str:
    text = str(path.resolve()).replace("\\", "/")
    return text.replace("'", "''")


def q(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def cutoff_list_sql(values: list[int]) -> str:
    if not values:
        return "NULL"
    return ", ".join(str(int(v)) for v in values)


def configure_duckdb(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(args.threads)}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    return con


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_input_dir(prediction_dir: Path) -> None:
    required = [
        prediction_dir / "prediction_feature_table.parquet",
        prediction_dir / "feature_columns.json",
        prediction_dir / "target_columns.json",
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(str(p) for p in missing)
        )


def get_parquet_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{duck_path(path)}')"
    ).fetchall()
    return [str(row[0]) for row in rows]


def infer_cutoff_column(columns: list[str]) -> str:
    candidates = [
        "cutoff_year",
        "cutoff_year_norm",
    ]

    lower_to_original = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    for column in columns:
        low = column.lower()
        if "cutoff" in low and "year" in low:
            return column

    raise ValueError(f"Could not infer cutoff year column from columns: {columns}")


def split_case_sql(
    *,
    cutoff_expr: str,
    train_cutoffs: list[int],
    validation_cutoffs: list[int],
    test_cutoffs: list[int],
) -> str:
    train_sql = cutoff_list_sql(train_cutoffs)
    val_sql = cutoff_list_sql(validation_cutoffs)
    test_sql = cutoff_list_sql(test_cutoffs)

    return f"""
    CASE
      WHEN {cutoff_expr} IN ({train_sql}) THEN 'train'
      WHEN {cutoff_expr} IN ({val_sql}) THEN 'validation'
      WHEN {cutoff_expr} IN ({test_sql}) THEN 'test'
      ELSE 'unused'
    END
    """


def resplit_parquet_table(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    output_path: Path,
    train_cutoffs: list[int],
    validation_cutoffs: list[int],
    test_cutoffs: list[int],
) -> dict[str, Any]:
    """Rewrite split columns for one parquet table."""

    columns = get_parquet_columns(con, input_path)
    cutoff_col = infer_cutoff_column(columns)

    old_split_cols = {
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

    kept_columns = [col for col in columns if col not in old_split_cols]

    if not kept_columns:
        raise ValueError(f"No columns left after removing split columns: {input_path}")

    select_cols = ",\n            ".join(q(col) for col in kept_columns)
    cutoff_expr = f"TRY_CAST({q(cutoff_col)} AS INTEGER)"

    split_expr = split_case_sql(
        cutoff_expr=cutoff_expr,
        train_cutoffs=train_cutoffs,
        validation_cutoffs=validation_cutoffs,
        test_cutoffs=test_cutoffs,
    )

    all_cutoffs = train_cutoffs + validation_cutoffs + test_cutoffs

    output_path.parent.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"""
        COPY (
          SELECT
            {select_cols},
            {split_expr} AS split,
            ({cutoff_expr} IN ({cutoff_list_sql(train_cutoffs)}))::BOOLEAN
              AS is_train,
            ({cutoff_expr} IN ({cutoff_list_sql(validation_cutoffs)}))::BOOLEAN
              AS is_validation,
            ({cutoff_expr} IN ({cutoff_list_sql(validation_cutoffs)}))::BOOLEAN
              AS is_val,
            ({cutoff_expr} IN ({cutoff_list_sql(test_cutoffs)}))::BOOLEAN
              AS is_test
          FROM read_parquet('{duck_path(input_path)}')
          WHERE {cutoff_expr} IN ({cutoff_list_sql(all_cutoffs)})
        )
        TO '{duck_path(output_path)}'
        (FORMAT PARQUET)
        """
    )

    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{duck_path(output_path)}')"
    ).fetchone()[0]

    cutoff_count = con.execute(
        f"""
        SELECT COUNT(DISTINCT TRY_CAST({q(cutoff_col)} AS INTEGER))
        FROM read_parquet('{duck_path(output_path)}')
        """
    ).fetchone()[0]

    split_counts = con.execute(
        f"""
        SELECT
          split,
          COUNT(*) AS row_count
        FROM read_parquet('{duck_path(output_path)}')
        GROUP BY split
        ORDER BY split
        """
    ).fetchdf()

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": int(row_count),
        "cutoff_count": int(cutoff_count),
        "cutoff_column": cutoff_col,
        "split_counts": split_counts.to_dict(orient="records"),
    }


def copy_spec_files(prediction_dir: Path, output_dir: Path) -> None:
    """Copy model specs and preserve optional source metadata."""

    for filename in [
        "feature_columns.json",
        "target_columns.json",
    ]:
        shutil.copy2(prediction_dir / filename, output_dir / filename)

    optional_files = [
        "run_config.json",
        "run_args.json",
        "prediction_dataset_summary.json",
        "prediction_leakage_audit_report.md",
        "prediction_dataset_report.md",
        "baseline_run_summary.json",
        "carrier_graph_structural_summary.json",
        "diffusion_proxy_summary.json",
        "latest_split_summary.json",
    ]

    provenance_dir = output_dir / "source_prediction_provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    for filename in optional_files:
        src = prediction_dir / filename
        if src.exists():
            shutil.copy2(src, provenance_dir / filename)


def write_task_views(
    *,
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> dict[str, Any]:
    """Regenerate task-specific eligible views from prediction_feature_table."""

    table_path = output_dir / "prediction_feature_table.parquet"

    tasks = {
        "patent": "eligible_patent_task",
        "trial": "eligible_trial_task",
        "translation": "eligible_translation_task",
    }

    results: dict[str, Any] = {}

    columns = get_parquet_columns(con, table_path)
    lower_to_original = {c.lower(): c for c in columns}

    for task, eligibility_col in tasks.items():
        actual_col = lower_to_original.get(eligibility_col.lower())

        if actual_col is None:
            results[task] = {
                "status": "skipped",
                "reason": f"Missing eligibility column: {eligibility_col}",
            }
            continue

        out_path = output_dir / f"prediction_examples_{task}.parquet"

        con.execute(
            f"""
            COPY (
              SELECT *
              FROM read_parquet('{duck_path(table_path)}')
              WHERE COALESCE(TRY_CAST({q(actual_col)} AS BOOLEAN), FALSE)
            )
            TO '{duck_path(out_path)}'
            (FORMAT PARQUET)
            """
        )

        row_count = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{duck_path(out_path)}')"
        ).fetchone()[0]

        split_counts = con.execute(
            f"""
            SELECT
              split,
              COUNT(*) AS row_count
            FROM read_parquet('{duck_path(out_path)}')
            GROUP BY split
            ORDER BY split
            """
        ).fetchdf()

        results[task] = {
            "status": "ok",
            "eligibility_column": actual_col,
            "path": str(out_path),
            "row_count": int(row_count),
            "split_counts": split_counts.to_dict(orient="records"),
        }

    return results


def write_summaries(
    *,
    con: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
    output_dir: Path,
    table_summary: dict[str, Any],
    examples_summary: dict[str, Any] | None,
    task_view_summary: dict[str, Any],
    runtime_seconds: float,
) -> None:
    table_path = output_dir / "prediction_feature_table.parquet"
    columns = get_parquet_columns(con, table_path)
    cutoff_col = infer_cutoff_column(columns)

    cutoff_summary = con.execute(
        f"""
        SELECT
          TRY_CAST({q(cutoff_col)} AS INTEGER) AS cutoff_year,
          split,
          COUNT(*) AS row_count,
          SUM(CASE WHEN is_train THEN 1 ELSE 0 END) AS train_rows,
          SUM(CASE WHEN is_validation THEN 1 ELSE 0 END) AS validation_rows,
          SUM(CASE WHEN is_test THEN 1 ELSE 0 END) AS test_rows
        FROM read_parquet('{duck_path(table_path)}')
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()

    cutoff_summary.to_csv(
        output_dir / "latest_split_cutoff_summary.csv",
        index=False,
    )

    task_rows = []
    for task, info in task_view_summary.items():
        task_rows.append(
            {
                "task": task,
                "status": info.get("status"),
                "eligibility_column": info.get("eligibility_column"),
                "row_count": info.get("row_count"),
                "path": info.get("path"),
                "reason": info.get("reason"),
            }
        )

    pd.DataFrame(task_rows).to_csv(
        output_dir / "latest_split_task_summary.csv",
        index=False,
    )

    max_test_cutoff = max(args.test_cutoffs)
    test_target_years = list(
        range(max_test_cutoff + 1, max_test_cutoff + args.horizon_years + 1)
    )

    feature_spec = read_json(output_dir / "feature_columns.json")
    target_spec = read_json(output_dir / "target_columns.json")

    summary = {
        "script": "scripts/33_make_knowledge_unit_prediction_latest_split.py",
        "status": "ok",
        "source_prediction_dir": str(args.prediction_dir),
        "output_dir": str(output_dir),
        "split_name": "latest_window",
        "train_cutoffs": args.train_cutoffs,
        "validation_cutoffs": args.validation_cutoffs,
        "test_cutoffs": args.test_cutoffs,
        "horizon_years": args.horizon_years,
        "dataset_end_year": args.dataset_end_year,
        "test_target_years": test_target_years,
        "important_note": (
            "The listed test target years are target years for the held-out "
            "test cutoff under the configured horizon. They are not cutoff years."
        ),
        "prediction_feature_table": table_summary,
        "prediction_examples": examples_summary,
        "task_views": task_view_summary,
        "feature_spec_keys": sorted(feature_spec.keys()),
        "target_spec_keys": sorted(target_spec.keys()),
        "runtime_seconds": runtime_seconds,
        "created_at_unix": time.time(),
    }

    write_json(output_dir / "latest_split_summary.json", summary)

    leakage_note = f"""# Latest-window split leakage note

This dataset was produced by rewriting the split assignment of an existing
prediction-style dataset.

Source dataset:

```text
{args.prediction_dir}
```

Output dataset:

```text
{output_dir}
```

The feature columns and targets are not recomputed by this script. This script
only changes the split assignment.

The latest-window split is:

```text
train cutoffs:
  {args.train_cutoffs}

validation cutoffs:
  {args.validation_cutoffs}

test cutoffs:
  {args.test_cutoffs}
```

With horizon:

```text
horizon_years = {args.horizon_years}
```

and dataset end year:

```text
dataset_end_year = {args.dataset_end_year}
```

The held-out test cutoff is:

```text
cutoff = {max_test_cutoff}
```

Its target window is:

```text
target years = {test_target_years}
```

Important:

```text
{test_target_years[0]}-{test_target_years[-1]} are target years, not cutoff years.
```

Using cutoff years 2022-2024 would require future data beyond 2024 under a
3-year horizon, so those years cannot be used as cutoff years with the current
dataset.

This output is a flat prediction-style dataset root. It intentionally does not
preserve any nested `feature_sets/...` directory structure from the source.
"""

    (output_dir / "latest_split_leakage_note.md").write_text(
        leakage_note,
        encoding="utf-8",
    )


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
        output_dir / "prediction_dataset_manifest.csv",
        index=False,
    )


def main() -> None:
    args = normalize_args(parse_args())
    validate_input_dir(args.prediction_dir)

    start = time.time()

    prepare_output_dir(args.output_dir, args.overwrite)

    log(args, "Creating latest-window split dataset ...")
    log(args, f"Input prediction-style dir:  {args.prediction_dir}")
    log(args, f"Output dataset root:         {args.output_dir}")
    log(args, f"Train cutoffs:               {args.train_cutoffs}")
    log(args, f"Validation cutoffs:          {args.validation_cutoffs}")
    log(args, f"Test cutoffs:                {args.test_cutoffs}")

    max_test_cutoff = max(args.test_cutoffs)

    log(
        args,
        (
            f"Test target years:           "
            f"{max_test_cutoff + 1}-"
            f"{max_test_cutoff + args.horizon_years}"
        ),
    )
    log(args, "")

    con = configure_duckdb(args)

    copy_spec_files(args.prediction_dir, args.output_dir)

    feature_input = args.prediction_dir / "prediction_feature_table.parquet"
    feature_output = args.output_dir / "prediction_feature_table.parquet"

    log(args, "Rewriting prediction_feature_table.parquet split assignment ...")
    table_summary = resplit_parquet_table(
        con=con,
        input_path=feature_input,
        output_path=feature_output,
        train_cutoffs=args.train_cutoffs,
        validation_cutoffs=args.validation_cutoffs,
        test_cutoffs=args.test_cutoffs,
    )
    log(args, f"  rows: {table_summary['row_count']:,}")

    examples_summary = None
    examples_input = args.prediction_dir / "prediction_examples.parquet"
    examples_output = args.output_dir / "prediction_examples.parquet"

    if examples_input.exists():
        log(args, "Rewriting prediction_examples.parquet split assignment ...")
        examples_summary = resplit_parquet_table(
            con=con,
            input_path=examples_input,
            output_path=examples_output,
            train_cutoffs=args.train_cutoffs,
            validation_cutoffs=args.validation_cutoffs,
            test_cutoffs=args.test_cutoffs,
        )
        log(args, f"  rows: {examples_summary['row_count']:,}")
    else:
        log(
            args,
            (
                "prediction_examples.parquet not found in source; "
                "copying rewritten prediction_feature_table.parquet as "
                "prediction_examples.parquet."
            ),
        )
        shutil.copy2(feature_output, examples_output)
        examples_summary = {
            "input_path": str(feature_output),
            "output_path": str(examples_output),
            "row_count": table_summary["row_count"],
            "cutoff_count": table_summary["cutoff_count"],
            "note": "Copied from rewritten prediction_feature_table.parquet",
        }

    log(args, "Regenerating task-specific eligible views ...")
    task_view_summary = write_task_views(
        con=con,
        output_dir=args.output_dir,
    )

    for task, info in task_view_summary.items():
        if info.get("status") == "ok":
            log(args, f"  {task}: {int(info.get('row_count', 0)):,} rows")
        else:
            log(args, f"  {task}: skipped ({info.get('reason')})")

    runtime_seconds = time.time() - start

    log(args, "Writing summaries and manifest ...")
    write_summaries(
        con=con,
        args=args,
        output_dir=args.output_dir,
        table_summary=table_summary,
        examples_summary=examples_summary,
        task_view_summary=task_view_summary,
        runtime_seconds=runtime_seconds,
    )
    write_manifest(args.output_dir)

    log(args, "")
    log(args, "Latest-window split dataset complete.")
    log(args, f"Output directory: {args.output_dir}")
    log(args, f"Runtime seconds: {runtime_seconds:.1f}")


if __name__ == "__main__":
    main()