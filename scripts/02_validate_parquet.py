"""Validate PKG24S4 Parquet outputs against inspected TSV schemas."""

from __future__ import annotations

import argparse
import codecs
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG = Path("config/pkg24s4.yaml")
DEFAULT_MANIFEST = Path("data/interim/parquet_manifest.csv")
DEFAULT_REPORT = Path("data/interim/parquet_validation.csv")
REPORT_COLUMNS = [
    "file_name",
    "expected_parquet_path",
    "parquet_exists",
    "readable",
    "status",
    "expected_column_count",
    "actual_column_count",
    "column_count_match",
    "column_match",
    "expected_column_names",
    "actual_column_names",
    "row_count",
    "parquet_size_mb",
    "manifest_status",
    "requested_engine",
    "actual_engine",
    "bad_row_count",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PKG24S4 Parquet files without loading them into memory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to PKG24S4 YAML config.",
    )
    parser.add_argument(
        "--schema-summary",
        type=Path,
        default=None,
        help="Path to schema_summary.csv. Defaults to config paths.schema_summary.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to parquet_manifest.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to write parquet_validation.csv.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        return parse_simple_yaml(config_file)


def parse_simple_yaml(lines: Iterable[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped.endswith(":"):
            current_key = stripped[:-1]
            config[current_key] = {}
            continue

        if current_key is None:
            continue

        if indent == 2 and stripped.startswith("- "):
            if not isinstance(config[current_key], list):
                config[current_key] = []
            config[current_key].append(parse_scalar(stripped[2:]))
            continue

        if indent == 2 and ":" in stripped and isinstance(config[current_key], dict):
            key, value = stripped.split(":", 1)
            config[current_key][key.strip()] = parse_scalar(value.strip())

    return config


def parse_scalar(value: str) -> Any:
    if not value:
        return None
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if normalized in {"null", "none", "~"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return codecs.decode(value[1:-1], "unicode_escape")
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def resolve_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def parquet_name(input_name: str) -> str:
    if input_name.endswith(".tsv.gz"):
        return input_name[: -len(".tsv.gz")] + ".parquet"
    if input_name.endswith(".tsv"):
        return input_name[: -len(".tsv")] + ".parquet"
    return Path(input_name).stem + ".parquet"


def file_size_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.3f}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {row.get("file_name", ""): row for row in rows}


def parse_expected_columns(row: dict[str, str]) -> list[str]:
    raw_columns = row.get("column_names", "[]")
    parsed = json.loads(raw_columns)
    if not isinstance(parsed, list):
        raise ValueError(f"column_names is not a JSON list for {row.get('file_name', '')}")
    return [str(column) for column in parsed]


def read_parquet_metadata(path: Path) -> tuple[list[str], int]:
    try:
        return read_parquet_metadata_with_pyarrow(path)
    except ModuleNotFoundError as pyarrow_error:
        try:
            return read_parquet_metadata_with_polars(path)
        except ModuleNotFoundError as polars_error:
            raise RuntimeError(
                "Neither pyarrow nor polars is installed; activate the PKG24S4 environment "
                "or install one of them to read Parquet metadata."
            ) from polars_error


def read_parquet_metadata_with_pyarrow(path: Path) -> tuple[list[str], int]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema_arrow.names
    row_count = parquet_file.metadata.num_rows
    return columns, row_count


def read_parquet_metadata_with_polars(path: Path) -> tuple[list[str], int]:
    import polars as pl

    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    columns = list(schema.names())
    row_count = int(scan.select(pl.len()).collect().item())
    return columns, row_count


def to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def validate_one(
    schema_row: dict[str, str],
    parquet_dir: Path,
    manifest_by_file: dict[str, dict[str, str]],
) -> dict[str, str]:
    file_name = schema_row["file_name"]
    expected_path = parquet_dir / parquet_name(file_name)
    expected_columns = parse_expected_columns(schema_row)
    expected_count = int(schema_row["column_count"])
    manifest_row = manifest_by_file.get(file_name, {})

    result = {
        "file_name": file_name,
        "expected_parquet_path": str(expected_path.resolve()),
        "parquet_exists": str(expected_path.exists()),
        "readable": "False",
        "status": "missing",
        "expected_column_count": str(expected_count),
        "actual_column_count": "",
        "column_count_match": "False",
        "column_match": "False",
        "expected_column_names": json.dumps(expected_columns, ensure_ascii=False),
        "actual_column_names": "",
        "row_count": "",
        "parquet_size_mb": "",
        "manifest_status": manifest_row.get("status", ""),
        "requested_engine": manifest_row.get("requested_engine", ""),
        "actual_engine": manifest_row.get("actual_engine", ""),
        "bad_row_count": manifest_row.get("bad_row_count", ""),
        "error_message": manifest_row.get("error_message", ""),
    }

    if not expected_path.exists():
        return result

    result["status"] = "unreadable"
    result["parquet_size_mb"] = file_size_mb(expected_path)
    try:
        actual_columns, row_count = read_parquet_metadata(expected_path)
    except Exception as error:
        result["error_message"] = str(error)
        return result

    actual_count = len(actual_columns)
    column_count_match = actual_count == expected_count
    column_match = actual_columns == expected_columns
    result.update(
        {
            "readable": "True",
            "status": "ok" if column_match else "column_mismatch",
            "actual_column_count": str(actual_count),
            "column_count_match": str(column_count_match),
            "column_match": str(column_match),
            "actual_column_names": json.dumps(actual_columns, ensure_ascii=False),
            "row_count": str(row_count),
        }
    )
    return result


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    expected_files = len(rows)
    existing_files = sum(row["parquet_exists"] == "True" for row in rows)
    readable_files = sum(row["readable"] == "True" for row in rows)
    failed_files = sum(row["status"] in {"missing", "unreadable"} for row in rows)
    column_mismatch_files = sum(row["status"] == "column_mismatch" for row in rows)
    manifest_failed = [row for row in rows if row["manifest_status"] == "error"]
    manifest_bad_rows = [row for row in rows if to_int(row["bad_row_count"]) > 0]

    print("PKG24S4 Parquet validation summary")
    print(f"Expected files: {expected_files}")
    print(f"Existing parquet files: {existing_files}")
    print(f"Readable files: {readable_files}")
    print(f"Failed files: {failed_files}")
    print(f"Column mismatch files: {column_mismatch_files}")
    print(f"Manifest failed conversions: {len(manifest_failed)}")
    print(f"Manifest files with bad rows: {len(manifest_bad_rows)}")
    print(f"Validation report: {output_path}")

    if manifest_failed:
        print("\nWARNING: manifest contains failed conversions")
        for row in manifest_failed:
            print(f"- {row['file_name']}: {row['error_message']}")

    if manifest_bad_rows:
        print("\nWARNING: manifest contains bad rows")
        for row in manifest_bad_rows:
            print(f"- {row['file_name']}: bad_row_count={row['bad_row_count']}")

    print()
    print("Total rows by table")
    for row in rows:
        row_count = row["row_count"] if row["row_count"] else "-"
        print(f"{row['file_name']}\t{row_count}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    paths_config = config.get("paths", {})

    parquet_dir = resolve_path(paths_config["parquet_dir"], project_root).resolve()
    schema_summary_path = (
        resolve_path(args.schema_summary, project_root).resolve()
        if args.schema_summary is not None
        else resolve_path(paths_config["schema_summary"], project_root).resolve()
    )
    manifest_path = resolve_path(args.manifest, project_root).resolve()
    output_path = resolve_path(args.output, project_root).resolve()

    schema_rows = read_csv_rows(schema_summary_path)
    manifest_by_file = load_manifest(manifest_path)
    report_rows = [
        validate_one(schema_row, parquet_dir, manifest_by_file) for schema_row in schema_rows
    ]
    write_report(output_path, report_rows)
    print_summary(report_rows, output_path)


if __name__ == "__main__":
    main()
