"""Inspect PKG24S4 TSV schemas without loading full raw files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import codecs
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CONFIG = Path("config/pkg24s4.yaml")
SUMMARY_COLUMNS = [
    "file_name",
    "file_path",
    "file_size_mb",
    "compression_type",
    "delimiter",
    "column_count",
    "column_names",
    "sample_row_count",
    "total_row_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PKG24S4 TSV/TSV.GZ files using only a small sample by default."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to PKG24S4 YAML config.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Number of data rows to sample per file. Defaults to config inspection.sample_rows.",
    )
    parser.add_argument(
        "--count-rows",
        action="store_true",
        help="Stream every file to count total data rows. Disabled by default.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
            config = parse_simple_yaml(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    return config


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


def resolve_config_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def normalize_encoding(encoding: str) -> tuple[str, str]:
    if encoding.lower() in {"utf8-lossy", "utf-8-lossy"}:
        return "utf-8", "replace"
    return encoding, "strict"


def display_delimiter(delimiter: str) -> str:
    if delimiter == "\t":
        return "\\t"
    if delimiter == "\n":
        return "\\n"
    return delimiter


def open_text(path: Path, encoding: str, errors: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding=encoding, errors=errors, newline="")
    return path.open("r", encoding=encoding, errors=errors, newline="")


def compression_type(path: Path) -> str:
    if path.name.endswith(".gz"):
        return "gzip"
    return "none"


def find_tsv_files(raw_dir: Path, patterns: dict[str, str]) -> list[Path]:
    matched: set[Path] = set()
    for key in ("raw_tsv_gz", "raw_tsv"):
        pattern = patterns.get(key)
        if pattern:
            matched.update(raw_dir.rglob(pattern))
    return sorted(path for path in matched if path.is_file())


def read_sample_rows(
    path: Path,
    delimiter: str,
    has_header: bool,
    sample_rows: int,
    encoding: str,
    errors: str,
) -> tuple[list[str], int]:
    with open_text(path, encoding, errors) as input_file:
        reader = csv.reader(input_file, delimiter=delimiter)
        try:
            first_row = next(reader)
        except StopIteration:
            return [], 0

        if has_header:
            column_names = first_row
            sample_count = sum(1 for _, _row in zip(range(sample_rows), reader))
            return column_names, sample_count

        column_names = [f"column_{index}" for index in range(1, len(first_row) + 1)]
        sample_count = 1 + sum(1 for _, _row in zip(range(max(sample_rows - 1, 0)), reader))
        return column_names, sample_count


def count_data_rows(path: Path, delimiter: str, has_header: bool, encoding: str, errors: str) -> int:
    with open_text(path, encoding, errors) as input_file:
        row_count = sum(1 for _row in csv.reader(input_file, delimiter=delimiter))
    if has_header and row_count > 0:
        return row_count - 1
    return row_count


def inspect_files(
    files: Iterable[Path],
    raw_dir: Path,
    output_path: Path,
    delimiter: str,
    has_header: bool,
    sample_rows: int,
    encoding: str,
    errors: str,
    should_count_rows: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in files:
        column_names, sample_row_count = read_sample_rows(
            file_path,
            delimiter=delimiter,
            has_header=has_header,
            sample_rows=sample_rows,
            encoding=encoding,
            errors=errors,
        )
        total_row_count = (
            count_data_rows(file_path, delimiter, has_header, encoding, errors)
            if should_count_rows
            else ""
        )
        rows.append(
            {
                "file_name": file_path.name,
                "file_path": str(file_path.resolve()),
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
                "compression_type": compression_type(file_path),
                "delimiter": display_delimiter(delimiter),
                "column_count": len(column_names),
                "column_names": json.dumps(column_names, ensure_ascii=False),
                "sample_row_count": sample_row_count,
                "total_row_count": total_row_count,
            }
        )
    rows.sort(key=lambda row: row["file_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_summary(rows: list[dict[str, Any]], raw_dir: Path, output_path: Path, counted_rows: bool) -> None:
    total_size_mb = sum(float(row["file_size_mb"]) for row in rows)
    print("PKG24S4 inspection summary")
    print(f"Raw directory: {raw_dir}")
    print(f"Files inspected: {len(rows)}")
    print(f"Total compressed size: {total_size_mb:.3f} MB")
    print(f"Row counting: {'enabled' if counted_rows else 'disabled'}")
    print(f"Schema summary: {output_path}")
    if rows:
        print()
        print(f"{'file_name':<45} {'columns':>7} {'sample_rows':>11} {'size_mb':>10}")
        for row in rows:
            print(
                f"{row['file_name']:<45} {row['column_count']:>7} "
                f"{row['sample_row_count']:>11} {row['file_size_mb']:>10}"
            )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)

    paths_config = config.get("paths", {})
    read_options = config.get("read_options", {})
    inspection = config.get("inspection", {})
    patterns = config.get("file_patterns", {})

    raw_dir = resolve_config_path(paths_config["raw_dir"], project_root).resolve()
    output_path = resolve_config_path(paths_config["schema_summary"], project_root).resolve()
    delimiter = read_options.get("separator", "\t")
    has_header = bool(read_options.get("has_header", True))
    encoding, errors = normalize_encoding(read_options.get("encoding", "utf-8"))
    sample_rows = args.sample_rows
    if sample_rows is None:
        sample_rows = int(inspection.get("sample_rows", 20))
    if sample_rows < 0:
        raise ValueError("--sample-rows must be greater than or equal to 0")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    files = find_tsv_files(raw_dir, patterns)
    rows = inspect_files(
        files,
        raw_dir=raw_dir,
        output_path=output_path,
        delimiter=delimiter,
        has_header=has_header,
        sample_rows=sample_rows,
        encoding=encoding,
        errors=errors,
        should_count_rows=args.count_rows,
    )
    print_summary(rows, raw_dir, output_path, args.count_rows)


if __name__ == "__main__":
    main()
