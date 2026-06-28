"""Convert PKG24S4 TSV/TSV.GZ files to compressed Parquet files."""

from __future__ import annotations

import argparse
import codecs
import csv
import gzip
import json
import time
from pathlib import Path
from typing import Any, Iterable, TextIO

DEFAULT_CONFIG = Path("config/pkg24s4.yaml")
MANIFEST_COLUMNS = [
    "file_name",
    "input_path",
    "output_path",
    "input_size_mb",
    "output_size_mb",
    "status",
    "elapsed_seconds",
    "error_message",
    "requested_engine",
    "actual_engine",
    "row_count",
    "bad_row_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PKG24S4 TSV/TSV.GZ files to Parquet without pandas."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to PKG24S4 YAML config.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional file names or paths under raw_dir to convert.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N rows per file for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Parquet outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing Parquet files.",
    )
    parser.add_argument(
        "--compression",
        default=None,
        help="Override config parquet.compression.",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "duckdb", "pyarrow-stream"),
        default="auto",
        help="Conversion engine. Defaults to auto.",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Skip rows that cannot be parsed. Use only when data loss is acceptable.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable DuckDB parallel CSV scanner. Disabled by default for null_padding with quoted newlines.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB thread count. Defaults to 1 to reduce memory peak.",
    )
    parser.add_argument(
        "--memory-limit",
        default="20GB",
        help="DuckDB memory_limit setting. Defaults to 20GB.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/interim/duckdb_tmp"),
        help="DuckDB temp_directory for spill files.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=32768,
        help="Parquet ROW_GROUP_SIZE. Defaults to 32768.",
    )
    parser.add_argument(
        "--stream-threshold-mb",
        type=float,
        default=4096,
        help="Use pyarrow-stream in auto mode when input size is at least this many MB.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Rows per pyarrow-stream batch. Defaults to 50000.",
    )
    parser.add_argument(
        "--bad-rows-dir",
        type=Path,
        default=Path("data/interim/bad_rows"),
        help="Directory for pyarrow-stream rows with too many fields.",
    )
    parser.add_argument(
        "--fail-on-bad-rows",
        action="store_true",
        help="Fail pyarrow-stream conversion when rows have too many fields.",
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


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parquet_name(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".tsv.gz"):
        return name[: -len(".tsv.gz")] + ".parquet"
    if name.endswith(".tsv"):
        return name[: -len(".tsv")] + ".parquet"
    return input_path.stem + ".parquet"


def file_size_mb(path: Path) -> str:
    if not path.exists():
        return ""
    return f"{file_size_mb_float(path):.3f}"


def file_size_mb_float(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def normalize_encoding(encoding: str) -> tuple[str, str]:
    if encoding.lower() in {"utf8-lossy", "utf-8-lossy"}:
        return "utf-8", "replace"
    return encoding, "strict"


def find_tsv_files(raw_dir: Path, patterns: dict[str, str]) -> list[Path]:
    matched: set[Path] = set()
    for key in ("raw_tsv_gz", "raw_tsv"):
        pattern = patterns.get(key)
        if pattern:
            matched.update(raw_dir.rglob(pattern))
    if not matched:
        matched.update(raw_dir.rglob("*.tsv.gz"))
        matched.update(raw_dir.rglob("*.tsv"))
    return sorted(path for path in matched if path.is_file())


def select_requested_files(raw_dir: Path, all_files: list[Path], requested: list[str] | None) -> list[Path]:
    if not requested:
        return all_files

    by_name: dict[str, list[Path]] = {}
    by_relative: dict[str, Path] = {}
    for file_path in all_files:
        by_name.setdefault(file_path.name, []).append(file_path)
        by_relative[str(file_path.relative_to(raw_dir))] = file_path

    selected: list[Path] = []
    for value in requested:
        requested_path = Path(value)
        candidates: list[Path] = []

        if requested_path.is_absolute() and requested_path.exists():
            candidates = [requested_path.resolve()]
        elif value in by_relative:
            candidates = [by_relative[value]]
        elif requested_path.name in by_name:
            candidates = by_name[requested_path.name]

        if not candidates:
            raise FileNotFoundError(f"Requested file was not found under raw_dir: {value}")
        if len(candidates) > 1:
            matches = ", ".join(str(path.relative_to(raw_dir)) for path in candidates)
            raise ValueError(f"Requested file name is ambiguous: {value}. Matches: {matches}")
        selected.append(candidates[0])

    return selected


def convert_with_duckdb(
    input_path: Path,
    output_path: Path,
    separator: str,
    compression: str,
    limit: int | None,
    ignore_errors: bool,
    parallel: bool,
    threads: int,
    memory_limit: str,
    temp_dir: Path,
    row_group_size: int,
) -> None:
    try:
        import duckdb
    except ModuleNotFoundError as error:
        raise RuntimeError("duckdb is not installed; install duckdb or use --engine polars") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    limit_clause = f" LIMIT {limit}" if limit is not None else ""
    query = (
        "COPY ("
        "SELECT * FROM read_csv("
        f"{sql_literal(str(input_path))}, "
        f"delim={sql_literal(separator)}, "
        "header=true, "
        "all_varchar=true, "
        "strict_mode=false, "
        "null_padding=true, "
        "quote='', "
        "escape='', "
        f"ignore_errors={'true' if ignore_errors else 'false'}, "
        f"parallel={'true' if parallel else 'false'}, "
        "compression='auto'"
        f"){limit_clause}"
        ") TO "
        f"{sql_literal(str(output_path))} "
        f"(FORMAT PARQUET, COMPRESSION {sql_literal(compression.upper())}, "
        f"ROW_GROUP_SIZE {row_group_size})"
    )
    with duckdb.connect() as connection:
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit={sql_literal(memory_limit)}")
        connection.execute(f"SET temp_directory={sql_literal(str(temp_dir))}")
        connection.execute(query)


def open_text_stream(path: Path, encoding: str, errors: str) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding=encoding, errors=errors, newline="")
    return path.open("r", encoding=encoding, errors=errors, newline="")


def bad_rows_name(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".tsv.gz"):
        return name[: -len(".tsv.gz")] + ".bad_rows.tsv"
    if name.endswith(".tsv"):
        return name[: -len(".tsv")] + ".bad_rows.tsv"
    return input_path.stem + ".bad_rows.tsv"


def convert_empty_fields_to_null(fields: list[str]) -> list[str | None]:
    return [field if field != "" else None for field in fields]


def write_pyarrow_batch(writer: Any, columns: list[str], batch_rows: list[list[str | None]]) -> None:
    import pyarrow as pa

    arrays = []
    for column_index in range(len(columns)):
        arrays.append(pa.array((row[column_index] for row in batch_rows), type=pa.string()))
    table = pa.Table.from_arrays(arrays, names=columns)
    writer.write_table(table)


def convert_with_pyarrow_stream(
    input_path: Path,
    output_path: Path,
    separator: str,
    compression: str,
    batch_size: int,
    bad_rows_dir: Path,
    fail_on_bad_rows: bool,
    limit: int | None,
    encoding: str,
    errors: str,
) -> tuple[int, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("pyarrow is not installed; install pyarrow or use --engine duckdb") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    bad_rows_path = bad_rows_dir / bad_rows_name(input_path)
    if bad_rows_path.exists():
        bad_rows_path.unlink()

    writer = None
    bad_rows_file: TextIO | None = None
    row_count = 0
    bad_row_count = 0
    batch_rows: list[list[str | None]] = []

    try:
        with open_text_stream(input_path, encoding, errors) as input_file:
            header_line = input_file.readline()
            if header_line == "":
                raise ValueError(f"Input file is empty: {input_path}")

            columns = header_line.rstrip("\r\n").split(separator)
            schema = pa.schema([(column, pa.string()) for column in columns])
            writer = pq.ParquetWriter(output_path, schema=schema, compression=compression.lower())

            for line_number, line in enumerate(input_file, start=2):
                if limit is not None and row_count >= limit:
                    break

                raw_line = line.rstrip("\r\n")
                fields = raw_line.split(separator)
                if len(fields) > len(columns):
                    bad_row_count += 1
                    bad_rows_dir.mkdir(parents=True, exist_ok=True)
                    if bad_rows_file is None:
                        bad_rows_file = bad_rows_path.open("w", encoding="utf-8", newline="")
                        bad_rows_file.write("line_number\traw_line\n")
                    bad_rows_file.write(f"{line_number}\t{raw_line}\n")
                    if fail_on_bad_rows:
                        raise ValueError(
                            f"Line {line_number} has {len(fields)} fields; expected {len(columns)}"
                        )
                    continue

                if len(fields) < len(columns):
                    fields.extend([None] * (len(columns) - len(fields)))

                batch_rows.append(convert_empty_fields_to_null(fields))
                row_count += 1

                if len(batch_rows) >= batch_size:
                    write_pyarrow_batch(writer, columns, batch_rows)
                    batch_rows.clear()

            if batch_rows:
                write_pyarrow_batch(writer, columns, batch_rows)
                batch_rows.clear()
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        if bad_rows_file is not None:
            bad_rows_file.close()
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()
        if bad_rows_file is not None and not bad_rows_file.closed:
            bad_rows_file.close()

    return row_count, bad_row_count


def choose_engine(requested_engine: str, input_size_mb: float, stream_threshold_mb: float) -> str:
    if requested_engine != "auto":
        return requested_engine
    if input_size_mb >= stream_threshold_mb:
        return "pyarrow-stream"
    return "duckdb"


def convert_one(
    input_path: Path,
    output_path: Path,
    separator: str,
    compression: str,
    engine: str,
    limit: int | None,
    overwrite: bool,
    dry_run: bool,
    ignore_errors: bool,
    parallel: bool,
    threads: int,
    memory_limit: str,
    temp_dir: Path,
    row_group_size: int,
    requested_engine: str,
    actual_engine: str,
    batch_size: int,
    bad_rows_dir: Path,
    fail_on_bad_rows: bool,
    encoding: str,
    errors: str,
) -> dict[str, str]:
    start = time.perf_counter()
    status = "success"
    error_message = ""
    input_size = file_size_mb(input_path)
    output_size = ""
    row_count = ""
    bad_row_count = ""

    try:
        if output_path.exists() and not overwrite:
            status = "skipped"
        elif dry_run:
            status = "dry_run"
        else:
            if actual_engine == "duckdb":
                convert_with_duckdb(
                    input_path,
                    output_path,
                    separator,
                    compression,
                    limit,
                    ignore_errors,
                    parallel,
                    threads,
                    memory_limit,
                    temp_dir,
                    row_group_size,
                )
            elif actual_engine == "pyarrow-stream":
                pyarrow_row_count, pyarrow_bad_row_count = convert_with_pyarrow_stream(
                    input_path,
                    output_path,
                    separator,
                    compression,
                    batch_size,
                    bad_rows_dir,
                    fail_on_bad_rows,
                    limit,
                    encoding,
                    errors,
                )
                row_count = str(pyarrow_row_count)
                bad_row_count = str(pyarrow_bad_row_count)
            else:
                raise ValueError(f"Unsupported actual engine: {actual_engine}")
            output_size = file_size_mb(output_path)
    except Exception as error:
        status = "error"
        error_message = str(error)

    elapsed = time.perf_counter() - start
    result = {
        "file_name": input_path.name,
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "input_size_mb": input_size,
        "output_size_mb": output_size,
        "status": status,
        "elapsed_seconds": f"{elapsed:.3f}",
        "error_message": error_message,
        "requested_engine": requested_engine,
        "actual_engine": actual_engine,
        "row_count": row_count,
        "bad_row_count": bad_row_count,
    }
    print_conversion_result(result)
    return result


def print_conversion_result(result: dict[str, str]) -> None:
    print(
        f"{result['status']:<8} input={result['input_path']} "
        f"output={result['output_path']} "
        f"input_mb={result['input_size_mb']} "
        f"output_mb={result['output_size_mb'] or '-'} "
        f"requested_engine={result['requested_engine']} "
        f"actual_engine={result['actual_engine']} "
        f"row_count={result['row_count'] or '-'} "
        f"bad_rows={result['bad_row_count'] or '-'} "
        f"elapsed_s={result['elapsed_seconds']}"
    )
    if result["error_message"]:
        print(f"  error: {result['error_message']}")


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be greater than or equal to 0")
    if args.threads < 1:
        raise ValueError("--threads must be greater than or equal to 1")
    if args.row_group_size < 1:
        raise ValueError("--row-group-size must be greater than or equal to 1")
    if args.stream_threshold_mb < 0:
        raise ValueError("--stream-threshold-mb must be greater than or equal to 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than or equal to 1")

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)

    paths_config = config.get("paths", {})
    read_options = config.get("read_options", {})
    patterns = config.get("file_patterns", {})
    parquet_config = config.get("parquet", {})

    raw_dir = resolve_config_path(paths_config["raw_dir"], project_root).resolve()
    parquet_dir = resolve_config_path(paths_config["parquet_dir"], project_root).resolve()
    manifest_path = resolve_config_path(
        paths_config.get("parquet_manifest", "data/interim/parquet_manifest.csv"),
        project_root,
    ).resolve()
    separator = read_options.get("separator", "\t")
    encoding, errors = normalize_encoding(read_options.get("encoding", "utf-8"))
    compression = args.compression or parquet_config.get("compression", "zstd")
    temp_dir = (args.temp_dir if args.temp_dir.is_absolute() else project_root / args.temp_dir).resolve()
    bad_rows_dir = (
        args.bad_rows_dir if args.bad_rows_dir.is_absolute() else project_root / args.bad_rows_dir
    ).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    all_files = find_tsv_files(raw_dir, patterns)
    selected_files = select_requested_files(raw_dir, all_files, args.files)
    print(
        json.dumps(
            {
                "engine": args.engine,
                "compression": compression,
                "raw_dir": str(raw_dir),
                "parquet_dir": str(parquet_dir),
                "manifest": str(manifest_path),
                "files": len(selected_files),
                "limit": args.limit,
                "dry_run": args.dry_run,
                "overwrite": args.overwrite,
                "ignore_errors": args.ignore_errors,
                "parallel": args.parallel,
                "threads": args.threads,
                "memory_limit": args.memory_limit,
                "temp_dir": str(temp_dir),
                "row_group_size": args.row_group_size,
                "batch_size": args.batch_size,
                "bad_rows_dir": str(bad_rows_dir),
                "fail_on_bad_rows": args.fail_on_bad_rows,
            },
            ensure_ascii=False,
        )
    )

    rows: list[dict[str, str]] = []
    for input_path in selected_files:
        output_path = parquet_dir / parquet_name(input_path)
        input_size = file_size_mb_float(input_path)
        actual_engine = choose_engine(args.engine, input_size, args.stream_threshold_mb)
        if args.dry_run:
            would_skip = output_path.exists() and not args.overwrite
            print(
                "dry_run_plan "
                f"file_name={input_path.name} "
                f"input_size_mb={input_size:.3f} "
                f"output_path={output_path.resolve()} "
                f"requested_engine={args.engine} "
                f"actual_engine={actual_engine} "
                f"output_exists={output_path.exists()} "
                f"would_skip={would_skip}"
            )
        rows.append(
            convert_one(
                input_path=input_path,
                output_path=output_path,
                separator=separator,
                compression=compression,
                engine=actual_engine,
                limit=args.limit,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                ignore_errors=args.ignore_errors,
                parallel=args.parallel,
                threads=args.threads,
                memory_limit=args.memory_limit,
                temp_dir=temp_dir,
                row_group_size=args.row_group_size,
                requested_engine=args.engine,
                actual_engine=actual_engine,
                batch_size=args.batch_size,
                bad_rows_dir=bad_rows_dir,
                fail_on_bad_rows=args.fail_on_bad_rows,
                encoding=encoding,
                errors=errors,
            )
        )

    write_manifest(manifest_path, rows)
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
