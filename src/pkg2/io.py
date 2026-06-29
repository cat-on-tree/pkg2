"""I/O, DuckDB, and lightweight reporting utilities.

This module contains small reusable helpers for the algorithm-stage pipeline.
The goal is to avoid copying DuckDB setup, SQL escaping, CSV writing, output
directory handling, and markdown table rendering logic across scripts.

The utilities here are intentionally lightweight and dependency-minimal. They
are used by dataset construction, baseline scripts, and future knowledge-unit
pipeline components.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


def sql_literal(value: str | Path | int | float | bool | None) -> str:
    """Return a SQL literal string safe for simple DuckDB query construction.

    This function is intended for controlled internal values such as file paths,
    memory-limit strings, and fixed configuration values. It is not a general
    SQL query builder.
    """

    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    if isinstance(value, int | float):
        return str(value)

    return "'" + str(value).replace("'", "''") + "'"


def qident(identifier: str) -> str:
    """Quote a SQL identifier for DuckDB."""

    return '"' + identifier.replace('"', '""') + '"'


def scan_sql(path: str | Path) -> str:
    """Return a DuckDB read_parquet(...) expression for a file path."""

    return f"read_parquet({sql_literal(Path(path))})"


def connect_duckdb(
    *,
    threads: int = 1,
    memory_limit: str = "20GB",
    temp_dir: str | Path = "data/interim/duckdb_tmp",
    database: str | Path | None = None,
) -> Any:
    """Create and configure a DuckDB connection.

    Parameters
    ----------
    threads:
        Number of DuckDB execution threads.
    memory_limit:
        DuckDB memory limit, for example ``"20GB"``.
    temp_dir:
        Directory used by DuckDB for temporary spill files.
    database:
        Optional DuckDB database path. If omitted, an in-memory database is used.
    """

    if threads < 1:
        raise ValueError("threads must be greater than or equal to 1")

    import duckdb

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if database is None:
        connection = duckdb.connect()
    else:
        connection = duckdb.connect(str(database))

    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET threads={int(threads)}")
    connection.execute(f"SET memory_limit={sql_literal(memory_limit)}")
    connection.execute(f"SET temp_directory={sql_literal(temp_dir)}")

    return connection


def execute(connection: Any, sql: str) -> None:
    """Execute a SQL statement."""

    connection.execute(sql)


def fetch_dicts(connection: Any, sql: str) -> list[dict[str, Any]]:
    """Execute SQL and return rows as dictionaries."""

    cursor = connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def fetch_one_dict(connection: Any, sql: str) -> dict[str, Any] | None:
    """Execute SQL and return the first row as a dictionary, or None."""

    cursor = connection.execute(sql)
    row = cursor.fetchone()
    if row is None:
        return None

    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row))


def table_exists(connection: Any, table_name: str) -> bool:
    """Return whether a table or view exists in the current DuckDB connection."""

    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchone()

    return bool(row and row[0] > 0)


def get_columns(connection: Any, table_name: str) -> set[str]:
    """Return column names for a table or view."""

    rows = connection.execute(f"DESCRIBE SELECT * FROM {qident(table_name)}").fetchall()
    return {str(row[0]) for row in rows}


def count_rows(connection: Any, table_name: str) -> int:
    """Return row count for a table or view."""

    return int(connection.execute(f"SELECT COUNT(*) FROM {qident(table_name)}").fetchone()[0])


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    """Write dictionaries to a CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Write JSON with UTF-8 encoding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=indent)


def read_json(path: str | Path) -> Any:
    """Read JSON with UTF-8 encoding."""

    with Path(path).open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a small list of dictionaries as a markdown table."""

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
            value_str = "" if value is None else str(value)
            value_str = value_str.replace("|", "\\|").replace("\n", " ")
            values.append(value_str)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def prepare_output_dir(output_dir: str | Path, *, overwrite: bool = False, allow_existing: bool = False) -> Path:
    """Create an output directory and optionally clear it first.

    Parameters
    ----------
    output_dir:
        Directory to create.
    overwrite:
        If true, remove an existing output directory before creating it again.
    allow_existing:
        If true, allow writing into an existing non-empty directory. This is
        useful for report-only scripts but should usually be false for dataset
        construction.
    """

    output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()) and not allow_existing:
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def require_paths_exist(paths: Iterable[str | Path]) -> None:
    """Raise FileNotFoundError if any path does not exist."""

    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def copy_file(source: str | Path, destination: str | Path, *, overwrite: bool = True) -> None:
    """Copy one file, creating the destination directory if needed."""

    source = Path(source)
    destination = Path(destination)

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_text(path: str | Path) -> str:
    """Read UTF-8 text."""

    return Path(path).read_text(encoding="utf-8")


def normalize_string_list(values: Iterable[str] | None) -> list[str]:
    """Normalize an optional iterable of strings into a clean list."""

    if values is None:
        return []

    normalized = []
    for value in values:
        value = str(value).strip()
        if value:
            normalized.append(value)

    return normalized