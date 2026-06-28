"""Lightweight heuristic profiling for PKG24S4 Parquet tables.

This is exploratory metadata profiling, not a final schema contract. Join candidates
must be manually confirmed before any subgraph extraction.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG = Path("config/pkg24s4.yaml")
DEFAULT_SCHEMA_SUMMARY = Path("data/interim/schema_summary.csv")
DEFAULT_VALIDATION = Path("data/interim/parquet_validation.csv")
DEFAULT_OUTPUT_DIR = Path("data/interim/profile")

TABLE_PROFILE_COLUMNS = [
    "file_name",
    "table_name",
    "parquet_path",
    "row_count",
    "column_count",
    "parquet_size_mb",
    "actual_engine",
    "bad_row_count",
    "columns_json",
    "id_like_columns_json",
    "date_like_columns_json",
    "year_like_columns_json",
    "text_like_columns_json",
]
COLUMN_PROFILE_COLUMNS = [
    "file_name",
    "table_name",
    "column_name",
    "dtype",
    "inferred_kind",
    "sample_non_null_values_json",
    "null_count",
    "null_fraction",
    "null_fraction_basis",
    "sample_n_unique",
    "approx_n_unique",
]
LINK_PROFILE_COLUMNS = [
    "file_name",
    "table_name",
    "row_count",
    "id_columns_json",
    "likely_source_columns_json",
    "likely_target_columns_json",
    "duplicate_row_count",
    "duplicate_count_basis",
    "sample_edges_json",
]
ENTITY_CANDIDATE_COLUMNS = [
    "table_name",
    "file_name",
    "primary_id_candidates_json",
    "label_columns_json",
    "year_columns_json",
    "row_count",
]
JOIN_CANDIDATE_COLUMNS = [
    "source_table",
    "source_column",
    "target_table",
    "target_column",
    "confidence",
    "reason",
]

EXACT_ID_COLUMNS = {
    "id",
    "pmid",
    "pmcid",
    "doi",
    "aid",
    "s2aid",
    "entityid",
    "entity_id",
    "projectid",
    "project_id",
    "patentid",
    "patent_id",
    "nctid",
    "nct_id",
    "clinicaltrialid",
    "clinical_trial_id",
    "authorid",
    "author_id",
    "journalid",
    "journal_id",
    "nlmuniqueid",
    "descriptorui",
}
TEXT_PATTERNS = [
    "title",
    "abstract",
    "mention",
    "name",
    "affiliation",
    "journal",
    "keyword",
    "mesh",
    "descriptor",
    "note",
]
ENTITY_TABLE_HINTS = [
    "Papers",
    "Authors",
    "Affiliations",
    "BioEntities",
    "Projects",
    "ClinicalTrials",
    "Patents",
    "Journals",
    "Investigators",
    "Inventors",
    "Assignees",
    "Descriptor",
    "Mesh",
    "Chemical",
    "Keyword",
    "Grant",
    "GeneSymbol",
    "PublicationType",
    "Qualifier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile PKG24S4 Parquet tables lightly.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema-summary", type=Path, default=DEFAULT_SCHEMA_SUMMARY)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rows", type=int, default=10000)
    parser.add_argument("--exact-null-count", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--exact-distinct", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--max-exact-distinct-rows", type=int, default=5000000)
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def table_name_from_file(file_name: str) -> str:
    if file_name.endswith(".tsv.gz"):
        return file_name[: -len(".tsv.gz")]
    if file_name.endswith(".parquet"):
        return file_name[: -len(".parquet")]
    return Path(file_name).stem


def parquet_name(file_name: str) -> str:
    return table_name_from_file(file_name) + ".parquet"


def file_size_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.3f}"


def json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def is_id_like(column: str) -> bool:
    lower = column.lower()
    if lower in EXACT_ID_COLUMNS:
        return True
    return column.endswith(("Id", "ID", "_id"))


def is_date_like(column: str) -> bool:
    lower = column.lower()
    return "date" in lower or "time" in lower


def is_year_like(column: str) -> bool:
    lower = column.lower()
    return "year" in lower or column == "PubYear"


def is_text_like(column: str) -> bool:
    lower = column.lower()
    return any(pattern in lower for pattern in TEXT_PATTERNS)


def infer_kind(column: str) -> str:
    if is_id_like(column):
        return "id"
    if is_date_like(column):
        return "date"
    if is_year_like(column):
        return "year"
    if is_text_like(column):
        return "text"
    return "other"


def read_parquet_metadata(path: Path) -> tuple[list[str], list[str], int]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "pyarrow is required for profiling Parquet metadata and samples. "
            "Activate the PKG24S4 environment that has pyarrow installed."
        ) from error

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    columns = schema.names
    dtypes = [str(schema.field(column).type) for column in columns]
    row_count = parquet_file.metadata.num_rows
    return columns, dtypes, row_count


def read_sample(path: Path, columns: list[str], sample_rows: int) -> tuple[dict[str, list[Any]], int]:
    import pyarrow.parquet as pq

    sample: dict[str, list[Any]] = {column: [] for column in columns}
    if sample_rows <= 0:
        return sample, 0
    remaining = sample_rows
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=min(sample_rows, 65536), columns=columns):
        data = batch.to_pydict()
        batch_len = batch.num_rows
        take = min(remaining, batch_len)
        for column in columns:
            sample[column].extend(data[column][:take])
        remaining -= take
        if remaining <= 0:
            break
    sample_count = sample_rows - remaining
    return sample, sample_count


def exact_null_counts(path: Path, columns: list[str]) -> dict[str, int]:
    import pyarrow.parquet as pq

    counts = {column: 0 for column in columns}
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=65536, columns=columns):
        for index, column in enumerate(columns):
            counts[column] += batch.column(index).null_count
    return counts


def exact_distinct_counts(path: Path, columns: list[str]) -> dict[str, int]:
    import pyarrow.parquet as pq

    values = {column: set() for column in columns}
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=65536, columns=columns):
        data = batch.to_pydict()
        for column in columns:
            values[column].update(data[column])
    return {column: len(column_values) for column, column_values in values.items()}


def non_null_sample(values: list[Any], limit: int = 10) -> list[Any]:
    output = []
    for value in values:
        if value is not None:
            output.append(value)
        if len(output) >= limit:
            break
    return output


def sample_n_unique(values: list[Any]) -> int:
    return len({value for value in values if value is not None})


def load_table_inputs(
    schema_summary_path: Path,
    validation_path: Path,
    parquet_dir: Path,
) -> list[dict[str, Any]]:
    schema_rows = read_csv_rows(schema_summary_path)
    validation_by_file = {row["file_name"]: row for row in read_csv_rows(validation_path)}
    tables = []
    for schema_row in schema_rows:
        file_name = schema_row["file_name"]
        validation_row = validation_by_file.get(file_name, {})
        parquet_path = validation_row.get("expected_parquet_path") or str(
            (parquet_dir / parquet_name(file_name)).resolve()
        )
        tables.append(
            {
                "file_name": file_name,
                "table_name": table_name_from_file(file_name),
                "parquet_path": Path(parquet_path),
                "schema_row": schema_row,
                "validation_row": validation_row,
            }
        )
    return tables


def profile_tables(
    tables: list[dict[str, Any]],
    sample_rows: int,
    exact_null_count: bool,
    exact_distinct: bool,
    max_exact_distinct_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    table_profiles = []
    column_profiles = []
    table_context: dict[str, dict[str, Any]] = {}

    for table in tables:
        path = table["parquet_path"]
        file_name = table["file_name"]
        table_name = table["table_name"]
        validation_row = table["validation_row"]
        if not path.exists():
            continue

        columns, dtypes, row_count = read_parquet_metadata(path)
        sample, sample_count = read_sample(path, columns, sample_rows)
        null_counts = exact_null_counts(path, columns) if exact_null_count else {}
        exact_distinct_for_table = exact_distinct and row_count <= max_exact_distinct_rows
        distinct_counts = exact_distinct_counts(path, columns) if exact_distinct_for_table else {}

        id_columns = [column for column in columns if is_id_like(column)]
        date_columns = [column for column in columns if is_date_like(column)]
        year_columns = [column for column in columns if is_year_like(column)]
        text_columns = [column for column in columns if is_text_like(column)]

        table_profiles.append(
            {
                "file_name": file_name,
                "table_name": table_name,
                "parquet_path": str(path.resolve()),
                "row_count": row_count,
                "column_count": len(columns),
                "parquet_size_mb": file_size_mb(path),
                "actual_engine": validation_row.get("actual_engine", ""),
                "bad_row_count": validation_row.get("bad_row_count", ""),
                "columns_json": json_list(columns),
                "id_like_columns_json": json_list(id_columns),
                "date_like_columns_json": json_list(date_columns),
                "year_like_columns_json": json_list(year_columns),
                "text_like_columns_json": json_list(text_columns),
            }
        )

        for column, dtype in zip(columns, dtypes):
            values = sample[column]
            if exact_null_count:
                null_count = null_counts[column]
                null_fraction = null_count / row_count if row_count else 0
                null_fraction_basis = "exact"
            else:
                null_count = ""
                null_fraction = values.count(None) / sample_count if sample_count else ""
                null_fraction_basis = "sample_first_n"
            if exact_distinct:
                approx_n_unique = (
                    distinct_counts[column] if exact_distinct_for_table else "skipped_large_table"
                )
                sample_unique = ""
            else:
                approx_n_unique = ""
                sample_unique = sample_n_unique(values)
            column_profiles.append(
                {
                    "file_name": file_name,
                    "table_name": table_name,
                    "column_name": column,
                    "dtype": dtype,
                    "inferred_kind": infer_kind(column),
                    "sample_non_null_values_json": json_list(non_null_sample(values)),
                    "null_count": null_count,
                    "null_fraction": null_fraction,
                    "null_fraction_basis": null_fraction_basis,
                    "sample_n_unique": sample_unique,
                    "approx_n_unique": approx_n_unique,
                }
            )

        table_context[table_name] = {
            "file_name": file_name,
            "path": path,
            "row_count": row_count,
            "columns": columns,
            "id_columns": id_columns,
            "year_columns": year_columns,
            "text_columns": text_columns,
            "sample": sample,
        }

    return table_profiles, column_profiles, table_context


def is_link_table(table_name: str, columns: list[str], id_columns: list[str]) -> bool:
    lower_name = table_name.lower()
    if "link" in lower_name:
        return True
    if any("link" in column.lower() for column in columns):
        return True
    return len(columns) <= 8 and len(id_columns) >= 2


def duplicate_count_from_sample(
    sample: dict[str, list[Any]], columns: list[str], row_count: int
) -> tuple[str, str]:
    if not columns:
        return "", "not_computed"
    sample_len = len(next(iter(sample.values()), []))
    if sample_len != row_count:
        return "", "not_computed"
    rows = [tuple(sample[column][index] for column in columns) for index in range(sample_len)]
    return str(len(rows) - len(set(rows))), "exact"


def sample_edges(sample: dict[str, list[Any]], id_columns: list[str], limit: int = 20) -> list[dict[str, Any]]:
    if len(id_columns) < 2:
        return []
    sample_len = len(next(iter(sample.values()), []))
    edges = []
    for index in range(min(sample_len, limit)):
        edges.append({column: sample[column][index] for column in id_columns})
    return edges


def build_link_profiles(table_context: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for table_name, context in table_context.items():
        columns = context["columns"]
        id_columns = context["id_columns"]
        if not is_link_table(table_name, columns, id_columns):
            continue
        midpoint = max(1, len(id_columns) // 2)
        duplicate_row_count, duplicate_count_basis = duplicate_count_from_sample(
            context["sample"], id_columns, context["row_count"]
        )
        profiles.append(
            {
                "file_name": context["file_name"],
                "table_name": table_name,
                "row_count": context["row_count"],
                "id_columns_json": json_list(id_columns),
                "likely_source_columns_json": json_list(id_columns[:midpoint]),
                "likely_target_columns_json": json_list(id_columns[midpoint:]),
                "duplicate_row_count": duplicate_row_count,
                "duplicate_count_basis": duplicate_count_basis,
                "sample_edges_json": json.dumps(
                    sample_edges(context["sample"], id_columns), ensure_ascii=False
                ),
            }
        )
    return profiles


def is_entity_candidate(table_name: str) -> bool:
    return any(hint.lower() in table_name.lower() for hint in ENTITY_TABLE_HINTS)


def build_entity_candidates(table_context: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for table_name, context in table_context.items():
        if not is_entity_candidate(table_name):
            continue
        rows.append(
            {
                "table_name": table_name,
                "file_name": context["file_name"],
                "primary_id_candidates_json": json_list(context["id_columns"]),
                "label_columns_json": json_list(context["text_columns"]),
                "year_columns_json": json_list(context["year_columns"]),
                "row_count": context["row_count"],
            }
        )
    return rows


def semantic_key(table_name: str, column: str) -> str:
    lower = column.lower()
    table_lower = table_name.lower()
    if lower in {"pmid", "paperid"} or "paperid" in lower:
        return "paper"
    if lower == "id" and "paper" in table_lower:
        return "paper"
    if "entityid" in lower or "bioentity" in lower:
        return "entity"
    if "project" in lower:
        return "project"
    if "clinicaltrial" in lower or "nct" in lower or "trial" in lower:
        return "clinical_trial"
    if "patent" in lower:
        return "patent"
    if lower in {"aid", "s2aid", "authorid"} or "author" in lower:
        return "author"
    if "journal" in lower or "nlmuniqueid" in lower:
        return "journal"
    if "descriptor" in lower and ("ui" in lower or "id" in lower):
        return "descriptor"
    return ""


def confidence_for(source_column: str, target_column: str, source_key: str, target_key: str) -> tuple[str, str]:
    source_lower = source_column.lower()
    target_lower = target_column.lower()
    generic_warning = " Generic id columns may be table-local and require manual confirmation."
    if source_lower == "id" or target_lower == "id":
        return "low", f"Generic id match.{generic_warning}"
    explicit_high_keys = {"paper", "entity", "project", "clinical_trial", "patent"}
    explicit_medium_keys = {"author", "journal", "descriptor"}
    if source_lower == target_lower and source_key in explicit_high_keys:
        return "high", f"Exact column name match for {source_key}"
    if source_lower == target_lower and source_key in explicit_medium_keys:
        return "medium", f"Exact column name match for {source_key}"
    if source_key == target_key:
        return "medium", f"Shared semantic key: {source_key}"
    return "low", "Weak column-name heuristic"


def build_join_candidates(table_context: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    columns_by_key: dict[str, list[tuple[str, str]]] = {}
    for table_name, context in table_context.items():
        for column in context["columns"]:
            key = semantic_key(table_name, column)
            if key:
                columns_by_key.setdefault(key, []).append((table_name, column))

    rows = []
    seen = set()
    for key, endpoints in columns_by_key.items():
        for source_table, source_column in endpoints:
            for target_table, target_column in endpoints:
                if source_table == target_table and source_column == target_column:
                    continue
                edge_key = (source_table, source_column, target_table, target_column)
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                confidence, reason = confidence_for(source_column, target_column, key, key)
                rows.append(
                    {
                        "source_table": source_table,
                        "source_column": source_column,
                        "target_table": target_table,
                        "target_column": target_column,
                        "confidence": confidence,
                        "reason": reason,
                    }
                )
    return rows


def print_summary(
    table_profiles: list[dict[str, Any]],
    link_profiles: list[dict[str, Any]],
    entity_candidates: list[dict[str, Any]],
    output_paths: list[Path],
) -> None:
    total_rows = sum(int(row["row_count"]) for row in table_profiles)
    largest_tables = sorted(table_profiles, key=lambda row: int(row["row_count"]), reverse=True)[:10]
    largest_files = sorted(
        table_profiles, key=lambda row: float(row["parquet_size_mb"]), reverse=True
    )[:10]

    print("PKG24S4 Parquet profile summary")
    print(f"Total tables: {len(table_profiles)}")
    print(f"Total rows across tables: {total_rows}")
    print("\nLargest 10 tables by row_count")
    for row in largest_tables:
        print(f"{row['table_name']}\t{row['row_count']}")
    print("\nLargest 10 files by parquet_size_mb")
    for row in largest_files:
        print(f"{row['table_name']}\t{row['parquet_size_mb']} MB")
    print("\nDetected link tables")
    for row in link_profiles:
        print(row["table_name"])
    print("\nDetected entity candidate tables")
    for row in entity_candidates:
        print(row["table_name"])
    print("\nOutput files written")
    for path in output_paths:
        print(path)


def main() -> None:
    args = parse_args()
    if args.sample_rows < 0:
        raise ValueError("--sample-rows must be greater than or equal to 0")
    if args.max_exact_distinct_rows < 0:
        raise ValueError("--max-exact-distinct-rows must be greater than or equal to 0")

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    parquet_dir = resolve_path(config["paths"]["parquet_dir"], project_root).resolve()
    schema_summary_path = resolve_path(args.schema_summary, project_root).resolve()
    validation_path = resolve_path(args.validation, project_root).resolve()
    output_dir = resolve_path(args.output_dir, project_root).resolve()

    tables = load_table_inputs(schema_summary_path, validation_path, parquet_dir)
    table_profiles, column_profiles, table_context = profile_tables(
        tables=tables,
        sample_rows=args.sample_rows,
        exact_null_count=args.exact_null_count,
        exact_distinct=args.exact_distinct,
        max_exact_distinct_rows=args.max_exact_distinct_rows,
    )
    link_profiles = build_link_profiles(table_context)
    entity_candidates = build_entity_candidates(table_context)
    join_candidates = build_join_candidates(table_context)

    output_paths = [
        output_dir / "table_profile.csv",
        output_dir / "column_profile.csv",
        output_dir / "link_table_profile.csv",
        output_dir / "entity_candidate_tables.csv",
        output_dir / "join_candidate_edges.csv",
    ]
    write_csv(output_paths[0], table_profiles, TABLE_PROFILE_COLUMNS)
    write_csv(output_paths[1], column_profiles, COLUMN_PROFILE_COLUMNS)
    write_csv(output_paths[2], link_profiles, LINK_PROFILE_COLUMNS)
    write_csv(output_paths[3], entity_candidates, ENTITY_CANDIDATE_COLUMNS)
    write_csv(output_paths[4], join_candidates, JOIN_CANDIDATE_COLUMNS)
    print_summary(table_profiles, link_profiles, entity_candidates, output_paths)


if __name__ == "__main__":
    main()
