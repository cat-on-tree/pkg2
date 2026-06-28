"""Aggregate mention-level BioEntity edges into pair-level weighted edges.

This script reads an extracted patent-centered/domain subgraph directory and
creates a new graph directory where BioEntity mention edges are aggregated by:

    source_id + target_id + edge_type

The goal is to convert repeated mention-level edges such as:

    Paper A -> diabetes
    Paper A -> diabetes
    Paper A -> diabetes

into one weighted pair-level edge:

    Paper A -> diabetes, mention_count = 3

This script does not modify the input graph. It writes a new output graph.

Recommended input:
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2/

Recommended output:
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/

The output keeps the same node/edge file names expected by
scripts/07_validate_extracted_subgraph.py.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Any


NODE_FILES = {
    "nodes_patent": "nodes_patent.parquet",
    "nodes_paper": "nodes_paper.parquet",
    "nodes_clinicaltrial": "nodes_clinicaltrial.parquet",
    "nodes_project": "nodes_project.parquet",
    "nodes_bioentity": "nodes_bioentity.parquet",
}

NON_BIOENTITY_EDGE_FILES = {
    "edges_patent_paper": "edges_patent_paper.parquet",
    "edges_patent_project": "edges_patent_project.parquet",
    "edges_paper_trial": "edges_paper_trial.parquet",
    "edges_paper_project": "edges_paper_project.parquet",
    "edges_trial_project": "edges_trial_project.parquet",
}

BIOENTITY_EDGE_FILES = {
    "edges_patent_bioentity": {
        "file_name": "edges_patent_bioentity.parquet",
        "carrier_id_column": "PatentId",
        "carrier_type": "Patent",
    },
    "edges_paper_bioentity": {
        "file_name": "edges_paper_bioentity.parquet",
        "carrier_id_column": "PMID",
        "carrier_type": "Paper",
    },
    "edges_trial_bioentity": {
        "file_name": "edges_trial_bioentity.parquet",
        "carrier_id_column": "nct_id",
        "carrier_type": "ClinicalTrial",
    },
}

OUTPUT_TABLES = [
    ("node", "nodes_patent", "nodes_patent.parquet", "Patent nodes copied from input graph."),
    ("node", "nodes_paper", "nodes_paper.parquet", "Paper nodes copied from input graph."),
    ("node", "nodes_clinicaltrial", "nodes_clinicaltrial.parquet", "ClinicalTrial nodes copied from input graph."),
    ("node", "nodes_project", "nodes_project.parquet", "Project nodes copied from input graph."),
    ("node", "nodes_bioentity", "nodes_bioentity.parquet", "BioEntity nodes copied from input graph."),
    ("edge", "edges_patent_paper", "edges_patent_paper.parquet", "Patent-paper edges copied from input graph."),
    ("edge", "edges_patent_bioentity", "edges_patent_bioentity.parquet", "Aggregated Patent-BioEntity pair-level weighted edges."),
    ("edge", "edges_patent_project", "edges_patent_project.parquet", "Patent-project edges copied from input graph."),
    ("edge", "edges_paper_trial", "edges_paper_trial.parquet", "Paper-trial edges copied from input graph."),
    ("edge", "edges_paper_project", "edges_paper_project.parquet", "Paper-project edges copied from input graph."),
    ("edge", "edges_paper_bioentity", "edges_paper_bioentity.parquet", "Aggregated Paper-BioEntity pair-level weighted edges."),
    ("edge", "edges_trial_project", "edges_trial_project.parquet", "Trial-project edges copied from input graph."),
    ("edge", "edges_trial_bioentity", "edges_trial_bioentity.parquet", "Aggregated Trial-BioEntity pair-level weighted edges."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate mention-level BioEntity edges in an extracted subgraph.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2"),
        help="Input extracted graph directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated"),
        help="Output graph directory with aggregated BioEntity edges.",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="20GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("data/interim/duckdb_tmp"))
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def qident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def scan_sql(path: Path) -> str:
    return f"read_parquet({sql_literal(path)})"


def connect_duckdb(args: argparse.Namespace):
    import duckdb

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET memory_limit={sql_literal(args.memory_limit)}")
    connection.execute(f"SET temp_directory={sql_literal(args.temp_dir)}")
    return connection


def execute(connection: Any, sql: str) -> None:
    connection.execute(sql)


def count_table(connection: Any, table_name: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {qident(table_name)}").fetchone()[0])


def get_columns(connection: Any, table_name: str) -> set[str]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {qident(table_name)}").fetchall()
    return {str(row[0]) for row in rows}


def require_files(input_dir: Path) -> None:
    required = list(NODE_FILES.values()) + list(NON_BIOENTITY_EDGE_FILES.values())
    required += [value["file_name"] for value in BIOENTITY_EDGE_FILES.values()]

    missing = [str(input_dir / file_name) for file_name in required if not (input_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError("Missing required input graph files:\n" + "\n".join(missing))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_step(args: argparse.Namespace, index: int, total: int, label: str, fn) -> None:
    if not args.no_progress:
        print(f"[{index}/{total}] {label} ...", flush=True)
    start = time.time()
    fn()
    elapsed = time.time() - start
    if not args.no_progress:
        print(f"[{index}/{total}] {label} done in {elapsed:.1f}s", flush=True)


def copy_table_to_parquet(connection: Any, table_name: str, path: Path, compression: str) -> None:
    compression_sql = compression.upper()
    execute(
        connection,
        f"""
        COPY (
            SELECT * FROM {qident(table_name)}
        )
        TO {sql_literal(path)}
        (FORMAT PARQUET, COMPRESSION {compression_sql})
        """,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
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
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def create_input_views(connection: Any, input_dir: Path) -> None:
    for table_name, file_name in NODE_FILES.items():
        input_view_name = f"input_{table_name}"
        execute(
            connection,
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(input_view_name)} AS
            SELECT * FROM {scan_sql(input_dir / file_name)}
            """,
        )

    for table_name, file_name in NON_BIOENTITY_EDGE_FILES.items():
        input_view_name = f"input_{table_name}"
        execute(
            connection,
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(input_view_name)} AS
            SELECT * FROM {scan_sql(input_dir / file_name)}
            """,
        )

    for table_name, meta in BIOENTITY_EDGE_FILES.items():
        input_view_name = f"input_{table_name}"
        execute(
            connection,
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(input_view_name)} AS
            SELECT * FROM {scan_sql(input_dir / meta["file_name"])}
            """,
        )


def copy_node_and_non_bioentity_tables(connection: Any) -> None:
    for table_name in NODE_FILES:
        input_view_name = f"input_{table_name}"
        execute(
            connection,
            f"""
            CREATE TEMP TABLE {qident(table_name)} AS
            SELECT * FROM {qident(input_view_name)}
            """,
        )

    for table_name in NON_BIOENTITY_EDGE_FILES:
        input_view_name = f"input_{table_name}"
        execute(
            connection,
            f"""
            CREATE TEMP TABLE {qident(table_name)} AS
            SELECT * FROM {qident(input_view_name)}
            """,
        )


def column_exists_expr(columns: set[str], column: str, expression_if_exists: str, expression_if_missing: str) -> str:
    if column in columns:
        return expression_if_exists
    return expression_if_missing


def varchar_col(columns: set[str], column: str, alias: str = "e") -> str:
    if column in columns:
        return f"NULLIF(TRIM(CAST({alias}.{qident(column)} AS VARCHAR)), '')"
    return "CAST(NULL AS VARCHAR)"


def double_col(columns: set[str], column: str, alias: str = "e") -> str:
    if column in columns:
        return f"TRY_CAST({alias}.{qident(column)} AS DOUBLE)"
    return "CAST(NULL AS DOUBLE)"


def bigint_col(columns: set[str], column: str, alias: str = "e") -> str:
    if column in columns:
        return f"TRY_CAST({alias}.{qident(column)} AS BIGINT)"
    return "CAST(NULL AS BIGINT)"


def nonempty_col_pred(columns: set[str], column: str, alias: str = "e") -> str:
    if column in columns:
        return f"NULLIF(TRIM(CAST({alias}.{qident(column)} AS VARCHAR)), '') IS NOT NULL"
    return "FALSE"


def any_value_expr(columns: set[str], column: str, output_name: str | None = None, alias: str = "e") -> str:
    out = output_name or column
    if column in columns:
        return f"ANY_VALUE({alias}.{qident(column)}) AS {qident(out)}"
    return f"CAST(NULL AS VARCHAR) AS {qident(out)}"


def count_distinct_expr(columns: set[str], column: str, output_name: str, alias: str = "e") -> str:
    if column in columns:
        return f"COUNT(DISTINCT NULLIF(TRIM(CAST({alias}.{qident(column)} AS VARCHAR)), '')) AS {qident(output_name)}"
    return f"CAST(0 AS BIGINT) AS {qident(output_name)}"


def string_agg_distinct_expr(columns: set[str], column: str, output_name: str, alias: str = "e") -> str:
    if column in columns:
        value_expr = f"NULLIF(TRIM(CAST({alias}.{qident(column)} AS VARCHAR)), '')"
        return (
            f"STRING_AGG(DISTINCT {value_expr}, '|') "
            f"FILTER (WHERE {value_expr} IS NOT NULL) AS {qident(output_name)}"
        )
    return f"CAST(NULL AS VARCHAR) AS {qident(output_name)}"


def flag_expr(columns: set[str], column: str, output_name: str, alias: str = "e") -> str:
    if column in columns:
        return (
            f"MAX(CASE WHEN NULLIF(TRIM(CAST({alias}.{qident(column)} AS VARCHAR)), '') IS NOT NULL "
            f"THEN 1 ELSE 0 END) AS {qident(output_name)}"
        )
    return f"CAST(0 AS INTEGER) AS {qident(output_name)}"


def aggregate_bioentity_edge_table(
    connection: Any,
    input_table_name: str,
    output_table_name: str,
    carrier_id_column: str,
    carrier_type: str,
) -> dict[str, Any]:
    input_view = f"input_{input_table_name}"
    columns = get_columns(connection, input_view)

    source_table_expr = any_value_expr(columns, "source_table", "source_table")
    carrier_id_expr = (
        any_value_expr(columns, carrier_id_column, carrier_id_column)
        if carrier_id_column in columns
        else f"ANY_VALUE(e.source_id) AS {qident(carrier_id_column)}"
    )

    entity_id_expr = (
        any_value_expr(columns, "EntityId", "EntityId")
        if "EntityId" in columns
        else "ANY_VALUE(e.target_id) AS EntityId"
    )

    sample_mention_expr = any_value_expr(columns, "Mention", "sample_mention")
    mention_type_sample_expr = any_value_expr(columns, "Type", "sample_mention_type")
    distinct_mention_expr = count_distinct_expr(columns, "Mention", "distinct_mention_count")
    mention_type_count_expr = count_distinct_expr(columns, "Type", "mention_type_count")
    text_field_count_expr = count_distinct_expr(columns, "TextFrom", "text_field_count")
    mention_types_expr = string_agg_distinct_expr(columns, "Type", "mention_types")
    text_fields_expr = string_agg_distinct_expr(columns, "TextFrom", "text_fields")

    min_start_expr = f"MIN({bigint_col(columns, 'StartPosition')}) AS min_start_position"
    max_end_expr = f"MAX({bigint_col(columns, 'EndPosition')}) AS max_end_position"
    avg_prob_expr = f"AVG({double_col(columns, 'prob')}) AS avg_prob"
    max_prob_expr = f"MAX({double_col(columns, 'prob')}) AS max_prob"

    has_neural_expr = flag_expr(columns, "is_neural_normalized", "has_neural_normalized")
    has_mesh_expr = flag_expr(columns, "mesh", "has_mesh")
    has_mim_expr = flag_expr(columns, "mim", "has_mim")
    has_cl_expr = flag_expr(columns, "CL", "has_cl")
    has_cellosaurus_expr = flag_expr(columns, "cellosaurus", "has_cellosaurus")
    has_ncbitaxon_expr = flag_expr(columns, "NCBITaxon", "has_ncbitaxon")
    has_ncbigene_expr = flag_expr(columns, "NCBIGene", "has_ncbigene")
    has_chebi_expr = flag_expr(columns, "CHEBI", "has_chebi")

    raw_row_count = count_table(connection, input_view)

    distinct_pair_count = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT source_id, target_id, edge_type
                FROM {qident(input_view)}
                GROUP BY source_id, target_id, edge_type
            )
            """
        ).fetchone()[0]
    )

    execute(
        connection,
        f"""
        CREATE TEMP TABLE {qident(output_table_name)} AS
        SELECT
            e.source_id,
            e.target_id,
            e.edge_type,
            {source_table_expr},
            {carrier_id_expr},
            {entity_id_expr},
            {sql_literal(carrier_type)} AS carrier_type,
            'source_target_pair' AS aggregation_level,
            COUNT(*) AS mention_count,
            {distinct_mention_expr},
            {sample_mention_expr},
            {mention_type_sample_expr},
            {mention_type_count_expr},
            {mention_types_expr},
            {text_field_count_expr},
            {text_fields_expr},
            {min_start_expr},
            {max_end_expr},
            {avg_prob_expr},
            {max_prob_expr},
            {has_neural_expr},
            {has_mesh_expr},
            {has_mim_expr},
            {has_cl_expr},
            {has_cellosaurus_expr},
            {has_ncbitaxon_expr},
            {has_ncbigene_expr},
            {has_chebi_expr}
        FROM {qident(input_view)} e
        WHERE e.source_id IS NOT NULL
          AND e.target_id IS NOT NULL
          AND e.edge_type IS NOT NULL
        GROUP BY
            e.source_id,
            e.target_id,
            e.edge_type
        """,
    )

    aggregated_row_count = count_table(connection, output_table_name)

    compression_ratio = (
        raw_row_count / aggregated_row_count
        if aggregated_row_count > 0
        else None
    )

    duplicate_basic_edge_count_removed = raw_row_count - distinct_pair_count

    return {
        "edge_table": output_table_name,
        "carrier_type": carrier_type,
        "raw_mention_edge_count": raw_row_count,
        "aggregated_pair_edge_count": aggregated_row_count,
        "distinct_pair_count_before": distinct_pair_count,
        "removed_duplicate_basic_edge_rows": duplicate_basic_edge_count_removed,
        "compression_ratio_raw_to_aggregated": f"{compression_ratio:.3f}" if compression_ratio is not None else "",
    }


def aggregate_all_bioentity_edges(connection: Any) -> list[dict[str, Any]]:
    stats = []

    for table_name, meta in BIOENTITY_EDGE_FILES.items():
        stat = aggregate_bioentity_edge_table(
            connection=connection,
            input_table_name=table_name,
            output_table_name=table_name,
            carrier_id_column=str(meta["carrier_id_column"]),
            carrier_type=str(meta["carrier_type"]),
        )
        stats.append(stat)

    return stats


def count_outputs(connection: Any) -> dict[str, int]:
    return {table_name: count_table(connection, table_name) for _, table_name, _, _ in OUTPUT_TABLES}


def write_outputs(connection: Any, args: argparse.Namespace, counts: dict[str, int]) -> list[dict[str, Any]]:
    manifest_rows = []

    for artifact_type, table_name, file_name, description in OUTPUT_TABLES:
        path = args.output_dir / file_name
        copy_table_to_parquet(connection, table_name, path, args.compression)
        manifest_rows.append(
            {
                "artifact_type": artifact_type,
                "name": table_name,
                "path": str(path),
                "row_count": counts.get(table_name, ""),
                "description": description,
            }
        )

    return manifest_rows


def write_manifest_files(output_dir: Path, manifest_rows: list[dict[str, Any]]) -> None:
    columns = ["artifact_type", "name", "path", "row_count", "description"]

    write_csv(output_dir / "aggregation_manifest.csv", manifest_rows, columns)

    # Compatibility manifest for scripts/07_validate_extracted_subgraph.py and
    # for consistency with extraction outputs.
    subgraph_rows = []
    for row in manifest_rows:
        subgraph_rows.append(
            {
                "artifact_type": row["artifact_type"],
                "name": row["name"],
                "path": row["path"],
                "row_count": row["row_count"],
                "source_tables": "aggregated_from_input_graph",
                "description": row["description"],
            }
        )

    write_csv(
        output_dir / "subgraph_manifest.csv",
        subgraph_rows,
        ["artifact_type", "name", "path", "row_count", "source_tables", "description"],
    )


def get_top_aggregated_edges(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            edge_table,
            source_id,
            target_id,
            edge_type,
            mention_count,
            distinct_mention_count,
            sample_mention,
            carrier_type
        FROM (
            SELECT
                'edges_patent_bioentity' AS edge_table,
                CAST(source_id AS VARCHAR) AS source_id,
                CAST(target_id AS VARCHAR) AS target_id,
                CAST(edge_type AS VARCHAR) AS edge_type,
                mention_count,
                distinct_mention_count,
                CAST(sample_mention AS VARCHAR) AS sample_mention,
                carrier_type
            FROM edges_patent_bioentity

            UNION ALL

            SELECT
                'edges_paper_bioentity' AS edge_table,
                CAST(source_id AS VARCHAR) AS source_id,
                CAST(target_id AS VARCHAR) AS target_id,
                CAST(edge_type AS VARCHAR) AS edge_type,
                mention_count,
                distinct_mention_count,
                CAST(sample_mention AS VARCHAR) AS sample_mention,
                carrier_type
            FROM edges_paper_bioentity

            UNION ALL

            SELECT
                'edges_trial_bioentity' AS edge_table,
                CAST(source_id AS VARCHAR) AS source_id,
                CAST(target_id AS VARCHAR) AS target_id,
                CAST(edge_type AS VARCHAR) AS edge_type,
                mention_count,
                distinct_mention_count,
                CAST(sample_mention AS VARCHAR) AS sample_mention,
                carrier_type
            FROM edges_trial_bioentity
        )
        ORDER BY mention_count DESC
        LIMIT 100
        """
    ).fetchall()

    return [
        {
            "edge_table": row[0],
            "source_id": row[1],
            "target_id": row[2],
            "edge_type": row[3],
            "mention_count": row[4],
            "distinct_mention_count": row[5],
            "sample_mention": row[6],
            "carrier_type": row[7],
        }
        for row in rows
    ]


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    counts: dict[str, int],
    aggregation_stats: list[dict[str, Any]],
    top_edges: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    runtime_seconds: float,
) -> None:
    node_rows = [
        {"node_type": "Patent", "count": counts.get("nodes_patent", 0)},
        {"node_type": "Paper", "count": counts.get("nodes_paper", 0)},
        {"node_type": "ClinicalTrial", "count": counts.get("nodes_clinicaltrial", 0)},
        {"node_type": "Project", "count": counts.get("nodes_project", 0)},
        {"node_type": "BioEntity", "count": counts.get("nodes_bioentity", 0)},
    ]

    edge_rows = [
        {"edge_type": "patent_paper", "count": counts.get("edges_patent_paper", 0)},
        {"edge_type": "patent_bioentity_aggregated", "count": counts.get("edges_patent_bioentity", 0)},
        {"edge_type": "patent_project", "count": counts.get("edges_patent_project", 0)},
        {"edge_type": "paper_trial", "count": counts.get("edges_paper_trial", 0)},
        {"edge_type": "paper_project", "count": counts.get("edges_paper_project", 0)},
        {"edge_type": "paper_bioentity_aggregated", "count": counts.get("edges_paper_bioentity", 0)},
        {"edge_type": "trial_project", "count": counts.get("edges_trial_project", 0)},
        {"edge_type": "trial_bioentity_aggregated", "count": counts.get("edges_trial_bioentity", 0)},
    ]

    parts = ["# BioEntity Mention Edge Aggregation Report\n"]

    parts.append("## 1. Parameters\n")
    parts.append(
        markdown_table(
            [
                {"parameter": "input_dir", "value": str(args.input_dir)},
                {"parameter": "output_dir", "value": str(args.output_dir)},
                {"parameter": "threads", "value": args.threads},
                {"parameter": "memory_limit", "value": args.memory_limit},
                {"parameter": "compression", "value": args.compression},
                {"parameter": "runtime_seconds", "value": f"{runtime_seconds:.1f}"},
            ],
            ["parameter", "value"],
        )
    )

    parts.append("\n## 2. Aggregation rule\n")
    parts.append(
        "BioEntity mention edges are aggregated by `source_id + target_id + edge_type`. "
        "The output keeps one weighted pair-level edge per carrier-BioEntity pair. "
        "The primary edge weight is `mention_count`. Additional attributes include "
        "`distinct_mention_count`, `sample_mention`, `mention_types`, `text_fields`, "
        "`avg_prob`, `max_prob`, and evidence flags such as `has_mesh`, `has_ncbigene`, "
        "and `has_chebi` where the source columns are available.\n"
    )

    parts.append("\n## 3. Aggregation statistics\n")
    parts.append(
        markdown_table(
            aggregation_stats,
            [
                "edge_table",
                "carrier_type",
                "raw_mention_edge_count",
                "aggregated_pair_edge_count",
                "distinct_pair_count_before",
                "removed_duplicate_basic_edge_rows",
                "compression_ratio_raw_to_aggregated",
            ],
        )
    )

    parts.append("\n## 4. Node counts\n")
    parts.append(markdown_table(node_rows, ["node_type", "count"]))

    parts.append("\n## 5. Edge counts\n")
    parts.append(markdown_table(edge_rows, ["edge_type", "count"]))

    parts.append("\n## 6. Top aggregated BioEntity edges by mention_count\n")
    parts.append(
        markdown_table(
            top_edges[:50],
            [
                "edge_table",
                "source_id",
                "target_id",
                "edge_type",
                "mention_count",
                "distinct_mention_count",
                "sample_mention",
                "carrier_type",
            ],
        )
    )

    parts.append("\n## 7. Output artifacts\n")
    parts.append(markdown_table(manifest_rows, ["artifact_type", "name", "path", "row_count", "description"]))

    parts.append("\n## 8. Recommended next step\n")
    parts.append(
        "- Run `scripts/07_validate_extracted_subgraph.py` on this aggregated output directory.\n"
        "- Expect duplicate-basic-edge warnings for BioEntity edges to drop substantially or disappear.\n"
        "- Use `mention_count` or `log1p(mention_count)` as a BioEntity edge weight in downstream graph analysis.\n"
        "- Keep the original mention-level graph for evidence tracing and the aggregated graph for statistics/modeling.\n"
    )

    report_text = "\n".join(parts)

    (output_dir / "aggregation_report.md").write_text(report_text, encoding="utf-8")

    # Compatibility report name.
    (output_dir / "subgraph_extraction_report.md").write_text(report_text, encoding="utf-8")


def print_summary(
    output_dir: Path,
    counts: dict[str, int],
    aggregation_stats: list[dict[str, Any]],
) -> None:
    print("BioEntity mention edge aggregation complete")
    print(f"Output directory: {output_dir}")

    print("Aggregation stats:")
    for row in aggregation_stats:
        print(
            f"- {row['edge_table']}: "
            f"raw={row['raw_mention_edge_count']:,}, "
            f"aggregated={row['aggregated_pair_edge_count']:,}, "
            f"compression={row['compression_ratio_raw_to_aggregated']}"
        )

    print("Node counts:")
    for key in ["nodes_patent", "nodes_paper", "nodes_clinicaltrial", "nodes_project", "nodes_bioentity"]:
        print(f"- {key}: {counts.get(key, 0):,}")

    print("Edge counts:")
    for key in [
        "edges_patent_paper",
        "edges_patent_bioentity",
        "edges_patent_project",
        "edges_paper_trial",
        "edges_paper_project",
        "edges_paper_bioentity",
        "edges_trial_project",
        "edges_trial_bioentity",
    ]:
        print(f"- {key}: {counts.get(key, 0):,}")

    print("Report: aggregation_report.md")
    print("Manifest: aggregation_manifest.csv")


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if args.threads < 1:
        raise ValueError("--threads must be greater than or equal to 1")

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()

    require_files(args.input_dir)
    prepare_output_dir(args.output_dir, args.overwrite)

    connection = connect_duckdb(args)

    run_step(args, 1, 6, "Creating input graph views", lambda: create_input_views(connection, args.input_dir))
    run_step(args, 2, 6, "Copying node and non-BioEntity edge tables", lambda: copy_node_and_non_bioentity_tables(connection))
    aggregation_stats: list[dict[str, Any]] = []
    run_step(args, 3, 6, "Aggregating BioEntity mention edges", lambda: aggregation_stats.extend(aggregate_all_bioentity_edges(connection)))

    counts = count_outputs(connection)
    top_edges = get_top_aggregated_edges(connection)

    run_step(
        args,
        4,
        6,
        "Writing graph Parquet outputs",
        lambda: None,
    )
    manifest_rows = write_outputs(connection, args, counts)

    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "aggregation_manifest",
            "path": str(args.output_dir / "aggregation_manifest.csv"),
            "row_count": len(manifest_rows) + 4,
            "description": "Manifest of aggregated graph output artifacts.",
        }
    )
    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "subgraph_manifest",
            "path": str(args.output_dir / "subgraph_manifest.csv"),
            "row_count": len(manifest_rows) + 4,
            "description": "Compatibility manifest for extracted subgraph validators.",
        }
    )
    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "aggregation_report",
            "path": str(args.output_dir / "aggregation_report.md"),
            "row_count": "",
            "description": "Human-readable BioEntity edge aggregation report.",
        }
    )
    manifest_rows.append(
        {
            "artifact_type": "report",
            "name": "subgraph_extraction_report",
            "path": str(args.output_dir / "subgraph_extraction_report.md"),
            "row_count": "",
            "description": "Compatibility report for extracted subgraph validators.",
        }
    )

    run_step(
        args,
        5,
        6,
        "Writing manifests and aggregation stats",
        lambda: (
            write_manifest_files(args.output_dir, manifest_rows),
            write_csv(
                args.output_dir / "bioentity_edge_aggregation_stats.csv",
                aggregation_stats,
                [
                    "edge_table",
                    "carrier_type",
                    "raw_mention_edge_count",
                    "aggregated_pair_edge_count",
                    "distinct_pair_count_before",
                    "removed_duplicate_basic_edge_rows",
                    "compression_ratio_raw_to_aggregated",
                ],
            ),
        ),
    )

    runtime_seconds = time.time() - start_time

    run_step(
        args,
        6,
        6,
        "Writing aggregation report",
        lambda: write_report(
            args.output_dir,
            args,
            counts,
            aggregation_stats,
            top_edges,
            manifest_rows,
            runtime_seconds,
        ),
    )

    print_summary(args.output_dir, counts, aggregation_stats)


if __name__ == "__main__":
    main()