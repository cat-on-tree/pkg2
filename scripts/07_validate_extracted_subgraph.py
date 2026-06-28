"""Validate a graph-ready patent-centered PKG24S4 subgraph.

This script validates outputs produced by scripts/06_extract_patent_centered_subgraph.py.

It checks:
- expected node/edge Parquet files exist
- required columns exist
- node_id uniqueness
- edge endpoint coverage
- duplicate edge rows by source_id/target_id/edge_type
- absence of CUI-less BioEntity nodes/edges
- degree summaries and top hub nodes
- BioEntity type distribution
- basic Project node normalization

It does not read raw data, does not modify source Parquet files, and does not
implement model code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


NODE_FILES = {
    "Patent": "nodes_patent.parquet",
    "Paper": "nodes_paper.parquet",
    "ClinicalTrial": "nodes_clinicaltrial.parquet",
    "Project": "nodes_project.parquet",
    "BioEntity": "nodes_bioentity.parquet",
}

EDGE_FILES = {
    "patent_links_paper": {
        "file": "edges_patent_paper.parquet",
        "source_type": "Patent",
        "target_type": "Paper",
    },
    "patent_mentions_bioentity": {
        "file": "edges_patent_bioentity.parquet",
        "source_type": "Patent",
        "target_type": "BioEntity",
    },
    "patent_linked_project": {
        "file": "edges_patent_project.parquet",
        "source_type": "Patent",
        "target_type": "Project",
    },
    "paper_linked_trial": {
        "file": "edges_paper_trial.parquet",
        "source_type": "Paper",
        "target_type": "ClinicalTrial",
    },
    "paper_linked_project": {
        "file": "edges_paper_project.parquet",
        "source_type": "Paper",
        "target_type": "Project",
    },
    "paper_mentions_bioentity": {
        "file": "edges_paper_bioentity.parquet",
        "source_type": "Paper",
        "target_type": "BioEntity",
    },
    "trial_linked_project": {
        "file": "edges_trial_project.parquet",
        "source_type": "ClinicalTrial",
        "target_type": "Project",
    },
    "trial_mentions_bioentity": {
        "file": "edges_trial_bioentity.parquet",
        "source_type": "ClinicalTrial",
        "target_type": "BioEntity",
    },
}

REPORT_FILES = [
    "subgraph_manifest.csv",
    "subgraph_extraction_report.md",
]

NODE_REQUIRED_COLUMNS = ["node_id"]
EDGE_REQUIRED_COLUMNS = ["source_id", "target_id", "edge_type", "source_table"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extracted patent-centered subgraph outputs.")
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="20GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("data/interim/duckdb_tmp"))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=20)
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
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def get_columns(connection: Any, path: Path) -> set[str]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {scan_sql(path)}").fetchall()
    return {str(row[0]) for row in rows}


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def validate_files(subgraph_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for node_type, file_name in NODE_FILES.items():
        path = subgraph_dir / file_name
        rows.append(
            {
                "artifact_type": "node",
                "name": node_type,
                "file_name": file_name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": f"{file_size_mb(path):.3f}",
                "status": "ok" if path.exists() else "error",
                "warning": "" if path.exists() else "missing_file",
            }
        )

    for edge_type, spec in EDGE_FILES.items():
        file_name = spec["file"]
        path = subgraph_dir / file_name
        rows.append(
            {
                "artifact_type": "edge",
                "name": edge_type,
                "file_name": file_name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": f"{file_size_mb(path):.3f}",
                "status": "ok" if path.exists() else "error",
                "warning": "" if path.exists() else "missing_file",
            }
        )

    for file_name in REPORT_FILES:
        path = subgraph_dir / file_name
        rows.append(
            {
                "artifact_type": "report",
                "name": file_name,
                "file_name": file_name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": f"{file_size_mb(path):.3f}",
                "status": "ok" if path.exists() else "warning",
                "warning": "" if path.exists() else "missing_report",
            }
        )

    return rows


def require_core_parquet_files(subgraph_dir: Path) -> None:
    missing = []
    for file_name in list(NODE_FILES.values()) + [spec["file"] for spec in EDGE_FILES.values()]:
        path = subgraph_dir / file_name
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required subgraph Parquet files:\n" + "\n".join(missing))


def validate_schemas(connection: Any, subgraph_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for node_type, file_name in NODE_FILES.items():
        path = subgraph_dir / file_name
        columns = get_columns(connection, path)
        missing = [column for column in NODE_REQUIRED_COLUMNS if column not in columns]
        rows.append(
            {
                "artifact_type": "node",
                "name": node_type,
                "file_name": file_name,
                "column_count": len(columns),
                "columns_json": str(sorted(columns)),
                "missing_required_columns": ";".join(missing),
                "status": "ok" if not missing else "error",
                "warning": "" if not missing else "missing_required_columns",
            }
        )

    for edge_type, spec in EDGE_FILES.items():
        file_name = spec["file"]
        path = subgraph_dir / file_name
        columns = get_columns(connection, path)
        missing = [column for column in EDGE_REQUIRED_COLUMNS if column not in columns]
        rows.append(
            {
                "artifact_type": "edge",
                "name": edge_type,
                "file_name": file_name,
                "column_count": len(columns),
                "columns_json": str(sorted(columns)),
                "missing_required_columns": ";".join(missing),
                "status": "ok" if not missing else "error",
                "warning": "" if not missing else "missing_required_columns",
            }
        )

    return rows


def create_views(connection: Any, subgraph_dir: Path) -> None:
    for node_type, file_name in NODE_FILES.items():
        view_name = f"nodes_{node_type.lower()}"
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(view_name)} AS
            SELECT * FROM {scan_sql(subgraph_dir / file_name)}
            """
        )

    for edge_type, spec in EDGE_FILES.items():
        view_name = f"edges_{edge_type}"
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(view_name)} AS
            SELECT * FROM {scan_sql(subgraph_dir / spec["file"])}
            """
        )


def validate_nodes(connection: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for node_type in NODE_FILES:
        view_name = f"nodes_{node_type.lower()}"
        row = connection.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(node_id) AS node_id_non_null_count,
                COUNT(*) - COUNT(node_id) AS node_id_null_count,
                COUNT(DISTINCT CAST(node_id AS VARCHAR)) AS distinct_node_id_count
            FROM {qident(view_name)}
            """
        ).fetchone()

        row_count, non_null, null_count, distinct_count = row
        duplicate_count = int(non_null) - int(distinct_count)
        warnings = []
        if int(row_count) == 0:
            warnings.append("empty_node_table")
        if int(null_count) > 0:
            warnings.append("node_id_has_nulls")
        if duplicate_count > 0:
            warnings.append("duplicate_node_ids")

        rows.append(
            {
                "node_type": node_type,
                "file_name": NODE_FILES[node_type],
                "row_count": row_count,
                "node_id_non_null_count": non_null,
                "node_id_null_count": null_count,
                "distinct_node_id_count": distinct_count,
                "duplicate_node_id_count": duplicate_count,
                "status": "ok" if not warnings else "warning",
                "warning": ";".join(warnings),
            }
        )

    return rows


def validate_edges(connection: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for edge_type, spec in EDGE_FILES.items():
        view_name = f"edges_{edge_type}"
        row = connection.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(source_id) AS source_non_null_count,
                COUNT(target_id) AS target_non_null_count,
                COUNT(edge_type) AS edge_type_non_null_count,
                COUNT(DISTINCT CAST(source_id AS VARCHAR)) AS distinct_source_count,
                COUNT(DISTINCT CAST(target_id AS VARCHAR)) AS distinct_target_count,
                COUNT(DISTINCT CAST(source_id AS VARCHAR) || '||' || CAST(target_id AS VARCHAR) || '||' || CAST(edge_type AS VARCHAR))
                    AS distinct_basic_edge_count
            FROM {qident(view_name)}
            """
        ).fetchone()

        (
            row_count,
            source_non_null,
            target_non_null,
            edge_type_non_null,
            distinct_source,
            distinct_target,
            distinct_basic_edges,
        ) = row
        source_null = int(row_count) - int(source_non_null)
        target_null = int(row_count) - int(target_non_null)
        edge_type_null = int(row_count) - int(edge_type_non_null)
        duplicate_basic_edges = int(row_count) - int(distinct_basic_edges)

        warnings = []
        if int(row_count) == 0:
            warnings.append("empty_edge_table")
        if source_null > 0:
            warnings.append("source_id_has_nulls")
        if target_null > 0:
            warnings.append("target_id_has_nulls")
        if edge_type_null > 0:
            warnings.append("edge_type_has_nulls")
        if duplicate_basic_edges > 0:
            warnings.append("duplicate_basic_edges")

        rows.append(
            {
                "edge_type": edge_type,
                "file_name": spec["file"],
                "source_type": spec["source_type"],
                "target_type": spec["target_type"],
                "row_count": row_count,
                "source_non_null_count": source_non_null,
                "source_null_count": source_null,
                "target_non_null_count": target_non_null,
                "target_null_count": target_null,
                "edge_type_null_count": edge_type_null,
                "distinct_source_count": distinct_source,
                "distinct_target_count": distinct_target,
                "distinct_basic_edge_count": distinct_basic_edges,
                "duplicate_basic_edge_count": duplicate_basic_edges,
                "status": "ok" if not warnings else "warning",
                "warning": ";".join(warnings),
            }
        )

    return rows


def node_view_name(node_type: str) -> str:
    return f"nodes_{node_type.lower()}"


def validate_endpoints(connection: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for edge_type, spec in EDGE_FILES.items():
        edge_view = f"edges_{edge_type}"
        source_view = node_view_name(spec["source_type"])
        target_view = node_view_name(spec["target_type"])

        source_unmatched = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {qident(edge_view)} e
            LEFT JOIN {qident(source_view)} n
                ON CAST(e.source_id AS VARCHAR) = CAST(n.node_id AS VARCHAR)
            WHERE e.source_id IS NOT NULL
              AND n.node_id IS NULL
            """
        ).fetchone()[0]

        target_unmatched = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {qident(edge_view)} e
            LEFT JOIN {qident(target_view)} n
                ON CAST(e.target_id AS VARCHAR) = CAST(n.node_id AS VARCHAR)
            WHERE e.target_id IS NOT NULL
              AND n.node_id IS NULL
            """
        ).fetchone()[0]

        edge_count = connection.execute(f"SELECT COUNT(*) FROM {qident(edge_view)}").fetchone()[0]
        warnings = []
        if int(source_unmatched) > 0:
            warnings.append("unmatched_source_endpoints")
        if int(target_unmatched) > 0:
            warnings.append("unmatched_target_endpoints")

        rows.append(
            {
                "edge_type": edge_type,
                "edge_file": spec["file"],
                "source_type": spec["source_type"],
                "target_type": spec["target_type"],
                "edge_count": edge_count,
                "source_unmatched_count": source_unmatched,
                "target_unmatched_count": target_unmatched,
                "source_match_rate": "" if int(edge_count) == 0 else f"{1 - int(source_unmatched) / int(edge_count):.6f}",
                "target_match_rate": "" if int(edge_count) == 0 else f"{1 - int(target_unmatched) / int(edge_count):.6f}",
                "status": "ok" if not warnings else "error",
                "warning": ";".join(warnings),
            }
        )

    return rows


def validate_cuiless(connection: Any) -> list[dict[str, Any]]:
    checks = [
        ("node", "nodes_bioentity", "node_id"),
        ("node", "nodes_bioentity", "EntityId"),
        ("edge", "edges_patent_mentions_bioentity", "target_id"),
        ("edge", "edges_patent_mentions_bioentity", "EntityId"),
        ("edge", "edges_paper_mentions_bioentity", "target_id"),
        ("edge", "edges_paper_mentions_bioentity", "EntityId"),
        ("edge", "edges_trial_mentions_bioentity", "target_id"),
        ("edge", "edges_trial_mentions_bioentity", "EntityId"),
    ]

    rows: list[dict[str, Any]] = []
    for artifact_type, view_name, column in checks:
        columns = {str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM {qident(view_name)}").fetchall()}
        if column not in columns:
            rows.append(
                {
                    "artifact_type": artifact_type,
                    "view_name": view_name,
                    "column": column,
                    "cuiless_count": "",
                    "status": "warning",
                    "warning": "missing_column",
                }
            )
            continue

        count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {qident(view_name)}
            WHERE UPPER(TRIM(CAST({qident(column)} AS VARCHAR))) = 'CUI-LESS'
            """
        ).fetchone()[0]

        rows.append(
            {
                "artifact_type": artifact_type,
                "view_name": view_name,
                "column": column,
                "cuiless_count": count,
                "status": "ok" if int(count) == 0 else "error",
                "warning": "" if int(count) == 0 else "cuiless_present_in_graph",
            }
        )

    return rows


def validate_project_normalization(connection: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    columns = {str(row[0]) for row in connection.execute("DESCRIBE SELECT * FROM nodes_project").fetchall()}
    if "node_id" not in columns:
        return [
            {
                "check_name": "project_node_id_normalization",
                "row_count": "",
                "bad_count": "",
                "status": "error",
                "warning": "missing_node_id",
            }
        ]

    bad_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM nodes_project
        WHERE node_id IS NULL
           OR TRIM(CAST(node_id AS VARCHAR)) = ''
           OR CAST(node_id AS VARCHAR) != UPPER(CAST(node_id AS VARCHAR))
           OR CAST(node_id AS VARCHAR) LIKE '% %'
           OR CAST(node_id AS VARCHAR) LIKE '%-%'
        """
    ).fetchone()[0]
    row_count = connection.execute("SELECT COUNT(*) FROM nodes_project").fetchone()[0]

    rows.append(
        {
            "check_name": "project_node_id_normalization",
            "row_count": row_count,
            "bad_count": bad_count,
            "status": "ok" if int(bad_count) == 0 else "warning",
            "warning": "" if int(bad_count) == 0 else "project_node_ids_not_fully_normalized",
        }
    )
    return rows


def create_all_edges_typed(connection: Any) -> None:
    union_parts = []
    for edge_type, spec in EDGE_FILES.items():
        view_name = f"edges_{edge_type}"
        union_parts.append(
            f"""
            SELECT
                {sql_literal(edge_type)} AS edge_type_name,
                {sql_literal(spec["source_type"])} AS source_type,
                CAST(source_id AS VARCHAR) AS source_id,
                {sql_literal(spec["target_type"])} AS target_type,
                CAST(target_id AS VARCHAR) AS target_id
            FROM {qident(view_name)}
            WHERE source_id IS NOT NULL AND target_id IS NOT NULL
            """
        )

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW all_edges_typed AS
        {" UNION ALL ".join(union_parts)}
        """
    )


def degree_summary(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        WITH degrees AS (
            SELECT source_type AS node_type, source_id AS node_id, COUNT(*) AS out_degree, 0 AS in_degree
            FROM all_edges_typed
            GROUP BY source_type, source_id
            UNION ALL
            SELECT target_type AS node_type, target_id AS node_id, 0 AS out_degree, COUNT(*) AS in_degree
            FROM all_edges_typed
            GROUP BY target_type, target_id
        ),
        node_degrees AS (
            SELECT
                node_type,
                node_id,
                SUM(out_degree) AS out_degree,
                SUM(in_degree) AS in_degree,
                SUM(out_degree) + SUM(in_degree) AS total_degree
            FROM degrees
            GROUP BY node_type, node_id
        )
        SELECT
            node_type,
            COUNT(*) AS nodes_with_degree,
            AVG(total_degree) AS avg_total_degree,
            MAX(total_degree) AS max_total_degree,
            AVG(out_degree) AS avg_out_degree,
            MAX(out_degree) AS max_out_degree,
            AVG(in_degree) AS avg_in_degree,
            MAX(in_degree) AS max_in_degree
        FROM node_degrees
        GROUP BY node_type
        ORDER BY node_type
        """
    ).fetchall()

    return [
        {
            "node_type": row[0],
            "nodes_with_degree": row[1],
            "avg_total_degree": f"{float(row[2]):.3f}" if row[2] is not None else "",
            "max_total_degree": row[3],
            "avg_out_degree": f"{float(row[4]):.3f}" if row[4] is not None else "",
            "max_out_degree": row[5],
            "avg_in_degree": f"{float(row[6]):.3f}" if row[6] is not None else "",
            "max_in_degree": row[7],
        }
        for row in rows
    ]


def top_degree_nodes(connection: Any, top_n: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"""
        WITH degrees AS (
            SELECT source_type AS node_type, source_id AS node_id, COUNT(*) AS out_degree, 0 AS in_degree
            FROM all_edges_typed
            GROUP BY source_type, source_id
            UNION ALL
            SELECT target_type AS node_type, target_id AS node_id, 0 AS out_degree, COUNT(*) AS in_degree
            FROM all_edges_typed
            GROUP BY target_type, target_id
        ),
        node_degrees AS (
            SELECT
                node_type,
                node_id,
                SUM(out_degree) AS out_degree,
                SUM(in_degree) AS in_degree,
                SUM(out_degree) + SUM(in_degree) AS total_degree
            FROM degrees
            GROUP BY node_type, node_id
        )
        SELECT node_type, node_id, out_degree, in_degree, total_degree
        FROM node_degrees
        ORDER BY total_degree DESC, node_type, node_id
        LIMIT {top_n}
        """
    ).fetchall()

    return [
        {
            "node_type": row[0],
            "node_id": row[1],
            "out_degree": row[2],
            "in_degree": row[3],
            "total_degree": row[4],
        }
        for row in rows
    ]


def bioentity_type_counts(connection: Any) -> list[dict[str, Any]]:
    columns = {str(row[0]) for row in connection.execute("DESCRIBE SELECT * FROM nodes_bioentity").fetchall()}
    if "Type" not in columns:
        return []

    rows = connection.execute(
        """
        SELECT
            COALESCE(CAST("Type" AS VARCHAR), '') AS bioentity_type,
            COUNT(*) AS node_count
        FROM nodes_bioentity
        GROUP BY COALESCE(CAST("Type" AS VARCHAR), '')
        ORDER BY node_count DESC
        """
    ).fetchall()

    return [{"bioentity_type": row[0], "node_count": row[1]} for row in rows]


def sample_endpoint_errors(connection: Any, output_dir: Path, sample_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for edge_type, spec in EDGE_FILES.items():
        edge_view = f"edges_{edge_type}"
        source_view = node_view_name(spec["source_type"])
        target_view = node_view_name(spec["target_type"])

        source_samples = connection.execute(
            f"""
            SELECT DISTINCT CAST(e.source_id AS VARCHAR)
            FROM {qident(edge_view)} e
            LEFT JOIN {qident(source_view)} n
                ON CAST(e.source_id AS VARCHAR) = CAST(n.node_id AS VARCHAR)
            WHERE e.source_id IS NOT NULL
              AND n.node_id IS NULL
            LIMIT {sample_limit}
            """
        ).fetchall()

        target_samples = connection.execute(
            f"""
            SELECT DISTINCT CAST(e.target_id AS VARCHAR)
            FROM {qident(edge_view)} e
            LEFT JOIN {qident(target_view)} n
                ON CAST(e.target_id AS VARCHAR) = CAST(n.node_id AS VARCHAR)
            WHERE e.target_id IS NOT NULL
              AND n.node_id IS NULL
            LIMIT {sample_limit}
            """
        ).fetchall()

        for sample in source_samples:
            rows.append(
                {
                    "edge_type": edge_type,
                    "endpoint": "source",
                    "expected_node_type": spec["source_type"],
                    "sample_id": sample[0],
                }
            )
        for sample in target_samples:
            rows.append(
                {
                    "edge_type": edge_type,
                    "endpoint": "target",
                    "expected_node_type": spec["target_type"],
                    "sample_id": sample[0],
                }
            )

    write_csv(
        output_dir / "subgraph_endpoint_unmatched_samples.csv",
        rows,
        ["edge_type", "endpoint", "expected_node_type", "sample_id"],
    )
    return rows


def summarize_status(*row_groups: list[dict[str, Any]]) -> dict[str, Any]:
    error_count = 0
    warning_count = 0
    for rows in row_groups:
        for row in rows:
            status = str(row.get("status", "")).lower()
            warning = str(row.get("warning", ""))
            if status == "error":
                error_count += 1
            elif status == "warning" or warning:
                warning_count += 1

    if error_count > 0:
        overall_status = "error"
    elif warning_count > 0:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "overall_status": overall_status,
        "error_count": error_count,
        "warning_count": warning_count,
    }


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    status_summary: dict[str, Any],
    file_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
    cuiless_rows: list[dict[str, Any]],
    project_rows: list[dict[str, Any]],
    degree_rows: list[dict[str, Any]],
    top_degree_rows: list[dict[str, Any]],
    bioentity_type_rows: list[dict[str, Any]],
) -> None:
    parts = ["# Extracted Subgraph Validation Report\n"]

    parts.append("## 1. Parameters\n")
    parts.append(
        markdown_table(
            [
                {"parameter": "subgraph_dir", "value": str(args.subgraph_dir)},
                {"parameter": "output_dir", "value": str(args.output_dir)},
                {"parameter": "threads", "value": args.threads},
                {"parameter": "memory_limit", "value": args.memory_limit},
                {"parameter": "top_n", "value": args.top_n},
                {"parameter": "sample_limit", "value": args.sample_limit},
            ],
            ["parameter", "value"],
        )
    )

    parts.append("\n## 2. Overall status\n")
    parts.append(
        markdown_table(
            [status_summary],
            ["overall_status", "error_count", "warning_count"],
        )
    )

    parts.append("\n## 3. File existence\n")
    parts.append(markdown_table(file_rows, ["artifact_type", "name", "file_name", "exists", "size_mb", "status", "warning"]))

    parts.append("\n## 4. Schema validation\n")
    parts.append(markdown_table(schema_rows, ["artifact_type", "name", "file_name", "column_count", "missing_required_columns", "status", "warning"]))

    parts.append("\n## 5. Node validation\n")
    parts.append(
        markdown_table(
            node_rows,
            [
                "node_type",
                "row_count",
                "node_id_non_null_count",
                "node_id_null_count",
                "distinct_node_id_count",
                "duplicate_node_id_count",
                "status",
                "warning",
            ],
        )
    )

    parts.append("\n## 6. Edge validation\n")
    parts.append(
        markdown_table(
            edge_rows,
            [
                "edge_type",
                "row_count",
                "source_null_count",
                "target_null_count",
                "distinct_source_count",
                "distinct_target_count",
                "duplicate_basic_edge_count",
                "status",
                "warning",
            ],
        )
    )

    parts.append("\n## 7. Endpoint coverage\n")
    parts.append(
        markdown_table(
            endpoint_rows,
            [
                "edge_type",
                "source_type",
                "target_type",
                "edge_count",
                "source_unmatched_count",
                "target_unmatched_count",
                "source_match_rate",
                "target_match_rate",
                "status",
                "warning",
            ],
        )
    )

    parts.append("\n## 8. CUI-less validation\n")
    parts.append(
        "`CUI-less` should not appear in `nodes_bioentity` or any BioEntity graph edge target.\n"
    )
    parts.append(markdown_table(cuiless_rows, ["artifact_type", "view_name", "column", "cuiless_count", "status", "warning"]))

    parts.append("\n## 9. Project normalization validation\n")
    parts.append(markdown_table(project_rows, ["check_name", "row_count", "bad_count", "status", "warning"]))

    parts.append("\n## 10. Degree summary\n")
    parts.append(
        markdown_table(
            degree_rows,
            [
                "node_type",
                "nodes_with_degree",
                "avg_total_degree",
                "max_total_degree",
                "avg_out_degree",
                "max_out_degree",
                "avg_in_degree",
                "max_in_degree",
            ],
        )
    )

    parts.append("\n## 11. Top degree nodes\n")
    parts.append(markdown_table(top_degree_rows, ["node_type", "node_id", "out_degree", "in_degree", "total_degree"]))

    parts.append("\n## 12. BioEntity type counts\n")
    parts.append(markdown_table(bioentity_type_rows, ["bioentity_type", "node_count"]))

    parts.append("\n## 13. Recommended next step\n")
    if status_summary["overall_status"] == "error":
        parts.append("- Fix validation errors before using this graph for downstream analysis.\n")
    elif status_summary["overall_status"] == "warning":
        parts.append("- Review validation warnings. Empty optional edge tables may be acceptable if the corresponding extraction flags were disabled.\n")
    else:
        parts.append("- Validation passed. Next, inspect graph size and degree distribution before temporal snapshot or knowledge unit extraction design.\n")
    parts.append("- Do not start model training until the extracted graph is validated at the intended full scale.\n")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def print_terminal_summary(status_summary: dict[str, Any], node_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]], endpoint_rows: list[dict[str, Any]], cuiless_rows: list[dict[str, Any]], output_dir: Path) -> None:
    print("Extracted subgraph validation complete")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {status_summary['overall_status']}")
    print(f"Errors: {status_summary['error_count']}")
    print(f"Warnings: {status_summary['warning_count']}")

    print("Node counts:")
    for row in node_rows:
        print(f"- {row['node_type']}: {row['row_count']}")

    print("Edge counts:")
    for row in edge_rows:
        print(f"- {row['edge_type']}: {row['row_count']}")

    endpoint_errors = [row for row in endpoint_rows if row.get("status") == "error"]
    if endpoint_errors:
        print("Endpoint validation errors:")
        for row in endpoint_errors:
            print(
                f"- {row['edge_type']}: "
                f"source_unmatched={row['source_unmatched_count']}, "
                f"target_unmatched={row['target_unmatched_count']}"
            )

    cuiless_errors = [row for row in cuiless_rows if row.get("status") == "error"]
    if cuiless_errors:
        print("CUI-less validation errors:")
        for row in cuiless_errors:
            print(f"- {row['view_name']}.{row['column']}: {row['cuiless_count']}")

    print("Report: subgraph_validation_report.md")


def main() -> None:
    args = parse_args()

    if args.threads < 1:
        raise ValueError("--threads must be greater than or equal to 1")
    if args.top_n < 1:
        raise ValueError("--top-n must be greater than or equal to 1")
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be greater than or equal to 1")

    args.subgraph_dir = args.subgraph_dir.resolve()
    args.output_dir = args.output_dir.resolve() if args.output_dir else args.subgraph_dir.resolve()
    args.temp_dir = args.temp_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    file_rows = validate_files(args.subgraph_dir)
    require_core_parquet_files(args.subgraph_dir)

    connection = connect_duckdb(args)
    create_views(connection, args.subgraph_dir)

    schema_rows = validate_schemas(connection, args.subgraph_dir)
    node_rows = validate_nodes(connection)
    edge_rows = validate_edges(connection)
    endpoint_rows = validate_endpoints(connection)
    cuiless_rows = validate_cuiless(connection)
    project_rows = validate_project_normalization(connection)

    create_all_edges_typed(connection)
    degree_rows = degree_summary(connection)
    top_degree_rows = top_degree_nodes(connection, args.top_n)
    bioentity_type_rows = bioentity_type_counts(connection)
    sample_endpoint_errors(connection, args.output_dir, args.sample_limit)

    status_summary = summarize_status(
        file_rows,
        schema_rows,
        node_rows,
        edge_rows,
        endpoint_rows,
        cuiless_rows,
        project_rows,
    )

    write_csv(
        args.output_dir / "subgraph_file_validation.csv",
        file_rows,
        ["artifact_type", "name", "file_name", "path", "exists", "size_mb", "status", "warning"],
    )
    write_csv(
        args.output_dir / "subgraph_schema_validation.csv",
        schema_rows,
        ["artifact_type", "name", "file_name", "column_count", "columns_json", "missing_required_columns", "status", "warning"],
    )
    write_csv(
        args.output_dir / "subgraph_node_validation.csv",
        node_rows,
        [
            "node_type",
            "file_name",
            "row_count",
            "node_id_non_null_count",
            "node_id_null_count",
            "distinct_node_id_count",
            "duplicate_node_id_count",
            "status",
            "warning",
        ],
    )
    write_csv(
        args.output_dir / "subgraph_edge_validation.csv",
        edge_rows,
        [
            "edge_type",
            "file_name",
            "source_type",
            "target_type",
            "row_count",
            "source_non_null_count",
            "source_null_count",
            "target_non_null_count",
            "target_null_count",
            "edge_type_null_count",
            "distinct_source_count",
            "distinct_target_count",
            "distinct_basic_edge_count",
            "duplicate_basic_edge_count",
            "status",
            "warning",
        ],
    )
    write_csv(
        args.output_dir / "subgraph_endpoint_validation.csv",
        endpoint_rows,
        [
            "edge_type",
            "edge_file",
            "source_type",
            "target_type",
            "edge_count",
            "source_unmatched_count",
            "target_unmatched_count",
            "source_match_rate",
            "target_match_rate",
            "status",
            "warning",
        ],
    )
    write_csv(
        args.output_dir / "subgraph_cuiless_validation.csv",
        cuiless_rows,
        ["artifact_type", "view_name", "column", "cuiless_count", "status", "warning"],
    )
    write_csv(
        args.output_dir / "subgraph_project_validation.csv",
        project_rows,
        ["check_name", "row_count", "bad_count", "status", "warning"],
    )
    write_csv(
        args.output_dir / "subgraph_degree_summary.csv",
        degree_rows,
        [
            "node_type",
            "nodes_with_degree",
            "avg_total_degree",
            "max_total_degree",
            "avg_out_degree",
            "max_out_degree",
            "avg_in_degree",
            "max_in_degree",
        ],
    )
    write_csv(
        args.output_dir / "subgraph_top_degree_nodes.csv",
        top_degree_rows,
        ["node_type", "node_id", "out_degree", "in_degree", "total_degree"],
    )
    write_csv(
        args.output_dir / "subgraph_bioentity_type_counts.csv",
        bioentity_type_rows,
        ["bioentity_type", "node_count"],
    )

    write_report(
        args.output_dir / "subgraph_validation_report.md",
        args,
        status_summary,
        file_rows,
        schema_rows,
        node_rows,
        edge_rows,
        endpoint_rows,
        cuiless_rows,
        project_rows,
        degree_rows,
        top_degree_rows,
        bioentity_type_rows,
    )

    print_terminal_summary(status_summary, node_rows, edge_rows, endpoint_rows, cuiless_rows, args.output_dir)


if __name__ == "__main__":
    main()