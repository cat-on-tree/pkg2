"""Compute graph statistics for an extracted / aggregated patent-centered subgraph.

This script is intended to run after:

- scripts/06_extract_patent_centered_subgraph.py
- scripts/08_extract_domain_subgraph.py
- scripts/09_aggregate_mention_edges.py

Recommended input:
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/

Main outputs:
- graph_statistics_report.md
- graph_node_type_counts.csv
- graph_edge_type_counts.csv
- graph_bioentity_type_counts.csv
- graph_edge_weight_summary.csv
- graph_degree_summary.csv
- graph_top_degree_nodes.csv
- graph_connected_components.csv
- graph_isolated_nodes.csv
- graph_top_weighted_bioentity_edges.csv

The script treats the graph as a typed heterogeneous graph.

BioEntity edges aggregated by scripts/09_aggregate_mention_edges.py should contain
`mention_count`, which is used as edge weight. Edges without `mention_count`
receive edge_weight = 1.0.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from collections import Counter, defaultdict
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
    "edges_patent_paper": {
        "file_name": "edges_patent_paper.parquet",
        "source_type": "Patent",
        "target_type": "Paper",
    },
    "edges_patent_bioentity": {
        "file_name": "edges_patent_bioentity.parquet",
        "source_type": "Patent",
        "target_type": "BioEntity",
    },
    "edges_patent_project": {
        "file_name": "edges_patent_project.parquet",
        "source_type": "Patent",
        "target_type": "Project",
    },
    "edges_paper_trial": {
        "file_name": "edges_paper_trial.parquet",
        "source_type": "Paper",
        "target_type": "ClinicalTrial",
    },
    "edges_paper_project": {
        "file_name": "edges_paper_project.parquet",
        "source_type": "Paper",
        "target_type": "Project",
    },
    "edges_paper_bioentity": {
        "file_name": "edges_paper_bioentity.parquet",
        "source_type": "Paper",
        "target_type": "BioEntity",
    },
    "edges_trial_project": {
        "file_name": "edges_trial_project.parquet",
        "source_type": "ClinicalTrial",
        "target_type": "Project",
    },
    "edges_trial_bioentity": {
        "file_name": "edges_trial_bioentity.parquet",
        "source_type": "ClinicalTrial",
        "target_type": "BioEntity",
    },
}

BIOENTITY_EDGE_TABLES = {
    "edges_patent_bioentity",
    "edges_paper_bioentity",
    "edges_trial_bioentity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute graph statistics for an extracted heterogeneous graph.")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated"),
        help="Input graph directory containing node and edge Parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for statistics outputs. "
            "Defaults to --graph-dir. If different from --graph-dir and exists, use --overwrite to replace it."
        ),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="20GB")
    parser.add_argument("--temp-dir", type=Path, default=Path("data/interim/duckdb_tmp"))
    parser.add_argument("--top-k", type=int, default=200)
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


def fetch_dicts(connection: Any, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_columns(connection: Any, table_name: str) -> set[str]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {qident(table_name)}").fetchall()
    return {str(row[0]) for row in rows}


def count_table(connection: Any, table_name: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {qident(table_name)}").fetchone()[0])


def require_files(graph_dir: Path) -> None:
    required = list(NODE_FILES.values()) + [meta["file_name"] for meta in EDGE_FILES.values()]
    missing = [str(graph_dir / file_name) for file_name in required if not (graph_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError("Missing required graph files:\n" + "\n".join(missing))


def prepare_output_dir(graph_dir: Path, output_dir: Path, overwrite: bool) -> None:
    graph_dir = graph_dir.resolve()
    output_dir = output_dir.resolve()

    if output_dir == graph_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return

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


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        column_set = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    column_set.append(key)
        columns = column_set

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


def safe_label_expr(columns: set[str], preferred_columns: list[str], alias: str = "n") -> str:
    pieces = []
    for column in preferred_columns:
        if column in columns:
            pieces.append(f"NULLIF(TRIM(CAST({alias}.{qident(column)} AS VARCHAR)), '')")

    pieces.append(f"CAST({alias}.node_id AS VARCHAR)")
    return "COALESCE(" + ", ".join(pieces) + ")"


def safe_type_expr(columns: set[str], column: str, alias: str = "n") -> str:
    if column in columns:
        return f"CAST({alias}.{qident(column)} AS VARCHAR)"
    return "CAST(NULL AS VARCHAR)"


def create_input_views(connection: Any, graph_dir: Path) -> None:
    for node_type, file_name in NODE_FILES.items():
        table_name = f"nodes_{node_type.lower()}"
        if node_type == "ClinicalTrial":
            table_name = "nodes_clinicaltrial"
        execute(
            connection,
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(table_name)} AS
            SELECT * FROM {scan_sql(graph_dir / file_name)}
            """,
        )

    for edge_table, meta in EDGE_FILES.items():
        execute(
            connection,
            f"""
            CREATE OR REPLACE TEMP VIEW {qident(edge_table)} AS
            SELECT * FROM {scan_sql(graph_dir / meta["file_name"])}
            """,
        )


def create_node_union(connection: Any) -> None:
    patent_cols = get_columns(connection, "nodes_patent")
    paper_cols = get_columns(connection, "nodes_paper")
    trial_cols = get_columns(connection, "nodes_clinicaltrial")
    project_cols = get_columns(connection, "nodes_project")
    bio_cols = get_columns(connection, "nodes_bioentity")

    patent_label = safe_label_expr(patent_cols, ["Title", "Abstract"])
    paper_label = safe_label_expr(paper_cols, ["ArticleTitle"])
    trial_label = safe_label_expr(trial_cols, ["brief_title", "official_title", "conditions"])
    project_label = safe_label_expr(project_cols, ["PROJECT_TITLE", "Abstract"])
    bio_label = safe_label_expr(bio_cols, ["Mention"])
    bio_type = safe_type_expr(bio_cols, "Type")

    execute(
        connection,
        f"""
        CREATE TEMP TABLE node_all AS
        SELECT
            'Patent' AS node_type,
            CAST(n.node_id AS VARCHAR) AS node_id,
            {patent_label} AS label,
            CAST(NULL AS VARCHAR) AS bioentity_type
        FROM nodes_patent n

        UNION ALL

        SELECT
            'Paper' AS node_type,
            CAST(n.node_id AS VARCHAR) AS node_id,
            {paper_label} AS label,
            CAST(NULL AS VARCHAR) AS bioentity_type
        FROM nodes_paper n

        UNION ALL

        SELECT
            'ClinicalTrial' AS node_type,
            CAST(n.node_id AS VARCHAR) AS node_id,
            {trial_label} AS label,
            CAST(NULL AS VARCHAR) AS bioentity_type
        FROM nodes_clinicaltrial n

        UNION ALL

        SELECT
            'Project' AS node_type,
            CAST(n.node_id AS VARCHAR) AS node_id,
            {project_label} AS label,
            CAST(NULL AS VARCHAR) AS bioentity_type
        FROM nodes_project n

        UNION ALL

        SELECT
            'BioEntity' AS node_type,
            CAST(n.node_id AS VARCHAR) AS node_id,
            {bio_label} AS label,
            {bio_type} AS bioentity_type
        FROM nodes_bioentity n
        """,
    )


def create_edge_union(connection: Any) -> None:
    union_parts = []

    for edge_table, meta in EDGE_FILES.items():
        columns = get_columns(connection, edge_table)
        if "mention_count" in columns:
            weight_expr = "COALESCE(TRY_CAST(e.mention_count AS DOUBLE), 1.0)"
            mention_count_expr = "TRY_CAST(e.mention_count AS BIGINT)"
        else:
            weight_expr = "1.0"
            mention_count_expr = "CAST(NULL AS BIGINT)"

        union_parts.append(
            f"""
            SELECT
                {sql_literal(edge_table)} AS edge_table,
                CAST(e.edge_type AS VARCHAR) AS edge_type,
                {sql_literal(meta["source_type"])} AS source_type,
                CAST(e.source_id AS VARCHAR) AS source_id,
                {sql_literal(meta["target_type"])} AS target_type,
                CAST(e.target_id AS VARCHAR) AS target_id,
                {weight_expr} AS edge_weight,
                {mention_count_expr} AS mention_count
            FROM {qident(edge_table)} e
            WHERE e.source_id IS NOT NULL
              AND e.target_id IS NOT NULL
              AND e.edge_type IS NOT NULL
            """
        )

    execute(
        connection,
        "CREATE TEMP TABLE edge_all AS\n" + "\nUNION ALL\n".join(union_parts),
    )


def create_degree_table(connection: Any) -> None:
    execute(
        connection,
        """
        CREATE TEMP TABLE node_degree AS
        SELECT
            node_type,
            node_id,
            SUM(out_degree) AS out_degree,
            SUM(in_degree) AS in_degree,
            SUM(total_degree) AS total_degree,
            SUM(weighted_out_degree) AS weighted_out_degree,
            SUM(weighted_in_degree) AS weighted_in_degree,
            SUM(weighted_total_degree) AS weighted_total_degree,
            COUNT(DISTINCT edge_table) AS incident_edge_table_count
        FROM (
            SELECT
                source_type AS node_type,
                source_id AS node_id,
                COUNT(*) AS out_degree,
                0 AS in_degree,
                COUNT(*) AS total_degree,
                SUM(edge_weight) AS weighted_out_degree,
                0.0 AS weighted_in_degree,
                SUM(edge_weight) AS weighted_total_degree,
                edge_table
            FROM edge_all
            GROUP BY source_type, source_id, edge_table

            UNION ALL

            SELECT
                target_type AS node_type,
                target_id AS node_id,
                0 AS out_degree,
                COUNT(*) AS in_degree,
                COUNT(*) AS total_degree,
                0.0 AS weighted_out_degree,
                SUM(edge_weight) AS weighted_in_degree,
                SUM(edge_weight) AS weighted_total_degree,
                edge_table
            FROM edge_all
            GROUP BY target_type, target_id, edge_table
        )
        GROUP BY node_type, node_id
        """,
    )


def compute_node_type_counts(connection: Any) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        """
        SELECT
            n.node_type,
            COUNT(*) AS node_count,
            COUNT(*) FILTER (WHERE d.node_id IS NULL) AS isolated_node_count,
            COUNT(*) FILTER (WHERE d.node_id IS NOT NULL) AS nonisolated_node_count
        FROM node_all n
        LEFT JOIN node_degree d
            ON n.node_type = d.node_type
           AND n.node_id = d.node_id
        GROUP BY n.node_type
        ORDER BY n.node_type
        """,
    )


def compute_edge_type_counts(connection: Any) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        """
        SELECT
            edge_table,
            edge_type,
            source_type,
            target_type,
            COUNT(*) AS edge_count,
            COUNT(DISTINCT source_id) AS distinct_source_count,
            COUNT(DISTINCT target_id) AS distinct_target_count,
            COUNT(*) - COUNT(DISTINCT source_id || '§' || target_id || '§' || edge_type) AS duplicate_basic_edge_count,
            SUM(edge_weight) AS total_edge_weight
        FROM edge_all
        GROUP BY edge_table, edge_type, source_type, target_type
        ORDER BY edge_table, edge_type
        """,
    )


def compute_bioentity_type_counts(connection: Any) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        """
        SELECT
            COALESCE(CAST(bioentity_type AS VARCHAR), '') AS bioentity_type,
            COUNT(*) AS node_count
        FROM node_all
        WHERE node_type = 'BioEntity'
        GROUP BY COALESCE(CAST(bioentity_type AS VARCHAR), '')
        ORDER BY node_count DESC, bioentity_type
        """,
    )


def compute_edge_weight_summary(connection: Any) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        """
        SELECT
            edge_table,
            edge_type,
            COUNT(*) AS edge_count,
            SUM(edge_weight) AS total_edge_weight,
            AVG(edge_weight) AS avg_edge_weight,
            MIN(edge_weight) AS min_edge_weight,
            quantile_cont(edge_weight, 0.50) AS p50_edge_weight,
            quantile_cont(edge_weight, 0.90) AS p90_edge_weight,
            quantile_cont(edge_weight, 0.95) AS p95_edge_weight,
            quantile_cont(edge_weight, 0.99) AS p99_edge_weight,
            MAX(edge_weight) AS max_edge_weight
        FROM edge_all
        GROUP BY edge_table, edge_type
        ORDER BY edge_table, edge_type
        """,
    )


def compute_degree_summary(connection: Any) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        """
        SELECT
            n.node_type,
            COUNT(*) AS node_count,
            AVG(COALESCE(d.total_degree, 0)) AS avg_total_degree,
            quantile_cont(COALESCE(d.total_degree, 0), 0.50) AS p50_total_degree,
            quantile_cont(COALESCE(d.total_degree, 0), 0.90) AS p90_total_degree,
            quantile_cont(COALESCE(d.total_degree, 0), 0.95) AS p95_total_degree,
            quantile_cont(COALESCE(d.total_degree, 0), 0.99) AS p99_total_degree,
            MAX(COALESCE(d.total_degree, 0)) AS max_total_degree,
            AVG(COALESCE(d.weighted_total_degree, 0)) AS avg_weighted_total_degree,
            quantile_cont(COALESCE(d.weighted_total_degree, 0), 0.50) AS p50_weighted_total_degree,
            quantile_cont(COALESCE(d.weighted_total_degree, 0), 0.90) AS p90_weighted_total_degree,
            quantile_cont(COALESCE(d.weighted_total_degree, 0), 0.95) AS p95_weighted_total_degree,
            quantile_cont(COALESCE(d.weighted_total_degree, 0), 0.99) AS p99_weighted_total_degree,
            MAX(COALESCE(d.weighted_total_degree, 0)) AS max_weighted_total_degree
        FROM node_all n
        LEFT JOIN node_degree d
            ON n.node_type = d.node_type
           AND n.node_id = d.node_id
        GROUP BY n.node_type
        ORDER BY n.node_type
        """,
    )


def compute_top_degree_nodes(connection: Any, top_k: int) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        f"""
        SELECT
            n.node_type,
            n.node_id,
            n.label,
            n.bioentity_type,
            COALESCE(d.out_degree, 0) AS out_degree,
            COALESCE(d.in_degree, 0) AS in_degree,
            COALESCE(d.total_degree, 0) AS total_degree,
            COALESCE(d.weighted_out_degree, 0) AS weighted_out_degree,
            COALESCE(d.weighted_in_degree, 0) AS weighted_in_degree,
            COALESCE(d.weighted_total_degree, 0) AS weighted_total_degree,
            COALESCE(d.incident_edge_table_count, 0) AS incident_edge_table_count
        FROM node_all n
        LEFT JOIN node_degree d
            ON n.node_type = d.node_type
           AND n.node_id = d.node_id
        ORDER BY
            COALESCE(d.weighted_total_degree, 0) DESC,
            COALESCE(d.total_degree, 0) DESC,
            n.node_type,
            n.node_id
        LIMIT {int(top_k)}
        """,
    )


def compute_isolated_nodes(connection: Any) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        """
        SELECT
            n.node_type,
            n.node_id,
            n.label,
            n.bioentity_type
        FROM node_all n
        LEFT JOIN node_degree d
            ON n.node_type = d.node_type
           AND n.node_id = d.node_id
        WHERE d.node_id IS NULL
        ORDER BY n.node_type, n.node_id
        """,
    )


def compute_top_weighted_bioentity_edges(connection: Any, top_k: int) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        f"""
        SELECT
            e.edge_table,
            e.edge_type,
            e.source_type,
            e.source_id,
            sn.label AS source_label,
            e.target_type,
            e.target_id,
            tn.label AS target_label,
            tn.bioentity_type AS target_bioentity_type,
            e.edge_weight,
            e.mention_count
        FROM edge_all e
        LEFT JOIN node_all sn
            ON e.source_type = sn.node_type
           AND e.source_id = sn.node_id
        LEFT JOIN node_all tn
            ON e.target_type = tn.node_type
           AND e.target_id = tn.node_id
        WHERE e.edge_table IN (
            'edges_patent_bioentity',
            'edges_paper_bioentity',
            'edges_trial_bioentity'
        )
        ORDER BY e.edge_weight DESC, e.edge_table, e.source_id, e.target_id
        LIMIT {int(top_k)}
        """,
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)

        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left

        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def typed_key(node_type: str, node_id: str) -> str:
    return f"{node_type}\x1f{node_id}"


def compute_connected_components(connection: Any) -> list[dict[str, Any]]:
    nodes = connection.execute(
        """
        SELECT node_type, node_id
        FROM node_all
        """
    ).fetchall()

    edges = connection.execute(
        """
        SELECT source_type, source_id, target_type, target_id
        FROM edge_all
        """
    ).fetchall()

    uf = UnionFind()

    node_key_to_type: dict[str, str] = {}

    for node_type, node_id in nodes:
        key = typed_key(str(node_type), str(node_id))
        uf.add(key)
        node_key_to_type[key] = str(node_type)

    for source_type, source_id, target_type, target_id in edges:
        source_key = typed_key(str(source_type), str(source_id))
        target_key = typed_key(str(target_type), str(target_id))
        uf.union(source_key, target_key)

    component_type_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for key, node_type in node_key_to_type.items():
        root = uf.find(key)
        component_type_counts[root][node_type] += 1

    component_rows = []

    sorted_components = sorted(
        component_type_counts.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    for index, (_root, type_counter) in enumerate(sorted_components, start=1):
        total = sum(type_counter.values())
        component_rows.append(
            {
                "component_rank": index,
                "component_id": f"C{index:06d}",
                "node_count": total,
                "patent_count": type_counter.get("Patent", 0),
                "paper_count": type_counter.get("Paper", 0),
                "clinicaltrial_count": type_counter.get("ClinicalTrial", 0),
                "project_count": type_counter.get("Project", 0),
                "bioentity_count": type_counter.get("BioEntity", 0),
            }
        )

    return component_rows


def write_all_outputs(
    output_dir: Path,
    node_type_counts: list[dict[str, Any]],
    edge_type_counts: list[dict[str, Any]],
    bioentity_type_counts: list[dict[str, Any]],
    edge_weight_summary: list[dict[str, Any]],
    degree_summary: list[dict[str, Any]],
    top_degree_nodes: list[dict[str, Any]],
    connected_components: list[dict[str, Any]],
    isolated_nodes: list[dict[str, Any]],
    top_weighted_bioentity_edges: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "graph_node_type_counts.csv", node_type_counts)
    write_csv(output_dir / "graph_edge_type_counts.csv", edge_type_counts)
    write_csv(output_dir / "graph_bioentity_type_counts.csv", bioentity_type_counts)
    write_csv(output_dir / "graph_edge_weight_summary.csv", edge_weight_summary)
    write_csv(output_dir / "graph_degree_summary.csv", degree_summary)
    write_csv(output_dir / "graph_top_degree_nodes.csv", top_degree_nodes)
    write_csv(output_dir / "graph_connected_components.csv", connected_components)
    write_csv(output_dir / "graph_isolated_nodes.csv", isolated_nodes)
    write_csv(output_dir / "graph_top_weighted_bioentity_edges.csv", top_weighted_bioentity_edges)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    runtime_seconds: float,
    node_type_counts: list[dict[str, Any]],
    edge_type_counts: list[dict[str, Any]],
    bioentity_type_counts: list[dict[str, Any]],
    edge_weight_summary: list[dict[str, Any]],
    degree_summary: list[dict[str, Any]],
    top_degree_nodes: list[dict[str, Any]],
    connected_components: list[dict[str, Any]],
    isolated_nodes: list[dict[str, Any]],
    top_weighted_bioentity_edges: list[dict[str, Any]],
) -> None:
    total_nodes = sum(int(row.get("node_count", 0) or 0) for row in node_type_counts)
    total_edges = sum(int(row.get("edge_count", 0) or 0) for row in edge_type_counts)
    isolated_count = len(isolated_nodes)
    component_count = len(connected_components)
    largest_component_size = connected_components[0]["node_count"] if connected_components else 0
    largest_component_ratio = (largest_component_size / total_nodes) if total_nodes else 0.0

    overview_rows = [
        {"metric": "graph_dir", "value": str(args.graph_dir)},
        {"metric": "output_dir", "value": str(args.output_dir)},
        {"metric": "total_nodes", "value": total_nodes},
        {"metric": "total_edges", "value": total_edges},
        {"metric": "connected_component_count", "value": component_count},
        {"metric": "largest_component_node_count", "value": largest_component_size},
        {"metric": "largest_component_node_ratio", "value": f"{largest_component_ratio:.4f}"},
        {"metric": "isolated_node_count", "value": isolated_count},
        {"metric": "threads", "value": args.threads},
        {"metric": "memory_limit", "value": args.memory_limit},
        {"metric": "runtime_seconds", "value": f"{runtime_seconds:.1f}"},
    ]

    output_rows = [
        {"artifact": "graph_node_type_counts.csv", "description": "Node counts and isolated node counts by node type."},
        {"artifact": "graph_edge_type_counts.csv", "description": "Edge counts, distinct endpoints, duplicate basic edge counts, and total weights by edge type."},
        {"artifact": "graph_bioentity_type_counts.csv", "description": "BioEntity node counts by BioEntity Type."},
        {"artifact": "graph_edge_weight_summary.csv", "description": "Edge weight distribution summary by edge table and edge type."},
        {"artifact": "graph_degree_summary.csv", "description": "Degree and weighted degree distribution summary by node type."},
        {"artifact": "graph_top_degree_nodes.csv", "description": "Top nodes by weighted total degree."},
        {"artifact": "graph_connected_components.csv", "description": "Undirected connected component sizes and node-type composition."},
        {"artifact": "graph_isolated_nodes.csv", "description": "Nodes with no incident edges."},
        {"artifact": "graph_top_weighted_bioentity_edges.csv", "description": "Top BioEntity edges by edge_weight / mention_count."},
        {"artifact": "graph_statistics_report.md", "description": "Human-readable graph statistics report."},
    ]

    parts = ["# Graph Statistics Report\n"]

    parts.append("## 1. Overview\n")
    parts.append(markdown_table(overview_rows, ["metric", "value"]))

    parts.append("\n## 2. Node type counts\n")
    parts.append(markdown_table(node_type_counts, ["node_type", "node_count", "isolated_node_count", "nonisolated_node_count"]))

    parts.append("\n## 3. Edge type counts\n")
    parts.append(
        markdown_table(
            edge_type_counts,
            [
                "edge_table",
                "edge_type",
                "source_type",
                "target_type",
                "edge_count",
                "distinct_source_count",
                "distinct_target_count",
                "duplicate_basic_edge_count",
                "total_edge_weight",
            ],
        )
    )

    parts.append("\n## 4. BioEntity type counts\n")
    parts.append(markdown_table(bioentity_type_counts[:50], ["bioentity_type", "node_count"]))

    parts.append("\n## 5. Edge weight summary\n")
    parts.append(
        markdown_table(
            edge_weight_summary,
            [
                "edge_table",
                "edge_type",
                "edge_count",
                "total_edge_weight",
                "avg_edge_weight",
                "min_edge_weight",
                "p50_edge_weight",
                "p90_edge_weight",
                "p95_edge_weight",
                "p99_edge_weight",
                "max_edge_weight",
            ],
        )
    )

    parts.append("\n## 6. Degree summary\n")
    parts.append(
        markdown_table(
            degree_summary,
            [
                "node_type",
                "node_count",
                "avg_total_degree",
                "p50_total_degree",
                "p90_total_degree",
                "p95_total_degree",
                "p99_total_degree",
                "max_total_degree",
                "avg_weighted_total_degree",
                "p50_weighted_total_degree",
                "p90_weighted_total_degree",
                "p95_weighted_total_degree",
                "p99_weighted_total_degree",
                "max_weighted_total_degree",
            ],
        )
    )

    parts.append("\n## 7. Top degree nodes\n")
    parts.append(
        markdown_table(
            top_degree_nodes[:50],
            [
                "node_type",
                "node_id",
                "label",
                "bioentity_type",
                "out_degree",
                "in_degree",
                "total_degree",
                "weighted_out_degree",
                "weighted_in_degree",
                "weighted_total_degree",
                "incident_edge_table_count",
            ],
        )
    )

    parts.append("\n## 8. Connected components\n")
    parts.append(
        markdown_table(
            connected_components[:50],
            [
                "component_rank",
                "component_id",
                "node_count",
                "patent_count",
                "paper_count",
                "clinicaltrial_count",
                "project_count",
                "bioentity_count",
            ],
        )
    )

    parts.append("\n## 9. Top weighted BioEntity edges\n")
    parts.append(
        markdown_table(
            top_weighted_bioentity_edges[:50],
            [
                "edge_table",
                "edge_type",
                "source_type",
                "source_id",
                "source_label",
                "target_type",
                "target_id",
                "target_label",
                "target_bioentity_type",
                "edge_weight",
                "mention_count",
            ],
        )
    )

    parts.append("\n## 10. Isolated node summary\n")
    isolated_by_type = Counter(str(row["node_type"]) for row in isolated_nodes)
    isolated_summary = [
        {"node_type": node_type, "isolated_node_count": count}
        for node_type, count in sorted(isolated_by_type.items())
    ]
    parts.append(markdown_table(isolated_summary, ["node_type", "isolated_node_count"]))

    parts.append("\n## 11. Output artifacts\n")
    parts.append(markdown_table(output_rows, ["artifact", "description"]))

    parts.append("\n## 12. Interpretation notes\n")
    parts.append(
        "- BioEntity edge weights use `mention_count` when available. Non-BioEntity edges use weight 1.0.\n"
        "- High-degree species nodes such as humans, patients, mice, and rat are common in biomedical NER outputs. "
        "They should be audited before algorithm experiments and may be filtered or downweighted in sensitivity analyses.\n"
        "- Cross-disease entities such as cancer or Alzheimer's disease may represent comorbidity, shared mechanisms, "
        "drug repurposing, or broader biomedical context. They should generally be tagged or filtered in derived "
        "analysis graphs rather than removed from the main context-preserving graph.\n"
        "- If the largest connected component contains most nodes, the graph is suitable for whole-graph representation "
        "learning. If many isolated nodes exist, consider filtering isolated nodes before link prediction or GNN training.\n"
    )

    (output_dir / "graph_statistics_report.md").write_text("\n".join(parts), encoding="utf-8")


def print_summary(
    output_dir: Path,
    node_type_counts: list[dict[str, Any]],
    edge_type_counts: list[dict[str, Any]],
    connected_components: list[dict[str, Any]],
    isolated_nodes: list[dict[str, Any]],
) -> None:
    total_nodes = sum(int(row.get("node_count", 0) or 0) for row in node_type_counts)
    total_edges = sum(int(row.get("edge_count", 0) or 0) for row in edge_type_counts)
    component_count = len(connected_components)
    largest_component = connected_components[0]["node_count"] if connected_components else 0

    print("Graph statistics complete")
    print(f"Output directory: {output_dir}")
    print("Graph summary:")
    print(f"- total_nodes: {total_nodes:,}")
    print(f"- total_edges: {total_edges:,}")
    print(f"- connected_components: {component_count:,}")
    print(f"- largest_component_node_count: {largest_component:,}")
    print(f"- isolated_nodes: {len(isolated_nodes):,}")

    print("Node counts:")
    for row in node_type_counts:
        print(f"- {row['node_type']}: {int(row['node_count']):,}")

    print("Edge counts:")
    for row in edge_type_counts:
        print(f"- {row['edge_table']}: {int(row['edge_count']):,}")

    print("Report: graph_statistics_report.md")


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if args.threads < 1:
        raise ValueError("--threads must be greater than or equal to 1")

    args.graph_dir = args.graph_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.graph_dir
    else:
        args.output_dir = args.output_dir.resolve()

    args.temp_dir = args.temp_dir.resolve()

    require_files(args.graph_dir)
    prepare_output_dir(args.graph_dir, args.output_dir, args.overwrite)

    connection = connect_duckdb(args)

    run_step(args, 1, 8, "Creating input views", lambda: create_input_views(connection, args.graph_dir))
    run_step(args, 2, 8, "Creating unified node table", lambda: create_node_union(connection))
    run_step(args, 3, 8, "Creating unified edge table", lambda: create_edge_union(connection))
    run_step(args, 4, 8, "Computing degree table", lambda: create_degree_table(connection))

    results: dict[str, Any] = {}

    def compute_sql_statistics() -> None:
        results["node_type_counts"] = compute_node_type_counts(connection)
        results["edge_type_counts"] = compute_edge_type_counts(connection)
        results["bioentity_type_counts"] = compute_bioentity_type_counts(connection)
        results["edge_weight_summary"] = compute_edge_weight_summary(connection)
        results["degree_summary"] = compute_degree_summary(connection)
        results["top_degree_nodes"] = compute_top_degree_nodes(connection, args.top_k)
        results["isolated_nodes"] = compute_isolated_nodes(connection)
        results["top_weighted_bioentity_edges"] = compute_top_weighted_bioentity_edges(connection, args.top_k)

    run_step(args, 5, 8, "Computing SQL graph statistics", compute_sql_statistics)

    def compute_components() -> None:
        results["connected_components"] = compute_connected_components(connection)

    run_step(args, 6, 8, "Computing connected components", compute_components)

    run_step(
        args,
        7,
        8,
        "Writing CSV outputs",
        lambda: write_all_outputs(
            args.output_dir,
            results["node_type_counts"],
            results["edge_type_counts"],
            results["bioentity_type_counts"],
            results["edge_weight_summary"],
            results["degree_summary"],
            results["top_degree_nodes"],
            results["connected_components"],
            results["isolated_nodes"],
            results["top_weighted_bioentity_edges"],
        ),
    )

    runtime_seconds = time.time() - start_time

    run_step(
        args,
        8,
        8,
        "Writing graph statistics report",
        lambda: write_report(
            args.output_dir,
            args,
            runtime_seconds,
            results["node_type_counts"],
            results["edge_type_counts"],
            results["bioentity_type_counts"],
            results["edge_weight_summary"],
            results["degree_summary"],
            results["top_degree_nodes"],
            results["connected_components"],
            results["isolated_nodes"],
            results["top_weighted_bioentity_edges"],
        ),
    )

    print_summary(
        args.output_dir,
        results["node_type_counts"],
        results["edge_type_counts"],
        results["connected_components"],
        results["isolated_nodes"],
    )


if __name__ == "__main__":
    main()