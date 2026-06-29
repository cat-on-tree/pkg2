"""Reporting utilities for algorithm-stage dataset construction.

This module writes lightweight manifests and Markdown reports for datasets
produced from the PKG2 Knowledge Carrier Graph prototype.

The current primary use case is Patent-Paper link prediction dataset
construction. The report explicitly records that this is a carrier-level proxy
task, not the final knowledge-unit-level translation prediction task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import markdown_table, write_csv, write_text


def format_int(value: Any) -> str:
    """Format an integer-like value with commas."""

    if value is None:
        return ""

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def format_float(value: Any, *, digits: int = 4) -> str:
    """Format a float-like value."""

    if value is None:
        return ""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_ratio(value: Any, *, digits: int = 2) -> str:
    """Format a ratio as a percentage string."""

    if value is None:
        return ""

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def normalize_manifest_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize manifest rows into a stable list of dictionaries."""

    normalized = []

    for row in rows:
        normalized.append(
            {
                "artifact": row.get("artifact", ""),
                "path": row.get("path", ""),
                "description": row.get("description", ""),
                "row_count": row.get("row_count", ""),
                "file_size_bytes": row.get("file_size_bytes", ""),
            }
        )

    return normalized


def write_manifest(
    output_dir: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    file_name: str = "dataset_manifest.csv",
) -> Path:
    """Write a dataset manifest CSV file.

    Parameters
    ----------
    output_dir:
        Dataset output directory.
    rows:
        Iterable of manifest rows. Expected keys are artifact, path,
        description, row_count, and file_size_bytes.
    file_name:
        Manifest file name.

    Returns
    -------
    Path
        Path to the written manifest file.
    """

    output_dir = Path(output_dir)
    output_path = output_dir / file_name

    normalized_rows = normalize_manifest_rows(rows)

    write_csv(
        output_path,
        normalized_rows,
        columns=[
            "artifact",
            "path",
            "description",
            "row_count",
            "file_size_bytes",
        ],
    )

    return output_path


def build_artifact_rows(
    output_dir: str | Path,
    artifacts: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build manifest rows for output artifacts.

    Each input artifact may contain:
    - artifact
    - file_name
    - description
    - row_count

    File size is filled when the file exists.
    """

    output_dir = Path(output_dir)
    rows = []

    for artifact in artifacts:
        file_name = str(artifact.get("file_name", ""))
        path = output_dir / file_name if file_name else output_dir

        rows.append(
            {
                "artifact": artifact.get("artifact", ""),
                "path": str(path),
                "description": artifact.get("description", ""),
                "row_count": artifact.get("row_count", ""),
                "file_size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
            }
        )

    return rows


def _parameters_table(parameters: dict[str, Any]) -> str:
    """Render parameters as a two-column Markdown table."""

    rows = [{"parameter": key, "value": value} for key, value in parameters.items()]
    return markdown_table(rows, ["parameter", "value"])


def _counts_table(counts: dict[str, Any]) -> str:
    """Render a dictionary of counts as a two-column Markdown table."""

    rows = [{"name": key, "value": value} for key, value in counts.items()]
    return markdown_table(rows, ["name", "value"])


def _filtering_summary_table(filtering_summary: dict[str, Any]) -> str:
    """Render graph filtering summary."""

    rows = [
        {
            "metric": "source_node_count",
            "value": format_int(filtering_summary.get("source_node_count")),
        },
        {
            "metric": "filtered_node_count",
            "value": format_int(filtering_summary.get("filtered_node_count")),
        },
        {
            "metric": "removed_node_count",
            "value": format_int(filtering_summary.get("removed_node_count")),
        },
        {
            "metric": "retained_node_ratio",
            "value": format_ratio(filtering_summary.get("retained_node_ratio")),
        },
        {
            "metric": "source_edge_count",
            "value": format_int(filtering_summary.get("source_edge_count")),
        },
        {
            "metric": "filtered_edge_count",
            "value": format_int(filtering_summary.get("filtered_edge_count")),
        },
        {
            "metric": "removed_edge_count",
            "value": format_int(filtering_summary.get("removed_edge_count")),
        },
        {
            "metric": "retained_edge_ratio",
            "value": format_ratio(filtering_summary.get("retained_edge_ratio")),
        },
    ]

    return markdown_table(rows, ["metric", "value"])


def _node_type_table(rows: list[dict[str, Any]]) -> str:
    """Render node type filtering counts."""

    formatted = []

    for row in rows:
        formatted.append(
            {
                "node_type": row.get("node_type", ""),
                "source_node_count": format_int(row.get("source_node_count")),
                "filtered_node_count": format_int(row.get("filtered_node_count")),
                "removed_node_count": format_int(row.get("removed_node_count")),
            }
        )

    return markdown_table(
        formatted,
        [
            "node_type",
            "source_node_count",
            "filtered_node_count",
            "removed_node_count",
        ],
    )


def _edge_table_filtering_table(rows: list[dict[str, Any]]) -> str:
    """Render edge table filtering counts."""

    formatted = []

    for row in rows:
        formatted.append(
            {
                "edge_table": row.get("edge_table", ""),
                "edge_type": row.get("edge_type", ""),
                "source_edge_count": format_int(row.get("source_edge_count")),
                "filtered_edge_count": format_int(row.get("filtered_edge_count")),
                "removed_edge_count": format_int(row.get("removed_edge_count")),
            }
        )

    return markdown_table(
        formatted,
        [
            "edge_table",
            "edge_type",
            "source_edge_count",
            "filtered_edge_count",
            "removed_edge_count",
        ],
    )


def _split_summary_table(
    positive_split_summary: list[dict[str, Any]],
    negative_split_summary: list[dict[str, Any]],
) -> str:
    """Render combined positive and negative split counts."""

    positive_by_split = {
        row.get("split"): row.get("positive_edge_count", 0)
        for row in positive_split_summary
    }
    negative_by_split = {
        row.get("split"): row.get("negative_edge_count", 0)
        for row in negative_split_summary
    }

    rows = []

    for split in ["train", "val", "test"]:
        positive_count = int(positive_by_split.get(split, 0) or 0)
        negative_count = int(negative_by_split.get(split, 0) or 0)
        total_count = positive_count + negative_count

        rows.append(
            {
                "split": split,
                "positive_edges": format_int(positive_count),
                "negative_edges": format_int(negative_count),
                "total_labeled_edges": format_int(total_count),
                "positive_ratio": format_ratio(
                    positive_count / total_count if total_count else 0.0
                ),
            }
        )

    return markdown_table(
        rows,
        [
            "split",
            "positive_edges",
            "negative_edges",
            "total_labeled_edges",
            "positive_ratio",
        ],
    )


def _negative_sampling_summary_table(negative_sampling_summary: dict[str, Any]) -> str:
    """Render negative sampling summary."""

    candidate_space = negative_sampling_summary.get("candidate_space", {})

    rows = [
        {
            "metric": "source_node_type",
            "value": negative_sampling_summary.get("source_node_type", ""),
        },
        {
            "metric": "target_node_type",
            "value": negative_sampling_summary.get("target_node_type", ""),
        },
        {
            "metric": "negative_ratio",
            "value": negative_sampling_summary.get("negative_ratio", ""),
        },
        {
            "metric": "source_node_count",
            "value": format_int(candidate_space.get("source_node_count")),
        },
        {
            "metric": "target_node_count",
            "value": format_int(candidate_space.get("target_node_count")),
        },
        {
            "metric": "candidate_pair_count",
            "value": format_int(candidate_space.get("candidate_pair_count")),
        },
        {
            "metric": "all_positive_pair_count",
            "value": format_int(negative_sampling_summary.get("all_positive_pair_count")),
        },
        {
            "metric": "negative_candidate_count",
            "value": format_int(negative_sampling_summary.get("negative_candidate_count")),
        },
    ]

    return markdown_table(rows, ["metric", "value"])


def _artifact_table(rows: list[dict[str, Any]]) -> str:
    """Render output artifact manifest rows."""

    formatted = []

    for row in rows:
        formatted.append(
            {
                "artifact": row.get("artifact", ""),
                "path": row.get("path", ""),
                "description": row.get("description", ""),
                "row_count": format_int(row.get("row_count"))
                if row.get("row_count") != ""
                else "",
                "file_size_bytes": format_int(row.get("file_size_bytes"))
                if row.get("file_size_bytes") != ""
                else "",
            }
        )

    return markdown_table(
        formatted,
        [
            "artifact",
            "path",
            "description",
            "row_count",
            "file_size_bytes",
        ],
    )


def write_dataset_report(
    output_dir: str | Path,
    *,
    parameters: dict[str, Any],
    graph_counts: dict[str, Any],
    filtering_summary: dict[str, Any],
    node_type_filtering: list[dict[str, Any]],
    edge_table_filtering: list[dict[str, Any]],
    positive_split_summary: list[dict[str, Any]],
    negative_split_summary: list[dict[str, Any]],
    negative_sampling_summary: dict[str, Any],
    artifact_rows: list[dict[str, Any]],
    file_name: str = "dataset_report.md",
) -> Path:
    """Write a Markdown report for a link prediction dataset."""

    output_dir = Path(output_dir)
    output_path = output_dir / file_name

    lines = [
        "# Link Prediction Dataset Report",
        "",
        "## 1. Dataset positioning",
        "",
        "This dataset is constructed from the PKG2 diabetes Knowledge Carrier Graph prototype.",
        "",
        "The current task is Patent-Paper link prediction. This is a carrier-level proxy task used to validate graph signal and establish reusable dataset construction infrastructure.",
        "",
        "It is not the final knowledge-unit-level translation prediction task. Later stages will move from carrier-level prediction to Knowledge Unit extraction, Knowledge State construction, and Knowledge Field dynamics.",
        "",
        "## 2. Parameters",
        "",
        _parameters_table(parameters),
        "",
        "## 3. Loaded graph counts",
        "",
        _counts_table(graph_counts),
        "",
        "## 4. Filtering summary",
        "",
        _filtering_summary_table(filtering_summary),
        "",
        "## 5. Node type filtering",
        "",
        _node_type_table(node_type_filtering),
        "",
        "## 6. Edge table filtering",
        "",
        _edge_table_filtering_table(edge_table_filtering),
        "",
        "## 7. Train / validation / test split summary",
        "",
        _split_summary_table(positive_split_summary, negative_split_summary),
        "",
        "## 8. Negative sampling summary",
        "",
        _negative_sampling_summary_table(negative_sampling_summary),
        "",
        "## 9. Output artifacts",
        "",
        _artifact_table(artifact_rows),
        "",
        "## 10. Leakage-control notes",
        "",
        "- Positive target edges are deduplicated before splitting.",
        "- Negative samples are drawn from source-target candidate pairs excluding all known positive target pairs.",
        "- Train, validation, and test negative samples are sampled from non-overlapping ranges of randomized negative candidates.",
        "- The context graph should exclude the target edge table to avoid target-edge leakage.",
        "",
        "## 11. Conceptual notes",
        "",
        "Paper, patent, and clinical trial nodes are treated as knowledge carriers rather than knowledge units.",
        "",
        "The long-term modeling target is not merely whether a patent links to a paper, but how reusable knowledge units, such as entity pairs or entity-relation triples, accumulate evidence, mature, diffuse, and translate across scientific, clinical, and technological carrier spaces.",
        "",
    ]

    write_text(output_path, "\n".join(lines))
    return output_path