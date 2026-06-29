"""I/O helpers for graph representation baseline artifacts.

This module centralizes saving and loading of artifacts produced by graph
representation baselines such as Node2Vec, MetaPath2Vec, KG embeddings, and GNNs.

Typical artifacts include:

- embeddings.npy
- node_index.csv
- relation_index.csv
- embedding_metadata.json
- predictions_train.parquet
- predictions_val.parquet
- predictions_test.parquet
- metrics.csv
- metrics_summary.csv
- baseline_manifest.csv
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkg2.graph_data import (
    NodeIndex,
    RelationIndex,
    node_index_to_frame,
    relation_index_to_frame,
)


def prepare_output_dir(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    allow_existing_empty: bool = True,
) -> Path:
    """Prepare an output directory.

    Parameters
    ----------
    output_dir:
        Directory to create.

    overwrite:
        If True, delete and recreate the directory when it already exists.

    allow_existing_empty:
        If True, allow an existing empty directory when overwrite=False.
    """

    output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            if not allow_existing_empty:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite to replace it."
                )

            existing = list(output_dir.iterdir())
            if existing:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Use --overwrite to replace it."
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure parent directory exists and return normalized path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save embedding matrix as .npy."""

    output_path = ensure_parent_dir(output_path)
    np.save(output_path, np.asarray(embeddings, dtype=np.float32))
    return output_path


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embedding matrix from .npy."""

    return np.load(Path(path))


def write_json(
    path: str | Path,
    data: dict[str, Any],
    *,
    indent: int = 2,
) -> Path:
    """Write JSON file."""

    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False, sort_keys=True)

    return path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON file."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_dataframe(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Write a DataFrame based on file suffix.

    Supported suffixes:

    - .parquet
    - .csv
    - .tsv
    - .json
    - .jsonl
    """

    path = ensure_parent_dir(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        frame.to_parquet(path, index=index)
    elif suffix == ".csv":
        frame.to_csv(path, index=index)
    elif suffix in {".tsv", ".tab"}:
        frame.to_csv(path, sep="\t", index=index)
    elif suffix == ".json":
        frame.to_json(path, orient="records", indent=2, force_ascii=False)
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported DataFrame output suffix: {path}")

    return path


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a DataFrame based on file suffix."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")

    if suffix == ".json":
        return pd.read_json(path)

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported DataFrame input suffix: {path}")


def save_node_index(
    node_index: NodeIndex,
    output_path: str | Path,
) -> Path:
    """Save NodeIndex as CSV or other DataFrame format."""

    frame = node_index_to_frame(node_index)
    return write_dataframe(frame, output_path)


def save_relation_index(
    relation_index: RelationIndex,
    output_path: str | Path,
) -> Path:
    """Save RelationIndex as CSV or other DataFrame format."""

    frame = relation_index_to_frame(relation_index)
    return write_dataframe(frame, output_path)


def save_predictions_by_split(
    predictions_by_split: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    file_prefix: str = "predictions",
    file_suffix: str = ".parquet",
) -> list[dict[str, Any]]:
    """Save prediction DataFrames by split.

    Returns artifact rows suitable for a manifest.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []

    for split, frame in predictions_by_split.items():
        path = output_dir / f"{file_prefix}_{split}{file_suffix}"
        write_dataframe(frame, path)

        artifacts.append(
            make_artifact_row(
                artifact=f"{file_prefix}_{split}",
                path=path,
                description=f"Prediction scores for the {split} split.",
                row_count=len(frame),
            )
        )

    return artifacts


def save_embedding_artifacts(
    *,
    output_dir: str | Path,
    embeddings: np.ndarray,
    node_index: NodeIndex,
    metadata: dict[str, Any] | None = None,
    relation_index: RelationIndex | None = None,
    embeddings_filename: str = "embeddings.npy",
    node_index_filename: str = "node_index.csv",
    relation_index_filename: str = "relation_index.csv",
    metadata_filename: str = "embedding_metadata.json",
) -> list[dict[str, Any]]:
    """Save common embedding artifacts and return manifest rows."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []

    embeddings_path = save_embeddings(embeddings, output_dir / embeddings_filename)
    artifacts.append(
        make_artifact_row(
            artifact="embeddings",
            path=embeddings_path,
            description="Node embedding matrix saved as NumPy .npy file.",
            row_count=int(np.asarray(embeddings).shape[0]),
        )
    )

    node_index_path = save_node_index(node_index, output_dir / node_index_filename)
    artifacts.append(
        make_artifact_row(
            artifact="node_index",
            path=node_index_path,
            description="Node index mapping embedding rows to node_type and node_id.",
            row_count=len(node_index.nodes),
        )
    )

    if relation_index is not None:
        relation_index_path = save_relation_index(
            relation_index,
            output_dir / relation_index_filename,
        )
        artifacts.append(
            make_artifact_row(
                artifact="relation_index",
                path=relation_index_path,
                description="Relation index mapping relation names to relation IDs.",
                row_count=len(relation_index.relation_to_idx),
            )
        )

    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("num_nodes", int(np.asarray(embeddings).shape[0]))
    metadata_payload.setdefault("embedding_dim", int(np.asarray(embeddings).shape[1]))

    metadata_path = write_json(output_dir / metadata_filename, metadata_payload)
    artifacts.append(
        make_artifact_row(
            artifact="embedding_metadata",
            path=metadata_path,
            description="Embedding training and data metadata.",
        )
    )

    return artifacts


def make_artifact_row(
    *,
    artifact: str,
    path: str | Path,
    description: str,
    row_count: int | str = "",
) -> dict[str, Any]:
    """Create one artifact manifest row."""

    path = Path(path)

    return {
        "artifact": artifact,
        "path": str(path),
        "description": description,
        "row_count": row_count,
        "file_size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
    }


def write_manifest(
    output_dir: str | Path,
    artifact_rows: list[dict[str, Any]],
    *,
    filename: str = "baseline_manifest.csv",
) -> Path:
    """Write artifact manifest CSV."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    frame = pd.DataFrame(
        artifact_rows,
        columns=[
            "artifact",
            "path",
            "description",
            "row_count",
            "file_size_bytes",
        ],
    )

    write_dataframe(frame, path)
    return path


def append_manifest_artifact(
    artifact_rows: list[dict[str, Any]],
    *,
    artifact: str,
    path: str | Path,
    description: str,
    row_count: int | str = "",
) -> list[dict[str, Any]]:
    """Append an artifact row and return the same list."""

    artifact_rows.append(
        make_artifact_row(
            artifact=artifact,
            path=path,
            description=description,
            row_count=row_count,
        )
    )

    return artifact_rows


def file_size_bytes(path: str | Path) -> int | str:
    """Return file size in bytes or empty string if path does not exist."""

    path = Path(path)

    if not path.exists() or not path.is_file():
        return ""

    return path.stat().st_size


def summarize_artifact_paths(
    artifact_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Create a compact mapping from artifact name to path."""

    return {
        str(row.get("artifact", "")): str(row.get("path", ""))
        for row in artifact_rows
    }


def save_training_history(
    history_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save training history rows."""

    frame = pd.DataFrame(history_rows)
    return write_dataframe(frame, output_path)


def save_metrics_dict(
    metrics_by_split: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save nested metrics dict as JSON."""

    serializable = {}

    for split, metrics in metrics_by_split.items():
        serializable[str(split)] = {
            str(key): _to_jsonable(value)
            for key, value in metrics.items()
        }

    return write_json(output_path, serializable)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalar values to JSON-serializable Python values."""

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str)) else False:
        return None

    return value


def load_embedding_artifacts(
    output_dir: str | Path,
    *,
    embeddings_filename: str = "embeddings.npy",
    node_index_filename: str = "node_index.csv",
    relation_index_filename: str = "relation_index.csv",
    metadata_filename: str = "embedding_metadata.json",
) -> dict[str, Any]:
    """Load common embedding artifacts if they exist.

    Returns a dictionary with keys:

    - embeddings
    - node_index_frame
    - relation_index_frame
    - metadata
    """

    output_dir = Path(output_dir)

    artifacts: dict[str, Any] = {}

    embeddings_path = output_dir / embeddings_filename
    node_index_path = output_dir / node_index_filename
    relation_index_path = output_dir / relation_index_filename
    metadata_path = output_dir / metadata_filename

    if embeddings_path.exists():
        artifacts["embeddings"] = load_embeddings(embeddings_path)

    if node_index_path.exists():
        artifacts["node_index_frame"] = read_dataframe(node_index_path)

    if relation_index_path.exists():
        artifacts["relation_index_frame"] = read_dataframe(relation_index_path)

    if metadata_path.exists():
        artifacts["metadata"] = read_json(metadata_path)

    return artifacts


def write_text(
    path: str | Path,
    text: str,
) -> Path:
    """Write text file."""

    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)

    return path


def read_text(path: str | Path) -> str:
    """Read text file."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return file.read()