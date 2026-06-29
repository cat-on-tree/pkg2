"""Reusable utilities for the PKG2 diabetes translation graph project.

This package contains reusable code for the algorithm-stage pipeline.

Current stage
-------------
The project has completed the initial Knowledge Carrier Graph construction
stage and is now entering carrier-level proxy task construction, starting with
Patent-Paper link prediction.

Long-term direction
-------------------
The final research target is knowledge-unit-level temporal dynamics:

    Carrier Graph
        -> Knowledge Unit
        -> Knowledge State
        -> Knowledge Field
        -> Translation Prediction

In the current implementation stage, the modules under ``src/pkg2`` provide
reusable infrastructure for:

- loading extracted / aggregated carrier graphs;
- filtering graph nodes and edges;
- constructing train / validation / test splits;
- sampling negative examples;
- exporting link prediction datasets;
- writing dataset manifests and reports.

The initial downstream task is Patent-Paper link prediction. This is a
carrier-level proxy task used to validate the graph and establish reusable
dataset-building infrastructure. It is not the final knowledge-unit-level
translation prediction task.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"