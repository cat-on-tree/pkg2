"""Reusable utilities for the PKG2 biomedical translation graph project.

This package contains reusable code for the algorithm-stage pipeline.

Current stage
-------------
The project has completed:

- Knowledge Carrier Graph construction;
- carrier-level Patent-Paper link prediction dataset construction;
- carrier-level Patent-Paper baseline experiments;
- Knowledge Unit construction MVP;
- Knowledge Unit audit;
- Knowledge Unit temporal panel construction;
- Knowledge Unit sanity checking;
- trial year parsing and temporal panel refresh.

The current official prototype is a diabetes-focused 2018-2019 patent-centered
Knowledge Carrier Graph with a validated typed BioEntity-pair Knowledge Unit
layer.

Current official graph input:

    data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/

Current official Knowledge Unit dataset:

    data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/

Current official dense temporal panel:

    data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel/

Current focus
-------------
The project is now moving from raw Knowledge Unit construction toward:

- rule-based Knowledge State proxy features;
- discovery-oriented Knowledge Unit ranking;
- future KU-level temporal translation task definition;
- temporal baselines and reaction-diffusion-source Knowledge Field modeling.

Long-term direction
-------------------
The final research target is knowledge-unit-level temporal dynamics:

    Carrier Graph
        -> Knowledge Unit
        -> Knowledge State
        -> Knowledge Field
        -> Translation Prediction
        -> Interpretable Knowledge Flow

Implementation convention
-------------------------
Modules under ``src/pkg2`` provide reusable infrastructure for:

- loading extracted / aggregated carrier graphs;
- filtering graph nodes and edges;
- constructing train / validation / test splits;
- sampling negative examples;
- exporting link prediction datasets;
- evaluating carrier-level baselines;
- writing dataset manifests and reports;
- constructing typed BioEntity-pair Knowledge Units;
- building KU-level temporal panels;
- constructing future Knowledge State proxy features.

Script files under ``scripts/`` should remain thin CLI orchestration layers.

Important interpretation note
-----------------------------
Carrier-level Patent-Paper link prediction is an intermediate validation task,
not the final project contribution.

The current Knowledge Units are typed BioEntity-pair co-mention units, not
asserted biomedical relation claims. For example:

- a chemical-disease pair is not necessarily a treatment claim;
- a gene-disease pair is not necessarily a causal claim;
- a chemical-gene pair is not necessarily a target claim.

The main project contribution should continue toward Knowledge State
construction, KU-level temporal translation modeling, discovery-oriented
candidate ranking, and interpretable Knowledge Field dynamics.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"