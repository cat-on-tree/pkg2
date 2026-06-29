# AGENTS.md

## Project status

This repository builds and analyzes a PKG Diabetes Translation Graph.

The graph construction stage is complete.

Current official algorithm prototype graph:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

The project has completed the carrier-level Patent-Paper link prediction dataset construction and baseline experiments.

Current completed stages:

```text
Stage 01: graph construction
Stage 02: Patent-Paper link prediction dataset construction
Stage 03: carrier-level link prediction baseline experiments
```

The project is now moving toward:

```text
knowledge-unit construction and temporal translation modeling
```

---

## Stage 01 archive

The completed graph construction documentation is archived at:

```text
docs/archive/stage_01_graph_construction_2026-06-28/
```

Do not modify the archived Stage 01 documentation unless explicitly requested.

---

## Current official graph input

Use the following processed graph as the official input for downstream experiments:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

Do not overwrite this directory.

If new processed graph variants are required, create versioned output directories instead.

---

## Completed downstream datasets

The current official Patent-Paper link prediction datasets are:

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full/
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/
```

Definitions:

- `full`: keeps all context graph BioEntity nodes.
- `no_species`: removes species / organism-level BioEntity nodes as a biologically motivated hub-noise sensitivity variant.

These datasets contain fixed train / validation / test splits. Do not re-split them unless explicitly requested.

---

## Completed baseline experiment families

Carrier-level Patent-Paper link prediction baselines have been implemented and evaluated, including:

```text
heuristic / overlap baselines
tabular ML baselines
Node2Vec / DeepWalk-like baselines
MetaPath2Vec baselines
KG embedding baselines: DistMult, TransE
supervised GNN baselines: GraphSAGE, R-GCN
```

The current strongest carrier-level baseline is:

```text
GraphSAGE no-species e50
Test AUROC = 0.880608
Test AUPRC = 0.888661
```

The full graph GraphSAGE result is nearly identical:

```text
GraphSAGE full e50
Test AUROC = 0.881377
Test AUPRC = 0.887874
```

These results suggest that the leakage-controlled carrier context graph contains strong Patent-Paper translation-related prediction signal.

---

## Key result files

Unified baseline comparison outputs:

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

Stage 3 summary report:

```text
reports/experiments/2026-06-29_patent_paper_link_prediction_stage3_summary.md
```

Important model output directories include:

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/gnn_graphsage_e50/
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/gnn_graphsage_e50/
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/gnn_rgcn_e50/
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/kg_distmult_e50/
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/kg_distmult_e50/
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/kg_transe_e50/
```

---

## Engineering rules

### General rules

- Do not refactor scripts `01`-`10` unless explicitly requested.
- Treat scripts `01`-`10` as completed Stage 01 graph construction pipeline scripts.
- From script `11` onward, reusable logic should live under:

```text
src/pkg2/
```

- Keep `scripts/` as thin CLI orchestration layers.
- Do not overwrite existing processed graph outputs unless explicitly requested.
- Prefer creating versioned output directories.
- Preserve fixed dataset splits unless explicitly asked to regenerate them.
- When adding new experiments, write reproducibility artifacts such as:

```text
run_config.json
baseline_manifest.csv
metrics_summary.csv
baseline_report.md
```

### Data leakage rules

For carrier-level Patent-Paper link prediction:

- Context graph construction may use context relations such as BioEntity, Project, Trial, and other carrier context edges.
- Target Patent-Paper edges must not be included in graph embedding training triples or GNN message-passing edges.
- Patent-Paper target edges may only be used as supervised labels in the fixed train / validation / test splits.
- Do not re-split train / validation / test unless explicitly requested.

### Baseline experiment rules

The carrier-level baseline phase is considered sufficiently complete.

Do not add more carrier-level baselines such as GAT, HAN, HGT, RotatE, ComplEx, or additional graph filtering variants unless explicitly requested.

If additional carrier-level experiments are requested, keep them minimal and clearly separate them from the completed baseline results.

---

## Current research direction

The next major research direction is no longer carrier-level Patent-Paper baseline expansion.

The next focus should be:

```text
knowledge-unit construction
knowledge-unit-level temporal translation task definition
temporal state features
temporal KG / dynamic graph baselines
interpretable higher-order graph dynamics
reaction-diffusion-source model
```

Potential next-stage tasks include:

```text
1. Define knowledge units from BioEntity / topic / mechanism / carrier evidence.
2. Construct temporal evidence windows.
3. Define future translation events at the knowledge-unit level.
4. Build static and temporal tabular baselines.
5. Build temporal KG or temporal GNN baselines if needed.
6. Implement the proposed reaction-diffusion-source model.
```

---

## Reporting style

Experiment reports should be written under:

```text
reports/experiments/
```

Reports should include:

```text
purpose
data inputs
leakage control
commands
hyperparameters
outputs
main results
interpretation
stopping criteria
next steps
```

Use Chinese for internal experiment reports unless otherwise requested.

---

## Important caution

The Patent-Paper link prediction task is an intermediate validation task.

Its purpose is to verify that the carrier graph contains predictive translational signal and to establish strong baselines.

The main project contribution should continue toward knowledge-unit-level temporal translation modeling, not further optimization of carrier-level link prediction.