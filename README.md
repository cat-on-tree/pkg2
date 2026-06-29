# PKG Diabetes Translation Graph

This repository builds and analyzes a PKG Diabetes Translation Graph for studying biomedical translation signals across patents, papers, clinical trials, projects, BioEntities, and related carrier evidence.

The project currently uses a diabetes-focused graph covering the 2018-2019 window.

---

## Environment

The current environment was initialized with:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
pip install polars duckdb pyarrow pyyaml typer rich tqdm loguru pytest ipykernel pandas scikit-learn xgboost lightgbm
```

---

## Current official graph

The current official algorithm prototype graph is:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

This graph is treated as the stable downstream input for link prediction and later knowledge-unit modeling.

Do not overwrite this directory. Create versioned directories for new graph variants.

---

## Project stages

### Stage 01: Graph construction

Status:

```text
completed
```

Stage 01 constructed the patent-centered diabetes carrier graph.

The completed Stage 01 documentation is archived at:

```text
docs/archive/stage_01_graph_construction_2026-06-28/
```

Scripts `01`-`10` are considered completed graph construction pipeline scripts.

They should not be refactored unless explicitly requested.

---

### Stage 02: Patent-Paper link prediction dataset construction

Status:

```text
completed
```

The first downstream validation task is carrier-level Patent-Paper link prediction.

The official dataset variants are:

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full/
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/
```

Dataset variant definitions:

| Dataset | Description |
|---|---|
| `diabetes_2018_2019_v1_full` | Full carrier context graph with all BioEntity types |
| `diabetes_2018_2019_v1_no_species` | Removes species / organism-level BioEntity nodes as a hub-noise sensitivity variant |

The fixed splits are reused across baseline experiments:

```text
labeled_edges_train
labeled_edges_val
labeled_edges_test
```

---

### Stage 03: Carrier-level Patent-Paper baseline experiments

Status:

```text
completed
```

Stage 03 evaluated multiple baseline families for Patent-Paper link prediction:

```text
heuristic / overlap baselines
tabular ML baselines
Node2Vec / DeepWalk-like baselines
MetaPath2Vec baselines
KG embedding baselines: DistMult, TransE
supervised GNN baselines: GraphSAGE, R-GCN
```

The unified comparison files are:

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

The Stage 03 summary report is:

```text
reports/experiments/2026-06-29_patent_paper_link_prediction_stage3_summary.md
```

Current top carrier-level baseline:

```text
GraphSAGE no-species e50
Test AUROC = 0.880608
Test AUPRC = 0.888661
```

GraphSAGE on the full graph produced a nearly identical result:

```text
GraphSAGE full e50
Test AUROC = 0.881377
Test AUPRC = 0.887874
```

These results show that the leakage-controlled carrier context graph contains strong Patent-Paper translation-related prediction signal.

---

## Key Stage 03 results

Top unified baseline results ranked by test AUPRC:

| Rank | Dataset | Run | Decoder | Test AUROC | Test AUPRC |
|---:|---|---|---|---:|---:|
| 1 | no-species | `gnn_graphsage_e50` | - | 0.880608 | 0.888661 |
| 2 | full | `gnn_graphsage_e50` | - | 0.881377 | 0.887874 |
| 3 | no-species | `gnn_rgcn_e50` | - | 0.880312 | 0.885240 |
| 4 | no-species | `kg_distmult_e50` | concat_logistic | 0.819824 | 0.826381 |
| 5 | full | `kg_distmult_e50` | concat_logistic | 0.818941 | 0.822387 |
| 6 | no-species | `kg_transe_e50` | concat_logistic | 0.790784 | 0.775526 |
| 7 | no-species | `node2vec_p1_q1` | concat_logistic | 0.778744 | 0.772752 |
| 8 | full | `node2vec_p1_q1` | concat_logistic | 0.772017 | 0.765871 |

Main conclusions:

1. Carrier context graph structure contains strong Patent-Paper prediction signal.
2. Relation-aware KG embedding improves substantially over random-walk and fixed-metapath embeddings.
3. Supervised message-passing GNNs are the strongest carrier-level baselines.
4. GraphSAGE and R-GCN perform similarly, with GraphSAGE slightly stronger in the current setup.
5. Full and no-species variants are nearly identical under supervised GNNs, suggesting robustness to species-level hubs.
6. The carrier-level link prediction baseline phase is sufficiently complete.

---

## Current project focus

The project is now moving beyond carrier-level link prediction baselines.

The next major focus is:

```text
knowledge-unit construction and temporal translation modeling
```

Expected next tasks:

```text
1. Define knowledge units from BioEntity, topic, mechanism, and carrier evidence.
2. Construct temporal evidence windows.
3. Define future translation events at the knowledge-unit level.
4. Build static and temporal feature baselines.
5. Evaluate temporal KG / temporal GNN baselines if needed.
6. Develop and test the proposed reaction-diffusion-source model.
```

---

## Repository engineering conventions

### Scripts

Scripts `01`-`10` are Stage 01 graph construction scripts and should be treated as stable.

From script `11` onward:

```text
scripts/
```

should contain thin CLI orchestration layers only.

Reusable logic should be implemented under:

```text
src/pkg2/
```

### Outputs

Do not overwrite completed processed graph outputs unless explicitly requested.

Prefer versioned output directories for new datasets, experiments, and model outputs.

### Reproducibility artifacts

Experiment outputs should include reproducibility files such as:

```text
run_config.json
baseline_manifest.csv
metrics_summary.csv
baseline_report.md
```

### Data leakage control

For Patent-Paper link prediction:

- Target Patent-Paper edges must not be included in graph embedding training triples.
- Target Patent-Paper edges must not be included in GNN message-passing graphs.
- Patent-Paper labels may be used only through fixed train / validation / test splits.
- Do not regenerate splits unless explicitly requested.

---

## Important note

Patent-Paper link prediction is an intermediate validation task.

It verifies that the constructed carrier graph contains predictive translational signal and provides strong baselines.

The main project contribution should now move toward:

```text
knowledge-unit-level temporal translation dynamics
interpretable higher-order graph modeling
reaction-diffusion-source modeling
```