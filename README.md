# PKG Diabetes Translation Graph

This repository builds and analyzes a PKG Diabetes Translation Graph for studying biomedical translation signals across patents, papers, clinical trials, projects, BioEntities, and related carrier evidence.

The current prototype uses a diabetes-focused carrier graph centered on the 2018-2019 patent window. Although the repository description may refer to broader biomedical or respiratory-disease translation modeling, the current implemented and validated prototype is diabetes-focused.

---

## Environment

The current environment was initialized with:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
pip install polars duckdb pyarrow pyyaml typer rich tqdm loguru pytest ipykernel pandas scikit-learn xgboost lightgbm tensorboard
```

Recommended short sanity check after setup:

```bash
python -m py_compile src/pkg2/*.py scripts/*.py
```

---

## Current official graph

The current official algorithm prototype graph is:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

This graph is treated as the stable downstream input for link prediction, Knowledge Unit construction, temporal panel generation, and Knowledge State proxy design.

Do not overwrite this directory. Create versioned directories for new graph variants.

---

## Project stages

### Stage 01: Graph construction

Status:

```text
completed
```

Stage 01 constructed the patent-centered diabetes Knowledge Carrier Graph.

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

The first downstream validation task was carrier-level Patent-Paper link prediction.

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

Do not regenerate these splits unless explicitly requested.

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

Carrier-level Patent-Paper link prediction is now considered an intermediate validation task, not the main project target.

---

### Stage 04: Knowledge Unit construction and temporal panel

Status:

```text
completed
```

Stage 04 projected the carrier graph into typed BioEntity-pair Knowledge Units and built a KU-level temporal panel.

Completed sub-stages:

```text
Stage 04A: Knowledge Unit construction MVP
Stage 04B: Knowledge Unit audit
Stage 04C: Knowledge Unit temporal panel MVP
Stage 04D: Knowledge Unit sanity check
Stage 04E: Trial year parsing patch and temporal panel refresh
```

Current official Knowledge Unit dataset:

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/
```

Current official dense temporal panel:

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel/
```

Current sanity-check output:

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/sanity_check/
```

#### Stage 04 Knowledge Unit definition

Current Knowledge Unit definition:

```text
Knowledge Unit = typed BioEntity pair
```

The current main translational KU layer uses:

```text
chemical-disease
gene-disease
chemical-gene
```

Filtering and construction policy:

```text
species excluded
disease-disease excluded from the main translational layer
min_carrier_count = 3
max_entities_per_carrier = None
carrier co-mention evidence does not imply an asserted biomedical relation
```

`disease-disease` pairs are intentionally excluded from the main layer because disease co-mentions can conflate comorbidity, complications, subtype hierarchy, near-synonymy, risk factors, and background co-mention. They may be built later as a separate exploratory disease-context or comorbidity layer.

#### Stage 04 results

Current official KU dataset summary:

```text
Knowledge Units:              52,476
Carrier-KU edges:             482,196
Paper-only KUs:               37,185
Patent-supported KUs:          8,489
Trial-supported KUs:           8,932
Translation-supported KUs:    15,291
Cross-modal KUs:              13,473
```

Current dense temporal panel summary after trial year patch:

```text
Knowledge Units:                  52,476
Panel rows:                       7,976,352
Temporal feature rows:            52,476
Year range:                       1874 - 2025
Year count:                       152
Carrier-KU edges:                 482,196
Dated edges:                      482,058
Undated edges:                        138
Dated edge fraction:              0.999714
KUs with dated paper evidence:    50,418
KUs with dated patent evidence:    8,489
KUs with dated trial evidence:     8,909
KUs with undated trial evidence:     128
KUs with dated translation ev.:   15,270
```

Sanity check status:

```text
Overall status:              PASS
Observed pair types:         chemical-disease, chemical-gene, gene-disease
Unexpected pair types:       []
Disallowed pair types:       []
Disallowed entity types:     []
Duplicate unordered pairs:   0
Invalid edge rows:           0
Count mismatch rows:         0
Failed checks:               0
Warnings:                    0
```

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

## Key Stage 04 outputs

### Build Knowledge Units

```bash
python scripts/20_build_knowledge_units.py \
  --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated \
  --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --exclude-bioentity-types species \
  --pair-types chemical-disease gene-disease chemical-gene \
  --min-carrier-count 3 \
  --max-entities-per-carrier 0 \
  --overwrite
```

### Audit Knowledge Units

```bash
python scripts/21_audit_knowledge_units.py \
  --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --top-n 100
```

### Build dense temporal panel

```bash
python scripts/22_build_knowledge_unit_temporal_panel.py \
  --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel \
  --dense \
  --overwrite
```

### Run KU sanity check

```bash
python scripts/23_check_knowledge_unit_sanity.py \
  --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --temporal-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel \
  --expected-pair-types chemical-disease gene-disease chemical-gene \
  --disallowed-pair-types disease-disease \
  --disallowed-entity-types species \
  --expect-no-duplicates \
  --expect-dense-temporal \
  --overwrite
```

---

## Current project focus

The project is now moving beyond both carrier-level link prediction and raw Knowledge Unit construction.

The current next focus is:

```text
Knowledge State proxy feature construction and discovery-oriented KU ranking
```

Recommended next task:

```text
Stage 05 preparation:
src/pkg2/knowledge_state.py
scripts/24_build_knowledge_state_features.py
```

Expected outputs:

```text
knowledge_unit_state_features.parquet
knowledge_unit_state_summary.json
knowledge_unit_state_report.md
mature_canonical_knowledge_units.csv
translation_supported_knowledge_units.csv
patent_supported_nonhub_candidates.csv
trial_supported_nonhub_candidates.csv
paper_only_discovery_candidates.csv
short_lag_translation_candidates.csv
long_lag_mature_translation_units.csv
```

The next stage should not rank discovery candidates by raw `carrier_count` alone. The highest-support KUs are dominated by mature canonical diabetes concepts such as:

```text
glucose - insulin
insulin - diabetes
glucose - diabetes
```

These are useful sanity checks, but not discovery outputs.

Discovery-oriented ranking should use features such as:

```text
novelty
specificity
entity hub penalty
recent growth
first_paper_year
first_patent_year
first_trial_year
paper_to_patent_lag
paper_to_trial_lag
translation support
commercialization support
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

Current reusable modules include carrier-graph loading, filtering, link-prediction dataset construction, metrics, reporting, graph baseline utilities, Knowledge Unit construction, and Knowledge Unit temporal panel construction.

Expected next reusable module:

```text
src/pkg2/knowledge_state.py
```

### Outputs

Do not overwrite completed processed graph outputs unless explicitly requested.

Prefer versioned output directories for new datasets, experiments, and model outputs.

### Reproducibility artifacts

Experiment outputs should include reproducibility files such as:

```text
run_config.json
dataset_manifest.csv
baseline_manifest.csv
metrics_summary.csv
knowledge_unit_summary.json
knowledge_unit_temporal_summary.json
knowledge_unit_sanity_summary.json
baseline_report.md
```

### Data leakage control

For Patent-Paper link prediction:

- Target Patent-Paper edges must not be included in graph embedding training triples.
- Target Patent-Paper edges must not be included in GNN message-passing graphs.
- Patent-Paper labels may be used only through fixed train / validation / test splits.
- Do not regenerate splits unless explicitly requested.

For future Knowledge Unit temporal prediction:

- History features must be computed only from evidence observed at or before the cutoff year.
- Future patent / trial labels must be defined strictly after the cutoff year.
- Carrier-level baseline scores, if projected to KU-level, must use only pre-cutoff carrier evidence.
- Trial and patent dates must be audited before constructing temporal labels.

---

## Important interpretation note

Patent-Paper link prediction is an intermediate validation task.

It verifies that the constructed carrier graph contains predictive translational signal and provides strong baselines.

Knowledge Units in the current Stage 04 dataset are typed BioEntity-pair co-mention units, not asserted biomedical relation claims.

Therefore:

```text
carrier co-mention evidence != proven relation
chemical-disease pair != treatment claim
gene-disease pair != causal claim
chemical-gene pair != target claim
```

The main project contribution should now move toward:

```text
knowledge-unit-level temporal translation dynamics
Knowledge State construction
discovery-oriented KU ranking
interpretable higher-order graph modeling
reaction-diffusion-source modeling
```

---

## Roadmap

Authoritative detailed roadmap:

```text
技术路线.md
```

Immediate next step:

```text
Stage 05 preparation:
Build rule-based Knowledge State proxy features and discovery-oriented candidate tables.
```

Later stages:

```text
1. Build longer-window graph variants, such as 2018-2025 or 2010-2025.
2. Define future patent / trial emergence labels at the KU level.
3. Build static and temporal KU-level tabular baselines.
4. Compare against temporal KG / temporal GNN baselines.
5. Implement and evaluate the reaction-diffusion-source Knowledge Field model.
6. Validate outputs with case studies and expert-interpretable flow paths.
```