# AGENTS.md

## Project status

This repository builds and analyzes a PKG Diabetes Translation Graph.

The current implemented and validated prototype is diabetes-focused, using a 2018-2019 patent-centered Knowledge Carrier Graph. Broader biomedical or respiratory-disease transfer can be considered later, but the current official pipeline and outputs are diabetes-based.

Current official algorithm prototype graph:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

Do not overwrite this directory.

If new processed graph variants are required, create versioned output directories instead.

---

## Completed stages

Current completed stages:

```text
Stage 01: Knowledge Carrier Graph construction
Stage 02: Patent-Paper link prediction dataset construction
Stage 03: Carrier-level Patent-Paper baseline experiments
Stage 04A: Knowledge Unit construction MVP
Stage 04B: Knowledge Unit audit
Stage 04C: Knowledge Unit temporal panel MVP
Stage 04D: Knowledge Unit sanity check
Stage 04E: Trial year parsing patch and temporal panel refresh
```

The project is now moving toward:

```text
Stage 05 preparation:
Knowledge State proxy feature construction and discovery-oriented KU ranking
```

Recommended next implementation targets:

```text
src/pkg2/knowledge_state.py
scripts/24_build_knowledge_state_features.py
```

---

## Stage 01 archive

The completed graph construction documentation is archived at:

```text
docs/archive/stage_01_graph_construction_2026-06-28/
```

Do not modify the archived Stage 01 documentation unless explicitly requested.

Scripts `01`-`10` are completed Stage 01 graph construction pipeline scripts.

Do not refactor scripts `01`-`10` unless explicitly requested.

---

## Current official graph input

Use the following processed graph as the official input for downstream experiments:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

This graph is a diabetes-domain context-preserving Knowledge Carrier Graph prototype.

It is not the final Knowledge Graph and not the Knowledge Unit graph.

---

## Completed carrier-level datasets

The current official Patent-Paper link prediction datasets are:

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full/
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/
```

Definitions:

- `full`: keeps all context graph BioEntity nodes.
- `no_species`: removes species / organism-level BioEntity nodes as a biologically motivated hub-noise sensitivity variant.

These datasets contain fixed train / validation / test splits. Do not re-split them unless explicitly requested.

Patent-Paper link prediction is a completed carrier-level proxy task. It should not be expanded unless explicitly requested.

---

## Completed Stage 03 baseline experiment families

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

## Key Stage 03 result files

Unified baseline comparison outputs:

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

Stage 03 summary report:

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

## Completed Stage 04 Knowledge Unit outputs

Current official Knowledge Unit dataset:

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/
```

Current official dense temporal panel:

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel/
```

Current sanity-check outputs:

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/sanity_check/
```

### Stage 04 Knowledge Unit definition

Current Knowledge Unit definition:

```text
Knowledge Unit = typed BioEntity pair
```

Current main translational pair types:

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
canonicalized pair orientation
```

Important interpretation rule:

```text
carrier co-mention evidence does not imply an asserted biomedical relation.
```

Therefore:

```text
chemical-disease pair != treatment claim
gene-disease pair != causal claim
chemical-gene pair != target claim
```

`disease-disease` is intentionally excluded from the main translational KU layer because it mixes comorbidity, complications, subtype hierarchy, near-synonymy, risk factors, and background co-mention. If needed later, build it as a separate disease-context or comorbidity layer.

### Stage 04 KU dataset summary

```text
Knowledge Units:              52,476
Carrier-KU edges:             482,196
Paper-only KUs:               37,185
Patent-supported KUs:          8,489
Trial-supported KUs:           8,932
Translation-supported KUs:    15,291
Cross-modal KUs:              13,473
```

### Stage 04 dense temporal panel summary

After trial year parsing patch:

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

### Stage 04 sanity check status

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

## Stage 04 canonical commands

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

## Current research direction

The next major research direction is no longer carrier-level Patent-Paper baseline expansion and no longer raw KU construction.

The next focus should be:

```text
Knowledge State proxy feature construction
discovery-oriented Knowledge Unit ranking
knowledge-unit-level temporal translation task definition
temporal state features
temporal KG / dynamic graph baselines
interpretable higher-order graph dynamics
reaction-diffusion-source model
```

Potential next-stage tasks include:

```text
1. Build rule-based Knowledge State proxy features.
2. Stratify KUs into mature canonical, translation-supported, nonhub, and paper-only discovery candidates.
3. Construct discovery-oriented candidate tables.
4. Define future translation events at the KU level for long-window graphs.
5. Build static and temporal tabular baselines.
6. Build temporal KG or temporal GNN baselines if needed.
7. Implement the proposed reaction-diffusion-source model.
```

The immediate next implementation target is:

```text
scripts/24_build_knowledge_state_features.py
```

with reusable logic in:

```text
src/pkg2/knowledge_state.py
```

---

## Discovery ranking guidance

Do not treat raw `carrier_count` top KUs as discovery outputs.

Top carrier-count KUs are dominated by mature canonical diabetes concepts, such as:

```text
glucose - insulin
insulin - diabetes
glucose - diabetes
```

These are useful for sanity checking but not for discovery.

Discovery-oriented ranking should include:

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

Recommended Stage 05 output tables:

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
dataset_manifest.csv
baseline_manifest.csv
metrics_summary.csv
knowledge_unit_summary.json
knowledge_unit_temporal_summary.json
knowledge_unit_sanity_summary.json
baseline_report.md
```

### Data leakage rules

For carrier-level Patent-Paper link prediction:

- Context graph construction may use context relations such as BioEntity, Project, Trial, and other carrier context edges.
- Target Patent-Paper edges must not be included in graph embedding training triples or GNN message-passing edges.
- Patent-Paper target edges may only be used as supervised labels in the fixed train / validation / test splits.
- Do not re-split train / validation / test unless explicitly requested.

For future KU-level temporal prediction:

- Features at cutoff year `t` must use only evidence observed at or before `t`.
- Future patent / trial labels must be defined strictly after the cutoff year.
- Trial and patent dates must be audited before constructing future labels.
- If carrier-level baseline scores are projected to KU-level, only pre-cutoff carrier evidence may be used.

### Baseline experiment rules

The carrier-level baseline phase is considered sufficiently complete.

Do not add more carrier-level baselines such as GAT, HAN, HGT, RotatE, ComplEx, or additional graph filtering variants unless explicitly requested.

If additional carrier-level experiments are requested, keep them minimal and clearly separate them from the completed baseline results.

Future modeling effort should focus on KU-level temporal baselines and Knowledge Field modeling.

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

Recommended next report:

```text
reports/experiments/2026-07-02_knowledge_unit_stage4_summary.md
```

---

## Authoritative roadmap

The detailed technical roadmap is maintained in:

```text
技术路线.md
```

If README.md, AGENTS.md, and `技术路线.md` disagree, treat `技术路线.md` as the research roadmap and update README.md / AGENTS.md to match the current completed stage.

---

## Important caution

Patent-Paper link prediction is an intermediate validation task.

Its purpose is to verify that the carrier graph contains predictive translational signal and to establish strong baselines.

Knowledge Units in the current Stage 04 dataset are typed BioEntity-pair co-mention units, not asserted biomedical relation claims.

The main project contribution should continue toward:

```text
knowledge-unit-level temporal translation modeling
Knowledge State construction
discovery-oriented KU ranking
interpretable higher-order graph dynamics
reaction-diffusion-source modeling
```