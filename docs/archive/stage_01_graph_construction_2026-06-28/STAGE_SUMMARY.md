# Stage 01 Archive: Graph Construction

Archived on: 2026-06-28

## Stage status

This archive captures the project state after completing the graph construction, diabetes domain extraction, BioEntity mention-edge aggregation, and graph statistics audit stages.

The project is now ready to move into algorithm dataset construction and baseline experiments.

## Current official algorithm prototype graph

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

## Graph summary

### Nodes

| Node type | Count |
| --- | ---: |
| Patent | 11,921 |
| Paper | 35,717 |
| ClinicalTrial | 4,180 |
| Project | 18,237 |
| BioEntity | 31,816 |
| Total | 101,871 |

### Edges

| Edge table | Count |
| --- | ---: |
| edges_patent_paper | 197,312 |
| edges_patent_bioentity | 43,366 |
| edges_patent_project | 2,527 |
| edges_paper_trial | 9,512 |
| edges_paper_project | 28,094 |
| edges_paper_bioentity | 347,466 |
| edges_trial_project | 350 |
| edges_trial_bioentity | 28,312 |
| Total | 656,939 |

## Validation status

Aggregated graph validation:

```text
Overall status: warning
Errors: 0
Warnings: 3
```

Remaining warnings are duplicate basic edges in non-BioEntity edge tables:

```text
patent_links_paper
patent_linked_project
paper_linked_trial
```

BioEntity mention-level duplicate edges were resolved by aggregation:

```text
patent_mentions_bioentity duplicate_basic_edge_count = 0
paper_mentions_bioentity duplicate_basic_edge_count = 0
trial_mentions_bioentity duplicate_basic_edge_count = 0
```

## Graph statistics status

Graph statistics were generated with:

```text
scripts/10_graph_statistics.py
```

Main observations:

- Total nodes: 101,871
- Total edges: 656,939
- Largest connected component: 99,383 nodes
- Largest component ratio: approximately 97.56%
- Isolated nodes: 2,488
- Isolated nodes are all Project nodes
- BioEntity edge weights use aggregated mention counts
- Major BioEntity hubs include patients, insulin, diabetes, glucose, mice, type 2 diabetes, cancer, rat, obesity, and mouse

## Completed pipeline scripts

The following stages have been completed:

```text
01 TSV to Parquet conversion
02 Parquet validation
03 Parquet profiling
04 Profile analysis / extraction planning
05 Core key audit
06 Patent-centered mother graph extraction
07 Extracted graph validation
08 Diabetes domain graph extraction
09 BioEntity mention-edge aggregation
10 Graph statistics audit
```

## Current recommended next step

Start the algorithm dataset construction stage.

Recommended next script:

```text
scripts/11_build_link_prediction_dataset.py
```

Recommended first algorithm task:

```text
Patent-Paper link prediction
```

Recommended dataset variants:

```text
1. full-context graph
2. no-species graph
```

The no-species variant should filter BioEntity nodes with:

```text
Type = species
```

or at least major species hubs such as:

```text
NCBITaxon9606
NCBITaxon10095
NCBITaxon10116
NCBITaxon10090
```

## Engineering decision for next stage

From script 11 onward, reusable logic should be placed under:

```text
src/pkg2/
```

Scripts should become CLI orchestration layers, while shared graph loading, filtering, splitting, negative sampling, and reporting logic should live in reusable modules.

Recommended initial modules:

```text
src/pkg2/__init__.py
src/pkg2/graph_schema.py
src/pkg2/io.py
src/pkg2/graph_loading.py
src/pkg2/graph_filtering.py
src/pkg2/splitting.py
src/pkg2/negative_sampling.py
src/pkg2/link_prediction.py
src/pkg2/reporting.py
```