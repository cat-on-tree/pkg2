# PKG Knowledge Field

This repository implements data preparation and modeling for a research project based on PKG24S4 / PubMed Knowledge Graph 2.0.

## Current stage

Stage 1: PKG24S4 data preparation.

The current goal is to inspect raw PKG24S4 TSV files, convert them to Parquet, and prepare clean intermediate datasets for building temporal biomedical knowledge carrier graphs.

No GNN model is implemented yet.

## Data layout

Raw data is stored locally and must not be committed.

```text
data/
  raw/
    pkg24s4/
      A01_Articles.tsv.gz
      ...
      C23_BioEntities.tsv.gz
  interim/
  processed/
```

## Environment

The current environment was initialized with:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
pip install polars duckdb pyarrow pyyaml typer rich tqdm loguru pytest ipykernel pandas
```

## Notes

- Do not commit raw data.
- Do not commit generated Parquet files.
- Use TSV/TSV.GZ files as immutable raw input.
- Use `data/interim/` for converted and cleaned intermediate files.
- Use `data/processed/` for final graph-ready datasets.

## Data inspection

Inspect local PKG24S4 TSV schemas without loading full files:

```bash
python scripts/00_inspect_pkg24s4.py
```

This writes `data/interim/schema_summary.csv`. Add `--count-rows` only when you explicitly want full streamed row counts.

## TSV to Parquet conversion

Preview conversions without writing Parquet files:

```bash
python scripts/01_tsv_to_parquet.py --dry-run
```

Automatically convert one large file:

```bash
python scripts/01_tsv_to_parquet.py --files A04_Abstract.tsv.gz --engine auto --overwrite
```

Force the low-memory streaming engine:

```bash
python scripts/01_tsv_to_parquet.py --files A04_Abstract.tsv.gz --engine pyarrow-stream --overwrite --batch-size 25000
```

Convert all TSV/TSV.GZ files automatically:

```bash
python scripts/01_tsv_to_parquet.py --engine auto --stream-threshold-mb 4096 --batch-size 25000 --overwrite
```

Overwrite existing Parquet outputs:

```bash
python scripts/01_tsv_to_parquet.py --overwrite
```

This writes Parquet files under `data/interim/parquet/` and a manifest at `data/interim/parquet_manifest.csv`. Do not commit `data/interim/parquet/` or `data/interim/bad_rows/`.

The converter defaults to `--engine auto`: files smaller than `--stream-threshold-mb` use DuckDB, while larger files use `pyarrow-stream`. The streaming engine reads TSV.GZ files line by line and writes a single Parquet file with low memory usage. TSV empty fields are written as Parquet NULL, trailing missing fields are padded on the right with NULL, and double quotes are treated as ordinary text characters rather than CSV quote syntax. Rows with more fields than the header are not silently truncated; they are written to `data/interim/bad_rows/` and skipped unless `--fail-on-bad-rows` is set.

The DuckDB engine uses `strict_mode=false`, `null_padding=true`, `quote=''`, `escape=''`, and disables the parallel CSV scanner by default to prioritize correctness for PKG24S4 TSVs. Large files such as `A04_Abstract.tsv.gz` may need lower thread counts and smaller Parquet row groups; the script defaults to `preserve_insertion_order=false`, `--threads 1`, `--memory-limit 20GB`, and `--row-group-size 32768`. `--temp-dir` configures DuckDB `temp_directory` for spill files. Use `--ignore-errors` only when you have confirmed that skipping unparsable rows and accepting data loss is appropriate.

Parquet profiling is heuristic analysis, not a final schema contract. By default, column `null_fraction` is estimated from the first `--sample-rows` rows; `join_candidate_edges.csv` should be manually confirmed before any subgraph extraction.

```bash
python scripts/02_validate_parquet.py
```

This validates the Parquet files generated under data/interim/parquet/ and writes a validation report at data/interim/parquet_validation.csv.

The validator checks whether expected Parquet outputs exist, compares converted files against the conversion manifest where available, and reports row counts, file sizes, source file names, conversion engines, and bad-row information. Run this script after TSV-to-Parquet conversion and before profiling or graph extraction.

Validation is a consistency check for the converted intermediate files. It does not prove that the PKG24S4 schema is semantically correct, but it helps catch missing outputs, failed conversions, suspicious row-count mismatches, and incomplete conversion runs.

```bash
python scripts/03_profile_parquet_tables.py
```

This profiles converted PKG24S4 Parquet tables under data/interim/parquet/ and writes profiling outputs under data/interim/profile/.

Main outputs include table_profile.csv, column_profile.csv, link_table_profile.csv, entity_candidate_tables.csv, and join_candidate_edges.csv. The script summarizes table sizes, row counts, column counts, ID-like columns, date/year-like columns, text-like columns, candidate entity tables, candidate link tables, and heuristic join candidates.

Parquet profiling is heuristic analysis, not a final schema contract. By default, column null_fraction is estimated from the first --sample-rows rows. Use --exact-null-count true for more accurate null fractions. Use --exact-distinct true only with caution on large tables, because exact distinct counts can be expensive. join_candidate_edges.csv should be manually reviewed and confirmed before any subgraph extraction.

```bash
python scripts/04_analyze_profile.py
```

This analyzes profiling CSV files under data/interim/profile/ and writes human-readable Markdown reports: data/interim/profile/profile_analysis.md and data/interim/profile/recommended_subgraph_plan.md.

The script summarizes dataset scale, largest tables, candidate node tables, candidate edge/link tables, and rule-based recommendations for a first-pass graph. It separates heuristic profiling outputs from manually defined graph-design recommendations, so the recommended subgraph plan should be treated as a planning document rather than an automatically inferred final schema.

The current recommended first-pass graph is a patent-centered biomedical translation subgraph with Patent, Paper, ClinicalTrial, Project, and grounded BioEntity nodes. Deferred components include author graphs, affiliation graphs, MeSH descriptor graphs, inventor/assignee graphs, citation graphs, BioEntity relationship graphs, and future text-derived knowledge units.

```bash
python scripts/05_audit_core_graph_keys.py --coverage-mode row --exact-distinct false --exact-duplicate-edges false
```

This audits the core graph keys and join behavior needed before extracting the first patent-centered subgraph. It writes audit outputs under data/interim/profile/.

Main outputs include core_node_key_audit.csv, core_edge_endpoint_audit.csv, core_fk_coverage.csv, project_number_join_audit.csv, bioentity_unmatched_audit.csv, bioentity_unmatched_entityid_audit.csv, and core_key_audit_summary.md.

The audit confirms that ProjectNumber in B03_Link_Papers_Projects, B04_Link_ClinicalTrials_Projects, and B05_Link_Patents_Projects should join to normalized B02_Projects.CORE_PROJECT_NUM, not to FULL_PROJECT_NUM, APPLICATION_ID, or SUBPROJECT_ID. The recommended normalization is UPPER(REPLACE(REPLACE(TRIM(CAST(value AS VARCHAR)), ' ', ''), '-', '')).

The audit also confirms that EntityId = CUI-less represents ungrounded entity mentions. CUI-less should not be used as a BioEntity graph node. For first-pass graph extraction, BioEntity nodes should come from grounded C23_BioEntities.EntityId, while CUI-less rows should be filtered from BioEntity graph edges or optionally summarized as carrier-level ungrounded mention features.

```bash
python scripts/06_extract_patent_centered_subgraph.py --start-year (2018-2026) --end-year (2018-2026) --output-dir data/processed/patent_centered_subgraph_xx
```
This extracts a graph-ready patent-centered biomedical translation subgraph from the converted Parquet files under data/interim/parquet/ and writes node/edge Parquet files plus reports under the specified --output-dir.

The script selects seed patents from C15_Patents using GrantedDate, expands to linked papers through C16_Link_Patents_Papers, expands to projects through B05_Link_Patents_Projects, expands to grounded BioEntities through C18_Link_Patents_BioEntities, then expands selected papers to clinical trials, projects, and BioEntities. If trial expansion is enabled, selected trials are further expanded to projects and BioEntities.

The extractor uses normalized CORE_PROJECT_NUM as the Project node key and filters EntityId = CUI-less from all BioEntity graph edges. Node outputs include nodes_patent.parquet, nodes_paper.parquet, nodes_clinicaltrial.parquet, nodes_project.parquet, and nodes_bioentity.parquet. Edge outputs include edges_patent_paper.parquet, edges_patent_bioentity.parquet, edges_patent_project.parquet, edges_paper_trial.parquet, edges_paper_project.parquet, edges_paper_bioentity.parquet, edges_trial_project.parquet, and edges_trial_bioentity.parquet.

The script also writes subgraph_manifest.csv and subgraph_extraction_report.md. Use explicit versioned output directories such as data/processed/patent_centered_subgraph_2018_2019_full_v1_1/ to avoid overwriting previous graph extractions. Do not commit generated graph outputs under data/processed/.

```bash
python scripts/07_validate_extracted_subgraph.py --subgraph-dir data/processed/patent_centered_subgraph_xx
```

This validates an extracted patent-centered subgraph directory created by scripts/06_extract_patent_centered_subgraph.py.

The validator checks that expected node, edge, manifest, and report files exist; required node and edge columns are present; node_id values are non-null and unique; edge source_id and target_id endpoints match the corresponding node tables; CUI-less does not appear in BioEntity nodes or BioEntity edge targets; Project node IDs are normalized; and duplicate basic edges are reported.

Validation outputs are written into the same subgraph directory by default. Main outputs include subgraph_validation_report.md, subgraph_file_validation.csv, subgraph_schema_validation.csv, subgraph_node_validation.csv, subgraph_edge_validation.csv, subgraph_endpoint_validation.csv, subgraph_cuiless_validation.csv, subgraph_project_validation.csv, subgraph_degree_summary.csv, subgraph_top_degree_nodes.csv, subgraph_bioentity_type_counts.csv, and subgraph_endpoint_unmatched_samples.csv.

Warnings about duplicate basic edges are expected for mention-level BioEntity edges, because the same carrier may mention the same BioEntity multiple times. Endpoint mismatches, CUI-less leakage into graph nodes or BioEntity edge targets, null node IDs, or duplicate node IDs should be treated as issues to fix before downstream graph analysis or modeling.

```bash
python scripts/08_extract_domain_subgraph.py \
  --input-dir data/processed/patent_centered_subgraph_2018_2019_full_v1_1 \
  --output-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2 \
  --domain diabetes \
  --carrier-expansion-mode conservative \
  --bioentity-mode all \
  --overwrite
```

This extracts a diabetes-specific domain subgraph from a validated patent-centered mother graph created by scripts/06_extract_patent_centered_subgraph.py.

The script does not read raw PKG24S4 TSV files and does not rescan the original large intermediate PKG Parquet tables. It works only from the already extracted mother graph node and edge files, such as nodes_patent.parquet, nodes_paper.parquet, nodes_clinicaltrial.parquet, nodes_project.parquet, nodes_bioentity.parquet, and the corresponding edge Parquet files.

The diabetes filter uses high-precision Tier 1 diabetes terms as seed evidence, including terms such as diabetes, diabetic, type 1 diabetes, type 2 diabetes, diabetes mellitus, gestational diabetes, insulin resistance, hyperglycemia, hypoglycemia, HbA1c, A1C, diabetic nephropathy, diabetic retinopathy, diabetic neuropathy, and MODY. Keyword matching uses boundary-aware regular expressions rather than raw substring matching, so short terms such as MODY, A1C, T1D, and T2D should not match inside unrelated words such as hemodynamic.

Tier 2 diabetes-related pharmacologic or mechanistic terms, such as metformin, GLP-1, SGLT2, DPP-4, semaglutide, liraglutide, insulin receptor, beta cell, and islet, are reported as keyword hits but do not create seed carriers by default. Use --include-tier2-seeds true only if you intentionally want these broader terms to create domain seed carriers. Broad Tier 3 metabolic terms such as glucose, insulin, metabolism, obesity, lipid, pancreas, and inflammation are disabled by default and should not be used as standalone seed terms without careful review.

The main expansion control is --carrier-expansion-mode. seed-only keeps only carriers directly matched by domain evidence. conservative keeps direct seed carriers and adds limited patent, paper, trial, and project context without allowing seed projects to reverse-expand into very large paper or patent neighborhoods. expanded reproduces the broader v1-style neighborhood expansion and is useful for exploration, but it can make the domain graph much larger and less specific.

The --bioentity-mode option controls BioEntity edge retention. --bioentity-mode all keeps all BioEntity mention edges from selected carriers, preserving broader mechanistic and contextual entities such as genes, drugs, complications, and comorbidities. --bioentity-mode matched keeps only BioEntity edges whose target BioEntity had domain keyword evidence, producing a smaller and higher-precision entity subgraph but potentially losing important diabetes-context entities such as drug targets, inflammatory markers, and complications.

The script writes a domain subgraph with the same node and edge file names used by the mother graph, so it can be validated with scripts/07_validate_extracted_subgraph.py. Outputs include nodes_patent.parquet, nodes_paper.parquet, nodes_clinicaltrial.parquet, nodes_project.parquet, nodes_bioentity.parquet, edges_patent_paper.parquet, edges_patent_bioentity.parquet, edges_patent_project.parquet, edges_paper_trial.parquet, edges_paper_project.parquet, edges_paper_bioentity.parquet, edges_trial_project.parquet, and edges_trial_bioentity.parquet.

The script also writes audit outputs: domain_seed_carriers.parquet, domain_keyword_hits.parquet, domain_manifest.csv, and domain_filter_report.md. domain_seed_carriers.parquet records the carriers selected by domain evidence and their evidence counts. domain_keyword_hits.parquet records the matched terms, evidence type, source field, and associated carrier for keyword/text and BioEntity mention hits. domain_filter_report.md summarizes parameters, seed carrier counts, node counts, edge counts, top keyword hits, and top BioEntities in the extracted domain subgraph.

After extraction, validate the domain subgraph with:

```bash
python scripts/07_validate_extracted_subgraph.py \
  --subgraph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2
```

Warnings about duplicate basic edges are expected because BioEntity edges are still mention-level edges. The same paper, patent, or clinical trial may mention the same BioEntity multiple times. Endpoint mismatches, CUI-less leakage into BioEntity nodes or BioEntity edge targets, null node IDs, or duplicate node IDs should be treated as issues to fix before downstream graph analysis or modeling.

For the current diabetes-focused prototype, the recommended default is --carrier-expansion-mode conservative --bioentity-mode all. This preserves diabetes-specific carriers and relevant biomedical context while avoiding the excessive graph expansion caused by project reverse-expansion. Cross-disease entities such as cancer or Alzheimer's disease may still appear because they can be biologically or clinically connected to diabetes through comorbidity, metabolism, inflammation, drug repurposing, or shared mechanisms. They should generally be tagged, downweighted, or filtered in derived analysis graphs rather than removed from the main context-preserving domain graph.

### `scripts/09_aggregate_mention_edges.py`

```bash
python scripts/09_aggregate_mention_edges.py \
  --input-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2 \
  --output-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated \
  --overwrite
```

This aggregates mention-level BioEntity edges into pair-level weighted BioEntity edges.

The input graph contains BioEntity mention edges where the same carrier can mention the same BioEntity multiple times. For example, one paper may mention `diabetes` several times, producing multiple `Paper -> BioEntity` rows. This script collapses repeated `source_id + target_id + edge_type` rows into a single weighted edge.

The main aggregated BioEntity edge weight is:

```text
mention_count
```

Additional aggregated attributes include `distinct_mention_count`, `sample_mention`, `mention_types`, `text_fields`, `avg_prob`, `max_prob`, and evidence flags such as `has_mesh`, `has_ncbigene`, and `has_chebi` when source columns are available.

The script aggregates:

```text
edges_patent_bioentity.parquet
edges_paper_bioentity.parquet
edges_trial_bioentity.parquet
```

It copies all node files and non-BioEntity edge files unchanged, then writes a new graph directory using the same expected graph file names so that the output can still be validated by `scripts/07_validate_extracted_subgraph.py`.

Main outputs include:

```text
nodes_patent.parquet
nodes_paper.parquet
nodes_clinicaltrial.parquet
nodes_project.parquet
nodes_bioentity.parquet
edges_patent_paper.parquet
edges_patent_bioentity.parquet
edges_patent_project.parquet
edges_paper_trial.parquet
edges_paper_project.parquet
edges_paper_bioentity.parquet
edges_trial_project.parquet
edges_trial_bioentity.parquet
aggregation_report.md
aggregation_manifest.csv
bioentity_edge_aggregation_stats.csv
subgraph_manifest.csv
subgraph_extraction_report.md
```

For the current diabetes graph, aggregation reduced BioEntity edges from mention-level to pair-level as follows:

```text
edges_patent_bioentity: 70,597 -> 43,366
edges_paper_bioentity: 817,320 -> 347,466
edges_trial_bioentity: 74,239 -> 28,312
```

After aggregation, validate the output graph:

```bash
python scripts/07_validate_extracted_subgraph.py \
  --subgraph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated
```

The validated aggregated graph is the current official algorithm prototype graph.

### `scripts/10_graph_statistics.py`

```bash
python scripts/10_graph_statistics.py \
  --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated
```

This computes graph-level statistics for an extracted or aggregated heterogeneous graph.

It is intended to run after graph extraction, domain filtering, and BioEntity edge aggregation. For the current project, it was run on the diabetes conservative aggregated graph:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

The script treats the graph as a typed heterogeneous graph with the following node types:

```text
Patent
Paper
ClinicalTrial
Project
BioEntity
```

and the following edge tables:

```text
edges_patent_paper
edges_patent_bioentity
edges_patent_project
edges_paper_trial
edges_paper_project
edges_paper_bioentity
edges_trial_project
edges_trial_bioentity
```

BioEntity edges aggregated by `scripts/09_aggregate_mention_edges.py` use `mention_count` as `edge_weight`. Non-BioEntity edges use weight `1.0`.

The script reports:

```text
node type counts
edge type counts
BioEntity type counts
edge weight distributions
degree distributions
top degree nodes
connected components
isolated nodes
top weighted BioEntity edges
```

Main outputs include:

```text
graph_statistics_report.md
graph_node_type_counts.csv
graph_edge_type_counts.csv
graph_bioentity_type_counts.csv
graph_edge_weight_summary.csv
graph_degree_summary.csv
graph_top_degree_nodes.csv
graph_connected_components.csv
graph_isolated_nodes.csv
graph_top_weighted_bioentity_edges.csv
```

For the current diabetes conservative aggregated graph, the main statistics are:

```text
total_nodes: 101,871
total_edges: 656,939
connected_components: 2,489
largest_component_node_count: 99,383
isolated_nodes: 2,488
```

The largest connected component contains approximately 97.56% of all nodes, indicating that the graph is suitable for whole-graph representation learning and link prediction experiments.

The isolated nodes are all `Project` nodes and can be filtered during downstream dataset construction.

Important modeling notes from the graph statistics stage:

```text
BioEntity hubs such as patients, insulin, diabetes, glucose, mice, rat, and mouse are expected in biomedical graphs.
Species nodes may need to be filtered or downweighted in sensitivity analyses.
Cross-disease entities such as cancer or Alzheimer's disease should generally be tagged or handled in derived analysis graphs rather than removed from the main context-preserving graph.
BioEntity edge weights should usually be transformed with log1p(mention_count) before use in graph algorithms.
```