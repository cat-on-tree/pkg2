# Stage 0 实验报告：Graph Construction and Data Processing

日期：2026-06-27

## 1. 阶段定位

本报告记录项目在进入算法实验之前的数据处理与图构建阶段状态。

本阶段完成了：

1. 原始 TSV 数据到 Parquet 的转换；
2. Parquet 数据验证与 profiling；
3. core key audit；
4. patent-centered mother graph 构建；
5. diabetes-domain 子图抽取；
6. BioEntity mention-level 边聚合；
7. 图验证；
8. 图统计审计。

本阶段完成后，项目正式进入：

```text
Patent-Paper carrier-level link prediction
```

的数据集构建和 baseline 实验阶段。

从整体实验链条看，本阶段对应：

```text
Stage 0: Graph construction and data processing
```

后续阶段包括：

```text
Stage 1: Patent-Paper link prediction dataset construction and heuristic baselines
Stage 2: Logistic / tree-based tabular baselines
Stage 3A: Node2Vec / DeepWalk-like graph embedding baseline
Stage 3B: MetaPath2Vec heterogeneous graph embedding baseline
Stage 3C: KG embedding baseline
Stage 3D: GraphSAGE / GNN baseline
```

---

## 2. 当前官方算法原型图

本阶段结束时，当前用于算法原型实验的官方图目录为：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

该图是：

```text
2018-2019 patent-centered diabetes-domain conservative subgraph
```

并且已经完成：

```text
BioEntity mention-edge aggregation
```

后续 Patent-Paper link prediction 数据集均从该图派生。

---

## 3. 图构建结果概览

### 3.1 Node summary

| Node type | Count |
| --- | ---: |
| Patent | 11,921 |
| Paper | 35,717 |
| ClinicalTrial | 4,180 |
| Project | 18,237 |
| BioEntity | 31,816 |
| Total | 101,871 |

当前图包含 5 类节点：

```text
Patent
Paper
ClinicalTrial
Project
BioEntity
```

其中 BioEntity 表示从 paper、patent、clinical trial 等 carrier 中抽取或标准化后的 biomedical entities。

---

### 3.2 Edge summary

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

当前图包含 8 类主要边表。

其中：

```text
edges_patent_paper
```

是后续 Patent-Paper link prediction 的 target edge table。

后续 link prediction 数据集构建时，需要将该 edge table 从 context graph 中排除，避免 target-edge leakage。

---

## 4. 图验证状态

聚合后图验证结果为：

```text
Overall status: warning
Errors: 0
Warnings: 3
```

当前没有 blocking errors。

剩余 warning 来自非 BioEntity 边表中的 duplicate basic edges：

```text
patent_links_paper
patent_linked_project
paper_linked_trial
```

BioEntity mention-level duplicate edges 已通过 aggregation 解决：

```text
patent_mentions_bioentity duplicate_basic_edge_count = 0
paper_mentions_bioentity duplicate_basic_edge_count = 0
trial_mentions_bioentity duplicate_basic_edge_count = 0
```

因此，本阶段结论是：

```text
图结构可以用于后续算法数据集构建。
```

剩余 warning 不阻塞后续实验，但需要在后续报告中注明。

---

## 5. 图统计审计

图统计使用以下脚本生成：

```bash
python scripts/10_graph_statistics.py \
  --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated
```

主要观察结果如下：

| Statistic | Value |
| --- | ---: |
| Total nodes | 101,871 |
| Total edges | 656,939 |
| Largest connected component | 99,383 nodes |
| Largest component ratio | approximately 97.56% |
| Isolated nodes | 2,488 |

主要结论：

1. 当前图的大部分节点处于最大连通分量中；
2. 最大连通分量占比约为 97.56%；
3. isolated nodes 共 2,488 个；
4. isolated nodes 全部为 Project nodes；
5. BioEntity 边权重使用聚合后的 mention counts；
6. 主要 BioEntity hubs 包括：

```text
patients
insulin
diabetes
glucose
mice
type 2 diabetes
cancer
rat
obesity
mouse
```

其中 species / organism 相关 hubs 后续被认为可能带来噪声，因此在 link prediction 阶段构建了：

```text
no-species
```

数据集变体。

---

## 6. 已完成的数据处理 pipeline

本阶段完成了以下 pipeline stages：

| Stage | Script / Step | Description |
| ---: | --- | --- |
| 01 | TSV to Parquet conversion | 将原始 TSV 数据转换为 Parquet |
| 02 | Parquet validation | 验证 Parquet 文件结构和基本字段 |
| 03 | Parquet profiling | 生成输入数据 profiling 信息 |
| 04 | Profile analysis / extraction planning | 根据 profiling 结果制定图抽取策略 |
| 05 | Core key audit | 审计核心实体 key 和跨表连接字段 |
| 06 | Patent-centered mother graph extraction | 构建 patent-centered mother graph |
| 07 | Extracted graph validation | 验证抽取后的 mother graph |
| 08 | Diabetes domain graph extraction | 抽取 diabetes-domain 子图 |
| 09 | BioEntity mention-edge aggregation | 聚合 BioEntity mention-level edges |
| 10 | Graph statistics audit | 生成图统计报告 |

---

## 7. 复现命令记录

本节记录 Stage 0 从原始 PKG24S4 TSV 数据到当前官方算法原型图的主要复现命令。

当前官方算法原型图为：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

后续所有 Patent-Paper carrier-level link prediction 数据集均应从该目录派生。

---

### 7.1 原始数据 schema inspection

首先检查本地 PKG24S4 TSV 文件 schema，不加载完整文件：

```bash
python scripts/00_inspect_pkg24s4.py
```

该命令输出：

```text
data/interim/schema_summary.csv
```

如需完整 streamed row counts，可额外使用：

```bash
python scripts/00_inspect_pkg24s4.py --count-rows
```

但默认不建议对所有大文件启用完整行数统计，除非明确需要。

---

### 7.2 TSV 到 Parquet 转换

转换前可以先 dry run：

```bash
python scripts/01_tsv_to_parquet.py --dry-run
```

正式转换全部 TSV / TSV.GZ 文件时，使用自动 engine：

```bash
python scripts/01_tsv_to_parquet.py \
  --engine auto \
  --stream-threshold-mb 4096 \
  --batch-size 25000 \
  --overwrite
```

该命令将原始 TSV / TSV.GZ 文件转换为 Parquet，并写入：

```text
data/interim/parquet/
```

同时生成 manifest：

```text
data/interim/parquet_manifest.csv
```

大文件会自动使用 low-memory streaming engine，小文件使用 DuckDB engine。

如需单独转换大文件，例如 `A04_Abstract.tsv.gz`，可使用：

```bash
python scripts/01_tsv_to_parquet.py \
  --files A04_Abstract.tsv.gz \
  --engine auto \
  --overwrite
```

或者强制使用低内存 streaming engine：

```bash
python scripts/01_tsv_to_parquet.py \
  --files A04_Abstract.tsv.gz \
  --engine pyarrow-stream \
  --overwrite \
  --batch-size 25000
```

注意：

```text
data/interim/parquet/
data/interim/bad_rows/
```

均为生成数据，不应提交到版本库。

---

### 7.3 Parquet validation

TSV 转换完成后，验证生成的 Parquet 文件：

```bash
python scripts/02_validate_parquet.py
```

该命令验证：

```text
data/interim/parquet/
```

下的 Parquet 文件，并输出：

```text
data/interim/parquet_validation.csv
```

该步骤用于检查：

```text
1. 预期 Parquet 输出是否存在；
2. manifest 中记录的转换是否完整；
3. row counts 是否异常；
4. file sizes 是否异常；
5. 是否存在 bad rows；
6. 是否存在转换失败或不完整输出。
```

---

### 7.4 Parquet profiling

然后对转换后的 Parquet tables 做 profiling：

```bash
python scripts/03_profile_parquet_tables.py
```

该命令读取：

```text
data/interim/parquet/
```

并输出 profiling 结果到：

```text
data/interim/profile/
```

主要输出包括：

```text
table_profile.csv
column_profile.csv
link_table_profile.csv
entity_candidate_tables.csv
join_candidate_edges.csv
```

这些 profiling 输出是 heuristic analysis，不是最终 schema contract。  
其中：

```text
join_candidate_edges.csv
```

需要人工确认后才能用于图抽取设计。

---

### 7.5 Profile analysis 与 first-pass graph plan

对 profiling 结果进行分析，并生成 graph extraction planning 文档：

```bash
python scripts/04_analyze_profile.py
```

主要输出：

```text
data/interim/profile/profile_analysis.md
data/interim/profile/recommended_subgraph_plan.md
```

该阶段给出的推荐 first-pass graph 是：

```text
patent-centered biomedical translation subgraph
```

包含节点类型：

```text
Patent
Paper
ClinicalTrial
Project
BioEntity
```

暂缓纳入的组件包括：

```text
author graph
affiliation graph
MeSH descriptor graph
inventor / assignee graph
citation graph
BioEntity relationship graph
future text-derived knowledge units
```

---

### 7.6 Core graph key audit

在正式抽图前，审计核心节点和边的 key / join behavior：

```bash
python scripts/05_audit_core_graph_keys.py \
  --coverage-mode row \
  --exact-distinct false \
  --exact-duplicate-edges false
```

该命令输出到：

```text
data/interim/profile/
```

主要输出包括：

```text
core_node_key_audit.csv
core_edge_endpoint_audit.csv
core_fk_coverage.csv
project_number_join_audit.csv
bioentity_unmatched_audit.csv
bioentity_unmatched_entityid_audit.csv
core_key_audit_summary.md
```

该阶段确认了 ProjectNumber 的 join 规则：

```text
B03_Link_Papers_Projects.ProjectNumber
B04_Link_ClinicalTrials_Projects.ProjectNumber
B05_Link_Patents_Projects.ProjectNumber
```

应 join 到：

```text
B02_Projects.CORE_PROJECT_NUM
```

而不是：

```text
FULL_PROJECT_NUM
APPLICATION_ID
SUBPROJECT_ID
```

推荐 normalization 为：

```text
UPPER(REPLACE(REPLACE(TRIM(CAST(value AS VARCHAR)), ' ', ''), '-', ''))
```

该阶段还确认：

```text
EntityId = CUI-less
```

代表未 grounding 的 entity mention，不应作为 BioEntity graph node。

因此 first-pass graph extraction 中：

```text
BioEntity nodes should come from grounded C23_BioEntities.EntityId
CUI-less rows should be filtered from BioEntity graph edges
```

---

### 7.7 Patent-centered mother graph extraction

接下来抽取 patent-centered mother graph。

根据后续 diabetes domain extraction 的输入目录，当前复现使用的 mother graph 目录为：

```text
data/processed/patent_centered_subgraph_2018_2019_full_v1_1
```

推荐复现命令为：

```bash
python scripts/06_extract_patent_centered_subgraph.py \
  --start-year 2018 \
  --end-year 2019 \
  --output-dir data/processed/patent_centered_subgraph_2018_2019_full_v1_1
```

该脚本从：

```text
data/interim/parquet/
```

中读取转换后的 PKG24S4 Parquet tables，并抽取 graph-ready patent-centered biomedical translation subgraph。

抽取逻辑为：

```text
1. 从 C15_Patents 中按 GrantedDate 选择 seed patents；
2. 通过 C16_Link_Patents_Papers 扩展 linked papers；
3. 通过 B05_Link_Patents_Projects 扩展 linked projects；
4. 通过 C18_Link_Patents_BioEntities 扩展 grounded BioEntities；
5. 对 selected papers 扩展 clinical trials、projects 和 BioEntities；
6. 如启用 trial expansion，则继续扩展 trials 到 projects 和 BioEntities。
```

该脚本输出节点文件：

```text
nodes_patent.parquet
nodes_paper.parquet
nodes_clinicaltrial.parquet
nodes_project.parquet
nodes_bioentity.parquet
```

输出边文件：

```text
edges_patent_paper.parquet
edges_patent_bioentity.parquet
edges_patent_project.parquet
edges_paper_trial.parquet
edges_paper_project.parquet
edges_paper_bioentity.parquet
edges_trial_project.parquet
edges_trial_bioentity.parquet
```

同时输出：

```text
subgraph_manifest.csv
subgraph_extraction_report.md
```

注意：

```text
data/processed/
```

下的 graph outputs 是生成数据，不应提交到版本库。

---

### 7.8 Mother graph validation

抽取 mother graph 后，运行 subgraph validation：

```bash
python scripts/07_validate_extracted_subgraph.py \
  --subgraph-dir data/processed/patent_centered_subgraph_2018_2019_full_v1_1
```

该脚本验证：

```text
1. expected node / edge / manifest / report files 是否存在；
2. required node / edge columns 是否存在；
3. node_id 是否 non-null and unique；
4. edge source_id / target_id 是否能匹配对应 node tables；
5. CUI-less 是否泄漏进 BioEntity nodes 或 BioEntity edge targets；
6. Project node IDs 是否已 normalization；
7. duplicate basic edges 情况。
```

主要输出包括：

```text
subgraph_validation_report.md
subgraph_file_validation.csv
subgraph_schema_validation.csv
subgraph_node_validation.csv
subgraph_edge_validation.csv
subgraph_endpoint_validation.csv
subgraph_cuiless_validation.csv
subgraph_project_validation.csv
subgraph_degree_summary.csv
subgraph_top_degree_nodes.csv
subgraph_bioentity_type_counts.csv
subgraph_endpoint_unmatched_samples.csv
```

BioEntity mention-level edges 存在 duplicate basic edges 是预期现象，因为同一 carrier 可能多次 mention 同一 BioEntity。

但以下情况应视为需要修复的问题：

```text
endpoint mismatches
CUI-less leakage
null node IDs
duplicate node IDs
```

---

### 7.9 Diabetes domain subgraph extraction

从 validated patent-centered mother graph 中抽取 diabetes-specific domain subgraph。

当前使用的命令为：

```bash
python scripts/08_extract_domain_subgraph.py \
  --input-dir data/processed/patent_centered_subgraph_2018_2019_full_v1_1 \
  --output-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2 \
  --domain diabetes \
  --carrier-expansion-mode conservative \
  --bioentity-mode all \
  --overwrite
```

该脚本不重新读取原始 PKG24S4 TSV，也不重新扫描大型中间 Parquet tables。  
它只从已经抽取好的 mother graph node / edge files 中工作。

当前 diabetes filter 使用 high-precision Tier 1 diabetes terms 作为 seed evidence，例如：

```text
diabetes
diabetic
type 1 diabetes
type 2 diabetes
diabetes mellitus
gestational diabetes
insulin resistance
hyperglycemia
hypoglycemia
HbA1c
A1C
diabetic nephropathy
diabetic retinopathy
diabetic neuropathy
MODY
```

当前推荐设置为：

```text
--carrier-expansion-mode conservative
--bioentity-mode all
```

含义：

```text
conservative:
  保留直接 diabetes seed carriers，并添加有限 patent / paper / trial / project context；
  不允许 seed projects 反向扩展到过大的 paper 或 patent neighborhoods。

bioentity-mode all:
  保留 selected carriers 的所有 BioEntity mention edges；
  不只保留 keyword-matched BioEntities；
  从而保留 genes、drugs、complications、comorbidities 等 broader biomedical context。
```

该脚本输出同样结构的 domain graph files：

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
```

并输出 audit files：

```text
domain_seed_carriers.parquet
domain_keyword_hits.parquet
domain_manifest.csv
domain_filter_report.md
```

---

### 7.10 Diabetes domain graph validation

抽取 diabetes conservative graph 后，进行验证：

```bash
python scripts/07_validate_extracted_subgraph.py \
  --subgraph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2
```

此时 BioEntity edges 仍是 mention-level edges，因此 duplicate basic edges warning 是预期现象。

但仍需确认不存在：

```text
endpoint mismatches
CUI-less leakage
null node IDs
duplicate node IDs
```

---

### 7.11 BioEntity mention-edge aggregation

由于原始 BioEntity edges 是 mention-level edges，同一 carrier 可能多次 mention 同一 BioEntity，因此需要聚合为 pair-level weighted BioEntity edges。

当前使用命令为：

```bash
python scripts/09_aggregate_mention_edges.py \
  --input-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2 \
  --output-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated \
  --overwrite
```

该脚本将 repeated：

```text
source_id + target_id + edge_type
```

聚合为单条 weighted edge。

主要边权为：

```text
mention_count
```

额外聚合属性包括：

```text
distinct_mention_count
sample_mention
mention_types
text_fields
avg_prob
max_prob
has_mesh
has_ncbigene
has_chebi
```

该脚本聚合以下 BioEntity edge tables：

```text
edges_patent_bioentity.parquet
edges_paper_bioentity.parquet
edges_trial_bioentity.parquet
```

非 BioEntity edge tables 和 node files 原样复制。

当前 diabetes graph 聚合前后边数变化为：

```text
edges_patent_bioentity: 70,597 -> 43,366
edges_paper_bioentity: 817,320 -> 347,466
edges_trial_bioentity: 74,239 -> 28,312
```

聚合输出目录为：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

主要输出包括：

```text
aggregation_report.md
aggregation_manifest.csv
bioentity_edge_aggregation_stats.csv
subgraph_manifest.csv
subgraph_extraction_report.md
```

---

### 7.12 Aggregated graph validation

BioEntity edge aggregation 后，再次验证 aggregated graph：

```bash
python scripts/07_validate_extracted_subgraph.py \
  --subgraph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated
```

该 validated aggregated graph 即当前官方算法原型图：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

当前验证结果为：

```text
Overall status: warning
Errors: 0
Warnings: 3
```

剩余 warnings 来自非 BioEntity 边表中的 duplicate basic edges：

```text
patent_links_paper
patent_linked_project
paper_linked_trial
```

BioEntity mention-level duplicate edges 已经通过 aggregation 消除：

```text
patent_mentions_bioentity duplicate_basic_edge_count = 0
paper_mentions_bioentity duplicate_basic_edge_count = 0
trial_mentions_bioentity duplicate_basic_edge_count = 0
```

---

### 7.13 Final graph statistics audit

最后，对 aggregated graph 生成图统计报告：

```bash
python scripts/10_graph_statistics.py \
  --graph-dir data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated
```

该命令用于计算当前官方算法原型图的 graph-level statistics。

主要输出包括：

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

当前 diabetes conservative aggregated graph 的关键统计为：

```text
total_nodes: 101,871
total_edges: 656,939
connected_components: 2,489
largest_component_node_count: 99,383
isolated_nodes: 2,488
largest_component_ratio: approximately 97.56%
```

最大连通分量包含约 97.56% 的节点，说明该图适合用于：

```text
whole-graph representation learning
link prediction
heterogeneous graph modeling
```

isolated nodes 全部为：

```text
Project nodes
```

后续算法数据集构建时可根据需要过滤或保留。

---

### 7.14 Stage 0 到 Stage 1 的入口

Stage 0 完成后，进入 Patent-Paper link prediction 数据集构建。

推荐入口脚本为：

```bash
python scripts/11_build_link_prediction_dataset.py
```

后续 dataset construction 应明确记录：

```text
input graph directory
target edge table
context edge tables
negative sampling strategy
train / validation / test split
random seed
full / no-species BioEntity filtering rule
```

当前 first algorithm task 为：

```text
Patent-Paper link prediction
```

目标边表为：

```text
edges_patent_paper
```

构建 context graph 时必须排除：

```text
edges_patent_paper
```

以避免 target-edge leakage。

推荐构建两个数据集变体：

```text
1. full-context graph
2. no-species graph
```

其中 no-species graph 应移除：

```text
Type = species
```

的 BioEntity 节点，或至少移除主要 species hubs，例如：

```text
NCBITaxon9606
NCBITaxon10095
NCBITaxon10116
NCBITaxon10090
```

## 8. 后续算法数据集设计

从该图派生的第一个算法数据集为：

```text
Patent-Paper link prediction dataset
```

任务目标：

```text
给定 leakage-controlled carrier context graph，预测 Patent-Paper link 是否存在。
```

输入图包含：

```text
Patent
Paper
ClinicalTrial
Project
BioEntity
```

但用于 embedding 或特征构建的 context graph 不应包含 target edge table：

```text
edges_patent_paper
```

监督标签来自：

```text
positive Patent-Paper edges
negative sampled Patent-Paper pairs
```

推荐构建两个主要数据集变体：

```text
1. full-context graph
2. no-species graph
```

---

## 9. Full-context dataset

Full-context 版本保留全部 BioEntity 节点，包括 species 类型 BioEntity。

该版本用于测试：

```text
保留完整 BioEntity context 时，模型性能如何。
```

潜在问题是：

```text
species BioEntity 节点可能是高频、泛化、hub-like context，
会引入噪声或造成 non-specific overlap signal。
```

---

## 10. No-species dataset

No-species 版本移除：

```text
Type = species
```

的 BioEntity 节点。

或者至少移除主要 species hubs，例如：

```text
NCBITaxon9606
NCBITaxon10095
NCBITaxon10116
NCBITaxon10090
```

该版本用于测试：

```text
移除高频 species context 后，Patent-Paper link prediction 是否更稳健。
```

后续 Stage 1、Stage 2、Stage 3A、Stage 3B 的实验已经表明：

```text
no-species dataset 通常优于 full dataset。
```

这说明 species BioEntity 在当前任务中更像 noise / hub context，而不是有效区分信号。

---

## 11. 工程设计决策

从 script 11 开始，项目进入算法数据集构建和 baseline 实验阶段。

工程设计原则为：

```text
scripts/ 只负责 CLI orchestration
src/pkg2/ 负责 reusable implementation
```

也就是说，后续脚本应尽量避免堆积大量不可复用逻辑。

推荐将公共逻辑放入：

```text
src/pkg2/
```

初始建议模块包括：

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

后续实验阶段已经进一步新增或使用了：

```text
src/pkg2/graph_data.py
src/pkg2/embedding_baselines.py
src/pkg2/graph_io.py
src/pkg2/metrics.py
```

---

## 12. 与后续报告的关系

本 Stage 0 报告是后续所有 Patent-Paper link prediction 实验的数据来源说明。

后续相关报告包括：

```text
reports/experiments/2026-06-28_patent_paper_link_prediction_stage2_basic_baselines.md
reports/experiments/2026-06-29_patent_paper_link_prediction_stage3a_node2vec_baseline.md
reports/experiments/2026-06-29_patent_paper_link_prediction_stage3b_metapath2vec_baseline.md
```

其中：

```text
Stage 2
```

总结了 heuristic、Logistic Regression 和 tree-based baselines。

```text
Stage 3A
```

总结了 Node2Vec / DeepWalk-like graph embedding baseline。

```text
Stage 3B
```

总结了 MetaPath2Vec heterogeneous graph embedding baseline。

这些实验均依赖本阶段构建的官方算法原型图。

---

## 13. 当前阶段结论

本阶段完成了从原始数据到算法原型图的构建流程。

最终官方图为：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

该图包含：

```text
101,871 nodes
656,939 edges
5 node types
8 edge tables
```

图验证结果为：

```text
Errors: 0
Warnings: 3
```

剩余 warnings 来自部分 non-BioEntity edge tables 的 duplicate basic edges，不阻塞后续实验。

图统计显示：

```text
largest connected component ratio ≈ 97.56%
```

说明当前图整体连通性较强，适合用于后续：

```text
link prediction
graph representation learning
heterogeneous graph modeling
knowledge-unit-level temporal translation modeling
```

本阶段之后，项目进入：

```text
Patent-Paper link prediction dataset construction
```

并开始构建：

```text
full-context
no-species
```

两个数据集变体。

---

## 14. 后续复现注意事项

为了保证后续实验可复现，需要特别注意以下几点：

1. 后续所有 Patent-Paper link prediction 数据集应明确记录其输入 graph directory；
2. target edge table `edges_patent_paper` 不应进入 context graph；
3. full 和 no-species 数据集应使用相同的 train / validation / test split 策略；
4. negative sampling 策略需要固定 random seed；
5. no-species 过滤规则需要明确记录；
6. 每个 baseline 结果目录应保存对应的 metrics、predictions、manifest 和 report；
7. 所有 graph representation baselines 应明确说明 embedding training graph 是否包含 target edge table；
8. 后续 temporal / knowledge-unit-level experiments 必须使用严格时间切分，避免 future information leakage。

---

## 15. 简短总结

Stage 0 完成了项目的数据处理和图构建基础工作。

当前官方算法原型图是：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

该图经过 BioEntity mention-edge aggregation 和 graph statistics audit，可以支持后续算法实验。

本阶段最重要的产出是：

```text
一个经过验证的 patent-centered diabetes-domain carrier graph
```

后续所有 carrier-level Patent-Paper link prediction baselines 均从该图派生。

下一步是：

```text
scripts/11_build_link_prediction_dataset.py
```

用于构建：

```text
full-context Patent-Paper link prediction dataset
no-species Patent-Paper link prediction dataset
```
