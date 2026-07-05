# Knowledge Unit Layer and Long-window Temporal Prediction Dataset Stage 4-5 实验总结报告

日期：2026-07-03  
任务：Knowledge Unit construction and KU-level temporal prediction dataset construction  
阶段：

- Stage 4: Knowledge Unit layer construction feasibility validation
- Stage 5: Formal long-window KU-level temporal prediction dataset construction

数据集：

- Stage 4 prototype graph / KU dataset:
  - `data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/`
  - `data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/`
  - `data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel/`
  - `data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_modeling_dataset/`

- Stage 5 formal long-window graph / KU / prediction dataset:
  - `data/processed/diabetes_2000_2024_v1_carrier_graph/`
  - `data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated/`
  - `data/datasets/knowledge_units/diabetes_2000_2024_v1_translational/`
  - `data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_dense_panel/`
  - `data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset/`
  - `data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/`

---

## 1. 实验目的

Stage 4-5 的目标是完成从 carrier-level prototype 到 formal long-window Knowledge Unit temporal prediction benchmark 的过渡。

Stage 3 已验证 carrier context graph 中存在强 Patent-Paper prediction signal，但 Patent-Paper link prediction 仍是 carrier-level proxy task，不是最终主任务。

Stage 4-5 的核心目标是：

1. 验证从 carrier graph 构造 Knowledge Unit layer 是否可行；
2. 确定第一版 Knowledge Unit 工程定义；
3. 构造 carrier-to-KU evidence mapping；
4. 构造 KU-year temporal panel；
5. 验证 trial year parsing、pair canonicalization、species / disease-disease filtering 等关键质量控制；
6. 将短窗口 prototype pipeline 迁移到 2000-2024 formal long-window diabetes graph；
7. 构造 long-window KU temporal modeling dataset；
8. 构造 leakage-controlled KU-level temporal prediction benchmark；
9. 为 Stage 6 baselines 和 Stage 7 Knowledge Field reaction-diffusion-source model 提供 frozen input dataset。

整体路线为：

```text
Carrier Graph
  -> Knowledge Unit Layer
  -> KU Temporal Panel
  -> KU Modeling Dataset
  -> KU Temporal Prediction Dataset
  -> Stage 6 Baselines
  -> Stage 7 Knowledge Field Model
```

---

## 2. Stage 4 定位：Knowledge Unit Layer Feasibility Validation

### 2.1 Stage 4 的角色

Stage 4 是 feasibility validation / prototype 阶段，使用已经完成的 2018-2019 patent-centered diabetes prototype graph：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

Stage 4 不是最终 temporal prediction benchmark。它的作用是验证 KU construction pipeline 的工程可行性和质量控制逻辑。

Stage 4 验证的问题包括：

1. KU 是否能从 carrier graph 中构造出来；
2. typed BioEntity pair 是否适合作为第一版 KU；
3. paper / patent / trial evidence 是否都能进入 KU 层；
4. carrier-KU evidence mapping 是否可追溯；
5. KU-year temporal panel 是否能构建；
6. trial year 是否能从 clinical trial node 中正确解析；
7. pair canonicalization 是否能消除 reversed duplicates；
8. species 和 disease-disease filtering 是否有效；
9. sanity check 是否 PASS；
10. 输出是否能进一步整理为 algorithm-ready modeling dataset。

---

## 3. Stage 4 Knowledge Unit 定义

### 3.1 当前工程定义

当前工程版本采用：

```text
Knowledge Unit = typed BioEntity pair
```

即：

```text
z = (entity_i, entity_j, pair_type)
```

当前 translational / mechanistic pair types 为：

```text
chemical-disease
gene-disease
chemical-gene
```

当前不把 `disease-disease` 放入主版。原因是 disease-disease 共现混合了多种语义，包括：

```text
comorbidity
complication
risk factor
subtype hierarchy
near-synonymy
background co-mention
```

因此，disease-disease pair 暂不适合作为第一版 translational KU 主层。

### 3.2 解释边界

当前 KU 是 carrier co-mention based unit，不是 asserted biomedical relation。

因此：

```text
carrier co-mention evidence does not imply an asserted biomedical relation.
chemical-disease pair != treatment claim
gene-disease pair != causal claim
chemical-gene pair != target claim
```

这一点对后续解释性分析非常重要。

---

## 4. Stage 4A：Knowledge Unit Construction MVP

### 4.1 构建命令

Stage 4 主版 KU dataset 构建命令为：

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

### 4.2 输出目录

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational/
```

核心文件：

```text
knowledge_units.parquet
carrier_knowledge_unit_edges.parquet
knowledge_unit_type_summary.csv
knowledge_unit_summary.json
knowledge_unit_report.md
dataset_manifest.csv
```

### 4.3 主版结果

Stage 4 主版 KU dataset 结果为：

```text
Knowledge Units:              52,476
Carrier-KU edges:             482,196
Paper-only KUs:               37,185
Patent-supported KUs:          8,489
Trial-supported KUs:           8,932
Translation-supported KUs:    15,291
Cross-modal KUs:              13,473
```

这表明在短窗口 prototype graph 上，typed BioEntity pair 可以形成规模合理且包含 paper / patent / trial evidence 的 KU layer。

---

## 5. Stage 4B：Knowledge Unit Audit

### 5.1 Audit 命令

```bash
python scripts/21_audit_knowledge_units.py \
  --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --top-n 100
```

### 5.2 Audit 输出

核心 audit 输出包括：

```text
knowledge_units_with_audit_flags.parquet
knowledge_unit_modality_summary.csv
knowledge_unit_audit_summary.json
knowledge_unit_audit_report.md
```

Audit 的作用是识别：

```text
paper-only KUs
patent-supported KUs
trial-supported KUs
translation-supported KUs
cross-modal KUs
hub entities
top KUs
```

Audit 结果表明 Stage 4 KU layer 中同时存在 paper-only discovery candidates 和 patent/trial translation-supported KUs，具备后续 temporal translation analysis 的基础。

---

## 6. Stage 4 min_carrier_count 阈值分析

### 6.1 Threshold comparison

在 `min_carrier_count=1` audit 版本中，threshold comparison 结果为：

| min_carrier_count | KU count | KU retention | Carrier-KU edge count | Edge retention |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 505,949 | 1.000000 | 989,516 | 1.000000 |
| 2 | 106,323 | 0.210146 | 589,890 | 0.596140 |
| 3 | 52,476 | 0.103718 | 482,196 | 0.487305 |
| 5 | 24,022 | 0.047479 | 387,517 | 0.391623 |
| 10 | 8,919 | 0.017628 | 291,388 | 0.294475 |
| 20 | 3,471 | 0.006860 | 219,467 | 0.221792 |
| 50 | 1,013 | 0.002002 | 147,365 | 0.148926 |
| 100 | 367 | 0.000725 | 103,982 | 0.105084 |

### 6.2 阈值选择

当前选择：

```text
min_carrier_count = 3
```

理由：

1. 去除了大量 singleton KU；
2. 将 KU 数量从 505,949 压缩到 52,476；
3. 仍保留 48.7% carrier-KU evidence edges；
4. 比 min=5 保留更宽的探索候选空间；
5. 适合作为 demo / prototype 阶段主版参数；
6. 后续 Stage 5 long-window KU construction 继续采用该阈值。

---

## 7. Stage 4C/4E：KU Temporal Panel and Trial Year Patch

### 7.1 Dense temporal panel 构建命令

```bash
python scripts/22_build_knowledge_unit_temporal_panel.py \
  --input-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel \
  --dense \
  --overwrite
```

### 7.2 输出目录

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel/
```

核心文件：

```text
knowledge_unit_year_panel.parquet
knowledge_unit_temporal_features.parquet
knowledge_unit_first_observed_events.parquet
knowledge_unit_year_summary.csv
knowledge_unit_temporal_summary.json
knowledge_unit_temporal_report.md
knowledge_unit_temporal_manifest.csv
```

### 7.3 Trial year parsing patch

Stage 4E 修复了 trial year parsing 问题。

此前：

```text
KUs with dated trial evidence:    0
KUs with undated trial evidence:  8,932
```

原因是：

```text
nodes_clinicaltrial.parquet 中 start_date 是日期字符串，
原始 year parser 只支持 numeric year。
```

修复后：

```text
trial edge rows:       26,666
trial dated rows:      26,528
trial undated rows:       138
trial year range:      1972 - 2025
```

当前 dense temporal panel 结果为：

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

### 7.4 关键结论

Trial year patch 证明 KU temporal panel 能正确处理 clinical trial 年份，并保留极少量 undated trial evidence。

这为 Stage 5 long-window temporal prediction dataset 构造提供了关键前置保证。

---

## 8. Stage 4D：KU Sanity Check

### 8.1 Sanity check 命令

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

### 8.2 Sanity check 结果

```text
Overall status:              PASS
Knowledge Units:             52,476
Carrier-KU edges:            482,196
Observed pair types:         chemical-disease, chemical-gene, gene-disease
Unexpected pair types:       []
Disallowed pair types:       []
Disallowed entity types:     []
Duplicate unordered pairs:   0
Invalid edge rows:           0
Count mismatch rows:         0
Paper-only KUs:              37,185
Patent-supported KUs:         8,489
Trial-supported KUs:          8,932
Translation-supported KUs:   15,291
Cross-modal KUs:             13,473
Failed checks:               0
Warnings:                    0
```

### 8.3 关键结论

Sanity check PASS 说明：

1. pair type 过滤正确；
2. disease-disease 未进入主版；
3. species entity 已移除；
4. reversed / unordered duplicate KUs 已消除；
5. carrier-KU edge 与 KU table 计数一致；
6. dense temporal panel 与 KU 数量和 year range 匹配。

---

## 9. Stage 4F：KU Temporal Modeling Dataset Preparation

### 9.1 构建命令

```bash
python scripts/24_build_knowledge_unit_modeling_dataset.py \
  --knowledge-unit-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational \
  --temporal-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_dense_panel \
  --output-dir data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_modeling_dataset \
  --overwrite
```

### 9.2 输出目录

```text
data/datasets/knowledge_units/diabetes_2018_2019_v2_translational_modeling_dataset/
```

输出文件：

```text
knowledge_unit_static_features.parquet
knowledge_unit_temporal_sequences.parquet
knowledge_unit_carrier_observations.parquet
knowledge_unit_modeling_summary.json
knowledge_unit_modeling_manifest.csv
knowledge_unit_modeling_report.md
future_label_plan.json
```

### 9.3 结果

```text
KU count:                        52,476
Static feature rows:             52,476
Temporal sequence rows:       7,976,352
Carrier observation rows:       482,196
Year range:                   1874-2025
Year count:                       152
Trains model:                   false
Defines rule-based state score: false
Has future labels:              false
```

### 9.4 关键结论

Stage 4F 将 KU evidence dataset 和 KU temporal panel 整理为 algorithm-ready modeling input package，但故意不生成 future labels，不训练模型，也不定义 rule-based state score。

Stage 4 完成后，pipeline 已验证：

```text
Carrier Graph
  -> KU evidence dataset
  -> KU temporal panel
  -> KU sanity check
  -> KU temporal modeling dataset
```

---

## 10. Stage 5 定位：Formal Long-window KU-level Temporal Prediction Dataset Construction

### 10.1 Stage 5 的角色

Stage 5 是正式实验数据构造阶段。

与 Stage 4 的区别是：

```text
Stage 4 = 在 2018-2019 prototype graph 上验证 KU layer 构造可行性
Stage 5 = 在正式 2000-2024 long-window graph 上构造后续模型训练 / 评估 / 消融所需的数据
```

Stage 5 的目标是构造：

```text
formal long-window KU-level temporal prediction benchmark
```

当前正式版本命名为：

```text
diabetes_2000_2024_v1
```

---

## 11. Stage 5A：Long-window Carrier Graph Scale / Schema Probe

### 11.1 Probe 目的

Stage 5A 的目的是在正式构图前确认 2000-2024 diabetes long-window graph 的规模、schema 和关键字段对齐是否可行。

### 11.2 Probe 结果

Probe 表明 2000-2024 diabetes graph 规模可控：

```text
selected_papers:                         757,819
selected_patents:                         71,940
selected_trials:                          42,691
candidate_project_numbers:                67,286
selected_bioentities_from_mentions:        92,571
selected_carrier_bioentity_mention_edges: 20,984,791
```

ProjectNumber 对齐规则明确为：

```text
B03/B04/B05.ProjectNumber == B02_Projects.CORE_PROJECT_NUM
```

### 11.3 关键结论

Stage 5A 证明正式长窗口 diabetes carrier graph 可构造，且 project link 的 join key 已明确。

---

## 12. Stage 5B：Long-window Carrier Graph Materialization

### 12.1 输出目录

正式 raw carrier graph：

```text
data/processed/diabetes_2000_2024_v1_carrier_graph/
```

### 12.2 构图原则

```text
domain = diabetes
time window = 2000-2024
seed entity type = disease
context expansion = one-hop patent-paper and paper-trial
project links = linked to selected carriers
project key = CORE_PROJECT_NUM
```

与 Stage 1 的 2018-2019 patent-centered graph 不同，Stage 5B 构造的是 domain-centered long-window carrier graph。

### 12.3 关键结论

该 graph 不再只是 patent-centered prototype，而是用于正式 KU temporal prediction 的 long-window carrier graph input。

---

## 13. Stage 5C：Carrier Graph Aggregation

### 13.1 输出目录

正式 aggregated graph：

```text
data/processed/diabetes_2000_2024_v1_carrier_graph_aggregated/
```

### 13.2 Aggregation 后规模

| Object | Count |
| --- | ---: |
| paper_nodes | 757,809 |
| patent_nodes | 71,933 |
| trial_nodes | 42,691 |
| project_nodes | 62,778 |
| bioentity_nodes | 92,563 |
| paper_bioentity_edges | 7,776,334 |
| patent_bioentity_edges | 327,162 |
| trial_bioentity_edges | 281,790 |
| patent_paper_edges | 833,839 |
| paper_trial_edges | 82,413 |
| paper_project_edges | 291,102 |
| patent_project_edges | 12,634 |
| trial_project_edges | 3,132 |

BioEntity mention edge aggregation：

```text
paper:  raw=19,670,103 -> aggregated=7,776,334
patent: raw=580,860    -> aggregated=327,162
trial:  raw=733,369    -> aggregated=281,790
```

### 13.3 关键结论

Aggregation 将 raw mentions 合并为 carrier-BioEntity level evidence，大幅降低后续 KU pair generation 的输入规模，同时保留 mention_count / pair_weight 所需信息。

---

## 14. Stage 5D：No-cap Streaming Long-window KU Construction

### 14.1 为什么需要 streaming builder

Long-window graph 明显大于 Stage 4 prototype。为了避免 pandas all-in-memory self-join 爆内存，Stage 5D 使用 DuckDB out-of-core / streaming-style KU builder：

```text
scripts/28_build_knowledge_units_streaming.py
```

该 builder 采用：

```text
Parquet scan
  -> filtered mentions
  -> carrier 内 self-join 生成 KU edges
  -> aggregation
  -> parquet output
```

### 14.2 构造参数

正式 KU dataset：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_translational/
```

构造参数：

```text
pair_types = chemical-disease, gene-disease, chemical-gene
exclude_bioentity_types = species
min_carrier_count = 3
max_entities_per_carrier = no cap
pair_weight_method = sqrt_product
```

### 14.3 运行结果

```text
raw_mention_edge_count:          8,385,286
filtered_mention_edge_count:     6,820,550
raw_carrier_ku_edge_count:      17,504,579
filtered_carrier_ku_edge_count: 13,772,584
knowledge_unit_count:              715,912
pair_type_count:                         3
carrier_count:                    590,602
paper_carrier_count:              533,276
patent_carrier_count:              31,325
trial_carrier_count:               26,001
first_year_min:                      2000
last_year_max:                       2024
carrier_count_min:                     3
carrier_count_median:                5.0
carrier_count_max:               123,724
runtime_seconds:                 148.43
```

Pair type summary：

| pair_type | knowledge_unit_count | carrier_count_sum | carrier_count_mean | paper_count_sum | patent_count_sum | trial_count_sum | total_pair_weight_sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gene-disease | 279,396 | 5,611,474 | 20.084303 | 5,375,843 | 121,889 | 113,742 | 1.147951e+07 |
| chemical-disease | 231,001 | 4,863,689 | 21.054840 | 4,594,161 | 124,156 | 145,372 | 9.242535e+06 |
| chemical-gene | 205,515 | 3,297,421 | 16.044673 | 3,195,452 | 53,645 | 48,324 | 6.517487e+06 |

### 14.4 关键结论

Stage 5D 成功完成 2000-2024 no-cap long-window KU construction，最终得到：

```text
715,912 KUs
13,772,584 filtered carrier-KU edges
590,602 supporting carriers
```

该规模适合后续 temporal modeling 和 prediction dataset construction。

---

## 15. Stage 5E：Long-window KU Audit

### 15.1 Audit 输出

正式 long-window KU audit 结果：

```text
Knowledge Units:             715,912
Carrier-KU edges:         13,772,584
Pair types:                        3
Carrier count:               590,602
Paper-only KUs:              563,562
Patent-supported KUs:         78,635
Trial-supported KUs:          91,408
Translation-supported KUs:   152,350
Cross-modal KUs:             136,812
All-modality KUs:             17,397
First year min:                 2000
Last year max:                  2024
```

### 15.2 Evidence modality summary

| evidence_pattern | observed_status | modality_count | ku_count | ku_fraction | median_carrier_count | max_carrier_count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P | paper_only | 1 | 563,562 | 0.787195 | 5 | 47,066 |
| A | patent_only | 1 | 14,869 | 0.020769 | 4 | 76 |
| T | trial_only | 1 | 669 | 0.000934 | 3 | 27 |
| PT | paper_trial | 2 | 73,046 | 0.102032 | 10 | 4,482 |
| PA | paper_patent | 2 | 46,073 | 0.064356 | 7 | 2,662 |
| AT | patent_trial | 2 | 296 | 0.000413 | 3 | 24 |
| PAT | paper_patent_trial | 3 | 17,397 | 0.024300 | 49 | 123,724 |

### 15.3 Pair type summary

| pair_type | ku_count | carrier_count_sum | median_carrier_count | max_carrier_count | paper_only_ku | patent_supported_ku | trial_supported_ku | cross_modal_ku |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gene-disease | 279,396 | 5,611,474 | 5 | 97,153 | 222,361 | 31,071 | 33,571 | 53,003 |
| chemical-disease | 231,001 | 4,863,689 | 5 | 123,724 | 166,083 | 30,833 | 41,774 | 56,988 |
| chemical-gene | 205,515 | 3,297,421 | 5 | 81,084 | 175,118 | 16,731 | 16,063 | 26,821 |

### 15.4 Hub observations

Top entity hubs include:

```text
diabetes
type 2 diabetes
glucose
insulin
cancer
hyperglycemia
metabolic syndrome
cardiovascular disease
obesity
hypertension
inflammation
```

这些 hubs 符合 diabetes biomedical domain 的实际结构，但也说明后续模型需要关注：

```text
hub effects
feature scaling
top-K discovery bias
case study interpretation
```

Stage 5 当前不直接删除这些 domain hubs，而是在 Stage 6/7 中通过 features、ablation 和解释性分析处理。

---

## 16. Stage 5E：Dense Temporal Panel

### 16.1 输出目录

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_dense_panel/
```

### 16.2 输出文件

```text
knowledge_unit_first_observed_events.parquet
knowledge_unit_temporal_features.parquet
knowledge_unit_temporal_manifest.csv
knowledge_unit_temporal_report.md
knowledge_unit_temporal_summary.json
knowledge_unit_year_panel.parquet
knowledge_unit_year_summary.csv
```

### 16.3 Temporal panel summary

```text
start_year: 2000
end_year: 2024
year_count: 25
knowledge_unit_count: 715,912
panel_row_count: 17,897,800
temporal_feature_row_count: 715,912
carrier_ku_edge_count: 13,772,584
dated_edge_count: 13,772,584
undated_edge_count: 0
dated_edge_fraction: 1.0
ku_with_dated_paper_evidence: 700,078
ku_with_dated_patent_evidence: 78,635
ku_with_dated_trial_evidence: 91,408
ku_with_dated_translation_evidence: 152,350
```

### 16.4 Lag summary

| Lag | Count | Min | Q25 | Median | Q75 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper_to_patent_lag | 63,470 | -24 | -1 | 6 | 12 | 24 |
| paper_to_trial_lag | 90,443 | -24 | 0 | 5 | 11 | 24 |
| patent_to_trial_lag | 17,693 | -24 | -7 | -1 | 5 | 24 |

### 16.5 关键结论

Dense panel 行数完全匹配：

```text
715,912 KUs × 25 years = 17,897,800 rows
```

所有 carrier-KU edges 都有有效年份：

```text
dated_edge_fraction = 1.0
undated_edge_count = 0
```

这说明 long-window temporal panel 可直接支持 cutoff-based feature / target construction。

---

## 17. Stage 5E：Long-window Sanity Check

### 17.1 Sanity check 命令

```bash
python scripts/23_check_knowledge_unit_sanity.py \
  --input-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational \
  --temporal-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_dense_panel \
  --expected-pair-types chemical-disease gene-disease chemical-gene \
  --disallowed-pair-types disease-disease \
  --disallowed-entity-types species \
  --expect-no-duplicates \
  --expect-dense-temporal \
  --overwrite
```

### 17.2 Sanity check 结果

```text
Overall status:              PASS
Knowledge Units:             715,912
Carrier-KU edges:         13,772,584
Observed pair types:         chemical-disease, chemical-gene, gene-disease
Unexpected pair types:       []
Disallowed pair types:       []
Disallowed entity types:     []
Duplicate unordered pairs:   0
Invalid edge rows:           0
Count mismatch rows:         0
Paper-only KUs:              563,562
Patent-supported KUs:         78,635
Trial-supported KUs:          91,408
Translation-supported KUs:   152,350
Cross-modal KUs:             136,812
Failed checks:               0
Warnings:                    0
```

### 17.3 关键结论

Long-window KU dataset 通过全部 sanity checks。它在规模上远大于 Stage 4 prototype，但仍满足 pair type、entity type、duplicate、edge consistency 和 dense temporal panel 检查。

---

## 18. Stage 5E：Long-window KU Modeling Dataset

### 18.1 构建命令

```bash
python scripts/24_build_knowledge_unit_modeling_dataset.py \
  --knowledge-unit-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational \
  --temporal-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_dense_panel \
  --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset \
  --overwrite
```

### 18.2 输出目录

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset/
```

输出文件：

```text
knowledge_unit_static_features.parquet
knowledge_unit_temporal_sequences.parquet
knowledge_unit_carrier_observations.parquet
knowledge_unit_modeling_summary.json
knowledge_unit_modeling_manifest.csv
knowledge_unit_modeling_report.md
future_label_plan.json
```

### 18.3 Modeling dataset summary

```text
ku_count:                        715,912
static_feature_rows:             715,912
temporal_sequence_rows:       17,897,800
carrier_observation_rows:     13,772,584
year_min:                         2000
year_max:                         2024
year_count:                         25
trains_model:                    False
defines_rule_based_state_scores: False
has_future_labels:               False
```

### 18.4 关键结论

该 dataset 是 Stage 5F prediction dataset construction 的 standardized input package。

它故意不包含：

```text
future labels
train / validation / test split
model training outputs
rule-based Knowledge State scores
```

---

## 19. Stage 5F：Formal Temporal Prediction Targets and Splits

### 19.1 Stage 5F 目标

Stage 5F 的目标是构造 formal supervised prediction benchmark：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/
```

该 benchmark 用于 Stage 6 baseline 和 Stage 7 main algorithm。

### 19.2 Forecasting protocol

Stage 5F 使用：

```text
expanding-window rolling-origin forecasting
```

样本单位：

```text
KU × cutoff_year
```

历史窗口：

```text
history = 2000 .. cutoff_year
```

未来窗口：

```text
future = cutoff_year + 1 .. cutoff_year + 3
```

horizon：

```text
3 years
```

时间划分：

```text
2000-2004 = warmup history
2005-2021 = prediction cutoff years
2022-2024 = latest future target years
```

### 19.3 Train / validation / test split

正式 split：

```text
train cutoffs:
  2005-2016

validation cutoffs:
  2017-2018

test cutoffs:
  2019-2021
```

对应：

```text
2005 -> future 2006-2008
...
2016 -> future 2017-2019

2017 -> future 2018-2020
2018 -> future 2019-2021

2019 -> future 2020-2022
2020 -> future 2021-2023
2021 -> future 2022-2024
```

该设计保证所有样本都有完整 3-year future window，且 train / validation / test 按 cutoff year 分离，不使用随机 split。

### 19.4 主任务：Future heat forecasting

Stage 5F 将主任务从 binary emergence prediction 升级为：

```text
future heat / intensity forecasting
```

原因是传统 patent analysis / technology forecasting 通常关注：

```text
future activity
growth
intensity
technology heat
innovation potential
ranking
```

而不是单纯预测下一年是否出现一条 evidence。

当前三类任务为：

```text
Clinical Translation Heat Forecasting
Patent Translation Heat Forecasting
Future Translation Heat Forecasting
```

其中 `Future Translation Heat Forecasting` 是第一主任务。

### 19.5 Targets and labels

Patent task：

```text
target_future_patent_count_3yr =
  sum patent_count_year over future window

target_future_patent_heat_3yr =
  log1p(target_future_patent_count_3yr)

label_future_patent_emergence_3yr =
  1[target_future_patent_count_3yr > 0]

eligible_patent_task =
  history_paper_count > 0
  AND history_patent_count == 0
```

Trial task：

```text
target_future_trial_count_3yr =
  sum trial_count_year over future window

target_future_trial_heat_3yr =
  log1p(target_future_trial_count_3yr)

label_future_trial_emergence_3yr =
  1[target_future_trial_count_3yr > 0]

eligible_trial_task =
  history_paper_count > 0
  AND history_trial_count == 0
```

Translation task：

```text
target_future_translation_count_3yr =
  target_future_patent_count_3yr
  + target_future_trial_count_3yr

target_future_translation_heat_3yr =
  log1p(target_future_translation_count_3yr)

label_future_translation_emergence_3yr =
  1[target_future_translation_count_3yr > 0]

eligible_translation_task =
  history_paper_count > 0
  AND history_patent_count == 0
  AND history_trial_count == 0
```

### 19.6 Stage 5F 构建命令

```bash
python scripts/29_build_knowledge_unit_prediction_dataset.py \
  --modeling-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_translational_modeling_dataset \
  --output-dir data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction \
  --history-start-year 2000 \
  --first-cutoff-year 2005 \
  --last-cutoff-year 2021 \
  --horizon-years 3 \
  --train-cutoffs 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 \
  --validation-cutoffs 2017 2018 \
  --test-cutoffs 2019 2020 2021 \
  --threads 4 \
  --memory-limit 24GB \
  --overwrite
```

### 19.7 Stage 5F 输出文件

```text
prediction_examples.parquet
prediction_feature_table.parquet

prediction_examples_patent.parquet
prediction_examples_trial.parquet
prediction_examples_translation.parquet

prediction_label_summary.csv
prediction_split_summary.csv
prediction_task_summary.csv
prediction_cutoff_summary.csv

prediction_dataset_summary.json
prediction_dataset_report.md
prediction_leakage_audit_report.md

feature_columns.json
target_columns.json
run_config.json
prediction_dataset_manifest.csv
```

### 19.8 Stage 5F 结果

```text
Prediction examples:        12,170,504
Prediction feature rows:    12,170,504
Patent task rows:            8,362,305
Trial task rows:             8,151,797
Translation task rows:       7,801,391
KU count:                      715,912
Cutoff count:                       17
Leakage audit status:             PASS
```

行数验证：

```text
715,912 KUs × 17 cutoff years = 12,170,504 prediction examples
```

### 19.9 Cutoff-level summary

| cutoff_year | split | eligible_patent_count | eligible_trial_count | eligible_translation_count | future_patent_positive_count | future_trial_positive_count | future_translation_positive_count | patent_heat_mean | trial_heat_mean | translation_heat_mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2005 | train | 301,685 | 299,486 | 294,400 | 3,162 | 6,815 | 9,043 | 0.003790 | 0.007688 | 0.010653 |
| 2006 | train | 332,424 | 329,273 | 322,891 | 3,246 | 7,339 | 9,502 | 0.003565 | 0.007470 | 0.010157 |
| 2007 | train | 361,390 | 356,513 | 349,064 | 4,241 | 7,834 | 10,773 | 0.004232 | 0.007286 | 0.010577 |
| 2008 | train | 388,298 | 381,443 | 372,965 | 5,512 | 7,977 | 11,932 | 0.005039 | 0.006841 | 0.010856 |
| 2009 | train | 412,676 | 404,467 | 394,711 | 6,245 | 8,346 | 12,871 | 0.005383 | 0.006692 | 0.011047 |
| 2010 | train | 436,896 | 427,374 | 415,730 | 6,972 | 8,021 | 13,096 | 0.005636 | 0.006014 | 0.010598 |
| 2011 | train | 460,335 | 450,145 | 436,268 | 7,447 | 7,885 | 13,278 | 0.005780 | 0.005610 | 0.010304 |
| 2012 | train | 482,540 | 471,174 | 455,390 | 8,031 | 8,230 | 14,026 | 0.005910 | 0.005653 | 0.010477 |
| 2013 | train | 502,485 | 490,949 | 472,569 | 7,675 | 8,888 | 14,196 | 0.005457 | 0.005863 | 0.010245 |
| 2014 | train | 520,477 | 508,761 | 487,768 | 7,609 | 9,870 | 14,931 | 0.005280 | 0.006243 | 0.010425 |
| 2015 | train | 536,679 | 523,943 | 500,437 | 7,631 | 10,721 | 15,583 | 0.004950 | 0.006528 | 0.010359 |
| 2016 | train | 567,322 | 551,777 | 524,425 | 8,272 | 11,755 | 16,893 | 0.005181 | 0.006749 | 0.010746 |
| 2017 | validation | 588,860 | 571,670 | 540,806 | 8,632 | 12,291 | 17,589 | 0.005102 | 0.006850 | 0.010781 |
| 2018 | validation | 604,222 | 584,926 | 551,105 | 8,353 | 11,667 | 16,651 | 0.004896 | 0.006382 | 0.010108 |
| 2019 | test | 614,777 | 594,362 | 557,718 | 7,826 | 11,666 | 16,115 | 0.004366 | 0.006294 | 0.009562 |
| 2020 | test | 622,836 | 600,845 | 561,611 | 6,484 | 10,807 | 14,230 | 0.003507 | 0.005671 | 0.008218 |
| 2021 | test | 628,403 | 604,689 | 563,533 | 4,521 | 7,872 | 10,233 | 0.002334 | 0.004064 | 0.005758 |

### 19.10 Right-edge observation note

Stage 5F cutoff summary 显示 test 期存在 future target 稀疏化，尤其是：

```text
cutoff = 2021
future = 2022-2024
```

这不是 Stage 5F 构造错误，而是 long-window temporal prediction 中需要处理的 right-edge observation issue。

可能原因包括：

```text
patent publication lag
grant / publication record lag
database update lag
paper-patent / paper-trial linkage lag
clinical trial record update lag
```

因此 Stage 6 应报告：

```text
test_all:
  cutoffs = 2019, 2020, 2021

test_by_cutoff:
  cutoff = 2019
  cutoff = 2020
  cutoff = 2021

test_stable:
  cutoffs = 2019, 2020
```

其中：

```text
2021 cutoff 可作为 right-edge stress test。
```

---

## 20. Stage 5F Leakage Audit

### 20.1 Audit requirements

Stage 5F leakage audit 检查：

```text
1. 所有 feature 使用 year <= cutoff_year；
2. 所有 target / label 使用 cutoff_year < year <= cutoff_year + horizon；
3. train / validation / test 只按 cutoff_year 分割；
4. test cutoff 不早于 validation cutoff；
5. validation cutoff 不早于 train cutoff；
6. latest cutoff + horizon <= dataset end_year；
7. eligible_*_task 中 prior patent/trial 条件只使用 cutoff 前累计；
8. future heat target 不进入 feature columns；
9. full-window summary columns 不进入 prediction_feature_table；
10. 每个 task view 只包含对应 eligible rows；
11. translation target count = patent target count + trial target count；
12. translation emergence label = patent emergence OR trial emergence。
```

### 20.2 Audit result

```text
Leakage audit status: PASS
```

### 20.3 关键结论

Stage 5F 输出的 prediction dataset 满足 leakage-controlled temporal forecasting 要求，可作为 Stage 6 和 Stage 7 的 frozen benchmark。

---

## 21. Stage 4-5 总体发现

### 21.1 Typed BioEntity pair 可作为第一版 Knowledge Unit

Stage 4 prototype 和 Stage 5 long-window dataset 都证明：

```text
Knowledge Unit = typed BioEntity pair
```

在工程上可构造、可审计、可追溯，并能形成 paper / patent / trial temporal evidence panel。

### 21.2 Stage 4 pipeline 成功迁移到 Stage 5 long-window setting

Stage 4 使用 2018-2019 patent-centered prototype graph 验证了 pipeline feasibility。

Stage 5 将该 pipeline 扩展到 2000-2024 domain-centered long-window graph，并成功构造：

```text
715,912 KUs
17,897,800 KU-year rows
12,170,504 KU × cutoff prediction examples
```

### 21.3 No-cap streaming KU construction 可扩展

DuckDB out-of-core streaming builder 在 no-cap setting 下完成 long-window KU construction：

```text
runtime_seconds: 148.43
filtered_carrier_ku_edge_count: 13,772,584
knowledge_unit_count: 715,912
```

说明该方法适合正式 long-window biomedical KU construction。

### 21.4 Long-window temporal panel 质量良好

Long-window dense panel：

```text
dated_edge_fraction = 1.0
undated_edge_count = 0
```

说明 2000-2024 long-window graph 中 paper / patent / trial evidence 均可进入 yearly panel。

### 21.5 Formal prediction benchmark 已冻结

Stage 5F 已经冻结：

```text
candidate / eligibility rules
cutoff years
train / validation / test split
future heat targets
binary emergence labels
feature whitelist
leakage audit
```

该 benchmark 是后续 Stage 6 / Stage 7 的统一评估基础。

### 21.6 Future heat forecasting 更符合 patent analysis 场景

相较于单纯二分类 emergence prediction，Stage 5F 将主任务定义为：

```text
future patent / trial / translation heat forecasting
```

更符合 patent analysis 和 technology forecasting 对 activity、growth、trend、heat 和 ranking 的关注。

### 21.7 Right-edge observation issue 需要在 Stage 6 报告中处理

Test cutoffs 中：

```text
2021 -> future 2022-2024
```

出现 future positive count 和 heat mean 明显下降。

这应在 Stage 6 中通过：

```text
pooled test
by-cutoff test
stable test subset
right-edge stress test
```

进行透明报告，而不是忽略。

---

## 22. 当前推荐的结果表述

### 中文表述

```text
Stage 4-5 完成了从 carrier-level prototype 到 formal long-window Knowledge Unit temporal prediction benchmark 的转换。Stage 4 在 2018-2019 diabetes prototype graph 上验证了 typed BioEntity pair 作为 Knowledge Unit 的可行性，并完成 KU construction、audit、temporal panel、sanity check 和 modeling dataset preparation。Stage 5 将该 pipeline 扩展到 2000-2024 long-window diabetes graph，构造了 715,912 个 Knowledge Units、17,897,800 条 KU-year temporal rows 和 12,170,504 条 KU × cutoff prediction examples。最终 Stage 5F 采用 expanding-window rolling-origin protocol，构造了 leakage-controlled future heat forecasting benchmark，并通过 leakage audit。该 benchmark 已冻结，可直接用于 Stage 6 baselines 和 Stage 7 Knowledge Field reaction-diffusion-source model。
```

### 英文表述

```text
Stages 4-5 completed the transition from a carrier-level prototype to a formal long-window Knowledge Unit temporal prediction benchmark. Stage 4 validated the feasibility of constructing Knowledge Units as typed BioEntity pairs on the 2018-2019 diabetes prototype graph, including KU construction, audit, temporal panel generation, sanity checking, and modeling dataset preparation. Stage 5 scaled the pipeline to the 2000-2024 long-window diabetes graph, yielding 715,912 Knowledge Units, 17,897,800 KU-year temporal rows, and 12,170,504 KU-by-cutoff prediction examples. The final Stage 5F benchmark uses an expanding-window rolling-origin protocol to define leakage-controlled future heat forecasting targets and time-based train/validation/test splits. This frozen benchmark is ready for Stage 6 baselines and the Stage 7 Knowledge Field reaction-diffusion-source model.
```

---

## 23. 结论

Stage 4-5 已完成 Knowledge Unit 层构造与 formal long-window temporal prediction dataset construction。

最终正式 benchmark 为：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/
```

核心结果：

```text
Knowledge Units:              715,912
Carrier-KU edges:          13,772,584
KU-year temporal rows:     17,897,800
Prediction examples:       12,170,504
Patent task rows:           8,362,305
Trial task rows:            8,151,797
Translation task rows:      7,801,391
Leakage audit:              PASS
```

Stage 4-5 的主要贡献是：

1. 验证 typed BioEntity pair 作为 Knowledge Unit 的工程可行性；
2. 建立 carrier-KU evidence mapping 和 KU temporal panel；
3. 修复并验证 trial year parsing；
4. 建立 KU sanity check 和 audit 流程；
5. 将 pipeline 扩展到 2000-2024 long-window diabetes graph；
6. 使用 streaming / out-of-core 方法完成 no-cap long-window KU construction；
7. 构造 formal KU-level temporal prediction benchmark；
8. 将主任务定义为 future patent / trial / translation heat forecasting；
9. 冻结 leakage-controlled train / validation / test split；
10. 为 Stage 6 baselines 和 Stage 7 main algorithm 提供统一数据基础。

下一步进入：

```text
Stage 6: KU-level temporal baseline and ablation experiments
```

第一轮建议优先运行：

```text
Task:
  Future Translation Heat Forecasting

Input:
  prediction_feature_table.parquet
  feature_columns.json
  target_columns.json

Task filter:
  eligible_translation_task == true

Primary target:
  target_future_translation_heat_3yr

Auxiliary label:
  label_future_translation_emergence_3yr
```

Stage 6 应优先输出：

```text
baseline_comparison.csv
baseline_comparison_report.md
test_by_cutoff_metrics.csv
test_stable_metrics.csv
```