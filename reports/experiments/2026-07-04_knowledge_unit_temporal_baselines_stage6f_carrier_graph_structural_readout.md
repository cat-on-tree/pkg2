# Stage 06F Carrier Graph Structural Readout Baseline 实验报告

日期：2026-07-04  
阶段：Stage 06F  
实验名称：Carrier graph structural readout baseline  
任务：KU-level temporal forecasting  
主要目的：在同一个 KU-level temporal forecasting benchmark 上，比较 **KU representation** 与 **原始 carrier graph 的直接结构读出表示**。

---

## 1. 实验目的

Stage 06E 已经回答了一个 feature-subset 问题：

```text
KU full representation 相比 numeric temporal evidence features only 是否有增量？
```

但 Stage 06E 并没有直接回答更关键的问题：

```text
KU abstraction 相比初始 carrier graph representation 是否更有预测价值？
```

因此，Stage 06F 设计了一个 same-task carrier graph baseline：

```text
原始 carrier graph
→ cutoff-safe structural readout
→ pool 到 KU × cutoff_year
→ 使用同一套 future forecasting labels / splits / metrics / learners
→ 与 KU full representation 比较
```

本实验重点回答三个问题：

```text
1. 直接从原始 carrier graph 读取结构特征，是否足以预测 KU future translation / patent / trial emergence？
2. KU full representation 相比 carrier graph structural readout 是否更强？
3. carrier graph structural readout 是否能在 KU full representation 之外提供额外增量？
```

---

## 2. 实验定位

本实验不是 GNN baseline，也不是 node2vec / DeepWalk embedding baseline。

本实验使用的是：

```text
direct carrier-graph structural readout baseline
```

即从原始 carrier graph 中提取 cutoff-safe topology statistics，例如：

```text
carrier node degree
weighted degree
unique neighbor count
neighbor type composition
edge type diversity
```

然后将这些 carrier-level structural features 聚合到：

```text
KU × cutoff_year
```

这个 baseline 的意义是：

```text
在不引入复杂 GNN / embedding 训练的情况下，
直接测试原始 carrier graph topology 对 KU-level future forecasting 的预测价值。
```

因此，本实验可以支持如下比较：

```text
KU full representation
vs
direct structural readout from the original carrier graph
```

但不能被表述为：

```text
KU representation beats all carrier graph GNNs.
```

---

## 3. 数据与任务设置

本实验沿用 Stage 05F 冻结的 temporal prediction benchmark。

预测样本单位：

```text
KU × cutoff_year
```

历史窗口：

```text
2000 .. cutoff_year
```

预测窗口：

```text
cutoff_year + 1 .. cutoff_year + 3
```

数据划分：

```text
train:
  cutoff_year = 2005-2016

validation:
  cutoff_year = 2017-2018

test_all:
  cutoff_year = 2019-2021

test_stable:
  cutoff_year = 2019-2020

right-edge stress cutoff:
  cutoff_year = 2021
```

任务包括：

| Task | Eligibility | Heat target | Emergence label |
|---|---|---|---|
| translation | `eligible_translation_task` | `target_future_translation_heat_3yr` | `label_future_translation_emergence_3yr` |
| patent | `eligible_patent_task` | `target_future_patent_heat_3yr` | `label_future_patent_emergence_3yr` |
| trial | `eligible_trial_task` | `target_future_trial_heat_3yr` | `label_future_trial_emergence_3yr` |

---

## 4. Representation 设计

本实验比较三组 representation。

### 4.1 Carrier graph structural only

结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_carrier_graph_structural_only_500k/
```

特征来源：

```text
原始 carrier graph 的 cutoff-safe structural readout
```

feature set：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/feature_sets/carrier_graph_structural_only/
```

feature count：

```text
47
```

该组不使用 KU full temporal/schema features，只使用从原始 carrier graph 读出的 structural features。

主要包括：

```text
cgs_support_count
cgs_coverage_count
cgs_coverage_fraction

cgs_support_paper_count
cgs_support_patent_count
cgs_support_trial_count
cgs_support_other_count

cgs_degree_mean / max / sum / std
cgs_weighted_degree_mean / max / sum / std
cgs_unique_neighbor_mean / max / sum / std
cgs_edge_type_count_mean / max / sum / std

cgs_paper_neighbor_*
cgs_patent_neighbor_*
cgs_trial_neighbor_*
cgs_project_neighbor_*
cgs_bioentity_neighbor_*
cgs_unknown_neighbor_*
```

该 representation 回答：

```text
直接 carrier graph topology readout 本身有多强？
```

---

### 4.2 KU full representation

结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_ku_full_500k/
```

feature set：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/
```

feature count：

```text
61
```

包括：

```text
KU temporal numeric features
+
KU schema categorical features
```

即：

```text
history / recent / growth / recency temporal evidence features
pair_type
entity_a_type
entity_b_type
```

该 representation 代表当前 KU abstraction 的完整 tabular baseline。

---

### 4.3 KU full + carrier graph structural

结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_ku_plus_carrier_graph_structural_500k/
```

feature set：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/feature_sets/ku_plus_carrier_graph_structural/
```

feature count：

```text
108
```

包括：

```text
KU full features
+
carrier graph structural readout features
```

该组用于测试：

```text
原始 carrier graph structural exposure 是否能在 KU full representation 之外提供额外增量？
```

---

## 5. Carrier graph structural readout 构造

结构特征由以下脚本构造：

```text
scripts/31_build_knowledge_unit_carrier_graph_structural_features.py
```

该脚本执行以下步骤：

```text
1. 读取 KU × cutoff_year prediction feature table；
2. 读取 carrier-KU observation table；
3. 读取原始 carrier graph edge parquet files；
4. 将原始 graph edges 归一化为 carrier-centric directed edge events；
5. 对每个 cutoff_year，只使用 edge_year <= cutoff_year 的边；
6. 对每个 carrier node 计算 cutoff-safe structural statistics；
7. 对每个 KU × cutoff_year，聚合 cutoff 前支持该 KU 的 carrier nodes 的 structural statistics；
8. 输出 script-30-compatible feature sets。
```

关键约束：

```text
所有 graph structural features 都只使用 cutoff_year 及以前的 graph edges；
不使用 future target window 内的信息；
因此是 cutoff-safe 的。
```

输出目录：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction_carrier_graph_structural/
```

主要 feature set：

```text
feature_sets/carrier_graph_structural_only/
feature_sets/ku_plus_carrier_graph_structural/
```

---

## 6. 训练协议

由于 KU + carrier graph structural representation 在 full-train 设置下超过本地内存限制，Stage 06F 使用 matched capped-training protocol。

所有 representation、所有 trainable learners 均使用相同训练上限：

```text
max_train_rows = 500000
```

评估仍然在完整 validation/test task frames 上进行。

也就是说：

```text
训练：
  每个 task/model 最多使用 500,000 training rows

评估：
  使用完整 validation/test rows
```

该设置保证了：

```text
representation 之间的比较是公平的；
不同 representation 使用相同 train cap；
不同 representation 使用相同 task / split / metrics / learners。
```

模型：

```text
ridge
logistic
logistic_unweighted
histgb
```

运行状态：

| Representation | Feature count | max_train_rows | error_count | Runtime seconds |
|---|---:|---:|---:|---:|
| carrier_graph_structural_only_500k | 47 | 500000 | 0 | 261.4 |
| ku_full_500k | 61 | 500000 | 0 | 350.8 |
| ku_plus_structural_500k | 108 | 500000 | 0 | 726.0 |

---

## 7. Evaluation metrics

本实验主要关注：

```text
Spearman
AUROC
AUPRC
NDCG@1000
Precision@1000
Enrichment@1000
Brier
```

其中：

```text
AUROC / AUPRC:
  binary emergence discrimination

Spearman:
  future heat ranking correlation

NDCG@1000:
  top-ranked heat ranking quality

Precision@1000 / Enrichment@1000:
  top-K discovery performance

Brier:
  probability calibration / binary probability error
```

由于 future heat target 极度 sparse / zero-inflated，主要解释重点放在：

```text
AUROC
AUPRC
Spearman
NDCG@1000
Precision@1000
Enrichment@1000
```

---

## 8. Test-all 结果

`test_all` 包含：

```text
cutoff_year = 2019, 2020, 2021
```

### 8.1 Patent task

#### HistGB

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0506 | 0.6462 | 0.0252 | 0.0557 | 0.108 | 10.70 |
| KU full | 0.0470 | 0.6348 | 0.0251 | 0.0769 | 0.121 | 11.99 |
| KU + structural | **0.0513** | **0.6482** | **0.0276** | **0.0858** | **0.149** | **14.76** |

#### Logistic

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0498 | 0.6440 | 0.0253 | 0.0485 | 0.088 | 8.72 |
| KU full | 0.0382 | 0.6104 | 0.0234 | 0.0784 | 0.125 | 12.39 |
| KU + structural | **0.0544** | **0.6571** | **0.0287** | **0.0816** | **0.129** | **12.78** |

主要观察：

```text
1. Patent task 中 carrier structural only 本身较强，global metrics 有时高于 KU full。
2. KU full 的 top-K precision 通常高于 carrier structural only。
3. KU + structural 在 patent 上整体最强，尤其 Precision@1000 和 Enrichment@1000 提升明显。
```

解释：

```text
Patent emergence 对 carrier graph topology / graph exposure 更敏感；
原始 carrier graph structural features 对 patent 有独立预测价值；
但将其与 KU full 结合后效果最好。
```

---

### 8.2 Translation task

Translation 是 primary task。

#### HistGB

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0625 | 0.6175 | 0.0408 | 0.0339 | 0.075 | 3.11 |
| KU full | 0.0854 | 0.6606 | 0.0480 | **0.0772** | **0.152** | **6.30** |
| KU + structural | **0.0890** | **0.6673** | **0.0519** | 0.0627 | 0.127 | 5.27 |

#### Logistic

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0551 | 0.6036 | 0.0392 | 0.0241 | 0.051 | 2.12 |
| KU full | 0.0614 | 0.6156 | 0.0427 | 0.0653 | 0.122 | 5.06 |
| KU + structural | **0.0741** | **0.6394** | **0.0473** | **0.0716** | **0.137** | **5.68** |

#### Ridge

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0484 | 0.5910 | 0.0376 | 0.0168 | 0.035 | 1.45 |
| KU full | 0.0525 | 0.5987 | 0.0397 | 0.0593 | 0.110 | 4.56 |
| KU + structural | **0.0656** | **0.6234** | **0.0436** | **0.0630** | **0.116** | **4.81** |

主要观察：

```text
1. 在 primary translation task 上，KU full 明显优于 carrier structural only。
2. KU + structural 在 AUROC / AUPRC / Spearman 上进一步优于 KU full。
3. HistGB 中 KU full 的 top-K Precision@1000 高于 KU + structural；但 logistic/ridge 中 KU + structural 的 top-K 更高。
```

解释：

```text
Translation forecasting 主要依赖 KU temporal organization；
carrier graph structural readout 单独不足以替代 KU full；
但 carrier structural features 在 KU full 之外仍提供额外 global discrimination signal。
```

---

### 8.3 Trial task

#### HistGB

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0598 | 0.6339 | 0.0373 | 0.1082 | 0.185 | 10.97 |
| KU full | 0.0842 | 0.6886 | 0.0418 | 0.0968 | 0.164 | 9.73 |
| KU + structural | **0.0867** | **0.6943** | **0.0459** | **0.1287** | **0.197** | **11.68** |

#### Logistic

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0692 | 0.6550 | 0.0341 | 0.0367 | 0.067 | 3.97 |
| KU full | 0.0717 | 0.6606 | 0.0392 | **0.0896** | **0.149** | **8.84** |
| KU + structural | **0.0848** | **0.6900** | **0.0438** | 0.0843 | 0.140 | 8.30 |

#### Ridge

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.0599 | 0.6342 | 0.0321 | 0.0504 | 0.081 | 4.80 |
| KU full | 0.0626 | 0.6402 | 0.0369 | **0.0953** | **0.164** | **9.73** |
| KU + structural | **0.0758** | **0.6698** | **0.0401** | 0.0945 | 0.157 | 9.31 |

主要观察：

```text
1. Trial task 中 KU full 的 global metrics 明显优于 carrier structural only。
2. Carrier structural only 在 HistGB 的 top-K precision 上已经很强。
3. KU + structural 在 HistGB 上同时获得最佳 global metrics 和最佳 top-K discovery。
```

解释：

```text
Trial emergence 既依赖 KU temporal trajectory，也依赖 carrier graph structural exposure。
KU full 提供整体排序和区分能力；
carrier graph structural features 对 extreme top-K tail 有独立信号；
二者结合效果最好。
```

---

## 9. Test-stable 结果

`test_stable` 排除 right-edge cutoff 2021，只包含：

```text
cutoff_year = 2019, 2020
```

### 9.1 Primary translation task / HistGB

| Representation | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.6191 | 0.0463 | 0.0671 | 0.0429 | 0.087 | 3.21 |
| KU full | 0.6627 | 0.0554 | 0.0916 | 0.0865 | 0.161 | 5.94 |
| KU + structural | **0.6687** | **0.0605** | **0.0949** | **0.0916** | **0.177** | **6.53** |

这是最清晰的主任务结果。

它说明：

```text
在 stable test 上，KU full 明显优于 carrier structural only；
KU + structural 进一步提升所有主要指标。
```

---

### 9.2 Patent task / HistGB

| Representation | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.6478 | 0.0292 | 0.0547 | 0.0645 | 0.122 | 10.55 |
| KU full | 0.6383 | 0.0296 | 0.0516 | 0.0853 | 0.136 | 11.76 |
| KU + structural | **0.6492** | **0.0319** | **0.0552** | **0.0933** | **0.161** | **13.92** |

Patent stable test 中：

```text
carrier structural only 的 AUROC / Spearman 接近或略高于 KU full；
KU full 的 top-K 好于 carrier structural only；
KU + structural 全面最好。
```

---

### 9.3 Trial task / HistGB

| Representation | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| carrier structural only | 0.6376 | 0.0422 | 0.0648 | 0.1130 | 0.185 | 9.84 |
| KU full | 0.6916 | 0.0478 | 0.0902 | 0.1069 | 0.174 | 9.25 |
| KU + structural | **0.6978** | **0.0529** | **0.0931** | **0.1361** | **0.200** | **10.64** |

Trial stable test 中：

```text
KU full 显著提升 global discrimination；
carrier structural only top-K 已经强；
KU + structural 同时取得最佳 global metrics 和 top-K metrics。
```

---

## 10. 主要发现

### 10.1 KU full 在 primary translation 上明显优于 carrier graph structural only

在 translation task 中，KU full 明显超过 carrier structural only。

例如 `test_stable / HistGB`：

```text
carrier structural only:
  AUROC = 0.6191
  AUPRC = 0.0463
  Spearman = 0.0671
  Precision@1000 = 0.087

KU full:
  AUROC = 0.6627
  AUPRC = 0.0554
  Spearman = 0.0916
  Precision@1000 = 0.161
```

这说明：

```text
KU abstraction 比直接 carrier graph topology readout 更好地组织了 future translation forecasting 所需的 temporal evidence。
```

---

### 10.2 KU full 在 trial 的 global metrics 上也明显强于 carrier structural only

例如 `test_all / HistGB`：

```text
carrier structural only:
  AUROC = 0.6339
  AUPRC = 0.0373
  Spearman = 0.0598

KU full:
  AUROC = 0.6886
  AUPRC = 0.0418
  Spearman = 0.0842
```

这说明：

```text
KU temporal trajectory 对 clinical translation forecasting 具有重要预测价值。
```

---

### 10.3 Patent task 对 carrier graph topology 更敏感

Patent task 中，carrier structural only 在部分 global metrics 上接近甚至超过 KU full。

例如 `test_all / logistic`：

```text
carrier structural only:
  AUROC = 0.6440
  AUPRC = 0.0253
  Spearman = 0.0498

KU full:
  AUROC = 0.6104
  AUPRC = 0.0234
  Spearman = 0.0382
```

但 KU full 在 top-K 上更强：

```text
carrier structural only Precision@1000 = 0.088
KU full Precision@1000 = 0.125
```

这说明：

```text
Patent emergence 更依赖 carrier graph structural exposure；
但 KU full 对 top-K candidate selection 仍有价值。
```

---

### 10.4 KU + carrier structural 几乎全面最强

最重要发现是：

```text
KU + carrier structural 在三类任务和多数模型上获得最佳 AUROC / AUPRC / Spearman。
```

例如：

#### Translation / test_stable / HistGB

```text
KU full:
  AUROC = 0.6627
  AUPRC = 0.0554
  Precision@1000 = 0.161

KU + structural:
  AUROC = 0.6687
  AUPRC = 0.0605
  Precision@1000 = 0.177
```

#### Patent / test_all / HistGB

```text
KU full:
  AUROC = 0.6348
  AUPRC = 0.0251
  Precision@1000 = 0.121

KU + structural:
  AUROC = 0.6482
  AUPRC = 0.0276
  Precision@1000 = 0.149
```

#### Trial / test_all / HistGB

```text
KU full:
  AUROC = 0.6886
  AUPRC = 0.0418
  Precision@1000 = 0.164

KU + structural:
  AUROC = 0.6943
  AUPRC = 0.0459
  Precision@1000 = 0.197
```

这说明：

```text
KU abstraction 与 carrier graph structural exposure 是互补的，而不是替代的。
```

---

## 11. 与 Stage 06E 的关系

Stage 06E 和 Stage 06F 共同形成了如下逻辑链：

```text
Stage 06E:
  KU temporal numeric features 是主要预测信号；
  KU schema features 在 temporal evidence features 之外提供增量；
  KU full representation 优于 numeric temporal only。

Stage 06F:
  KU full representation 在 primary translation 和 trial 上优于直接 carrier graph structural readout；
  carrier graph structural readout 在 patent 和部分 top-K 指标上保留独立信号；
  KU full + carrier structural readout 效果最好。
```

因此，两个实验共同支持：

```text
KU construction 是有效的预测抽象；
但原始 carrier graph topology 中仍有可利用的结构信息；
后续模型应同时利用 KU-level temporal abstraction 和 graph-level structural / diffusion signal。
```

---

## 12. 对后续 Knowledge Field model 的含义

Stage 06F 对后续模型设计有直接启发。

结果显示：

```text
1. KU abstraction 不应被简单 carrier graph topology 替代；
2. carrier graph topology 也不应被完全丢弃；
3. 最强表示来自 KU temporal/schema features 与 carrier graph structural exposure 的结合。
```

这正好支持后续 Knowledge Field / reaction-diffusion-source model 的设计方向：

```text
Knowledge Field model 应以 KU abstraction 作为主要预测单元，
同时引入 graph structural exposure / neighborhood diffusion / source terms。
```

也就是说，后续模型不应只做：

```text
KU-only tabular forecasting
```

也不应只做：

```text
raw carrier graph topology readout
```

更合理的方向是：

```text
KU-level temporal dynamics
+
graph-field structural/diffusion information
```

---

## 13. 局限性

### 13.1 不是 GNN baseline

本实验使用的是：

```text
carrier graph structural readout
```

而不是：

```text
GraphSAGE
heterogeneous GNN
temporal GNN
node2vec / DeepWalk
```

因此不能声称：

```text
KU representation outperforms carrier graph GNNs.
```

更准确表述是：

```text
KU representation outperforms a direct cutoff-safe structural readout from the original carrier graph on primary translation and trial forecasting.
```

---

### 13.2 使用 500k capped training

由于：

```text
KU + carrier structural representation
```

在 full-train 设置下超过本地内存限制，本实验采用：

```text
max_train_rows = 500000
```

所有 representation 使用相同 cap，因此比较是 matched 的。

但这仍意味着：

```text
Stage 06F 不是 full-train comparison；
它是 matched capped-training comparison。
```

评估集仍然是完整 validation/test task frames。

---

### 13.3 Carrier structural readout 是 handcrafted topology baseline

Carrier structural features 包括：

```text
degree
weighted degree
neighbor type composition
edge type diversity
```

这些是手工设计的 topology summaries。

它们不能完全代表：

```text
learned graph representation
```

也不能捕捉复杂高阶 graph dynamics。

后续如果需要更强图对照，可以考虑：

```text
carrier graph embedding baseline
heterogeneous GNN
temporal GNN
KU-KU diffusion proxy
```

---

### 13.4 Top-K 指标存在任务和模型差异

虽然 KU + structural 在 global metrics 上几乎全面最好，但 top-K 指标仍存在差异。

例如：

```text
translation / test_all / HistGB:
  KU full Precision@1000 = 0.152
  KU + structural Precision@1000 = 0.127
```

但在 stable test 中：

```text
translation / test_stable / HistGB:
  KU full Precision@1000 = 0.161
  KU + structural Precision@1000 = 0.177
```

因此，top-K tail ordering 需要按任务、模型和 evaluation slice 谨慎解释。

---

## 14. 推荐论文表述

### 中文

```text
我们进一步在同一 KU-level temporal forecasting benchmark 上构造了一个 direct carrier-graph structural readout baseline。该 baseline 对原始 carrier graph 进行 cutoff-safe 结构读出，提取 carrier node degree、weighted degree、neighbor type composition 和 edge-type diversity 等 topology features，并将其聚合到每个 KU × cutoff_year。所有 trainable learners 使用相同的 500k training-row cap，并在完整 validation/test task frames 上评估。

结果显示，KU full representation 在 primary translation 和 trial 任务上明显优于 carrier graph structural only，说明 KU 抽象比简单的原始 carrier graph topology readout 更好地组织了 future translation forecasting 所需的 temporal evidence。与此同时，carrier structural readout 在 patent 任务和 trial 的部分 top-K 指标上表现较强，说明原始 carrier graph topology 中仍包含 KU temporal features 之外的独立结构信号。将 KU full 与 carrier structural features 结合后，三类任务上的 AUROC、AUPRC 和 Spearman 几乎全面达到最佳，并在多个 top-K discovery 指标上进一步提升。这表明 KU abstraction 与 carrier graph structural exposure 是互补关系，而非替代关系。
```

### English

```text
We further constructed a direct carrier-graph structural readout baseline under the same KU-level temporal forecasting benchmark. This baseline extracts cutoff-safe topology features from the original carrier graph, including carrier node degree, weighted degree, neighbor-type composition, and edge-type diversity, and pools them to each KU × cutoff_year example. All trainable learners use a matched 500k-row training cap and are evaluated on the full validation/test task frames.

The results show that the full KU representation outperforms the carrier-graph structural readout on the primary translation task and trial forecasting, indicating that KU abstraction better organizes the temporal evidence needed for future translation prediction than simple carrier-graph topology alone. At the same time, carrier-graph structural readout remains competitive for patent forecasting and some top-K trial metrics, showing that the original carrier graph retains independent structural signal beyond KU temporal features. Combining KU features with carrier-graph structural features yields the strongest overall AUROC, AUPRC, and Spearman across tasks and often improves top-K discovery as well. These results suggest that KU abstraction and carrier-graph structural exposure are complementary rather than substitutable.
```

---

## 15. 结论

Stage 06F 已完成。

完成的 representation：

```text
carrier_graph_structural_only_500k
ku_full_500k
ku_plus_carrier_graph_structural_500k
```

完成的 tasks：

```text
translation
patent
trial
```

完成的 models：

```text
ridge
logistic
logistic_unweighted
histgb
```

运行状态：

```text
max_train_rows = 500000
error_count = 0
```

核心结论：

```text
1. KU full 在 primary translation 和 trial 的整体指标上明显优于 carrier graph structural only；
2. carrier graph structural only 在 patent 和部分 trial top-K 指标上具有独立信号；
3. KU + carrier structural 几乎在三类任务的 AUROC / AUPRC / Spearman 上全面最强；
4. 因此，KU abstraction 与 carrier graph structural exposure 是互补的；
5. 后续 Knowledge Field / diffusion model 应结合 KU-level temporal abstraction 与 graph structural / diffusion information。
```

下一步建议：

```text
将 Stage 06E 和 Stage 06F 结果整合进总实验报告，
然后进入 Knowledge Field / diffusion proxy / reaction-diffusion-source 模型设计。
```