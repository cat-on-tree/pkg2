# Patent-Paper Link Prediction 阶段 1 实验报告

日期：2026-06-28

## 1. 实验目的

本报告总结基于 PKG2 diabetes Knowledge Carrier Graph 原型图完成的第一阶段 Patent-Paper link prediction 实验。

本阶段实验的目的并不是直接解决最终的 knowledge-unit-level translation prediction 任务，而是验证以下问题：

1. 当前构建的 carrier graph 是否包含可用于预测的图信号；
2. link prediction 数据集构建流程是否能够稳定运行；
3. target-edge leakage control 是否有效；
4. 负采样、train/validation/test 划分、baseline 评估、结果汇总流程是否能够形成完整闭环；
5. BioEntity context 是否能够解释一部分 Patent-Paper link。

当前预测任务为：

```text
给定一个 Patent 节点和一个 Paper 节点，预测二者之间是否存在 Patent-Paper link。
```

这是一个 carrier-level proxy task。  
其中 Paper、Patent、ClinicalTrial、Project 被视为 knowledge carrier，BioEntity 被视为生物医学上下文节点。

---

## 2. 数据与图来源

本阶段实验基于以下 diabetes-centered aggregated carrier graph：

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated
```

图中包含的主要节点类型包括：

```text
Patent
Paper
ClinicalTrial
Project
BioEntity
```

当前任务的目标边表为：

```text
edges_patent_paper
```

为了避免 target-edge leakage，在构建 baseline 特征时，context graph 会排除该目标边表。

---

## 3. 数据集构建

本阶段构建了两个 Patent-Paper link prediction 数据集。

---

### 3.1 Full BioEntity context 数据集

数据集路径：

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full
```

该版本保留全部 BioEntity context。

核心统计如下：

| 指标 | 数值 |
| --- | ---: |
| Source nodes | 101,871 |
| Filtered nodes | 99,383 |
| Removed nodes | 2,488 |
| Source edges | 656,939 |
| Filtered edges | 656,939 |
| Removed edges | 0 |
| Context edges | 459,627 |
| 去重后 target positive edges | 197,283 |

原始 `edges_patent_paper` 表中包含：

```text
197,312
```

条边。  
按 Patent-Paper pair 去重后，剩余：

```text
197,283
```

条唯一正样本。  
因此当前数据中存在：

```text
29
```

条重复 Patent-Paper pair，在划分 train/validation/test 之前被去除。

正样本与负样本划分如下：

| Split | Positive edges | Negative edges | Total labeled edges |
| --- | ---: | ---: | ---: |
| Train | 157,826 | 157,826 | 315,652 |
| Validation | 19,728 | 19,728 | 39,456 |
| Test | 19,729 | 19,729 | 39,458 |

负采样比例为：

```text
1:1
```

---

### 3.2 No-species BioEntity context 数据集

数据集路径：

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species
```

该版本移除了 `bioentity_type = species` 的 BioEntity 节点。

核心统计如下：

| 指标 | 数值 |
| --- | ---: |
| Source nodes | 101,871 |
| Filtered nodes | 98,337 |
| Removed nodes | 3,534 |
| Source edges | 656,939 |
| Filtered edges | 614,028 |
| Removed edges | 42,911 |

与 full 版本相比，no-species 版本额外移除了：

```text
1,046
```

个 species 类型 BioEntity 节点，以及：

```text
42,911
```

条与 species BioEntity 相关的边。

该版本的正负样本数量与 full 版本保持一致：

| Split | Positive edges | Negative edges | Total labeled edges |
| --- | ---: | ---: | ---: |
| Train | 157,826 | 157,826 | 315,652 |
| Validation | 19,728 | 19,728 | 39,456 |
| Test | 19,729 | 19,729 | 39,458 |

这说明 no-species 版本只改变 context graph，不改变监督学习目标样本。

---

## 4. Target-edge leakage control

当前预测目标边表是：

```text
edges_patent_paper
```

在 full dataset 中：

```text
total edges = 656,939
target Patent-Paper edges = 197,312
context edges = 656,939 - 197,312 = 459,627
```

实际输出中的 `context_edges` 数量为：

```text
459,627
```

因此可以确认：

```text
context graph 已排除 edges_patent_paper。
```

这保证了在计算 BioEntity overlap、degree 等 baseline 特征时，不会直接使用待预测的 Patent-Paper target edge。

no-species 版本同样遵循该 leakage-control 原则。

---

## 5. Baseline 设置

本阶段测试的第一组 baseline 是 BioEntity overlap heuristic baselines。

对于每一个候选 Patent-Paper pair，baseline 会比较：

```text
Patent 关联的 BioEntity 集合
Paper 关联的 BioEntity 集合
```

并根据二者的重叠或加权相似度计算预测分数。

本阶段测试了以下 score function：

| Score column | 含义 |
| --- | --- |
| `shared_bioentity_count` | Patent 和 Paper 共享的 BioEntity 数量 |
| `bioentity_jaccard` | Patent-BioEntity 集合与 Paper-BioEntity 集合的 Jaccard 相似度 |
| `weighted_shared_bioentity_min` | 基于 mention weight 的加权共享 BioEntity overlap |
| `weighted_bioentity_jaccard` | 基于 mention weight 的加权 Jaccard 相似度 |
| `weighted_bioentity_cosine` | Patent 与 Paper 的加权 BioEntity 向量余弦相似度 |

Baseline 输出目录位于：

```text
data/results/link_prediction/patent_paper
```

汇总结果文件位于：

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

---

## 6. 实验结果

---

### 6.1 Shared BioEntity count baseline

最简单的 baseline 使用 Patent 和 Paper 共享的 BioEntity 数量作为预测分数。

结果如下：

| Dataset | Score | Test AUROC | Test AUPRC | Test P@100 |
| --- | --- | ---: | ---: | ---: |
| `diabetes_2018_2019_v1_full` | `shared_bioentity_count` | 0.597568 | 0.583886 | 0.970000 |
| `diabetes_2018_2019_v1_no_species` | `shared_bioentity_count` | 0.601840 | 0.590658 | 0.970000 |

由于当前数据集采用 1:1 正负样本比例，随机预测的期望结果大致为：

```text
AUROC ≈ 0.50
AUPRC ≈ 0.50
```

因此，简单的 shared BioEntity count baseline 已经明显优于随机预测。  
这说明 BioEntity context 对 Patent-Paper link prediction 具有可用的预测信号。

---

### 6.2 BioEntity overlap baselines 汇总结果

按 test AUPRC 排序，当前前 10 个结果如下：

| Rank | Dataset | Score column | Test AUROC | Test AUPRC |
| ---: | --- | --- | ---: | ---: |
| 1 | `diabetes_2018_2019_v1_full` | `weighted_bioentity_cosine` | 0.601600 | 0.608473 |
| 2 | `diabetes_2018_2019_v1_no_species` | `weighted_bioentity_cosine` | 0.603843 | 0.608470 |
| 3 | `diabetes_2018_2019_v1_no_species` | `weighted_bioentity_jaccard` | 0.603211 | 0.603249 |
| 4 | `diabetes_2018_2019_v1_full` | `weighted_bioentity_jaccard` | 0.600558 | 0.602308 |
| 5 | `diabetes_2018_2019_v1_no_species` | `bioentity_jaccard` | 0.602590 | 0.599788 |
| 6 | `diabetes_2018_2019_v1_no_species` | `weighted_shared_bioentity_min` | 0.602678 | 0.596956 |
| 7 | `diabetes_2018_2019_v1_full` | `bioentity_jaccard` | 0.598374 | 0.594721 |
| 8 | `diabetes_2018_2019_v1_no_species` | `shared_bioentity_count` | 0.601840 | 0.590658 |
| 9 | `diabetes_2018_2019_v1_full` | `weighted_shared_bioentity_min` | 0.598721 | 0.589920 |
| 10 | `diabetes_2018_2019_v1_full` | `shared_bioentity_count` | 0.597568 | 0.583886 |

当前最强的 heuristic baseline 是：

```text
weighted_bioentity_cosine
```

其 test AUPRC 约为：

```text
0.608
```

test AUROC 约为：

```text
0.602 - 0.604
```

---

## 7. 结果解释

---

### 7.1 BioEntity context 含有有效预测信号

所有 BioEntity overlap baseline 都明显优于随机预测期望。

这说明 Patent 和 Paper 之间共享的生物医学实体上下文确实能够部分解释 Patent-Paper link。

因此，本阶段结果支持以下判断：

```text
当前构建的 carrier graph 中包含可用于 link prediction 的有效图信号。
```

---

### 7.2 加权相似度优于原始重叠计数

当前最强分数不是：

```text
shared_bioentity_count
```

而是：

```text
weighted_bioentity_cosine
```

这说明：

```text
Patent 和 Paper 是否链接，不仅取决于二者共享多少 BioEntity，
也取决于这些 BioEntity 在二者上下文中的权重分布是否相似。
```

换言之，BioEntity mention weight 的分布信息比简单的共享数量更有预测价值。

---

### 7.3 Species 节点不是关键有效信号

移除 species 类型 BioEntity 后，baseline 性能没有下降，部分指标反而略有提升。

例如：

| Score | Full test AUPRC | No-species test AUPRC |
| --- | ---: | ---: |
| `shared_bioentity_count` | 0.583886 | 0.590658 |
| `weighted_bioentity_cosine` | 0.608473 | 0.608470 |
| `weighted_bioentity_jaccard` | 0.602308 | 0.603249 |

这说明 species 类型 BioEntity 很可能更接近：

```text
高频 hub / 泛化上下文 / 弱区分度节点
```

它们虽然提供了广义生物学背景，但对当前 Patent-Paper link prediction 的任务特异性贡献有限。

---

### 7.4 Cosine normalization 能削弱 hub effect

`weighted_bioentity_cosine` 在 full 和 no-species 数据集上的表现几乎一致：

```text
full test AUPRC       = 0.608473
no-species test AUPRC = 0.608470
```

这说明 weighted cosine 对高频 hub 节点更稳健。

可能原因是 cosine similarity 对向量长度进行了归一化，从而削弱了高连接度 BioEntity 对分数的直接放大作用。

---

## 8. 阶段性结论

本阶段实验可以得出以下结论：

1. Patent-Paper carrier-level link prediction 数据集构建流程已经跑通；
2. `v1_full` 和 `v1_no_species` 两个数据集均构建成功；
3. `edges_patent_paper` 已从 context graph 中排除，target-edge leakage control 生效；
4. 原始 `edges_patent_paper` 中存在 29 条重复 Patent-Paper pair，已在构建正样本时去重；
5. BioEntity overlap baseline 明显优于随机预测，说明 BioEntity context 含有有效预测信号；
6. `weighted_bioentity_cosine` 是当前最强的 heuristic baseline；
7. 移除 species 节点不会削弱模型效果，说明 species 很可能不是任务特异性的关键预测信号；
8. 当前结果支持继续进入 feature-based machine learning baseline 阶段。

---

## 9. 当前局限性

本阶段实验仍然有若干限制。

第一，当前任务仍然是 carrier-level proxy task。  
它预测的是 Patent-Paper link，而不是最终目标中的 knowledge-unit-level translation。

第二，当前划分方式是基于 Patent-Paper pair 的随机划分。  
它还不是 temporal prediction setup。

第三，负样本来自未观测到的 Patent-Paper pair。  
其中部分 pair 可能是潜在但缺失的真实链接，因此负样本可能存在噪声。

第四，BioEntity overlap baseline 只使用局部上下文特征。  
它没有使用更长路径、多关系结构、图表示学习或 temporal dynamics。

第五，当前 baseline 尚未区分 Patent-Paper link 背后的不同机制，例如：

```text
直接引用
共享 funding / project context
共享 clinical evidence
共享 BioEntity context
技术转化关系
```

---

## 10. 下一步计划

下一阶段建议进入 feature-based machine learning baselines。

推荐新增脚本：

```text
scripts/14_baseline_logistic_regression.py
scripts/15_baseline_tree_models.py
```

Logistic regression baseline 可以使用以下特征：

```text
shared_bioentity_count
bioentity_jaccard
weighted_shared_bioentity_min
weighted_bioentity_jaccard
weighted_bioentity_cosine
source_bioentity_count
target_bioentity_count
source_context_degree
target_context_degree
source_target_degree_product
```

在 logistic regression 之后，可以继续测试：

```text
Random forest
XGBoost / LightGBM
```

再之后进入 graph representation baseline：

```text
Node2Vec
Metapath2Vec
TransE
RotatE
GraphSAGE
R-GCN
HAN
HGT
```

最终阶段应转向 knowledge-unit-level temporal translation modeling，包括：

```text
Knowledge Unit extraction
Knowledge State construction
Temporal evidence accumulation
Knowledge Field dynamics
Reaction-diffusion-source modeling
```

---

## 11. 本阶段生成文件

### 11.1 数据集

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species
```

### 11.2 Baseline 结果

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species
```

### 11.3 汇总比较结果

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

### 11.4 建议人工报告保存位置

```text
reports/experiments/2026-06-28_patent_paper_link_prediction_stage1.md
```

---

## 12. 简短总结

本阶段 Patent-Paper link prediction 实验成功验证了 carrier graph 与 baseline pipeline 的可用性。简单的 BioEntity overlap 已经优于随机预测，而 `weighted_bioentity_cosine` 取得了当前最好的 heuristic baseline 表现，test AUPRC 约为 0.608。移除 species 节点没有导致性能下降，说明 broad species hub 并不是当前任务的关键有效信号。下一步应进入 logistic regression 和 tree-based models 等 feature-based machine learning baseline 阶段。