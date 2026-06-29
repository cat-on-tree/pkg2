# Patent-Paper Link Prediction 阶段 2 实验报告：Basic Algorithm Baselines

日期：2026-06-28

## 1. 实验目的

本报告总结 Patent-Paper carrier-level link prediction 的第二阶段实验。

第一阶段已经完成了：

1. Patent-Paper link prediction 数据集构建；
2. full 与 no-species 两个 BioEntity context 版本；
3. target-edge leakage control；
4. BioEntity overlap heuristic baselines；
5. baseline 结果汇总流程。

第二阶段的目标是在第一阶段基础上，进一步测试更标准的机器学习 baseline，判断手工构造的 BioEntity overlap 与 carrier degree/context richness 特征能否支持更强的预测模型。

本阶段重点回答以下问题：

1. 多个 BioEntity overlap 特征组合是否优于单一 heuristic score？
2. Logistic Regression 是否能显著超过 heuristic baseline？
3. tree-based models 是否能进一步利用非线性特征交互？
4. no-species 数据集是否仍然优于 full 数据集？
5. 当前最强的 basic algorithm baseline 是什么？
6. 当前基础特征工程是否足够支撑进入 graph embedding / GNN 阶段？

当前任务仍然是 carrier-level proxy task：

```text
给定一个 Patent 节点和一个 Paper 节点，预测二者之间是否存在 Patent-Paper link。
```

---

## 2. 输入数据

本阶段继续使用第一阶段构建的两个数据集。

### 2.1 Full BioEntity context dataset

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full
```

该版本保留全部 BioEntity context。

### 2.2 No-species BioEntity context dataset

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species
```

该版本移除了 `bioentity_type = species` 的 BioEntity 节点。

两个数据集使用相同的监督学习样本划分：

| Split | Positive edges | Negative edges | Total labeled edges |
| --- | ---: | ---: | ---: |
| Train | 157,826 | 157,826 | 315,652 |
| Validation | 19,728 | 19,728 | 39,456 |
| Test | 19,729 | 19,729 | 39,458 |

负样本比例为：

```text
1:1
```

因此随机预测的期望大致为：

```text
AUROC ≈ 0.50
AUPRC ≈ 0.50
```

---

## 3. Leakage control

当前预测目标边表为：

```text
edges_patent_paper
```

在构建 BioEntity overlap、degree、context richness 等特征时，context graph 已排除该目标边表。

也就是说：

```text
用于构造特征的 context_edges 不包含待预测的 Patent-Paper target edges。
```

因此，本阶段的 feature-based baseline 不直接使用目标边作为上下文证据。

需要注意的是，本阶段的特征仍然可能包含合理的 node popularity / carrier activity 信号，例如：

```text
source_context_degree
target_context_degree
source_bioentity_count
target_bioentity_count
```

这些不是 target-edge leakage，但属于 link prediction 中常见的 degree / popularity signal。

---

## 4. 特征工程

本阶段使用第一阶段已经实现的 BioEntity overlap feature builder。

对于每一个候选 Patent-Paper pair，构造以下基础特征：

| Feature | 含义 |
| --- | --- |
| `shared_bioentity_count` | Patent 与 Paper 共享的 BioEntity 数量 |
| `bioentity_jaccard` | Patent-BioEntity 集合与 Paper-BioEntity 集合的 Jaccard 相似度 |
| `weighted_shared_bioentity_min` | 基于 mention weight 的加权共享 BioEntity overlap |
| `weighted_bioentity_jaccard` | 基于 mention weight 的加权 Jaccard 相似度 |
| `weighted_bioentity_cosine` | Patent 与 Paper 的加权 BioEntity 向量余弦相似度 |
| `source_bioentity_count` | Patent 关联的 BioEntity 数量 |
| `target_bioentity_count` | Paper 关联的 BioEntity 数量 |
| `source_context_degree` | Patent 在 context graph 中的总 degree |
| `target_context_degree` | Paper 在 context graph 中的总 degree |
| `source_target_degree_product` | Patent 与 Paper context degree 的乘积 |

---

## 5. Feature set 设计

为了区分不同类型特征的贡献，本阶段测试了多个 feature set。

### 5.1 Overlap-only feature set

只包含 Patent-Paper pairwise BioEntity overlap / similarity 特征：

```text
shared_bioentity_count
bioentity_jaccard
weighted_shared_bioentity_min
weighted_bioentity_jaccard
weighted_bioentity_cosine
```

该组特征主要用于测试：

```text
纯 BioEntity semantic overlap signal 的上限。
```

---

### 5.2 Degree-only feature set

只包含 carrier activity / context richness 相关特征：

```text
source_bioentity_count
target_bioentity_count
source_context_degree
target_context_degree
source_target_degree_product
```

该组特征主要用于测试：

```text
degree / popularity / carrier activity signal 本身能够解释多少 Patent-Paper link。
```

---

### 5.3 No-product feature set

包含 overlap 特征和单节点 degree/context richness 特征，但不包含乘积项：

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
```

该组特征是当前最重要的 feature set。

它保留了：

```text
pairwise semantic overlap signal
carrier context richness signal
```

同时去掉了可能过度放大 popularity bias 的：

```text
source_target_degree_product
```

---

### 5.4 All feature set

包含全部基础特征：

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

该组特征用于测试加入 degree product 是否能进一步提升性能。

---

## 6. Baseline models

本阶段测试了三类 baseline。

### 6.1 Heuristic baselines

第一阶段已经完成，包括：

```text
shared_bioentity_count
bioentity_jaccard
weighted_shared_bioentity_min
weighted_bioentity_jaccard
weighted_bioentity_cosine
```

其中最强 heuristic baseline 是：

```text
weighted_bioentity_cosine
```

---

### 6.2 Logistic Regression

使用 `scikit-learn` 的 Logistic Regression。

处理流程：

1. 用 DuckDB 构建 BioEntity overlap feature tables；
2. 将 train / validation / test feature tables 读入 pandas DataFrame；
3. 对特征做 median imputation；
4. 对特征做 standardization；
5. 在 train split 上训练 Logistic Regression；
6. 在 train / validation / test 上输出 label=1 的概率作为 score；
7. 复用统一 metrics pipeline 计算 AUROC、AUPRC、Precision@K、Recall@K。

主要脚本：

```text
scripts/14_baseline_logistic_regression.py
```

---

### 6.3 Tree-based models

本阶段测试了以下 tree-based baselines：

```text
RandomForestClassifier
HistGradientBoostingClassifier
XGBoost
LightGBM
```

主要脚本：

```text
scripts/15_baseline_tree_models.py
```

Tree models 不需要标准化特征，只使用 median imputation。

XGBoost、LightGBM、HistGradientBoosting 属于 gradient boosting tree 类模型，适合捕捉非线性特征交互。

---

## 7. 主要结果

### 7.1 当前 overall top results

按 test AUPRC 排序，当前前 10 个结果如下：

| Rank | Dataset | Baseline | Feature set | Test AUROC | Test AUPRC |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/xgboost` | `no_product` | 0.719088 | 0.743793 |
| 2 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/hist_gradient_boosting` | `no_product` | 0.716625 | 0.741950 |
| 3 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/lightgbm` | `no_product` | 0.716008 | 0.741653 |
| 4 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/random_forest` | `no_product` | 0.708044 | 0.736105 |
| 5 | `diabetes_2018_2019_v1_full` | `tree_models_no_product/xgboost` | `no_product` | 0.708314 | 0.729275 |
| 6 | `diabetes_2018_2019_v1_no_species` | `logistic_regression_no_product` | `bioentity_features` | 0.696907 | 0.726131 |
| 7 | `diabetes_2018_2019_v1_no_species` | `logistic_regression_bioentity_features` | `bioentity_features` | 0.696675 | 0.725441 |
| 8 | `diabetes_2018_2019_v1_full` | `tree_models_no_product/lightgbm` | `no_product` | 0.703580 | 0.725235 |
| 9 | `diabetes_2018_2019_v1_full` | `tree_models_no_product/hist_gradient_boosting` | `no_product` | 0.704110 | 0.725097 |
| 10 | `diabetes_2018_2019_v1_full` | `tree_models_no_product/random_forest` | `no_product` | 0.694036 | 0.717128 |

当前最强 baseline 是：

```text
diabetes_2018_2019_v1_no_species
tree_models_no_product/xgboost
```

其 test-set 结果为：

```text
AUROC = 0.719088
AUPRC = 0.743793
```

---

## 8. Baseline 层级比较

本阶段形成了非常清晰的 baseline 梯度。

| Baseline type | Best model / score | Dataset | Test AUROC | Test AUPRC |
| --- | --- | --- | ---: | ---: |
| Heuristic | `weighted_bioentity_cosine` | full / no-species | about 0.602 - 0.604 | about 0.608 |
| Linear ML | `logistic_regression_no_product` | no-species | 0.696907 | 0.726131 |
| Tree ML | `xgboost no_product` | no-species | 0.719088 | 0.743793 |

可以概括为：

```text
Tree-based models > Logistic Regression > single-score heuristic baselines
```

这说明：

1. 单一 BioEntity overlap score 已经包含有效信号；
2. 多个 overlap 与 degree/context richness 特征组合后，性能显著提升；
3. 非线性 tree models 能进一步利用特征之间的交互关系。

---

## 9. Logistic Regression 结果解读

### 9.1 Logistic Regression 明显超过 heuristic baseline

第一阶段最强 heuristic baseline 为：

| Dataset | Score | Test AUROC | Test AUPRC |
| --- | --- | ---: | ---: |
| full | `weighted_bioentity_cosine` | 0.601600 | 0.608473 |
| no-species | `weighted_bioentity_cosine` | 0.603843 | 0.608470 |

Logistic Regression no-product 在 no-species 上达到：

| Dataset | Model | Test AUROC | Test AUPRC |
| --- | --- | ---: | ---: |
| no-species | Logistic Regression no-product | 0.696907 | 0.726131 |

AUPRC 提升约为：

```text
0.726131 - 0.608470 = 0.117661
```

这说明：

```text
多个 BioEntity overlap / weighted similarity / carrier context richness 特征之间存在互补信息。
```

单一 heuristic score 无法充分利用这些互补信号。

---

### 9.2 No-product 略优于 all features

Logistic Regression 中：

```text
no_species logistic_regression_no_product:
AUPRC = 0.726131

no_species logistic_regression_bioentity_features:
AUPRC = 0.725441
```

两者非常接近，但 no-product 略高。

这说明：

```text
source_target_degree_product 不是必要特征。
```

它可能会放大 high-degree Patent 与 high-degree Paper 的组合，从而引入一定 popularity bias。

因此，本阶段后续 tree models 默认采用：

```text
no_product feature set
```

作为主实验设置。

---

### 9.3 Overlap-only 的性能接近 heuristic 上限

Logistic Regression overlap-only 的结果进入了中等 baseline 区间，但没有接近 no-product 模型。

已观察到：

```text
full logistic_regression_overlap_only:
AUPRC = 0.615077

no_species logistic_regression_overlap_only:
AUPRC = 0.609194
```

这与最强 heuristic `weighted_bioentity_cosine` 的 AUPRC 约 0.608 非常接近。

说明：

```text
纯 pairwise BioEntity semantic overlap signal 的上限大约在 AUPRC 0.61 附近。
```

Logistic Regression 的大幅提升主要来自加入：

```text
source_bioentity_count
target_bioentity_count
source_context_degree
target_context_degree
```

这些 carrier context richness / activity 特征。

---

## 10. Tree-based model 结果解读

### 10.1 Tree models 进一步超过 Logistic Regression

当前最强 Logistic Regression：

```text
no_species logistic_regression_no_product
AUROC = 0.696907
AUPRC = 0.726131
```

当前最强 tree model：

```text
no_species tree_models_no_product/xgboost
AUROC = 0.719088
AUPRC = 0.743793
```

提升为：

```text
AUROC improvement = 0.719088 - 0.696907 = 0.022181
AUPRC improvement = 0.743793 - 0.726131 = 0.017662
```

这说明：

```text
BioEntity overlap、weighted similarity、carrier degree/context richness 特征之间存在非线性 interaction。
```

Tree-based models 能够比线性模型更好地捕捉这些 interaction。

---

### 10.2 Gradient boosting models 表现接近

在 no-species + no-product 设置下：

| Model | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| XGBoost | 0.719088 | 0.743793 |
| HistGradientBoosting | 0.716625 | 0.741950 |
| LightGBM | 0.716008 | 0.741653 |
| Random Forest | 0.708044 | 0.736105 |

XGBoost、HistGradientBoosting、LightGBM 的结果非常接近。

这说明：

```text
性能提升主要来自 gradient boosting tree 这一类非线性模型，
而不是某个具体实现的偶然优势。
```

Random Forest 也超过 Logistic Regression，但略低于 gradient boosting models。

---

## 11. Full vs no-species 对比

本阶段最稳定的发现之一是：

```text
no-species 数据集系统性优于 full 数据集。
```

### 11.1 Tree models 中的 full vs no-species

| Model | Full test AUPRC | No-species test AUPRC | Difference |
| --- | ---: | ---: | ---: |
| XGBoost no-product | 0.729275 | 0.743793 | +0.014518 |
| LightGBM no-product | 0.725235 | 0.741653 | +0.016418 |
| HistGradientBoosting no-product | 0.725097 | 0.741950 | +0.016853 |
| Random Forest no-product | 0.717128 | 0.736105 | +0.018977 |

所有 tree models 都显示：

```text
no-species > full
```

### 11.2 Logistic Regression 中的 full vs no-species

| Model | Full test AUPRC | No-species test AUPRC | Difference |
| --- | ---: | ---: | ---: |
| Logistic Regression no-product | 0.695735 | 0.726131 | +0.030396 |
| Logistic Regression all features | 0.695897 | 0.725441 | +0.029544 |

Logistic Regression 中 no-species 的优势更加明显。

---

## 12. Species hub 的解释

no-species 在 heuristic、Logistic Regression 和 tree-based models 中均表现稳定或更好。

这说明 species 类型 BioEntity 更可能表现为：

```text
high-frequency hub
broad biological context
low-specificity signal
potential noise source
```

它们虽然提供了广义生物学背景，但对当前 Patent-Paper link prediction 的任务特异性帮助有限。

去除 species 后，模型可能更依赖更具体的 BioEntity context，例如：

```text
disease
gene
protein
chemical
drug
pathway
phenotype
```

因此，本阶段结果支持将：

```text
diabetes_2018_2019_v1_no_species
```

作为后续 graph embedding / GNN baseline 的主数据集版本之一。

---

## 13. Feature engineering 阶段性判断

本阶段的基础特征工程已经足够支持 basic algorithm baseline。

已完成的 carrier-level basic features 包括：

```text
BioEntity overlap
weighted BioEntity overlap
BioEntity Jaccard
weighted BioEntity Jaccard
weighted BioEntity cosine
source / target BioEntity count
source / target context degree
degree product
```

并完成了以下 feature set 对比：

```text
overlap_only
degree_only
no_product
all features
```

因此，可以认为：

```text
Stage 2 所需的基础特征工程已经完成。
```

但这并不意味着全部特征工程已经结束。

如果后续继续增强传统 ML baseline，还可以进一步加入：

```text
BioEntity type-specific overlap
edge-type-specific degree
metapath count features
Adamic-Adar / resource allocation features
Project / ClinicalTrial proximity features
Paper citation/activity features
Patent assignee/inventor features
time-aware features
```

这些更适合作为后续增强实验，而不是当前 basic baseline 阶段的必要内容。

---

## 14. 当前最强 baseline

截至本阶段结束，当前最强 baseline 为：

```text
Dataset:
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species

Model:
XGBoost

Feature set:
no_product

Output directory:
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/tree_models_no_product/xgboost
```

Test-set performance：

| Metric | Value |
| --- | ---: |
| AUROC | 0.719088 |
| AUPRC | 0.743793 |

这是后续 Node2Vec、KG embedding、GNN、temporal model 需要超过或对比的重要 basic algorithm baseline。

---

## 15. 局限性

本阶段实验仍然有以下限制。

### 15.1 仍然是 carrier-level proxy task

当前预测的是：

```text
Patent-Paper link
```

而不是最终目标中的：

```text
knowledge-unit-level translation
```

因此，当前结果只能说明 carrier graph 中存在可预测结构信号，不能直接等价于知识单元层面的转化预测能力。

---

### 15.2 当前 split 是 random split

当前 train / validation / test 是基于 Patent-Paper pair 的随机划分。

这适合验证 graph signal 和 baseline pipeline，但还不是 temporal prediction setup。

后续需要设计：

```text
temporal split
future link prediction
prospective evaluation
```

才能更接近真实知识转化预测任务。

---

### 15.3 Negative samples 可能包含潜在 missing links

负样本来自未观测到的 Patent-Paper pairs。

其中一部分可能是尚未被记录或未被观测到的潜在真实链接。

因此，当前 negative label 应理解为：

```text
unobserved link
```

而不是绝对意义上的 no link。

---

### 15.4 Degree / popularity signal 需要谨慎解释

Logistic Regression 和 tree models 的强性能部分来自：

```text
source_bioentity_count
target_bioentity_count
source_context_degree
target_context_degree
```

这些特征代表 carrier activity / context richness。

它们是合理的 link prediction signal，但也可能带来 popularity bias。

因此，后续报告和论文中需要区分：

```text
semantic overlap signal
degree / activity signal
graph structural signal
```

---

### 15.5 还未使用真正的图表示学习

本阶段模型只使用手工构造的 tabular features。

它们没有直接学习：

```text
multi-hop paths
heterogeneous relation types
metapath structures
node embeddings
edge-type-aware graph representations
temporal dynamics
```

因此，下一阶段需要进入 graph embedding 和 GNN baselines。

---

## 16. 下一步计划

建议下一阶段进入：

```text
Stage 3: Graph embedding baselines
```

推荐顺序如下：

### 16.1 Node2Vec baseline

先从最简单的 graph embedding baseline 开始：

```text
Node2Vec
```

Node2Vec 可以回答：

```text
仅使用图结构 embedding，是否能超过当前手工特征 tree baseline？
```

需要注意，Node2Vec 通常会把异构图同构化，因此它不能充分利用 edge type 和 node type，但适合作为第一组 graph embedding baseline。

---

### 16.2 Metapath2Vec / heterogeneous random walk

如果 Node2Vec 跑通，可以继续测试：

```text
Metapath2Vec
```

用于考虑异构图中的 node type sequence。

例如：

```text
Patent - BioEntity - Paper
Patent - BioEntity - BioEntity - Paper
Patent - Project - Paper
Patent - ClinicalTrial - Paper
```

---

### 16.3 KG embedding baselines

可以进一步测试：

```text
TransE
RotatE
DistMult
ComplEx
```

这些模型更适合多关系知识图谱 link prediction。

---

### 16.4 GNN baselines

最后进入 GNN：

```text
GraphSAGE
R-GCN
HAN
HGT
```

其中：

```text
GraphSAGE
```

可以作为最简单的 GNN baseline；

```text
R-GCN / HAN / HGT
```

更适合异构图和多关系建模。

---

## 17. 本阶段生成文件

### 17.1 Logistic Regression 脚本

```text
scripts/14_baseline_logistic_regression.py
```

### 17.2 Tree models 脚本

```text
scripts/15_baseline_tree_models.py
```

### 17.3 Baseline 汇总脚本

```text
scripts/13_collect_baseline_results.py
```

### 17.4 主要结果目录

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species
```

### 17.5 汇总结果

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

### 17.6 当前最强 baseline 输出目录

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/tree_models_no_product/xgboost
```

---

## 18. 简短总结

本阶段完成了 Patent-Paper link prediction 的 basic algorithm baseline 测试。

实验结果显示，单一 BioEntity overlap heuristic 虽然已经优于随机预测，但性能上限有限。将 BioEntity overlap、weighted similarity 与 carrier context richness 特征组合后，Logistic Regression 显著提升性能。进一步使用 tree-based models 后，XGBoost、LightGBM 和 HistGradientBoosting 均超过 Logistic Regression，说明当前特征之间存在明显的非线性交互。

当前最强 baseline 是 no-species 数据集上的 XGBoost no-product 特征模型：

```text
Test AUROC = 0.719088
Test AUPRC = 0.743793
```

此外，no-species 数据集在 Logistic Regression 和所有 tree-based models 中均系统性优于 full 数据集，说明 species 类型 BioEntity 更可能作为高频 hub noise，而不是提供任务特异性的预测信号。

本阶段的基础特征工程已经足够支持 basic algorithm baseline。下一阶段建议进入 Node2Vec、Metapath2Vec、KG embedding 和 GNN 等 graph representation baselines。

## 复现命令

### 汇总所有 baseline 结果

Stage 2 及后续 graph embedding baseline 的统一结果汇总使用以下命令：

```bash
python scripts/13_collect_baseline_results.py \
  --results-root data/results/link_prediction/patent_paper \
  --sort-split test \
  --sort-metric auprc
```

该命令会递归扫描：

```text
data/results/link_prediction/patent_paper/
```

下各 baseline 输出目录中的：

```text
metrics_summary.csv
```

并生成统一比较结果。

主要输出文件包括：

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

该汇总命令用于按 test split 的 AUPRC 对所有 baseline 进行排序，便于识别当前 strongest baseline。