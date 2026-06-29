# Patent-Paper Link Prediction 阶段 3B 实验报告：MetaPath2Vec Heterogeneous Graph Embedding Baseline

日期：2026-06-29

## 1. 实验目的

本报告总结 Patent-Paper carrier-level link prediction 的阶段 3B 实验。

前序阶段已经完成：

1. Patent-Paper link prediction 数据集构建；
2. full 与 no-species 两个 BioEntity context 版本；
3. target-edge leakage control；
4. heuristic baselines；
5. Logistic Regression baselines；
6. tree-based model baselines；
7. Node2Vec / DeepWalk-like graph embedding baseline。

Stage 3A 中，Node2Vec 已成为当前 strongest baseline：

```text
Dataset: diabetes_2018_2019_v1_no_species
Model: Node2Vec p=1,q=1
Decoder: concat_logistic
Test AUROC = 0.778744
Test AUPRC = 0.772752
```

Stage 3B 的目标是测试 heterogeneous random-walk embedding baseline：

```text
MetaPath2Vec
```

本阶段重点回答以下问题：

1. 显式使用 node type / edge type 的 typed random walk 是否有助于 Patent-Paper link prediction？
2. BioEntity bridge metapath 是否能够捕捉 Patent-Paper 之间的 translation signal？
3. MetaPath2Vec 是否能够超过或接近 Stage 2 的 tabular XGBoost baseline？
4. MetaPath2Vec 是否能够超过 Stage 3A 的 unrestricted homogeneous Node2Vec？
5. no-species 是否仍然优于 full？

---

## 2. 输入数据

本阶段继续使用之前构建的两个数据集。

### 2.1 Full BioEntity context dataset

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full
```

该版本保留所有 BioEntity，包括：

```text
bioentity_type = species
```

### 2.2 No-species BioEntity context dataset

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species
```

该版本移除了：

```text
bioentity_type = species
```

的 BioEntity 节点。

两个数据集继续使用相同的 labeled split：

| Split | Positive edges | Negative edges | Total labeled edges |
| --- | ---: | ---: | ---: |
| Train | 157,826 | 157,826 | 315,652 |
| Validation | 19,728 | 19,728 | 39,456 |
| Test | 19,729 | 19,729 | 39,458 |

负样本比例为：

```text
1:1
```

因此随机预测的期望大约为：

```text
AUROC ≈ 0.50
AUPRC ≈ 0.50
```

---

## 3. Leakage control

本阶段继续遵守前序实验的 leakage control 设定。

目标预测边表为：

```text
edges_patent_paper
```

MetaPath2Vec 的训练图来自：

```text
context_edges
```

该 context graph 已排除 target Patent-Paper edge table。

因此：

```text
MetaPath2Vec embedding training 不使用 target Patent-Paper edges。
```

监督评估使用已有的 labeled splits：

```text
labeled_edges_train
labeled_edges_val
labeled_edges_test
```

本阶段不重新划分 train / validation / test。

---

## 4. 本阶段新增脚本

本阶段新增脚本：

```text
scripts/17_baseline_metapath2vec.py
```

该脚本完成以下流程：

1. 从 dataset directory 加载 `nodes`、`context_edges` 和 labeled splits；
2. 构建 heterogeneous `edge_index_dict`；
3. 列出可用 heterogeneous edge types；
4. 自动解析或手动指定 metapath；
5. 调用 PyTorch Geometric `MetaPath2Vec` 训练 node embeddings；
6. 将 type-specific embeddings 合并回 global embedding matrix；
7. 使用多个 decoder 对 Patent-Paper labeled pairs 打分；
8. 对每个 decoder 单独输出 predictions、metrics、summary report；
9. 输出结构与 Node2Vec baseline 保持一致，方便统一汇总。

脚本复用了以下通用模块：

```text
src/pkg2/graph_data.py
src/pkg2/embedding_baselines.py
src/pkg2/graph_io.py
src/pkg2/metrics.py
src/pkg2/io.py
```

---

## 5. 可用 heterogeneous edge types

在 no-species dataset 上，dry run 得到的 heterogeneous edge types 如下：

| Edge type | Edge count |
| --- | ---: |
| BioEntity → Paper | 310,949 |
| BioEntity → Patent | 40,326 |
| BioEntity → ClinicalTrial | 24,958 |
| ClinicalTrial → Paper | 9,512 |
| ClinicalTrial → Project | 350 |
| ClinicalTrial → BioEntity | 24,958 |
| Paper → Project | 28,094 |
| Paper → ClinicalTrial | 9,512 |
| Paper → BioEntity | 310,949 |
| Patent → Project | 2,527 |
| Patent → BioEntity | 40,326 |
| Project → Paper | 28,094 |
| Project → Patent | 2,527 |
| Project → ClinicalTrial | 350 |

具体 edge type names 为：

```text
BioEntity|paper_mentions_bioentity__rev|Paper
BioEntity|patent_mentions_bioentity__rev|Patent
BioEntity|trial_mentions_bioentity__rev|ClinicalTrial
ClinicalTrial|paper_linked_trial__rev|Paper
ClinicalTrial|trial_linked_project|Project
ClinicalTrial|trial_mentions_bioentity|BioEntity
Paper|paper_linked_project|Project
Paper|paper_linked_trial|ClinicalTrial
Paper|paper_mentions_bioentity|BioEntity
Patent|patent_linked_project|Project
Patent|patent_mentions_bioentity|BioEntity
Project|paper_linked_project__rev|Paper
Project|patent_linked_project__rev|Patent
Project|trial_linked_project__rev|ClinicalTrial
```

---

## 6. MetaPath2Vec metapath 设计

本阶段选择的核心 metapath 为 BioEntity bridge path：

```text
Patent -> BioEntity -> Paper -> BioEntity -> Patent
```

对应具体 edge sequence：

```text
Patent|patent_mentions_bioentity|BioEntity
BioEntity|paper_mentions_bioentity__rev|Paper
Paper|paper_mentions_bioentity|BioEntity
BioEntity|patent_mentions_bioentity__rev|Patent
```

该 metapath 的语义为：

```text
Patent
  mentions BioEntity
BioEntity
  is mentioned by Paper
Paper
  mentions BioEntity
BioEntity
  is mentioned by Patent
Patent
```

也就是说，它显式建模：

```text
Patent 和 Paper 通过共享 BioEntity context 产生潜在关联。
```

---

## 7. 与 Node2Vec 的区别

Stage 3A 的 Node2Vec 将异构 graph 同构化：

```text
Patent
Paper
BioEntity
ClinicalTrial
Project
```

都被视为同一类 node，edge type 被忽略。

而 Stage 3B 的 MetaPath2Vec 显式限制 random walk 只能沿着指定 typed path 行走：

```text
Patent - BioEntity - Paper - BioEntity - Patent
```

因此两者测试的是不同假设：

| Model | Graph usage | Hypothesis |
| --- | --- | --- |
| Node2Vec | Unrestricted homogeneous random walk | 更宽的 multi-hop graph context 有助于预测 |
| MetaPath2Vec | Typed metapath-constrained random walk | BioEntity-mediated typed path 是核心 translation signal |

---

## 8. No-species 输入图统计

No-species 数据集的 MetaPath2Vec 输入图统计如下：

| Item | Count |
| --- | ---: |
| num_nodes | 98,337 |
| num_node_types | 5 |
| num_heterogeneous_edge_types | 14 |
| num_typed_edges | 833,432 |
| BioEntity nodes | 30,770 |
| ClinicalTrial nodes | 4,180 |
| Paper nodes | 35,717 |
| Patent nodes | 11,921 |
| Project nodes | 15,749 |
| train labeled edges | 315,652 |
| validation labeled edges | 39,456 |
| test labeled edges | 39,458 |

---

## 9. 初始 5-epoch 训练结果与欠拟合诊断

首先使用与 Node2Vec 相同的 epoch 数：

```text
epochs = 5
```

训练 loss 为：

| Epoch | Loss |
| ---: | ---: |
| 1 | 7.108703 |
| 2 | 5.821978 |
| 3 | 4.896211 |
| 4 | 4.143783 |
| 5 | 3.536478 |

5 epoch 后，loss 仍在快速下降，尚未收敛。

对应 no-species test 结果如下：

| Decoder | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| cosine | 0.504189 | 0.500476 |
| dot | 0.503832 | 0.498674 |
| negative_l2 | 0.518660 | 0.515328 |
| hadamard_logistic | 0.500411 | 0.497075 |
| concat_logistic | 0.616539 | 0.608386 |

该结果说明：

```text
5 epoch 下 MetaPath2Vec 明显欠拟合。
```

原因是 MetaPath2Vec 每个 epoch 的 batch 数远少于 Node2Vec。

Node2Vec 每个 epoch 约：

```text
385 batches
```

MetaPath2Vec 当前 metapath 从 Patent 开始，因此每个 epoch 约：

```text
11921 Patent nodes / batch_size 256 ≈ 47 batches
```

因此 5 epoch 的 optimizer steps 为：

```text
47 * 5 = 235 steps
```

而 Node2Vec 5 epoch 的 optimizer steps 为：

```text
385 * 5 = 1925 steps
```

二者相差约：

```text
1925 / 235 ≈ 8.2x
```

因此，为了与 Node2Vec 的训练步数更接近，本阶段将 MetaPath2Vec 训练扩展到：

```text
epochs = 40
```

---

## 10. MetaPath2Vec e40 训练设置

最终采用的主要参数如下：

| Parameter | Value |
| --- | ---: |
| embedding_dim | 128 |
| walk_length | 20 |
| context_size | 10 |
| walks_per_node | 10 |
| epochs | 40 |
| batch_size | 256 |
| num_negative_samples | 1 |
| device | auto |

训练一次 MetaPath2Vec embedding 后，评估以下 decoder：

```text
cosine
dot
negative_l2
hadamard_logistic
concat_logistic
```

其中：

- `cosine`、`dot`、`negative_l2` 是 unsupervised similarity / distance decoder；
- `hadamard_logistic` 和 `concat_logistic` 是 supervised Logistic Regression decoder；
- supervised decoders 只使用 train split 训练；
- validation 和 test split 只用于评估。

---

## 11. No-species e40 training result

No-species 上，40 epoch 训练最终 loss 为：

```text
Epoch 040/40: loss = 0.843630
```

相较于 5 epoch 时的：

```text
loss = 3.536478
```

下降显著，说明 e40 训练更加充分。

---

## 12. No-species e40 test results

No-species dataset 上，MetaPath2Vec BioEntity bridge e40 的 test 结果如下：

| Decoder | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| cosine | 0.606048 | 0.643368 |
| dot | 0.600212 | 0.625365 |
| negative_l2 | 0.541317 | 0.554750 |
| hadamard_logistic | 0.641345 | 0.652222 |
| concat_logistic | 0.741189 | 0.741545 |

最强结果来自：

```text
MetaPath2Vec BioEntity bridge e40 + concat_logistic
```

其 test-set performance 为：

```text
AUROC = 0.741189
AUPRC = 0.741545
```

相较于 5 epoch 的 concat_logistic：

```text
AUPRC = 0.608386
```

提升为：

```text
0.741545 - 0.608386 = 0.133159
```

这确认了初始结果较弱的主要原因是训练不足。

---

## 13. No-species split-level precision

No-species 的 concat_logistic 在 train / validation / test 上的 Precision@K 表现较稳定。

观察到的 precision 约为：

| Split | Precision |
| --- | ---: |
| train | 0.99 |
| validation | 0.96 |
| test | 0.97 |

这说明：

```text
top-ranked predictions 的质量较高；
train / validation / test 差距不大；
没有明显 train 高、test 崩的过拟合迹象。
```

需要注意，若该 precision 指的是 Precision@K，则 Recall@K 的绝对值可能仍然较低，因为每个 split 中正样本总数较大，而 K 相对较小。

因此本任务更适合同时关注：

```text
AUROC
AUPRC
Precision@K
```

---

## 14. Full dataset e40 test results

Full dataset 上，MetaPath2Vec BioEntity bridge e40 的 test 结果如下：

| Decoder | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| cosine | 0.600113 | 0.632727 |
| dot | 0.594995 | 0.617739 |
| negative_l2 | 0.544621 | 0.550227 |
| hadamard_logistic | 0.636495 | 0.649835 |
| concat_logistic | 0.734642 | 0.729932 |

最强结果同样来自：

```text
MetaPath2Vec BioEntity bridge e40 + concat_logistic
```

其 test-set performance 为：

```text
AUROC = 0.734642
AUPRC = 0.729932
```

---

## 15. Full vs no-species 对比

### 15.1 Decoder-level comparison

| Decoder | No-species AUROC | No-species AUPRC | Full AUROC | Full AUPRC | AUPRC Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| cosine | 0.606048 | 0.643368 | 0.600113 | 0.632727 | +0.010641 |
| dot | 0.600212 | 0.625365 | 0.594995 | 0.617739 | +0.007626 |
| negative_l2 | 0.541317 | 0.554750 | 0.544621 | 0.550227 | +0.004523 |
| hadamard_logistic | 0.641345 | 0.652222 | 0.636495 | 0.649835 | +0.002387 |
| concat_logistic | 0.741189 | 0.741545 | 0.734642 | 0.729932 | +0.011613 |

No-species 在所有 AUPRC 指标下均优于 full。

最强 decoder 下的差异为：

```text
AUPRC difference = 0.741545 - 0.729932 = 0.011613
```

这与前序实验结论一致：

```text
species BioEntity 节点作为高频、泛化、hub-like context，会对 Patent-Paper link prediction 引入噪声。
```

---

## 16. 与 Stage 2 XGBoost baseline 对比

### 16.1 No-species

Stage 2 no-species 最强 tabular baseline：

```text
XGBoost no-product
Test AUROC = 0.719088
Test AUPRC = 0.743793
```

Stage 3B no-species MetaPath2Vec 最强结果：

```text
MetaPath2Vec BioEntity bridge e40 + concat_logistic
Test AUROC = 0.741189
Test AUPRC = 0.741545
```

对比：

| Method | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| XGBoost no-product | 0.719088 | 0.743793 |
| MetaPath2Vec BioEntity bridge + concat_logistic | 0.741189 | 0.741545 |

差异：

```text
AUROC difference = +0.022101
AUPRC difference = -0.002248
```

说明 MetaPath2Vec 的 AUPRC 几乎追平 XGBoost，AUROC 则明显更高。

---

### 16.2 Full

Stage 2 full 最强 tabular baseline：

```text
XGBoost no-product
Test AUROC = 0.708314
Test AUPRC = 0.729275
```

Stage 3B full MetaPath2Vec 最强结果：

```text
MetaPath2Vec BioEntity bridge e40 + concat_logistic
Test AUROC = 0.734642
Test AUPRC = 0.729932
```

对比：

| Method | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| XGBoost no-product | 0.708314 | 0.729275 |
| MetaPath2Vec BioEntity bridge + concat_logistic | 0.734642 | 0.729932 |

差异：

```text
AUROC difference = +0.026328
AUPRC difference = +0.000657
```

说明 full dataset 上 MetaPath2Vec 与 XGBoost 基本持平，并略高于 XGBoost 的 AUPRC。

---

## 17. 与 Stage 3A Node2Vec baseline 对比

Stage 3A 最强 Node2Vec baseline：

```text
Node2Vec p=1,q=1 + concat_logistic
```

对比如下：

| Dataset | Method | Test AUROC | Test AUPRC |
| --- | --- | ---: | ---: |
| no-species | Node2Vec p1q1 + concat_logistic | 0.778744 | 0.772752 |
| no-species | MetaPath2Vec BioEntity bridge + concat_logistic | 0.741189 | 0.741545 |
| full | Node2Vec p1q1 + concat_logistic | 0.772017 | 0.765871 |
| full | MetaPath2Vec BioEntity bridge + concat_logistic | 0.734642 | 0.729932 |

Node2Vec 在 full 和 no-species 上均明显优于 MetaPath2Vec。

No-species 上差异为：

```text
AUROC difference = 0.778744 - 0.741189 = 0.037555
AUPRC difference = 0.772752 - 0.741545 = 0.031207
```

Full 上差异为：

```text
AUROC difference = 0.772017 - 0.734642 = 0.037375
AUPRC difference = 0.765871 - 0.729932 = 0.035939
```

这说明：

```text
BioEntity bridge 是强信号，但 unrestricted homogeneous random walk 能利用更宽的 graph context，因此表现更强。
```

可能提供额外信息的 context 包括：

```text
Patent - Project
Paper - Project
Paper - ClinicalTrial
ClinicalTrial - Project
ClinicalTrial - BioEntity
```

---

## 18. Decoder 结果解读

### 18.1 Pure similarity decoders

No-species e40 上：

| Decoder | Test AUPRC |
| --- | ---: |
| cosine | 0.643368 |
| dot | 0.625365 |
| negative_l2 | 0.554750 |

Full e40 上：

| Decoder | Test AUPRC |
| --- | ---: |
| cosine | 0.632727 |
| dot | 0.617739 |
| negative_l2 | 0.550227 |

训练充分后，pure similarity decoders 明显高于随机预测，说明 MetaPath2Vec embedding 确实学习到了 typed structural signal。

不过，它们仍然明显低于 concat_logistic。

这说明：

```text
Patent embedding 和 Paper embedding 在 heterogeneous embedding space 中不一定适合直接用 cosine/dot 比较。
```

---

### 18.2 Hadamard logistic decoder

No-species e40 上：

```text
hadamard_logistic AUPRC = 0.652222
```

Full e40 上：

```text
hadamard_logistic AUPRC = 0.649835
```

Hadamard logistic 明显优于 pure dot / negative_l2，但仍低于 concat_logistic。

这说明单纯逐维交互：

```text
u * v
```

不能充分利用 embedding 中的 source-side / target-side signal。

---

### 18.3 Concat logistic decoder

No-species e40 上：

```text
concat_logistic AUPRC = 0.741545
```

Full e40 上：

```text
concat_logistic AUPRC = 0.729932
```

该 decoder 使用：

```text
[u, v, |u - v|, u * v]
```

因此它同时利用：

```text
Patent embedding signal
Paper embedding signal
Patent-Paper embedding distance
Patent-Paper embedding interaction
```

这与 Node2Vec 阶段的结论一致：

```text
concat_logistic 是当前 embedding baseline 中最有效的 decoder。
```

---

## 19. 当前 overall baseline 位置

截至 Stage 3B，主要 baseline 位置如下：

| Family | Best method | Dataset | Test AUROC | Test AUPRC |
| --- | --- | --- | ---: | ---: |
| Tree/tabular | XGBoost no-product | no-species | 0.719088 | 0.743793 |
| Node2Vec | p1q1 + concat_logistic | no-species | 0.778744 | 0.772752 |
| MetaPath2Vec | BioEntity bridge e40 + concat_logistic | no-species | 0.741189 | 0.741545 |
| Node2Vec | p1q1 + concat_logistic | full | 0.772017 | 0.765871 |
| MetaPath2Vec | BioEntity bridge e40 + concat_logistic | full | 0.734642 | 0.729932 |

当前 strongest baseline 仍然是：

```text
diabetes_2018_2019_v1_no_species / node2vec_p1_q1 / concat_logistic
Test AUROC = 0.778744
Test AUPRC = 0.772752
```

MetaPath2Vec 的主要价值在于：

```text
它证明了 BioEntity-mediated typed path 本身就是强 signal；
但单一 metapath 的信息覆盖不如 unrestricted multi-relation graph context。
```

---

## 20. 解释注意事项

### 20.1 当前仍然是 random pair split

当前 carrier-level experiments 仍然基于 random Patent-Paper pair split。

因此，同一个 Patent 或 Paper 可能出现在 train 和 test 的不同 pairs 中。

这意味着 supervised decoder，尤其是：

```text
concat_logistic
```

可能利用 transductive node-level structural position signal。

这不是 target-edge leakage，因为 embedding training graph 没有使用 target Patent-Paper edges。

但结果应解释为：

```text
random pair split 下的 transductive carrier-level link prediction baseline。
```

而不应直接解释为：

```text
prospective future Patent-Paper link prediction。
```

未来需要 temporal split 或 inductive split 进一步验证。

---

### 20.2 Metapath 中未出现的 node types 没有直接 embedding

当前 BioEntity bridge metapath 只包含：

```text
Patent
BioEntity
Paper
```

因此：

```text
ClinicalTrial
Project
```

未直接出现在 metapath 中。

脚本中对未出现在 metapath 的 node types 使用 zero-filled embeddings。

这不会影响当前 Patent-Paper 评估，因为 labeled pairs 只需要：

```text
Patent embedding
Paper embedding
```

但这也说明当前 MetaPath2Vec baseline 没有直接利用 Project 和 ClinicalTrial context。

---

### 20.3 单一 metapath 可能过窄

当前仅测试了：

```text
Patent - BioEntity - Paper - BioEntity - Patent
```

该 metapath 语义清楚，但信息面有限。

未测试的可能路径包括：

```text
Patent - Project - Paper - Project - Patent
Paper - ClinicalTrial - BioEntity - Patent
Patent - Project - ClinicalTrial - Paper
```

不过，本阶段目标是建立一个代表性 typed random-walk baseline，而不是大规模搜索 metapath。

---

## 21. 本阶段结论

本阶段完成了 Stage 3B MetaPath2Vec heterogeneous graph embedding baseline。

主要结论如下：

1. 5 epoch 的 MetaPath2Vec 明显欠拟合，因为每个 epoch 只有约 47 batches；
2. 训练到 40 epochs 后，MetaPath2Vec loss 从 3.536478 进一步下降到 0.843630；
3. 训练充分后，BioEntity bridge MetaPath2Vec 的性能显著提升；
4. no-species 上，MetaPath2Vec + concat_logistic 达到：

```text
Test AUROC = 0.741189
Test AUPRC = 0.741545
```

5. full 上，MetaPath2Vec + concat_logistic 达到：

```text
Test AUROC = 0.734642
Test AUPRC = 0.729932
```

6. no-species 在所有 AUPRC 指标下均优于 full；
7. MetaPath2Vec 的 AUPRC 接近 Stage 2 最强 XGBoost baseline；
8. MetaPath2Vec 仍明显低于 Stage 3A Node2Vec baseline；
9. 结果说明 BioEntity-mediated typed path 是强 signal，但 broader multi-relation context 仍提供额外预测信息；
10. 当前阶段结果足够稳定，可以进入下一个 baseline。

---

## 22. 局限性

### 22.1 单一 BioEntity bridge metapath

本阶段只使用一个主要 metapath：

```text
Patent - BioEntity - Paper - BioEntity - Patent
```

未系统搜索其他 metapath。

因此，本阶段结论应理解为：

```text
BioEntity bridge MetaPath2Vec baseline
```

而不是所有 possible MetaPath2Vec variants 的最优结果。

---

### 22.2 未直接利用 Project / ClinicalTrial context

当前 metapath 不包含：

```text
Project
ClinicalTrial
```

因此不能充分利用：

```text
Patent - Project
Paper - Project
Paper - ClinicalTrial
ClinicalTrial - BioEntity
```

等额外 context。

这可能是 MetaPath2Vec 弱于 Node2Vec 的重要原因。

---

### 22.3 Random split 的 transductive 性质

同 Node2Vec 阶段一样，当前评估仍基于 random pair split。

后续需要在 temporal prediction setting 下重新评估。

---

### 22.4 尚未测试 paper-start metapath

当前 metapath 从 Patent 开始：

```text
Patent -> BioEntity -> Paper -> BioEntity -> Patent
```

尚未测试反向起点：

```text
Paper -> BioEntity -> Patent -> BioEntity -> Paper
```

该 variant 可作为后续 sensitivity check，但不是当前主线必需。

---

## 23. 下一步计划

建议进入：

```text
Stage 3C: KG embedding baseline
```

推荐实现：

```text
scripts/18_baseline_kg_embedding.py
```

首轮建议测试：

```text
DistMult
TransE
```

其中 KG embedding 训练应只使用：

```text
context_edges triples
```

不使用 target Patent-Paper edges，避免 leakage。

KG embedding 训练完成后，继续复用：

```text
src/pkg2/embedding_baselines.py
```

中的 decoder：

```text
cosine
dot
negative_l2
hadamard_logistic
concat_logistic
```

用于 Patent-Paper link prediction。

随后可进入：

```text
Stage 3D: GraphSAGE baseline
```

建议实现一个简洁的 homogeneous GraphSAGE supervised edge prediction baseline。

---

## 24. 本阶段生成文件

### 24.1 新增脚本

```text
scripts/17_baseline_metapath2vec.py
```

### 24.2 No-species MetaPath2Vec 结果目录

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/metapath2vec_bioentity_bridge_e40
```

主要子目录：

```text
cosine
dot
negative_l2
hadamard_logistic
concat_logistic
```

当前 no-species 最强结果目录：

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/metapath2vec_bioentity_bridge_e40/concat_logistic
```

### 24.3 Full MetaPath2Vec 结果目录

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/metapath2vec_bioentity_bridge_e40
```

当前 full 最强结果目录：

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/metapath2vec_bioentity_bridge_e40/concat_logistic
```

### 24.4 主要 summary files

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/metapath2vec_bioentity_bridge_e40/metapath2vec_scoring_summary.csv
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/metapath2vec_bioentity_bridge_e40/metapath2vec_summary_report.md

data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/metapath2vec_bioentity_bridge_e40/metapath2vec_scoring_summary.csv
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/metapath2vec_bioentity_bridge_e40/metapath2vec_summary_report.md
```

---

## 25. 简短总结

本阶段完成了 MetaPath2Vec BioEntity bridge baseline。

初始 5 epoch 结果较弱，主要原因是每个 epoch 只有约 47 batches，训练步数远少于 Node2Vec。将训练扩展到 40 epochs 后，模型 loss 下降到 0.843630，性能显著提升。

最终 no-species 上：

```text
MetaPath2Vec BioEntity bridge e40 + concat_logistic
Test AUROC = 0.741189
Test AUPRC = 0.741545
```

Full 上：

```text
MetaPath2Vec BioEntity bridge e40 + concat_logistic
Test AUROC = 0.734642
Test AUPRC = 0.729932
```

MetaPath2Vec 接近 Stage 2 XGBoost baseline，但仍低于 Stage 3A Node2Vec baseline。这说明 BioEntity-mediated typed path 是强预测信号，但 broader multi-relation context 对 Patent-Paper link prediction 仍有额外贡献。

## 复现命令

### Dry run：检查 heterogeneous edge types 和自动 metapath 解析

在正式训练前，可以先运行 dry run 检查可用 edge types 和自动解析出的 BioEntity bridge metapath：

```bash
python scripts/17_baseline_metapath2vec.py \
  --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
  --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/metapath2vec_probe \
  --list-edge-types \
  --dry-run \
  --overwrite
```

该命令确认自动解析出的 metapath 为：

```text
Patent|patent_mentions_bioentity|BioEntity
BioEntity|paper_mentions_bioentity__rev|Paper
Paper|paper_mentions_bioentity|BioEntity
BioEntity|patent_mentions_bioentity__rev|Patent
```

也就是：

```text
Patent -> BioEntity -> Paper -> BioEntity -> Patent
```

---

### No-species MetaPath2Vec BioEntity bridge e40

No-species dataset 的正式 Stage 3B 运行命令为：

```bash
python scripts/17_baseline_metapath2vec.py \
  --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
  --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/metapath2vec_bioentity_bridge_e40 \
  --auto-metapath bioentity_bridge \
  --embedding-dim 128 \
  --walk-length 20 \
  --context-size 10 \
  --walks-per-node 10 \
  --epochs 40 \
  --batch-size 256 \
  --scoring-methods cosine dot negative_l2 hadamard_logistic concat_logistic \
  --device auto \
  --seed 42 \
  --overwrite
```

该命令训练：

```text
MetaPath2Vec BioEntity bridge
epochs = 40
```

对应最强 decoder 为：

```text
concat_logistic
```

Test-set 结果为：

```text
Test AUROC = 0.741189
Test AUPRC = 0.741545
```

---

### Full MetaPath2Vec BioEntity bridge e40

Full dataset 的正式 Stage 3B 运行命令为：

```bash
python scripts/17_baseline_metapath2vec.py \
  --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full \
  --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/metapath2vec_bioentity_bridge_e40 \
  --auto-metapath bioentity_bridge \
  --embedding-dim 128 \
  --walk-length 20 \
  --context-size 10 \
  --walks-per-node 10 \
  --epochs 40 \
  --batch-size 256 \
  --scoring-methods cosine dot negative_l2 hadamard_logistic concat_logistic \
  --device auto \
  --seed 42 \
  --overwrite
```

该命令对应最强 decoder 为：

```text
concat_logistic
```

Test-set 结果为：

```text
Test AUROC = 0.734642
Test AUPRC = 0.729932
```

---

### 结果汇总命令

MetaPath2Vec 结果完成后，可以继续使用统一 baseline 汇总命令：

```bash
python scripts/13_collect_baseline_results.py \
  --results-root data/results/link_prediction/patent_paper \
  --sort-split test \
  --sort-metric auprc
```

该命令用于将 Stage 2、Stage 3A、Stage 3B 及后续 baseline 的 `metrics_summary.csv` 汇总到统一 ranking 表中。
