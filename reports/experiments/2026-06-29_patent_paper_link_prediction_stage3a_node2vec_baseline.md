# Patent-Paper Link Prediction 阶段 3A 实验报告：Node2Vec / DeepWalk-like Graph Embedding Baseline

日期：2026-06-29

## 1. 实验目的

本报告总结 Patent-Paper carrier-level link prediction 的阶段 3A 实验。

阶段 1 和阶段 2 已经完成：

1. Patent-Paper link prediction 数据集构建；
2. full 与 no-species 两个 BioEntity context 版本；
3. target-edge leakage control；
4. BioEntity overlap heuristic baselines；
5. Logistic Regression baselines；
6. tree-based model baselines；
7. feature set ablation；
8. baseline result collection pipeline。

阶段 2 的最强 baseline 为：

```text
Dataset: diabetes_2018_2019_v1_no_species
Model: XGBoost
Feature set: no_product
Test AUROC = 0.719088
Test AUPRC = 0.743793
```

阶段 3A 的目标是进入 graph representation baselines，首先测试同构化 random-walk embedding：

```text
Node2Vec / DeepWalk-like baseline
```

本阶段重点回答以下问题：

1. 在 leakage-controlled context graph 上训练的 node embeddings 是否包含 Patent-Paper link prediction signal？
2. 纯 embedding similarity decoder 是否能超过 BioEntity overlap heuristic？
3. embedding + supervised decoder 是否能超过 Stage 2 的 tabular feature baselines？
4. no-species 是否仍然优于 full？
5. Node2Vec 是否值得作为后续 MetaPath2Vec / KG embedding / GNN 的参照 baseline？

---

## 2. 输入数据

本阶段继续使用前两个阶段构建的两个数据集。

### 2.1 Full BioEntity context dataset

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_full
```

该版本保留全部 BioEntity context，包括 species 类型 BioEntity。

### 2.2 No-species BioEntity context dataset

```text
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species
```

该版本移除了：

```text
bioentity_type = species
```

的 BioEntity 节点。

两个数据集继续使用相同的监督学习 split：

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

本阶段继续遵守前两个阶段的 leakage control 设定。

目标预测边表为：

```text
edges_patent_paper
```

Node2Vec 训练图来自：

```text
context_edges
```

该 context graph 已经排除了 target Patent-Paper edge table。

也就是说：

```text
Node2Vec embedding training 不使用 edges_patent_paper。
```

监督评估仍然使用 dataset 中已经构建好的：

```text
labeled_edges_train
labeled_edges_val
labeled_edges_test
```

本阶段不重新划分 train / validation / test。

---

## 4. 本阶段新增可复用模块

为了支持 Node2Vec、MetaPath2Vec、KG embedding 和后续 GNN，本阶段先新增了 graph representation baseline 的基础模块。

### 4.1 `src/pkg2/graph_data.py`

负责通用图数据加载和索引构建：

```text
load graph tables
build global node index
build homogeneous edge_index
build heterogeneous edge_index_dict
build KG triples
load labeled train / val / test pairs
add graph node indices to labeled pairs
```

该模块不绑定 PyTorch Geometric 或 DGL。

核心原则是：

```text
从 data/datasets/... 读取已有 dataset；
不重新划分 train / val / test。
```

---

### 4.2 `src/pkg2/embedding_baselines.py`

负责将 node embeddings 转换为 Patent-Paper link prediction scores。

支持两类 decoder。

#### Direct similarity decoders

```text
dot
cosine
negative_l2
negative_l1
```

这些 decoder 不使用监督训练，只根据 Patent embedding 与 Paper embedding 直接计算 score。

#### Supervised Logistic Regression decoders

```text
hadamard_logistic
l1_logistic
l2_logistic
concat_logistic
```

这些 decoder 使用已有的 train split 训练 Logistic Regression，再在 train / validation / test 上评估。

---

### 4.3 `src/pkg2/graph_io.py`

负责统一保存 graph representation baseline artifacts：

```text
embeddings.npy
node_index.csv
relation_index.csv
embedding_metadata.json
predictions_train.parquet
predictions_val.parquet
predictions_test.parquet
baseline_manifest.csv
```

---

## 5. Node2Vec baseline 脚本

本阶段新增脚本：

```text
scripts/16_baseline_node2vec.py
```

该脚本完成以下流程：

1. 从 dataset directory 加载 `nodes`、`context_edges` 和已有 labeled splits；
2. 构建 homogeneous graph；
3. 调用 PyTorch Geometric `Node2Vec` 训练 node embeddings；
4. 使用多个 decoder 对 Patent-Paper labeled pairs 打分；
5. 对每个 decoder 单独输出 predictions、metrics、summary report；
6. 输出目录结构与之前 baseline 保持一致，方便 `scripts/13_collect_baseline_results.py` 汇总。

脚本不会重新划分 train / validation / test。

---

## 6. Node2Vec 设置

本阶段首先测试：

```text
p = 1.0
q = 1.0
```

该设置可以视为 DeepWalk-like random walk baseline。

主要参数为：

| Parameter | Value |
| --- | ---: |
| embedding_dim | 128 |
| walk_length | 20 |
| context_size | 10 |
| walks_per_node | 10 |
| p | 1.0 |
| q | 1.0 |
| epochs | 5 |
| batch_size | 256 |
| device | auto |

训练一次 Node2Vec embedding 后，同时评估以下 decoder：

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
- Logistic Regression decoder 只使用 train split 训练；
- validation 和 test split 只用于评估。

---

## 7. No-species 数据集图统计

No-species 数据集的 Node2Vec 输入图统计如下：

| Item | Count |
| --- | ---: |
| num_nodes | 98,337 |
| num_node_types | 5 |
| num_homogeneous_edges | 833,432 |
| BioEntity nodes | 30,770 |
| ClinicalTrial nodes | 4,180 |
| Paper nodes | 35,717 |
| Patent nodes | 11,921 |
| Project nodes | 15,749 |
| train labeled edges | 315,652 |
| validation labeled edges | 39,456 |
| test labeled edges | 39,458 |

其中 test split 包含：

```text
positive_edges:test = 19,729
negative_edges:test = 19,729
```

---

## 8. No-species Node2Vec 训练结果

No-species 数据集上，Node2Vec 训练 loss 如下：

| Epoch | Loss |
| ---: | ---: |
| 1 | 5.294438 |
| 2 | 1.976501 |
| 3 | 1.246909 |
| 4 | 1.072834 |
| 5 | 1.020479 |

训练过程收敛较快，5 个 epoch 后 loss 已明显下降。

---

## 9. No-species Node2Vec test results

No-species 数据集上，`p=1,q=1` Node2Vec 的 test 结果如下：

| Decoder | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| cosine | 0.656173 | 0.687331 |
| dot | 0.649908 | 0.669619 |
| negative_l2 | 0.582395 | 0.603846 |
| hadamard_logistic | 0.674845 | 0.690585 |
| concat_logistic | 0.778744 | 0.772752 |

最强结果来自：

```text
Node2Vec p=1,q=1 + concat_logistic
```

其 test-set performance 为：

```text
AUROC = 0.778744
AUPRC = 0.772752
```

---

## 10. No-species concat_logistic train / validation / test 对比

`concat_logistic` 的 split-level metrics 如下：

| Split | AUROC | AUPRC | Precision@100 | Recall@100 | Precision@1000 | Recall@1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.783143 | 0.781036 | 0.890 | 0.000564 | 0.922 | 0.005842 |
| validation | 0.777282 | 0.775226 | 0.920 | 0.004663 | 0.938 | 0.047547 |
| test | 0.778744 | 0.772752 | 0.880 | 0.004460 | 0.915 | 0.046378 |

Train、validation、test 的 AUROC / AUPRC 非常接近：

```text
train AUPRC = 0.781036
val AUPRC   = 0.775226
test AUPRC  = 0.772752
```

这说明：

```text
concat_logistic decoder 没有出现明显过拟合。
```

需要注意，Recall@100 和 Recall@1000 的数值较低，主要是因为分母为该 split 中全部正样本数。

以 test split 为例：

```text
positive_count = 19,729
Precision@100 = 0.88
Top 100 中命中正样本数 = 88
Recall@100 = 88 / 19,729 = 0.004460
```

因此，在当前 balanced global ranking setting 下，更适合重点关注：

```text
AUROC
AUPRC
Precision@K
```

而不是直接用 Recall@100 的绝对数值判断模型强弱。

---

## 11. Full dataset Node2Vec test results

Full 数据集上，`p=1,q=1` Node2Vec 的 test 结果如下：

| Decoder | Test AUROC | Test AUPRC |
| --- | ---: | ---: |
| cosine | 0.638522 | 0.667400 |
| dot | 0.632619 | 0.651817 |
| negative_l2 | 0.577495 | 0.593451 |
| hadamard_logistic | 0.658835 | 0.675995 |
| concat_logistic | 0.772017 | 0.765871 |

最强结果同样来自：

```text
Node2Vec p=1,q=1 + concat_logistic
```

其 test-set performance 为：

```text
AUROC = 0.772017
AUPRC = 0.765871
```

---

## 12. Full vs no-species 对比

### 12.1 Decoder-level 对比

| Decoder | No-species AUROC | No-species AUPRC | Full AUROC | Full AUPRC | AUPRC Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| cosine | 0.656173 | 0.687331 | 0.638522 | 0.667400 | +0.019931 |
| dot | 0.649908 | 0.669619 | 0.632619 | 0.651817 | +0.017802 |
| negative_l2 | 0.582395 | 0.603846 | 0.577495 | 0.593451 | +0.010395 |
| hadamard_logistic | 0.674845 | 0.690585 | 0.658835 | 0.675995 | +0.014590 |
| concat_logistic | 0.778744 | 0.772752 | 0.772017 | 0.765871 | +0.006881 |

No-species 在所有 decoder 下均优于 full。

不过，与 Stage 2 tabular baselines 相比，Node2Vec 中 no-species 相对 full 的优势变小。

这说明：

```text
species 类型 BioEntity 对 random-walk graph embedding 仍然有一定噪声影响，
但 Node2Vec 对 species hub noise 的鲁棒性强于手工 overlap / degree 特征模型。
```

---

## 13. 与 Stage 2 最强 baseline 对比

Stage 2 最强 baseline 为：

```text
diabetes_2018_2019_v1_no_species
tree_models_no_product/xgboost
Test AUROC = 0.719088
Test AUPRC = 0.743793
```

Stage 3A 最强 baseline 为：

```text
diabetes_2018_2019_v1_no_species
node2vec_p1_q1/concat_logistic
Test AUROC = 0.778744
Test AUPRC = 0.772752
```

提升为：

```text
AUROC improvement = 0.778744 - 0.719088 = 0.059656
AUPRC improvement = 0.772752 - 0.743793 = 0.028959
```

这说明：

```text
Node2Vec random-walk embeddings 捕捉到了手工 BioEntity overlap / degree / context richness 特征没有完全覆盖的 multi-hop graph structural signal。
```

---

## 14. 当前 overall baseline ranking

截至本阶段结束，统一汇总后的 top results 如下：

| Rank | Dataset | Baseline | Decoder / feature set | Test AUROC | Test AUPRC |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `diabetes_2018_2019_v1_no_species` | `node2vec_p1_q1/concat_logistic` | `concat_logistic` | 0.778744 | 0.772752 |
| 2 | `diabetes_2018_2019_v1_full` | `node2vec_p1_q1/concat_logistic` | `concat_logistic` | 0.772017 | 0.765871 |
| 3 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/xgboost` | `no_product` | 0.719088 | 0.743793 |
| 4 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/hist_gradient_boosting` | `no_product` | 0.716625 | 0.741950 |
| 5 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/lightgbm` | `no_product` | 0.716008 | 0.741653 |
| 6 | `diabetes_2018_2019_v1_no_species` | `tree_models_no_product/random_forest` | `no_product` | 0.708044 | 0.736105 |
| 7 | `diabetes_2018_2019_v1_full` | `tree_models_no_product/xgboost` | `no_product` | 0.708314 | 0.729275 |
| 8 | `diabetes_2018_2019_v1_no_species` | `logistic_regression_no_product` | `bioentity_features` | 0.696907 | 0.726131 |
| 9 | `diabetes_2018_2019_v1_no_species` | `logistic_regression_bioentity_features` | `bioentity_features` | 0.696675 | 0.725441 |
| 10 | `diabetes_2018_2019_v1_full` | `tree_models_no_product/lightgbm` | `no_product` | 0.703580 | 0.725235 |

当前新的 strongest baseline 是：

```text
diabetes_2018_2019_v1_no_species / node2vec_p1_q1/concat_logistic
```

---

## 15. Decoder 结果解读

### 15.1 Pure similarity decoders

No-species 上：

| Decoder | Test AUPRC |
| --- | ---: |
| cosine | 0.687331 |
| dot | 0.669619 |
| negative_l2 | 0.603846 |

这些结果说明：

```text
Node2Vec embedding 空间本身已经包含 Patent-Paper link prediction signal。
```

尤其是 cosine：

```text
AUPRC = 0.687331
```

明显超过 Stage 1 的单一 BioEntity overlap heuristic baseline。

但 pure similarity decoders 仍然低于 Stage 2 的 XGBoost tabular baseline。

---

### 15.2 Hadamard logistic decoder

No-species 上：

```text
hadamard_logistic AUPRC = 0.690585
```

它只比 cosine 略高。

说明单纯逐维交互：

```text
u * v
```

未能充分利用 Node2Vec embedding 中的信息。

---

### 15.3 Concat logistic decoder

No-species 上：

```text
concat_logistic AUPRC = 0.772752
```

它显著超过所有其他 Node2Vec decoders。

`concat_logistic` 使用的 pair feature 为：

```text
[u, v, |u - v|, u * v]
```

这使 Logistic Regression decoder 能同时利用：

```text
Patent node embedding signal
Paper node embedding signal
Patent-Paper embedding distance
Patent-Paper embedding interaction
```

因此，`concat_logistic` 不只是学习二者的相似度，也能学习 node-level activity / position signal。

---

## 16. 解释注意事项

当前实验仍然是 random pair split。

在这种 split 下，同一个 Patent 或 Paper 节点可能同时出现在 train 和 test 的不同 labeled pairs 中。

因此，`concat_logistic` 可能学习到一定的 transductive node-level signal，例如：

```text
某些 Patent embedding 本身更容易与 Paper 连接
某些 Paper embedding 本身更容易被 Patent 连接
```

这不是 target-edge leakage，因为 Node2Vec 训练图没有使用 target Patent-Paper edges。

但它说明当前结果应解释为：

```text
在 random pair split 的 transductive setting 下，Node2Vec + concat Logistic Regression decoder 显著超过 tabular baseline。
```

而不应直接解释为：

```text
Node2Vec 已经证明能预测未来 Patent-Paper links。
```

后续仍需要 temporal split 或 inductive split 来验证 prospective prediction 能力。

---

## 17. 本阶段结论

本阶段完成了 Stage 3A Node2Vec / DeepWalk-like graph embedding baseline。

主要结论如下：

1. `p=1,q=1` 的 Node2Vec，即 DeepWalk-like random-walk embedding，在 full 和 no-species 数据集上都有效；
2. pure similarity decoders 能够超过 Stage 1 heuristic baselines，但不足以超过 Stage 2 XGBoost tabular baseline；
3. `concat_logistic` decoder 显著超过 Stage 2 最强 baseline；
4. no-species 在所有 Node2Vec decoders 下均优于 full；
5. graph embedding 捕捉到了手工 BioEntity overlap / degree 特征未完全覆盖的 multi-hop structural signal；
6. 当前新的 strongest baseline 是 no-species 上的 `Node2Vec p=1,q=1 + concat_logistic`。

当前最佳结果为：

```text
Dataset:
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species

Model:
Node2Vec p=1,q=1

Decoder:
concat_logistic

Test AUROC:
0.778744

Test AUPRC:
0.772752
```

---

## 18. 局限性

### 18.1 同构化处理丢失 node type / edge type 信息

Node2Vec 将异构图同构化处理：

```text
Patent
Paper
BioEntity
ClinicalTrial
Project
```

都被映射到同一 node index space。

同时，edge type 信息也没有显式进入 random walk transition。

因此，Node2Vec 无法直接区分：

```text
Patent - BioEntity
Paper - BioEntity
Paper - Project
ClinicalTrial - BioEntity
```

等不同关系类型。

---

### 18.2 当前仍然是 random pair split

当前 split 是基于 Patent-Paper labeled pairs 的 random split。

该设置适合验证 graph signal 和 baseline pipeline，但还不是 future link prediction setup。

后续需要设计：

```text
temporal split
prospective evaluation
inductive node split
```

来验证模型在真实预测场景中的泛化能力。

---

### 18.3 Concat decoder 可能利用 transductive node-level signal

`concat_logistic` 使用：

```text
[u, v, |u-v|, u*v]
```

其中包含 Patent embedding 和 Paper embedding 本身。

因此，在 random pair split 下，它可能学习：

```text
node identity-like structural position
node activity / popularity
transductive node-level propensity
```

这些信号在当前任务设定下是允许的，但在 temporal 或 inductive setting 下需要重新验证。

---

### 18.4 尚未测试 biased Node2Vec variants

本阶段只测试了：

```text
p=1.0
q=1.0
```

即 DeepWalk-like 设置。

尚未系统测试：

```text
p / q biased random walks
embedding dimension
walk length
context size
walks per node
```

不过本阶段目标是建立 graph embedding baseline，而不是进行大规模 Node2Vec hyperparameter search。

---

## 19. 下一步计划

下一阶段建议进入：

```text
Stage 3B: MetaPath2Vec baseline
```

MetaPath2Vec 的目标是测试：

```text
显式利用 heterogeneous node types / relation types 的 typed random walk 是否优于同构 Node2Vec。
```

推荐下一步实现：

```text
scripts/17_baseline_metapath2vec.py
```

该脚本应支持：

1. 读取相同 dataset directory；
2. 不重新划分 train / validation / test；
3. 构建 heterogeneous edge_index_dict；
4. 输出可用 heterogeneous edge types；
5. 用户指定 metapath；
6. 调用 PyG MetaPath2Vec；
7. 复用 `embedding_baselines.py` 中的 decoders；
8. 输出与 Node2Vec 一致的 metrics 和 artifacts。

首个推荐 metapath 应围绕 BioEntity 桥接路径：

```text
Patent - BioEntity - Paper - BioEntity - Patent
```

该路径直接对应当前任务中的核心语义假设：

```text
Patent 与 Paper 可能因共享 BioEntity context 而存在 link。
```

---

## 20. 本阶段生成文件

### 20.1 新增基础模块

```text
src/pkg2/graph_data.py
src/pkg2/embedding_baselines.py
src/pkg2/graph_io.py
```

### 20.2 新增 Node2Vec baseline 脚本

```text
scripts/16_baseline_node2vec.py
```

### 20.3 No-species Node2Vec 结果目录

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/node2vec_p1_q1
```

主要子目录：

```text
cosine
dot
negative_l2
hadamard_logistic
concat_logistic
```

当前最强结果目录：

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/node2vec_p1_q1/concat_logistic
```

### 20.4 Full Node2Vec 结果目录

```text
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_full/node2vec_p1_q1
```

### 20.5 汇总结果

```text
data/results/link_prediction/patent_paper/baseline_comparison.csv
data/results/link_prediction/patent_paper/baseline_comparison_test.csv
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

---

## 21. 简短总结

本阶段完成了 Node2Vec / DeepWalk-like graph embedding baseline。

结果表明，在 leakage-controlled context graph 上训练的 Node2Vec embeddings 已经包含明显的 Patent-Paper link prediction signal。Pure similarity decoders 能超过 Stage 1 heuristic baselines，但不能超过 Stage 2 tree-based tabular baselines。

当使用 `concat_logistic` decoder 后，Node2Vec 显著超过此前最强的 XGBoost no-product baseline，成为当前新的 strongest baseline：

```text
diabetes_2018_2019_v1_no_species / node2vec_p1_q1 / concat_logistic
Test AUROC = 0.778744
Test AUPRC = 0.772752
```

No-species 数据集在所有 Node2Vec decoders 下均优于 full 数据集，说明 species 类型 BioEntity 对 random-walk embedding 仍有一定噪声影响，但 Node2Vec 相比手工 overlap features 更鲁棒。

下一步建议进入 MetaPath2Vec baseline，用 typed random walks 测试

## 复现命令

### No-species Node2Vec p=1,q=1 最佳 baseline

当前 Stage 3A 的最佳 Node2Vec baseline 来自 no-species dataset：

```text
Dataset:
data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species

Output:
data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/node2vec_p1_q1

Best decoder:
concat_logistic
```

推荐复现命令如下：

```bash
python scripts/16_baseline_node2vec.py \
  --dataset-dir data/datasets/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species \
  --output-dir data/results/link_prediction/patent_paper/diabetes_2018_2019_v1_no_species/node2vec_p1_q1 \
  --embedding-dim 128 \
  --walk-length 20 \
  --context-size 10 \
  --walks-per-node 10 \
  --p 1.0 \
  --q 1.0 \
  --epochs 5 \
  --batch-size 256 \
  --scoring-methods cosine dot negative_l2 hadamard_logistic concat_logistic \
  --device auto \
  --seed 42 \
  --overwrite
```

该命令训练：

```text
Node2Vec p=1,q=1
```

即 DeepWalk-like random-walk embedding baseline。

该实验中最强 decoder 为：

```text
concat_logistic
```

对应 test-set 结果为：

```text
Test AUROC = 0.778744
Test AUPRC = 0.772752
```

注意：该命令不重新划分 train / validation / test，而是读取 dataset directory 中已经固定保存的 labeled splits。