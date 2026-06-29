# Patent-Paper Link Prediction Stage 3 实验总结报告

日期：2026-06-29  
任务：Carrier-level Patent-Paper link prediction  
阶段：Stage 3 graph / KG / GNN baselines  
数据集：

- `diabetes_2018_2019_v1_full`
- `diabetes_2018_2019_v1_no_species`

---

## 1. 实验目的

Stage 3 的目标是系统评估 carrier context graph 是否包含可用于 Patent-Paper link prediction 的结构信号，并比较不同图表示学习方法对该信号的利用能力。

本阶段不是最终主模型贡献，而是为后续 knowledge-unit temporal translation modeling 提供以下支撑：

1. 验证构建的 carrier graph 中确实存在可预测的 translational signal；
2. 比较 homogeneous random-walk、metapath、KG embedding 和 supervised GNN 等不同图建模范式；
3. 检验 full graph 与 no-species graph 的差异，评估 species-level BioEntity hubs 的影响；
4. 为后续 knowledge unit construction 和 temporal dynamics modeling 选择合理的数据基础与参考 baseline。

---

## 2. 数据集与任务设置

### 2.1 Link prediction 任务

任务为 carrier-level Patent-Paper link prediction：

```text
给定 Patent 与 Paper pair，预测二者之间是否存在目标 Patent-Paper link。
```

数据集包含固定划分：

```text
labeled_edges_train
labeled_edges_val
labeled_edges_test
```

所有 Stage 3 实验均复用同一划分，不重新 split。

### 2.2 Full 与 no-species 数据集

本阶段主要比较两个 graph variants：

| Dataset | 说明 |
|---|---|
| `diabetes_2018_2019_v1_full` | 保留全部 context graph，包括所有 BioEntity 类型 |
| `diabetes_2018_2019_v1_no_species` | 移除 species / organism-level BioEntity nodes，用于评估高频 species hubs 的影响 |

no-species variant 的依据是：species BioEntities 通常表示广泛实验背景或研究对象，例如 human、mouse、rat 等。这些节点在 biomedical documents 中高频出现，容易形成 high-degree hubs，增加图连通性但降低任务相关语义特异性。

因此，no-species graph 被定位为：

```text
一个 biologically motivated noise-control / sensitivity-analysis variant
```

而不是最终最优数据集。

### 2.3 泄漏控制

所有图表示学习实验均遵守以下原则：

```text
context graph / message passing graph 不包含目标 Patent-Paper edges。
```

也就是说，Patent-Paper target links 只用于：

```text
1. downstream decoder training；
2. supervised GNN edge prediction loss；
3. validation / test evaluation。
```

它们不会作为 context graph edges 出现在 embedding training 或 GNN message passing 中。

---

## 3. Stage 3 baseline families

本阶段覆盖以下 baseline families：

| Family | Representative methods | 目的 |
|---|---|---|
| Homogeneous graph embedding | Node2Vec / DeepWalk-like | 检验同构化 carrier graph 的 random-walk proximity 是否有效 |
| Metapath embedding | MetaPath2Vec | 检验固定异构路径模式是否有效 |
| KG embedding | DistMult / TransE | 检验 relation-aware context triples 是否优于 random-walk / metapath embedding |
| Supervised GNN | GraphSAGE / R-GCN | 检验 supervised message passing 是否进一步提升 Patent-Paper link prediction |

---

## 4. Stage 3A：Node2Vec / DeepWalk-like baselines

### 4.1 方法说明

Node2Vec / DeepWalk-like baseline 将异构 carrier context graph 同构化，然后通过 random walks 学习节点 embedding。

训练完成后，对 Patent-Paper pair 使用 embedding decoder 进行预测，例如：

```text
cosine
dot
negative_l2
hadamard_logistic
concat_logistic
```

前序实验中，最强 Node2Vec 配置为：

```text
node2vec_p1_q1 + concat_logistic
```

### 4.2 关键发现

Node2Vec 结果显示：

1. carrier graph 的纯结构信息具有明显预测力；
2. supervised pair decoder 明显优于 raw similarity decoder；
3. no-species graph 通常优于 full graph，说明 species hubs 对 random-walk methods 有一定噪声影响。

代表结果：

| Dataset | Run | Decoder | Test AUROC | Test AUPRC |
|---|---|---|---:|---:|
| no-species | node2vec_p1_q1 | concat_logistic | 0.778744 | 0.772752 |
| full | node2vec_p1_q1 | concat_logistic | 0.772017 | 0.765871 |

---

## 5. Stage 3B：MetaPath2Vec baselines

### 5.1 方法说明

MetaPath2Vec 使用异构图中的固定 metapath 进行 random walk，从而保留部分 node type / path type 信息。

本阶段使用的核心 metapath 思路包括：

```text
Patent - BioEntity - Paper - BioEntity - Patent
```

或其他围绕 Patent、Paper、BioEntity 的 carrier-level metapath。

### 5.2 关键发现

MetaPath2Vec 证明 fixed metapath representation 在该任务中有一定效果，但整体未超过 Node2Vec best 和 KG embedding best。

可能原因包括：

1. 固定 metapath 难以覆盖 project、trial、BioEntity 等多种 context relations；
2. metapath 设计本身会限制可利用的图结构；
3. Patent-Paper translation signal 可能来自多种 heterogeneous context signals，而非单一 metapath。

代表结果：

| Dataset | Run | Decoder | Test AUROC | Test AUPRC |
|---|---|---|---:|---:|
| no-species | metapath2vec best | concat_logistic | 约 0.741 | 约 0.742 |

---

## 6. Stage 3C-1：KG embedding baselines

### 6.1 方法说明

KG embedding baseline 将 carrier context graph 转化为多关系 triples：

```text
(head, relation, tail)
```

训练时只使用 context relations，不使用目标 Patent-Paper edges。

本阶段实现并评估：

```text
DistMult
TransE
```

训练完成后，使用与前序 embedding baseline 一致的 downstream decoders：

```text
cosine
dot
negative_l2
hadamard_logistic
concat_logistic
```

### 6.2 DistMult 结果

DistMult 是最强 KG embedding baseline。

| Dataset | Model | Decoder | Test AUROC | Test AUPRC |
|---|---|---|---:|---:|
| no-species | DistMult | concat_logistic | 0.819824 | **0.826381** |
| full | DistMult | concat_logistic | 0.818941 | 0.822387 |
| no-species | DistMult | hadamard_logistic | 0.754067 | 0.762838 |
| full | DistMult | hadamard_logistic | 0.751490 | 0.754887 |

DistMult no-species 相比 Node2Vec no-species best：

```text
AUROC: 0.819824 - 0.778744 = +0.041080
AUPRC: 0.826381 - 0.772752 = +0.053629
```

这说明 relation-aware KG embedding 能够更好地利用 heterogeneous carrier context graph 中的 relation type 信息。

### 6.3 TransE 结果

TransE no-species 结果如下：

| Dataset | Model | Decoder | Test AUROC | Test AUPRC |
|---|---|---|---:|---:|
| no-species | TransE | concat_logistic | 0.790784 | 0.775526 |

TransE 明显弱于 DistMult：

```text
DistMult no-species AUPRC = 0.826381
TransE no-species AUPRC   = 0.775526
差值                       = 0.050855
```

这说明当前 carrier graph 中，bilinear relation compatibility 可能比 translation-distance geometry 更适合建模 heterogeneous biomedical carrier relations。

### 6.4 KG embedding 关键发现

KG embedding 阶段得到以下结论：

1. DistMult 显著优于 Node2Vec / MetaPath2Vec；
2. TransE 也具有一定效果，但明显低于 DistMult；
3. raw similarity decoders 仍然较弱，concat_logistic 是最强 downstream decoder；
4. no-species 略优于 full，但差距不大；
5. relation-aware modeling 能部分缓解 species hubs 的噪声影响。

---

## 7. Stage 3C-2：Supervised GNN baselines

### 7.1 方法说明

在 KG embedding 之后，本阶段进一步补充 supervised message-passing GNN baselines：

```text
GraphSAGE
R-GCN
```

与 DistMult / TransE 不同，GraphSAGE / R-GCN 是 supervised edge prediction baselines：

```text
encoder: 在 context_edges 上 message passing
decoder: 使用 labeled_edges_train 训练 Patent-Paper edge predictor
validation: labeled_edges_val
test: labeled_edges_test
```

因此，GraphSAGE / R-GCN 与 KG embedding 的训练范式不同：

| Method family | Training signal | 是否直接使用 Patent-Paper train labels 训练 encoder |
|---|---|---|
| DistMult / TransE | context triples | 否 |
| GraphSAGE / R-GCN | context graph + Patent-Paper train labels | 是 |

因此，Supervised GNN 应被理解为：

```text
strong supervised GNN reference point
```

而不是纯无监督 embedding baseline。

### 7.2 GNN 模型设置

节点特征：

```text
trainable node embeddings + node type embeddings
```

Decoder：

```text
concat MLP decoder
score(patent, paper) = MLP([h_patent, h_paper])
```

主要超参数：

| 参数 | 值 |
|---|---:|
| hidden_dim | 128 |
| num_layers | 2 |
| learning_rate | 0.001 |
| weight_decay | 0.00001 |
| dropout | 0.2 |
| epochs | 50 |
| early_stopping_patience | 10 |
| seed | 42 |
| R-GCN num_bases | 8 |

checkpoint 根据 validation AUPRC 选择。

### 7.3 GraphSAGE 与 R-GCN 结果

| Model | Dataset | Best epoch | Val AUPRC | Test AUROC | Test AUPRC |
|---|---|---:|---:|---:|---:|
| GraphSAGE | no-species | 22 | 0.885794 | 0.880608 | **0.888661** |
| GraphSAGE | full | 23 | 0.884919 | **0.881377** | 0.887874 |
| R-GCN | no-species | 19 | 0.881654 | 0.880312 | 0.885240 |

最佳结果来自：

```text
GraphSAGE no-species e50
Test AUROC = 0.880608
Test AUPRC = 0.888661
```

GraphSAGE full 几乎相同：

```text
GraphSAGE full e50
Test AUROC = 0.881377
Test AUPRC = 0.887874
```

R-GCN no-species 表现接近但未超过 GraphSAGE：

```text
R-GCN no-species e50
Test AUROC = 0.880312
Test AUPRC = 0.885240
```

### 7.4 GNN 关键发现

#### 7.4.1 Supervised GNN 是目前最强 baseline

GraphSAGE no-species 相比 DistMult no-species：

```text
GraphSAGE no-species AUPRC = 0.888661
DistMult no-species AUPRC  = 0.826381
提升                         = +0.062280
```

这说明 supervised message passing 能够更有效地利用 carrier context graph 和 Patent-Paper train labels。

#### 7.4.2 R-GCN 未明显超过 GraphSAGE

no-species 上：

```text
GraphSAGE AUPRC = 0.888661
R-GCN AUPRC     = 0.885240
差值             = +0.003421
```

这说明在当前轻量设置下，显式 relation-aware message passing 没有带来明显增益。

可能原因包括：

1. GraphSAGE 通过 node type embeddings 和 supervised decoder 已经能捕捉大部分任务相关信号；
2. Patent-Paper train labels 的监督信号较强；
3. 当前 R-GCN 仅作为合理 baseline，并未进行复杂调参。

因此不能简单解释为 relation information 不重要，而应表述为：

```text
在当前 supervised GNN 设置下，R-GCN 未明显优于 GraphSAGE。
```

#### 7.4.3 full 与 no-species 差距极小

GraphSAGE：

```text
no-species AUPRC = 0.888661
full AUPRC       = 0.887874
差值              = +0.000787
```

AUROC 则 full 略高：

```text
full AUROC       = 0.881377
no-species AUROC = 0.880608
差值              = +0.000769
```

这表明 supervised GraphSAGE 对 species-level hubs 较为鲁棒。

---

## 8. 统一结果汇总

运行统一汇总脚本：

```bash
python scripts/13_collect_baseline_results.py \
  --results-root data/results/link_prediction/patent_paper \
  --sort-split test \
  --sort-metric auprc
```

汇总输出：

```text
Metric rows collected: 231
Ranking split: test
Ranking metric: auprc

Comparison CSV:
data/results/link_prediction/patent_paper/baseline_comparison.csv

Test CSV:
data/results/link_prediction/patent_paper/baseline_comparison_test.csv

Report:
data/results/link_prediction/patent_paper/baseline_comparison_report.md
```

Top runs：

| Rank | Dataset | Run | Decoder | Test AUROC | Test AUPRC |
|---:|---|---|---|---:|---:|
| 1 | no-species | gnn_graphsage_e50 | - | 0.880608 | **0.888661** |
| 2 | full | gnn_graphsage_e50 | - | **0.881377** | 0.887874 |
| 3 | no-species | gnn_rgcn_e50 | - | 0.880312 | 0.885240 |
| 4 | no-species | kg_distmult_e50 | concat_logistic | 0.819824 | 0.826381 |
| 5 | full | kg_distmult_e50 | concat_logistic | 0.818941 | 0.822387 |
| 6 | no-species | kg_transe_e50 | concat_logistic | 0.790784 | 0.775526 |
| 7 | no-species | node2vec_p1_q1 | concat_logistic | 0.778744 | 0.772752 |
| 8 | full | node2vec_p1_q1 | concat_logistic | 0.772017 | 0.765871 |
| 9 | no-species | kg_distmult_e50 | hadamard_logistic | 0.754067 | 0.762838 |
| 10 | full | kg_distmult_e50 | hadamard_logistic | 0.751490 | 0.754887 |

---

## 9. Stage 3 总体发现

### 9.1 Carrier context graph 包含强预测信号

从 heuristic / tabular ML 到 graph embedding，再到 supervised GNN，多个方法均显著高于随机预测，说明 carrier graph 中确实存在可预测的 Patent-Paper translation-related signal。

### 9.2 Relation-aware KG embedding 明显优于 random-walk / metapath embedding

DistMult 显著优于 Node2Vec 和 MetaPath2Vec，说明 context graph 中的 relation type 对该任务很重要。

### 9.3 Supervised GNN 是最强 carrier-level baseline

GraphSAGE 取得最高 Test AUPRC：

```text
0.888661
```

这说明当模型能够通过 Patent-Paper train labels 进行 supervised message passing 时，可以更充分地利用 context graph 结构。

### 9.4 R-GCN 与 GraphSAGE 接近，但没有明显提升

R-GCN no-species 接近 GraphSAGE，但略低。这表明当前任务中，普通 GraphSAGE + node type embeddings + supervised decoder 已经非常强。

### 9.5 full 与 no-species 差异随模型增强而减小

在 random-walk embedding 中，no-species 更明显有帮助；在 DistMult 和 GraphSAGE 中，full 与 no-species 差距很小。

这说明：

```text
更强的 relation-aware 或 supervised learning 能力可以缓解 species hubs 的负面影响。
```

---

## 10. 为什么不继续扩展更多 baseline？

本阶段已经覆盖了：

```text
Node2Vec
MetaPath2Vec
DistMult
TransE
GraphSAGE
R-GCN
```

因此不再继续扩展以下模型：

```text
GAT
HAN
HGT
RotatE
ComplEx
R-GCN full
更多 graph filtering variants
```

原因是：

1. 当前任务主要用于验证 carrier graph 和提供 baseline，不是最终主贡献；
2. GraphSAGE 已经取得很强结果，继续调参或扩展模型边际收益有限；
3. HAN / HGT 等复杂 heterogeneous GNN 会显著增加工程和调参成本；
4. 更精细的 graph filtering 会增加设计自由度，可能偏离主线；
5. 后续研究重点应转向 knowledge-unit-level temporal translation modeling。

---

## 11. 当前推荐的结果表述

### 中文表述

```text
Stage 3 系统比较了 Node2Vec、MetaPath2Vec、KG embedding 和 supervised GNN 等 carrier-level Patent-Paper link prediction baselines。结果显示，relation-aware KG embedding 明显优于 random-walk 和 fixed-metapath embedding，其中 DistMult no-species + concat_logistic 达到 test AUPRC = 0.826381。进一步地，监督式 message-passing GNN 取得最强结果，GraphSAGE no-species 达到 test AUROC = 0.880608，test AUPRC = 0.888661。GraphSAGE full 结果几乎相同，说明监督式 GNN 对 species-level hubs 较为鲁棒。R-GCN 表现接近但未超过 GraphSAGE。整体结果证明，leakage-controlled carrier context graph 中包含强 translational prediction signal，并为后续 knowledge-unit temporal dynamics modeling 提供了可靠的数据和方法基础。
```

### 英文表述

```text
Stage 3 systematically evaluated carrier-level Patent-Paper link prediction baselines, including Node2Vec, MetaPath2Vec, KG embeddings, and supervised GNNs. Relation-aware KG embedding substantially outperformed random-walk and fixed-metapath embeddings, with DistMult on the no-species graph achieving test AUPRC = 0.826381 using a concat logistic decoder. Supervised message-passing GNNs provided the strongest performance: GraphSAGE on the no-species graph achieved test AUROC = 0.880608 and test AUPRC = 0.888661, while the full graph produced a nearly identical result. R-GCN performed comparably but did not improve over GraphSAGE. Overall, these results demonstrate that the leakage-controlled carrier context graph contains strong translational prediction signals and provides a reliable foundation for downstream knowledge-unit temporal dynamics modeling.
```

---

## 12. 结论

Stage 3 已经完成 carrier-level Patent-Paper link prediction 的主要 baseline 验证。

最终排序中，最强模型为：

```text
GraphSAGE no-species e50
Test AUROC = 0.880608
Test AUPRC = 0.888661
```

其次为：

```text
GraphSAGE full e50
Test AUROC = 0.881377
Test AUPRC = 0.887874
```

R-GCN no-species 也达到接近水平：

```text
Test AUROC = 0.880312
Test AUPRC = 0.885240
```

这些结果说明：

1. carrier context graph 的结构信号非常强；
2. relation-aware KG embedding 能显著优于普通 random-walk / metapath embedding；
3. supervised message passing 是当前最强 carrier-level baseline；
4. full 与 no-species 差异在强模型下很小；
5. 当前阶段不需要继续扩展更多 baseline 或更复杂的数据划分。

下一步应将重点转向主线任务：

```text
knowledge-unit construction
temporal translation state definition
interpretable higher-order graph dynamics
reaction-diffusion-source model
```