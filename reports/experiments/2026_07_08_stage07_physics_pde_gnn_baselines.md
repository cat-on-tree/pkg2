# Stage 07E 实验总结：RDGNN-style 与 GRAND-style Physics / PDE GNN Baselines

日期：2026-07-08  
阶段：Stage 07E  
主题：Physics / PDE-style GNN baselines 与当前 Knowledge Field 模型比较  
用途：内部实验总结、DynamicRDS 设计依据、后续论文 baseline 组织依据  
注意：本文档总结 RDGNN-style 和 GRAND-style 两类 physics / PDE GNN baseline 的当前结果。DynamicRDS 尚未纳入本轮实验。

---

## 1. 背景与目的

Stage 07C / 07D 已经完成了第一批 Temporal Knowledge Field GNN 实验，包括：

```text
MLP no-graph baseline
Temporal GraphSAGE
WeightedDiffusion GNN
Reaction-Diffusion-Source GNN v0
```

上一轮结果显示：

```text
1. GraphSAGE 明显优于 no-graph MLP；
2. WeightedDiffusion 在 local_source_proxy 输入下显著提升 top-K discovery；
3. Reaction-Diffusion-Source GNN v0 是当前 task-specific Knowledge Field 模型中的最强模型；
4. RDS v0 同时利用 local reaction、carrier source、diffusion proxy 和 graph diffusion。
```

但是，仅与普通 MLP / GraphSAGE / WeightedDiffusion 比较还不够。由于本研究的理论主张是：

```text
未来 translational emergence 更像是 KU graph 上的 source-driven diffusion / field process
```

因此必须补充更接近现有 physics / PDE GNN 思路的 baseline，用于回答：

```text
1. 通用 reaction-diffusion GNN 是否已经足以解释当前任务？
2. 通用 graph diffusion / neural PDE-style smoothing 是否已经足够强？
3. RDS v0 的 task-specific source/reaction/diffusion 分解是否仍有增量？
4. 后续 DynamicRDS 是否有必要？
```

因此，Stage 07E 补充了两类 physics / PDE-style baseline：

```text
RDGNN-style reaction-diffusion baseline
GRAND-style graph diffusion / neural PDE baseline
```

---

## 2. 总体实验设定

### 2.1 数据版本

当前实验仍使用 diabetes long-window formal benchmark：

```text
diabetes_2000_2024_v1
```

GNN-ready 数据目录包括：

```text
data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_latest_split/
data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_latest_split/
data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_proxy_latest_split/
```

三类输入定义：

```text
local:
  focal KU temporal state features x_i(t)

local_source:
  focal KU temporal state x_i(t)
  + carrier graph source / exposure features s_i(t)

local_source_proxy:
  focal KU temporal state x_i(t)
  + carrier graph source / exposure features s_i(t)
  + handcrafted KU-neighborhood diffusion proxy features p_i(t)
```

---

### 2.2 图结构

Stage 07E 继续使用 KU-KU shared-entity graph：

```text
node = Knowledge Unit
edge = two KUs share at least one entity
```

边权表示 shared-entity proximity：

```text
edge_weight = shared-entity weighted proximity
```

当前 KU graph 规模约为：

```text
knowledge_unit_count = 715,912
edge_count ≈ 17M directed / sampled edge entries
```

---

### 2.3 预测任务

当前第一主任务仍为：

```text
Future Translation Heat Forecasting
```

Eligibility：

```text
eligible_translation_task =
  history_paper_count > 0
  AND history_patent_count == 0
  AND history_trial_count == 0
```

Heat target：

```text
target_future_translation_heat_3yr =
  log1p(target_future_patent_count_3yr + target_future_trial_count_3yr)
```

Emergence label：

```text
label_future_translation_emergence_3yr =
  1[target_future_patent_count_3yr + target_future_trial_count_3yr > 0]
```

---

### 2.4 Temporal split

Stage 07E 使用与 Stage 07C / 07D 相同的 latest-window split：

```text
train cutoffs:
  2005-2018

validation cutoffs:
  2019-2020

test cutoff:
  2021

test target years:
  2022-2024
```

说明：

```text
2022-2024 是 test target window，不是 cutoff years。
```

---

### 2.5 训练协议

本轮 RDGNN-style 和 GRAND-style 模型使用与前序 Stage 07 模型一致的训练配置：

```text
hidden_dim = 128
num_layers = 2
num_neighbors = 15 10
batch_size = 16384
optimizer = AdamW
lr = 1e-3
weight_decay = 1e-4
dropout = 0.1
loss = SmoothL1
monitor_metric = validation AUPRC
patience = 8
epochs = 50
seed = 42
device = cuda
amp = enabled
num_workers = 4
```

主要 evaluation metrics：

```text
AUPRC
AUROC
Spearman
Precision@100
Precision@1000
Precision@5000
Recall@1000
NDCG@1000
Enrichment@1000
```

Primary model selection metric：

```text
validation AUPRC
```

Primary reporting split：

```text
test cutoff = 2021
target years = 2022-2024
```

---

## 3. 模型定义

## 3.1 RDGNN-style baseline

RDGNN-style baseline 用于表示通用 reaction-diffusion GNN 思路。

它不是 task-specific Knowledge Field decomposition，而是一个通用的 reaction-diffusion representation dynamics baseline。

核心形式：

```text
h_i^{l+1}
=
h_i^l
+
α_l Rθ(h_i^l)
+
β_l Σ_j A_ij · Φψ(h_j^l - h_i^l)
```

其中：

```text
h_i^l:
  第 l 层的 KU hidden representation

Rθ:
  reaction MLP

Φψ:
  diffusion message MLP

A_ij:
  KU-KU edge weight

h_j^l - h_i^l:
  hidden-state difference message
```

最终预测：

```text
ŷ_i(t) = Head(h_i^L)
```

RDGNN-style baseline 的作用是回答：

```text
通用 reaction-diffusion representation dynamics 是否已经足以解释当前 Knowledge Field 任务？
```

它与 RDS v0 的区别：

```text
RDS v0:
  显式 local/source/proxy/diffusion 分支，面向 Knowledge Field 任务设计

RDGNN-style:
  通用 reaction-diffusion representation learning，不显式区分 source/proxy/decay/knowledge state
```

---

## 3.2 GRAND-style baseline

GRAND-style baseline 用于表示 graph diffusion / neural PDE-style GNN。

它将图传播看作一种 diffusion / smoothing process。

当前离散化形式可概括为：

```text
h_i^{l+1}
=
h_i^l
+
δ · [
    Σ_j Ã_ij h_j^l - h_i^l
    +
    Rθ(h_i^l)
  ]
```

其中：

```text
Σ_j Ã_ij h_j^l - h_i^l:
  normalized graph diffusion / smoothing term

Rθ(h_i^l):
  feature-wise reaction / nonlinear transformation

δ:
  learnable or fixed diffusion step size
```

最终预测：

```text
ŷ_i(t) = Head(h_i^L)
```

GRAND-style baseline 的作用是回答：

```text
简单 graph diffusion / PDE-style smoothing 是否已经足以捕捉 KU field 中的 diffusion-like signal？
```

它与 RDS v0 的区别：

```text
RDS v0:
  local reaction + carrier source + diffusion proxy + weighted graph diffusion

GRAND-style:
  通用 graph diffusion / smoothing + reaction，不显式区分 source/proxy/field-state/decay
```

---

## 4. 当前完成的实验

本轮完成 6 组模型：

```text
local_grand_style_v0_seed42
local_rdgnn_style_v0_seed42
local_source_grand_style_v0_seed42
local_source_rdgnn_style_v0_seed42
local_source_proxy_grand_style_v0_seed42
local_source_proxy_rdgnn_style_v0_seed42
```

这些实验覆盖三类输入：

```text
local
local_source
local_source_proxy
```

以及两类 physics / PDE GNN：

```text
grand_style
rdgnn_style
```

---

## 5. 当前 Stage 07E 结果汇总

### 5.1 Test cutoff = 2021 主结果

| Run | Model | Best Epoch | Val AUPRC | Test AUPRC | AUROC | Spearman | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_grand_style_v0_seed42 | grand_style | 27 | 0.065481 | 0.038674 | 0.683351 | 0.084861 | 0.110 | 0.082 | 0.0728 | 0.088485 | 4.515754 |
| local_rdgnn_style_v0_seed42 | rdgnn_style | 32 | 0.067626 | 0.040439 | 0.688386 | 0.087194 | 0.120 | 0.101 | 0.0840 | 0.109211 | 5.562087 |
| local_source_grand_style_v0_seed42 | grand_style | 30 | 0.072380 | 0.042239 | 0.689759 | 0.087825 | 0.130 | 0.136 | 0.0920 | 0.140702 | 7.489542 |
| local_source_rdgnn_style_v0_seed42 | rdgnn_style | 49 | 0.074082 | 0.043171 | 0.696701 | 0.091040 | 0.120 | 0.129 | 0.0920 | 0.127278 | 7.104051 |
| local_source_proxy_grand_style_v0_seed42 | grand_style | 32 | 0.081023 | 0.047503 | 0.718492 | 0.101115 | 0.200 | 0.115 | 0.0912 | 0.119242 | 6.333069 |
| local_source_proxy_rdgnn_style_v0_seed42 | rdgnn_style | 48 | 0.083581 | 0.045454 | 0.714192 | 0.099127 | 0.120 | 0.103 | 0.0904 | 0.105720 | 5.672227 |

Test positive rate：

```text
test_positive_rate = 0.018159
```

解释：

```text
Enrichment@1000 = Precision@1000 / test_positive_rate
```

---

## 6. 主要结果分析

## 6.1 local 输入下：RDGNN-style 优于 GRAND-style

对比：

```text
local_grand_style_v0_seed42
local_rdgnn_style_v0_seed42
```

结果：

| Metric | GRAND-style | RDGNN-style | 更优 |
|---|---:|---:|---|
| Test AUPRC | 0.038674 | 0.040439 | RDGNN |
| AUROC | 0.683351 | 0.688386 | RDGNN |
| Spearman | 0.084861 | 0.087194 | RDGNN |
| P@100 | 0.110 | 0.120 | RDGNN |
| P@1000 | 0.082 | 0.101 | RDGNN |
| P@5000 | 0.0728 | 0.0840 | RDGNN |
| NDCG@1000 | 0.088485 | 0.109211 | RDGNN |
| Enrich@1000 | 4.515754 | 5.562087 | RDGNN |

结论：

```text
在只使用 focal KU temporal state 的情况下，
通用 reaction-diffusion representation dynamics 比单纯 GRAND-style diffusion smoothing 更强。
```

解释：

```text
local 输入只包含 KU 自身状态，
因此 RDGNN-style 的 hidden-state difference message h_j - h_i
能够提供比普通 smoothing 更丰富的 neighbor contrast。
```

这说明：

```text
reaction term 和 pairwise difference message 在 local-only 场景下有价值。
```

---

## 6.2 加入 source features 后，两类 physics-style GNN 都明显增强

从 local 到 local_source：

### GRAND-style

```text
AUPRC:     0.038674 -> 0.042239
AUROC:     0.683351 -> 0.689759
Spearman:  0.084861 -> 0.087825
P@1000:    0.082    -> 0.136
NDCG@1000: 0.088485 -> 0.140702
```

### RDGNN-style

```text
AUPRC:     0.040439 -> 0.043171
AUROC:     0.688386 -> 0.696701
Spearman:  0.087194 -> 0.091040
P@1000:    0.101    -> 0.129
NDCG@1000: 0.109211 -> 0.127278
```

结论：

```text
carrier source / exposure features 对 physics-style GNN 同样非常重要。
```

尤其是：

```text
local_source_grand_style_v0_seed42
```

取得了很强的 top-K 表现：

```text
P@1000 = 0.136
NDCG@1000 = 0.140702
Enrichment@1000 = 7.489542
```

这说明：

```text
source exposure 不只是普通辅助特征，
而是 high-confidence translational candidate discovery 的核心驱动之一。
```

对后续 DynamicRDS 的启发是：

```text
必须显式建模 source injection term S_i(t)。
```

---

## 6.3 local_source 输入下：RDGNN-style 整体 ranking 略强，GRAND-style top-K 更强

对比：

```text
local_source_grand_style_v0_seed42
local_source_rdgnn_style_v0_seed42
```

结果：

| Metric | GRAND-style | RDGNN-style | 更优 |
|---|---:|---:|---|
| Test AUPRC | 0.042239 | 0.043171 | RDGNN |
| AUROC | 0.689759 | 0.696701 | RDGNN |
| Spearman | 0.087825 | 0.091040 | RDGNN |
| P@100 | 0.130 | 0.120 | GRAND |
| P@1000 | 0.136 | 0.129 | GRAND |
| P@5000 | 0.0920 | 0.0920 | Tie |
| NDCG@1000 | 0.140702 | 0.127278 | GRAND |
| Enrich@1000 | 7.489542 | 7.104051 | GRAND |

结论：

```text
RDGNN-style 在 global ranking 上更强；
GRAND-style 在 top-K discovery 上更强。
```

解释：

```text
RDGNN-style 的 h_j - h_i difference message 可能更适合改善整体排序；
GRAND-style 的 smoothing 结合 source exposure 后，更容易把 source-driven high-confidence candidates 推到 top-K。
```

这再次说明：

```text
global ranking 和 extreme top-K discovery 可能依赖不同机制。
```

---

## 6.4 加入 proxy features 后：GRAND-style 明显优于 RDGNN-style

对比：

```text
local_source_proxy_grand_style_v0_seed42
local_source_proxy_rdgnn_style_v0_seed42
```

结果：

| Metric | GRAND-style | RDGNN-style | 更优 |
|---|---:|---:|---|
| Test AUPRC | 0.047503 | 0.045454 | GRAND |
| AUROC | 0.718492 | 0.714192 | GRAND |
| Spearman | 0.101115 | 0.099127 | GRAND |
| P@100 | 0.200 | 0.120 | GRAND |
| P@1000 | 0.115 | 0.103 | GRAND |
| P@5000 | 0.0912 | 0.0904 | GRAND |
| NDCG@1000 | 0.119242 | 0.105720 | GRAND |
| Enrich@1000 | 6.333069 | 5.672227 | GRAND |

结论：

```text
在 local_source_proxy 输入下，GRAND-style 全面优于 RDGNN-style。
```

可能解释：

```text
1. proxy features 已经包含 handcrafted KU-neighborhood diffusion signal；
2. RDGNN-style 在 hidden representation 上再做 h_j - h_i difference message，可能与 proxy signal 重复或产生扰动；
3. GRAND-style 的 diffusion smoothing 更适合整合已有 source/proxy field features；
4. 高维 local_source_proxy 输入下，简单稳定的 PDE-style smoothing 反而比通用 reaction-diffusion message 更稳。
```

这对后续 DynamicRDS 很关键：

```text
DynamicRDS 不应简单复制 RDGNN-style 的 hidden representation difference；
而应在显式 knowledge state u_i(t) 上做物理 diffusion flux。
```

---

## 6.5 proxy features 对 GRAND-style 的 global ranking 提升非常明显

GRAND-style 从 local_source 到 local_source_proxy：

```text
AUPRC:     0.042239 -> 0.047503
AUROC:     0.689759 -> 0.718492
Spearman:  0.087825 -> 0.101115
P@100:     0.130    -> 0.200
```

结论：

```text
handcrafted diffusion proxy 与 GRAND-style PDE propagation 高度互补。
```

解释：

```text
proxy features 提供了 cutoff-safe 的邻域 diffusion prior；
GRAND-style graph diffusion 再对这些 field features 做平滑传播，
从而显著提升 global field ranking。
```

这说明后续 DynamicRDS 可以考虑保留：

```text
P_i(t)
```

作为：

```text
proxy injection term
prior field term
diffusion prior
```

---

# 7. 与 Stage 07D 既有模型的比较

## 7.1 与 RDS v0 比较

当前 RDS v0：

```text
local_source_proxy_rds_v0_seed42
```

此前结果：

```text
Test AUPRC       = 0.047322
AUROC            = 0.715591
Spearman         = 0.099775
P@100            = 0.160
P@1000           = 0.132
P@5000           = 0.0974
NDCG@1000        = 0.136672
Enrichment@1000  = 7.269262
```

当前最强 GRAND-style：

```text
local_source_proxy_grand_style_v0_seed42
```

结果：

```text
Test AUPRC       = 0.047503
AUROC            = 0.718492
Spearman         = 0.101115
P@100            = 0.200
P@1000           = 0.115
P@5000           = 0.0912
NDCG@1000        = 0.119242
Enrichment@1000  = 6.333069
```

对比：

| Metric | RDS v0 | GRAND-style local_source_proxy | 更优 |
|---|---:|---:|---|
| Test AUPRC | 0.047322 | 0.047503 | GRAND |
| AUROC | 0.715591 | 0.718492 | GRAND |
| Spearman | 0.099775 | 0.101115 | GRAND |
| P@100 | 0.160 | 0.200 | GRAND |
| P@1000 | 0.132 | 0.115 | RDS |
| P@5000 | 0.0974 | 0.0912 | RDS |
| NDCG@1000 | 0.136672 | 0.119242 | RDS |
| Enrich@1000 | 7.269262 | 6.333069 | RDS |

结论：

```text
GRAND-style local_source_proxy 在 global ranking 和 P@100 上超过 RDS v0；
RDS v0 在 P@1000 / P@5000 / NDCG@1000 / Enrichment@1000 上仍更强。
```

解释：

```text
GRAND-style 更擅长改善整体 field ranking；
RDS v0 更擅长稳定的 top-K candidate discovery。
```

因此，当前不能简单说 RDS v0 在所有指标上最强。更准确是：

```text
GRAND-style 是当前最强 global ranking baseline；
RDS v0 是当前更稳的 top-K discovery model。
```

---

## 7.2 与 WeightedDiffusion 比较

此前 `local_source_proxy_weighted_diffusion_v0_seed42` 结果：

```text
AUPRC            = 0.045404
AUROC            = 0.712645
Spearman         = 0.098410
P@100            = 0.180
P@1000           = 0.121
NDCG@1000        = 0.122069
Enrichment@1000  = 6.663490
```

当前 `local_source_proxy_grand_style_v0_seed42`：

```text
AUPRC            = 0.047503
AUROC            = 0.718492
Spearman         = 0.101115
P@100            = 0.200
P@1000           = 0.115
NDCG@1000        = 0.119242
Enrichment@1000  = 6.333069
```

结论：

```text
GRAND-style 比 WeightedDiffusion 的 global metrics 更强；
WeightedDiffusion 的 P@1000 / NDCG@1000 略强或接近。
```

这说明：

```text
PDE-style propagation 比简单 weighted aggregation 更有利于整体 field ranking；
但 top-K discovery 仍需要更专门的 source / diffusion / ranking 机制。
```

---

## 7.3 与 GraphSAGE 比较

此前 `local_source_proxy_graphsage_v0_seed42`：

```text
AUPRC            = 0.045131
AUROC            = 0.713521
Spearman         = 0.098816
P@100            = 0.050
P@1000           = 0.097
NDCG@1000        = 0.090821
Enrichment@1000  = 5.341806
```

当前 `local_source_proxy_grand_style_v0_seed42`：

```text
AUPRC            = 0.047503
AUROC            = 0.718492
Spearman         = 0.101115
P@100            = 0.200
P@1000           = 0.115
NDCG@1000        = 0.119242
Enrichment@1000  = 6.333069
```

结论：

```text
GRAND-style 明显优于普通 GraphSAGE。
```

这说明：

```text
在当前任务中，PDE-style diffusion smoothing 比 generic GraphSAGE message passing 更适合整合 local_source_proxy field features。
```

---

# 8. 当前模型定位更新

基于 Stage 07C / 07D / 07E 当前结果，可以更新模型定位。

## 8.1 当前 global ranking 最强模型

当前 global ranking 最强是：

```text
local_source_proxy_grand_style_v0_seed42
```

代表指标：

```text
Test AUPRC    = 0.047503
AUROC         = 0.718492
Spearman      = 0.101115
P@100         = 0.200
```

---

## 8.2 当前 stable top-K discovery 最强模型

当前 stable top-K discovery 更强的是：

```text
local_source_proxy_rds_v0_seed42
```

代表指标：

```text
P@1000          = 0.132
P@5000          = 0.0974
NDCG@1000       = 0.136672
Enrichment@1000 = 7.269262
```

---

## 8.3 当前最重要竞争关系

当前最重要的模型竞争关系是：

```text
RDS v0
vs
GRAND-style local_source_proxy
```

二者代表两种路线：

```text
RDS v0:
  task-specific reaction/source/proxy/diffusion decomposition

GRAND-style:
  generic graph diffusion / PDE-style smoothing over rich field features
```

当前结果说明：

```text
通用 PDE-style baseline 已经非常强；
但 task-specific Knowledge Field decomposition 在 top-K discovery 上仍有优势。
```

---

# 9. 对 Knowledge Field 假设的支持

Stage 07E 进一步支持了 Knowledge Field 假设。

## 9.1 KU graph 中存在 diffusion-like predictive structure

证据：

```text
GRAND-style 和 RDGNN-style 均能在不同输入下取得接近或超过 GraphSAGE / WeightedDiffusion 的表现。
```

尤其：

```text
local_source_proxy_grand_style_v0_seed42
```

取得当前最强 global ranking：

```text
AUPRC = 0.047503
AUROC = 0.718492
Spearman = 0.101115
```

说明：

```text
KU graph 不是普通辅助结构；
它具有可被 PDE-style diffusion 利用的场结构。
```

---

## 9.2 source term 是不可忽略的核心因素

从 local 到 local_source，两类模型都提升明显。

尤其：

```text
local_source_grand_style_v0_seed42
P@1000 = 0.136
Enrichment@1000 = 7.489542
```

说明：

```text
carrier graph source / exposure 对 high-confidence candidate discovery 非常重要。
```

这直接支持：

```text
DynamicRDS 必须显式包含 source injection S_i(t)。
```

---

## 9.3 proxy features 是有用的 field prior

GRAND-style 加入 proxy 后 global ranking 大幅提升：

```text
local_source_grand_style:
  AUPRC = 0.042239
  AUROC = 0.689759
  Spearman = 0.087825

local_source_proxy_grand_style:
  AUPRC = 0.047503
  AUROC = 0.718492
  Spearman = 0.101115
```

说明：

```text
handcrafted diffusion proxy 能作为有效的 field prior / prior diffusion state。
```

这支持后续 DynamicRDS 中保留：

```text
P_i(t)
```

或者用 proxy features 辅助初始化：

```text
u_i(t)
```

---

## 9.4 通用 hidden-state reaction-diffusion 不一定最优

RDGNN-style 在 local 输入下强于 GRAND-style，但在 local_source_proxy 输入下明显弱于 GRAND-style。

说明：

```text
hidden representation 上的 h_j - h_i reaction-diffusion message
不一定适合高维 local_source_proxy field features。
```

这提示后续 DynamicRDS 不应只是：

```text
RDGNN-style + source features
```

而应采用更明确的物理状态变量：

```text
u_i(t)
```

并在该状态上做：

```text
D_i(t) = Σ_j κ_ij(t) [u_j(t) - u_i(t)]
```

---

# 10. 对 DynamicRDS 的设计启发

Stage 07E 的结果为 DynamicRDS 提供了直接依据。

## 10.1 DynamicRDS 应吸收 GRAND-style 的稳定 diffusion smoothing

GRAND-style 在 global ranking 上很强，说明：

```text
normalized graph diffusion / PDE smoothing 是有效的。
```

DynamicRDS 应保留类似形式：

```text
D_i(t) = Σ_j κ_ij(t) [u_j(t) - u_i(t)]
```

并使用稳定的 edge normalization。

---

## 10.2 DynamicRDS 不应简单复制 RDGNN-style hidden difference

RDGNN-style 的结果说明：

```text
h_j - h_i hidden representation difference
在 local_source_proxy 输入下不如 GRAND-style 稳定。
```

因此 DynamicRDS 应使用：

```text
explicit scalar or low-dimensional knowledge state u_i(t)
```

而不是直接在高维 hidden representation 上做差分。

---

## 10.3 DynamicRDS 必须包含 source injection

source features 对 physics-style GNN 的提升非常明显。

因此 DynamicRDS 应包含：

```text
S_i(t) = Sφ(s_i(t))
```

并将其解释为：

```text
carrier graph evidence substrate 对 KU field 的外部注入。
```

---

## 10.4 DynamicRDS 可保留 proxy prior

proxy features 对 GRAND-style 的 global ranking 提升明显。

DynamicRDS 可保留：

```text
P_i(t) = Pη(p_i(t))
```

作为：

```text
prior field term
diffusion proxy injection
neighborhood prior
```

或者将 proxy features 用于：

```text
u_i(t) initialization
```

---

## 10.5 DynamicRDS 应加入 decay / dissipation

当前模型主要建模 positive diffusion / reaction / source，但知识场中也存在：

```text
attention decay
research cooling
translation failure
topic aging
```

因此 DynamicRDS 应显式建模：

```text
Λ_i(t) = λ_i(t) · u_i(t)
```

以提高物理解释性和长期稳定性。

---

# 11. 推荐下一步实验路线

当前不建议继续扩展更多普通 GNN baseline。

原因：

```text
1. MLP / GraphSAGE / WeightedDiffusion / RDS v0 已完成；
2. RDGNN-style / GRAND-style physics baselines 已完成；
3. GRAND-style 已经是强竞争 baseline；
4. RDGNN-style 说明通用 reaction-diffusion 不一定充分；
5. DynamicRDS 的设计动机已经明确。
```

下一步推荐进入：

```text
Stage 07F: Dynamic Reaction-Diffusion-Source Knowledge Field Dynamics
```

推荐模型名：

```text
dynamic_rds
```

推荐第一版形式：

```text
u_i(t + Δt)
=
u_i(t)
+
Δt · [
    R_i(t)
  + S_i(t)
  + P_i(t)
  + D_i(t)
  - Λ_i(t)
]
```

其中：

```text
R_i(t) = Rθ(u_i(t), x_i(t))
S_i(t) = Sφ(s_i(t))
P_i(t) = Pη(p_i(t))
D_i(t) = Σ_j κ_ij(t) · [u_j(t) - u_i(t)]
Λ_i(t) = λ_i(t) · u_i(t)
```

---

# 12. 后续汇总时建议主表模型

后续 Stage 07 总表建议包含：

```text
MLP no-graph
GraphSAGE
WeightedDiffusion
RDS v0
RDGNN-style
GRAND-style
DynamicRDS
```

建议主比较输入：

```text
local_source_proxy
```

当前已完成的关键 runs：

```text
local_source_mlp_v0_seed42
local_source_proxy_graphsage_v0_seed42
local_source_proxy_weighted_diffusion_v0_seed42
local_source_proxy_rds_v0_seed42
local_source_proxy_rdgnn_style_v0_seed42
local_source_proxy_grand_style_v0_seed42
```

后续待补充：

```text
local_source_proxy_dynamic_rds_v0_seed42
```

---

# 13. 当前结论

Stage 07E 的结果具有很高参考价值。

主要结论：

```text
1. Physics / PDE-style GNN baseline 是必须纳入的强 baseline；
2. GRAND-style local_source_proxy 在 global ranking 上达到当前最强；
3. RDS v0 在 P@1000 / P@5000 / NDCG@1000 / Enrichment@1000 上仍更稳；
4. RDGNN-style 在 local 输入下优于 GRAND-style，但在 local_source_proxy 输入下不如 GRAND-style；
5. source features 对 physics-style GNN 的提升非常明显；
6. proxy features 与 GRAND-style PDE smoothing 高度互补；
7. 当前结果直接支持开发 DynamicRDS。
```

---

# 14. 最终一句话

Stage 07E 证明：

```text
KU graph 中确实存在可被 physics / PDE-style GNN 利用的 diffusion-like field structure。
```

但同时也说明：

```text
通用 graph diffusion 虽然能提升 global ranking，
但还不能完全替代 task-specific reaction-source-diffusion decomposition 在 top-K discovery 上的优势。
```

因此，下一步最有价值的模型开发方向是：

```text
DynamicRDS：
显式定义 knowledge state u_i(t)，
在 u_i(t) 上进行 source-driven reaction-diffusion-decay 更新，
从而将当前 physics-inspired Knowledge Field GNN
推进为真正的 Knowledge Field dynamical
