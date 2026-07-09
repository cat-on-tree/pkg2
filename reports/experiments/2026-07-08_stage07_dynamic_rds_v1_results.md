# Stage 07F 实验总结：DynamicRDS v1 显式 Knowledge Field Dynamics 模型

日期：2026-07-08  
阶段：Stage 07F  
主题：Dynamic Reaction-Diffusion-Source Knowledge Field Dynamics v1 实验结果  
用途：内部实验总结、DynamicRDS v2 设计依据、后续论文方法演进记录  
注意：本文档总结 DynamicRDS v1 的第一版结果。DynamicRDS v1 是显式 knowledge-state dynamics 的最小可验证版本，不代表最终算法形态。

---

## 1. 背景与目的

Stage 07C / 07D / 07E 已经完成了以下模型系列：

```text
MLP no-graph baseline
Temporal GraphSAGE
WeightedDiffusion GNN
Reaction-Diffusion-Source GNN v0
RDGNN-style reaction-diffusion baseline
GRAND-style graph diffusion / neural PDE baseline
```

这些结果共同表明：

```text
1. KU-KU graph message passing 明显优于 no-graph baseline；
2. edge_weight 对 top-K discovery 有帮助；
3. RDS v0 的 reaction/source/proxy/diffusion 分支结构在 top-K discovery 上表现较稳；
4. GRAND-style physics / PDE baseline 在 global ranking 上非常强；
5. RDGNN-style 在 local 输入下有效，但在 local_source_proxy 输入下不如 GRAND-style；
6. KU graph 中确实存在 diffusion-like field structure。
```

然而，前述模型仍主要是：

```text
features at cutoff t -> future score
```

即静态监督预测器。即使 RDS v0 引入了 reaction-source-diffusion 的物理归纳偏置，它也没有显式定义知识状态变量并建模：

```text
u_i(t) -> u_i(t + Δt)
```

因此，Stage 07F 的目标是开发并验证第一版：

```text
Dynamic Reaction-Diffusion-Source Knowledge Field Dynamics
```

即：

```text
DynamicRDS v1
```

本轮实验核心问题：

```text
1. 显式 knowledge state u_i(t) 是否可训练？
2. true graph diffusion flux Σ κ_ij(u_j - u_i) 是否有效？
3. DynamicRDS 是否能达到或接近 RDS / GRAND-style 等强 baseline？
4. DynamicRDS 是否优于通用 RDGNN-style hidden representation dynamics？
5. 当前 scalar-state DynamicRDS v1 的不足在哪里？
```

---

## 2. DynamicRDS v1 的理论定位

DynamicRDS v1 是当前项目中第一个显式建模 knowledge state temporal evolution 的模型。

它与前序模型的区别如下：

```text
MLP / GraphSAGE / WeightedDiffusion:
  直接从 features at t 预测 future score

RDS v0:
  引入 reaction/source/proxy/diffusion 分支，但仍是 static predictor

RDGNN-style:
  在 hidden representation 上做 reaction-diffusion dynamics

GRAND-style:
  在 hidden representation 上做 graph diffusion / PDE-style smoothing

DynamicRDS v1:
  显式定义 knowledge state u_i(t)，并通过 reaction/source/proxy/diffusion/decay
  更新到 u_i(t + Δt)
```

因此，DynamicRDS v1 的方法定位是：

```text
explicit knowledge-field dynamical system
```

而不是普通 GNN layer 或静态预测器。

---

## 3. DynamicRDS v1 模型形式

### 3.1 显式知识状态变量

DynamicRDS v1 定义每个 KU 在 cutoff year `t` 的显式状态：

```text
u_i(t)
```

表示：

```text
KU_i 的 knowledge field intensity / translational potential / learned scalar state
```

第一版使用 learned scalar state：

```text
u_i(t) = StateEncoderθ(x_i(t))
```

其中：

```text
x_i(t) = focal KU local temporal state features
```

当前 v1 中：

```text
u_i(t) ∈ R
```

即 scalar-state DynamicRDS。

---

### 3.2 核心动力学方程

DynamicRDS v1 使用 Euler-style update：

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
R_i(t):
  local reaction term

S_i(t):
  carrier source / exposure injection term

P_i(t):
  diffusion proxy / prior field term

D_i(t):
  graph diffusion flux

Λ_i(t):
  decay / dissipation term
```

具体定义：

```text
R_i(t) = Rθ(u_i(t), x_i(t))

S_i(t) = Sφ(s_i(t))

P_i(t) = Pη(p_i(t))

D_i(t) = Σ_j κ_ij(t) · [u_j(t) - u_i(t)]

Λ_i(t) = λ_i(t) · u_i(t)
```

其中：

```text
s_i(t):
  carrier source / exposure features

p_i(t):
  handcrafted KU-neighborhood diffusion proxy features

κ_ij(t):
  edge conductance, v1 中由 normalized edge_weight 给出

λ_i(t):
  learned nonnegative decay rate
```

最终监督预测：

```text
ŷ_i(t) = Readout(u_i(t), Δu_i(t), u_i(t + Δt), context_i(t))
```

其中：

```text
Δu_i(t) = u_i(t + Δt) - u_i(t)
```

---

### 3.3 与 RDGNN-style 的关键区别

RDGNN-style 使用：

```text
h_j - h_i
```

即 hidden representation difference。

DynamicRDS v1 使用：

```text
u_j(t) - u_i(t)
```

即显式知识状态差。

因此，DynamicRDS v1 更接近物理扩散：

```text
D_i(t) = Σ_j κ_ij [u_j(t) - u_i(t)]
```

这是真正的 flux diffusion，而不是普通 neighbor aggregation：

```text
Σ_j A_ij h_j
```

---

## 4. 总体实验设定

### 4.1 数据版本

当前实验使用 diabetes long-window formal benchmark：

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

### 4.2 图结构

DynamicRDS v1 使用 KU-KU shared-entity graph：

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

### 4.3 预测任务

当前第一主任务为：

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

### 4.4 Temporal split

DynamicRDS v1 使用与 Stage 07C / 07D / 07E 相同的 latest-window split：

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

### 4.5 训练协议

本轮 DynamicRDS v1 使用与前序 Stage 07 模型一致的训练配置：

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

## 5. 当前完成的 DynamicRDS v1 实验

本轮完成 3 组模型：

```text
local_dynamic_rds_v0_seed42
local_source_dynamic_rds_v0_seed42
local_source_proxy_dynamic_rds_v0_seed42
```

覆盖三类输入：

```text
local
local_source
local_source_proxy
```

---

## 6. DynamicRDS v1 结果汇总

### 6.1 Test cutoff = 2021 主结果

| Run | Model | Best Epoch | Val AUPRC | Test AUPRC | AUROC | Spearman | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_dynamic_rds_v0_seed42 | dynamic_rds | 27 | 0.064360 | 0.037622 | 0.672150 | 0.079679 | 0.120 | 0.092 | 0.0736 | 0.099948 | 5.066455 |
| local_source_dynamic_rds_v0_seed42 | dynamic_rds | 17 | 0.070149 | 0.038652 | 0.681206 | 0.083868 | 0.010 | 0.081 | 0.0832 | 0.073847 | 4.460683 |
| local_source_proxy_dynamic_rds_v0_seed42 | dynamic_rds | 22 | 0.082189 | 0.046314 | 0.719937 | 0.101782 | 0.140 | 0.101 | 0.0820 | 0.115910 | 5.562087 |

Test positive rate：

```text
test_positive_rate = 0.018159
```

解释：

```text
Enrichment@1000 = Precision@1000 / test_positive_rate
```

---

## 7. DynamicRDS v1 内部结果分析

## 7.1 local_source_proxy 输入显著提升 global ranking

三组 DynamicRDS v1 中，`local_source_proxy_dynamic_rds_v0_seed42` 显著最强。

| Run | AUPRC | AUROC | Spearman | P@1000 | NDCG@1000 |
|---|---:|---:|---:|---:|---:|
| local_dynamic_rds | 0.037622 | 0.672150 | 0.079679 | 0.092 | 0.099948 |
| local_source_dynamic_rds | 0.038652 | 0.681206 | 0.083868 | 0.081 | 0.073847 |
| local_source_proxy_dynamic_rds | 0.046314 | 0.719937 | 0.101782 | 0.101 | 0.115910 |

从 local 到 local_source_proxy：

```text
AUPRC:
  0.037622 -> 0.046314

AUROC:
  0.672150 -> 0.719937

Spearman:
  0.079679 -> 0.101782

NDCG@1000:
  0.099948 -> 0.115910
```

结论：

```text
DynamicRDS v1 对 rich field features 非常敏感；
local + source + proxy 的组合显著改善显式动力学模型的 global ranking。
```

这说明：

```text
显式 u(t) dynamics 需要足够的 field context 才能发挥作用。
```

---

## 7.2 local_source 版本的 top-K 表现异常偏弱

`local_source_dynamic_rds_v0_seed42` 的结果：

```text
P@100 = 0.010
P@1000 = 0.081
NDCG@1000 = 0.073847
Enrichment@1000 = 4.460683
```

它在 global metrics 上比 local 版本略强：

```text
AUPRC:
  0.037622 -> 0.038652

AUROC:
  0.672150 -> 0.681206

Spearman:
  0.079679 -> 0.083868
```

但 top-K 明显变差。

可能解释：

```text
1. 当前 v1 source term 被压缩为 scalar injection，表达能力不足；
2. source features 本身维度和尺度复杂，简单 scalar S_i(t) 可能损失 top-K 信息；
3. 缺少 proxy prior 时，source injection 可能更容易扰动 u dynamics；
4. validation AUPRC 选择出的 epoch 不一定对应最佳 top-K discovery；
5. 当前 scalar-state dynamics 对 source-driven extreme tail candidates 不够敏感。
```

这说明：

```text
DynamicRDS v1 的 source branch 设计还不够稳定。
```

---

## 7.3 local_source_proxy 版本证明 proxy prior 对 DynamicRDS 很关键

加入 proxy 后：

```text
local_source_dynamic_rds:
  AUPRC = 0.038652
  AUROC = 0.681206
  Spearman = 0.083868

local_source_proxy_dynamic_rds:
  AUPRC = 0.046314
  AUROC = 0.719937
  Spearman = 0.101782
```

提升非常明显。

结论：

```text
handcrafted diffusion proxy / neighborhood prior 对 DynamicRDS v1 非常关键。
```

解释：

```text
DynamicRDS 的 scalar u_i(t) 需要 proxy features 提供更强的 prior field context，
否则 source / local 信息不足以稳定构造有效的 future dynamics。
```

这支持后续继续保留：

```text
P_i(t)
```

作为：

```text
proxy prior field term
diffusion prior injection
neighborhood prior
```

---

# 8. 与已有 Stage 07 模型的比较

## 8.1 与 RDS v0 比较

当前 RDS v0：

```text
local_source_proxy_rds_v0_seed42
```

此前结果：

```text
AUPRC            = 0.047322
AUROC            = 0.715591
Spearman         = 0.099775
P@100            = 0.160
P@1000           = 0.132
P@5000           = 0.0974
NDCG@1000        = 0.136672
Enrichment@1000  = 7.269262
```

DynamicRDS v1：

```text
local_source_proxy_dynamic_rds_v0_seed42
```

结果：

```text
AUPRC            = 0.046314
AUROC            = 0.719937
Spearman         = 0.101782
P@100            = 0.140
P@1000           = 0.101
P@5000           = 0.0820
NDCG@1000        = 0.115910
Enrichment@1000  = 5.562087
```

对比：

| Metric | RDS v0 | DynamicRDS v1 | 更优 |
|---|---:|---:|---|
| AUPRC | 0.047322 | 0.046314 | RDS v0 |
| AUROC | 0.715591 | 0.719937 | DynamicRDS |
| Spearman | 0.099775 | 0.101782 | DynamicRDS |
| P@100 | 0.160 | 0.140 | RDS v0 |
| P@1000 | 0.132 | 0.101 | RDS v0 |
| P@5000 | 0.0974 | 0.0820 | RDS v0 |
| NDCG@1000 | 0.136672 | 0.115910 | RDS v0 |
| Enrich@1000 | 7.269262 | 5.562087 | RDS v0 |

结论：

```text
DynamicRDS v1 在 AUROC / Spearman 上超过 RDS v0；
但在 AUPRC 和 top-K discovery 上仍弱于 RDS v0。
```

解释：

```text
显式 scalar-state dynamics 有助于整体 field ranking；
但当前 v1 对 rare positive top-tail candidate discovery 还不够强。
```

---

## 8.2 与 GRAND-style 比较

当前最强 GRAND-style：

```text
local_source_proxy_grand_style_v0_seed42
```

此前结果：

```text
AUPRC            = 0.047503
AUROC            = 0.718492
Spearman         = 0.101115
P@100            = 0.200
P@1000           = 0.115
P@5000           = 0.0912
NDCG@1000        = 0.119242
Enrichment@1000  = 6.333069
```

DynamicRDS v1：

```text
AUPRC            = 0.046314
AUROC            = 0.719937
Spearman         = 0.101782
P@100            = 0.140
P@1000           = 0.101
P@5000           = 0.0820
NDCG@1000        = 0.115910
Enrichment@1000  = 5.562087
```

对比：

| Metric | GRAND-style | DynamicRDS v1 | 更优 |
|---|---:|---:|---|
| AUPRC | 0.047503 | 0.046314 | GRAND |
| AUROC | 0.718492 | 0.719937 | DynamicRDS |
| Spearman | 0.101115 | 0.101782 | DynamicRDS |
| P@100 | 0.200 | 0.140 | GRAND |
| P@1000 | 0.115 | 0.101 | GRAND |
| P@5000 | 0.0912 | 0.0820 | GRAND |
| NDCG@1000 | 0.119242 | 0.115910 | GRAND |
| Enrich@1000 | 6.333069 | 5.562087 | GRAND |

结论：

```text
DynamicRDS v1 在 AUROC / Spearman 上超过 GRAND-style；
但在 AUPRC 和 top-K discovery 上仍弱于 GRAND-style。
```

解释：

```text
DynamicRDS v1 更适合改善全局连续排序；
GRAND-style smoothing 更适合把部分高置信样本推到 top region。
```

---

## 8.3 与 RDGNN-style 比较

当前 RDGNN-style：

```text
local_source_proxy_rdgnn_style_v0_seed42
```

此前结果：

```text
AUPRC            = 0.045454
AUROC            = 0.714192
Spearman         = 0.099127
P@100            = 0.120
P@1000           = 0.103
P@5000           = 0.0904
NDCG@1000        = 0.105720
Enrichment@1000  = 5.672227
```

DynamicRDS v1：

```text
AUPRC            = 0.046314
AUROC            = 0.719937
Spearman         = 0.101782
P@100            = 0.140
P@1000           = 0.101
P@5000           = 0.0820
NDCG@1000        = 0.115910
Enrichment@1000  = 5.562087
```

对比：

| Metric | RDGNN-style | DynamicRDS v1 | 更优 |
|---|---:|---:|---|
| AUPRC | 0.045454 | 0.046314 | DynamicRDS |
| AUROC | 0.714192 | 0.719937 | DynamicRDS |
| Spearman | 0.099127 | 0.101782 | DynamicRDS |
| P@100 | 0.120 | 0.140 | DynamicRDS |
| P@1000 | 0.103 | 0.101 | RDGNN-style |
| P@5000 | 0.0904 | 0.0820 | RDGNN-style |
| NDCG@1000 | 0.105720 | 0.115910 | DynamicRDS |
| Enrich@1000 | 5.672227 | 5.562087 | RDGNN-style |

结论：

```text
DynamicRDS v1 基本优于 RDGNN-style，尤其在 AUROC、Spearman 和 NDCG@1000 上更强。
```

这很重要，因为它说明：

```text
显式 u_i(t) state + true flux diffusion
比通用 hidden representation h_j - h_i reaction-diffusion 更适合当前任务。
```

---

# 9. DynamicRDS v1 的主要价值

## 9.1 验证了显式 u(t) 动力学的可训练性

DynamicRDS v1 成功跑通并取得接近强 baseline 的结果。

这说明：

```text
显式定义 KU-level knowledge state u_i(t)，
并用 reaction/source/proxy/diffusion/decay 更新它，
在当前大规模 KU graph 上是可训练的。
```

这是从：

```text
physics-inspired static predictor
```

迈向：

```text
explicit knowledge field dynamical system
```

的重要一步。

---

## 9.2 证明 true graph flux diffusion 有效

DynamicRDS v1 使用：

```text
D_i(t) = Σ_j κ_ij [u_j(t) - u_i(t)]
```

而不是普通 aggregation：

```text
Σ_j A_ij h_j
```

在 local_source_proxy 输入下取得：

```text
AUROC = 0.719937
Spearman = 0.101782
```

这是当前所有模型中非常强的 global ranking 表现。

说明：

```text
物理 flux diffusion 在 KU field 上是有意义的。
```

---

## 9.3 支持“显式状态优于 hidden representation difference”的方向

RDGNN-style 使用：

```text
h_j - h_i
```

DynamicRDS v1 使用：

```text
u_j - u_i
```

当前结果显示 DynamicRDS v1 在多个 global metrics 上优于 RDGNN-style。

这支持后续主张：

```text
知识应被显式建模为 field state variable，
而不只是 GNN hidden representation。
```

---

## 9.4 提供了 DynamicRDS v2 的明确改进方向

DynamicRDS v1 的不足也非常清楚：

```text
1. scalar-state formulation 表达能力可能不足；
2. source term 被压缩成 scalar 后 top-K 表现不稳定；
3. AUPRC 和 top-K discovery 仍弱于 RDS v0 / GRAND-style；
4. validation AUPRC early stopping 不一定对应最佳 top-K discovery；
5. current readout 对 rare positive top-tail ranking 仍不够敏感。
```

这些不足不是路线失败，而是说明：

```text
v1 是有效原型；
v2 需要增强状态维度和 top-K sensitivity。
```

---

# 10. 当前不足与可能原因

## 10.1 AUPRC 未超过 RDS v0 / GRAND-style

DynamicRDS v1：

```text
AUPRC = 0.046314
```

RDS v0：

```text
AUPRC = 0.047322
```

GRAND-style：

```text
AUPRC = 0.047503
```

说明：

```text
DynamicRDS v1 对 rare positive ranking 的 early precision 还不足。
```

可能原因：

```text
1. scalar u_i(t) 表达能力有限；
2. dynamics loss 仍是 heat regression loss，没有专门优化 positive ranking；
3. readout 虽有 context，但 dynamic state 对 top-tail 区分不够强；
4. source / proxy / reaction 被压缩成 scalar term，可能损失细粒度信号。
```

---

## 10.2 top-K discovery 明显弱于 RDS v0

DynamicRDS v1：

```text
P@1000 = 0.101
NDCG@1000 = 0.115910
```

RDS v0：

```text
P@1000 = 0.132
NDCG@1000 = 0.136672
```

说明：

```text
RDS v0 的 hidden branch combination 对 top-tail candidate discovery 更有效。
```

可能解释：

```text
RDS v0 保留了 high-dimensional hidden branch representation；
DynamicRDS v1 将核心 dynamics 压缩到 scalar u / scalar du，
因此在 high-confidence top-K 排序上表达能力不足。
```

---

## 10.3 local_source 版本异常偏弱

`local_source_dynamic_rds_v0_seed42` 的 top-K 结果：

```text
P@100 = 0.010
P@1000 = 0.081
NDCG@1000 = 0.073847
```

说明：

```text
source branch 当前设计可能不稳定。
```

可能原因：

```text
1. source features 很复杂，不适合压缩为单一 scalar injection；
2. source term 没有非负约束或 gate，可能产生不稳定扰动；
3. 缺少 proxy prior 时，source injection 缺乏邻域 field context；
4. top-K discovery 更依赖 source/proxy 和 hidden branch interaction，而当前 v1 表达不足。
```

---

# 11. 对 DynamicRDS v2 的建议

## 11.1 升级为 vector-state DynamicRDS

当前：

```text
u_i(t) ∈ R
```

建议 v2 改为：

```text
u_i(t) ∈ R^d
```

例如：

```text
state_dim = 8 or 16
```

对应：

```text
R_i(t), S_i(t), P_i(t), D_i(t), Λ_i(t) ∈ R^d
```

Diffusion 仍保持 flux 形式：

```text
D_i(t) = Σ_j κ_ij [u_j(t) - u_i(t)]
```

这样既保留物理结构，又提高表达能力。

---

## 11.2 source / proxy term 输出 vector injection

当前：

```text
S_i(t) ∈ R
P_i(t) ∈ R
```

v2 建议：

```text
S_i(t) ∈ R^d
P_i(t) ∈ R^d
```

这样可以减少 source/proxy 信息被 scalar bottleneck 压缩的问题。

---

## 11.3 保留 high-dimensional context readout

v2 仍应保留：

```text
context_i(t) = MLP_context([x_i(t), s_i(t), p_i(t)])
```

最终 readout：

```text
score_i = Head([u_i, du_i, u_next_i, context_i])
```

这样动态状态负责物理演化，context 负责补充监督预测信息。

---

## 11.4 尝试 top-K-oriented model selection

当前 early stopping 使用：

```text
validation AUPRC
```

如果目标强调 top-K discovery，可额外尝试：

```text
monitor_metric = ndcg_at_1000
monitor_metric = precision_at_1000
```

这可能改善：

```text
P@1000
NDCG@1000
Enrichment@1000
```

但这属于训练策略优化，不应替代 v2 的结构改进。

---

## 11.5 可选加入 RDS-style residual

如果 v2 仍然 top-K 不足，可以加入：

```text
score = dynamic_score + residual_rds_score
```

或：

```text
score = Head([u, du, u_next, context, rds_hidden])
```

这可以结合：

```text
DynamicRDS 的 explicit dynamics
RDS v0 的 top-K discovery 表达能力
```

但第一步建议先做 vector-state v2，不要马上加入复杂 residual。

---

# 12. 当前阶段结论

DynamicRDS v1 的结论不是：

```text
DynamicRDS v1 已经全面超过 RDS / GRAND-style。
```

而是：

```text
DynamicRDS v1 成功验证了显式 knowledge-state dynamics 的可行性；
它在 AUROC / Spearman 等 global ranking 指标上已经达到甚至超过强 baseline；
但在 AUPRC 和 stable top-K discovery 上仍弱于 RDS v0 / GRAND-style。
```

因此，DynamicRDS v1 应被定位为：

```text
feasibility-validated explicit dynamics prototype
```

而不是最终版本。

---

# 13. 推荐写法

可以在后续报告或论文草稿中这样描述：

```text
DynamicRDS v1 validates the feasibility of explicitly evolving a learned knowledge-state variable over the KU graph. Under local+source+proxy inputs, DynamicRDS v1 achieved AUROC = 0.7199 and Spearman = 0.1018, surpassing both RDS v0 and GRAND-style baselines on these global ranking metrics. It also outperformed RDGNN-style on AUPRC, AUROC, Spearman, P@100, and NDCG@1000, suggesting that flux diffusion over an explicit state u_i(t) is more suitable than hidden-representation difference dynamics. However, DynamicRDS v1 still underperformed RDS v0 and GRAND-style on AUPRC and stable top-K discovery, indicating that the scalar-state formulation is likely too restrictive and motivating a vector-state DynamicRDS v2.
```

中文版本：

```text
DynamicRDS v1 验证了在 KU graph 上显式演化 learned knowledge-state variable 的可行性。在 local+source+proxy 输入下，DynamicRDS v1 取得 AUROC = 0.7199、Spearman = 0.1018，在这两个 global ranking 指标上超过 RDS v0 和 GRAND-style baseline。同时，相比 RDGNN-style，DynamicRDS v1 在 AUPRC、AUROC、Spearman、P@100 和 NDCG@1000 上更优，说明在显式状态 u_i(t) 上做 flux diffusion 比在 hidden representation 上做差分更适合当前任务。但 DynamicRDS v1 在 AUPRC 和稳定 top-K discovery 上仍弱于 RDS v0 和 GRAND-style，说明 scalar-state formulation 可能过于受限，因此后续应开发 vector-state DynamicRDS v2。
```

---

# 14. 最终一句话

DynamicRDS v1 是一个有意义的结果：

```text
它没有最终赢下所有指标，
但它证明了“知识作为显式物理状态变量进行图动力学演化”这条路线是可训练、有效、且优于通用 RDGNN-style hidden-dynamics baseline 的。
```

因此，下一步推荐进入：

```text
DynamicRDS v2:
  vector-state dynamics
  vector source/proxy injection
  true graph flux diffusion
  stronger top-K discovery optimization
```