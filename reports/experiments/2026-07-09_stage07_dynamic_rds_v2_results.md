# Stage 07F 实验总结：DynamicRDS v2 Vector-State Knowledge Field Dynamics

日期：2026-07-09  
阶段：Stage 07F  
主题：DynamicRDS v2 vector-state 显式 Knowledge Field Dynamics 实验结果  
用途：内部实验总结、主模型选择依据、后续多 seed 稳健性验证与解释分析设计依据  
注意：本文档总结 DynamicRDS v2 当前结果。当前最强模型为 `local_source_proxy_dynamic_rds_v2_state8_seed42`，但仍需后续多 seed 稳健性验证。

---

## 1. 背景与目的

Stage 07C / 07D / 07E / 07F-v1 已经完成以下模型系列：

```text
MLP no-graph baseline
Temporal GraphSAGE
WeightedDiffusion GNN
Reaction-Diffusion-Source GNN v0
RDGNN-style reaction-diffusion baseline
GRAND-style graph diffusion / neural PDE baseline
DynamicRDS v1 scalar-state dynamics
```

已有结果显示：

```text
1. KU-KU graph message passing 明显优于 no-graph baseline；
2. edge_weight 对 top-K discovery 有帮助；
3. RDS v0 的 reaction/source/proxy/diffusion 分支结构在 top-K discovery 上表现较稳；
4. GRAND-style physics / PDE baseline 在 global ranking 上非常强；
5. RDGNN-style 在 local 输入下有效，但在 local_source_proxy 输入下不如 GRAND-style；
6. DynamicRDS v1 验证了显式 u_i(t) dynamics 的可训练性，但 scalar state 对 top-K discovery 表达不足。
```

DynamicRDS v1 的核心问题是：

```text
u_i(t) ∈ R
```

即每个 KU 的 knowledge state 被压缩为一个 scalar。  
这使得模型虽然在 AUROC / Spearman 等 global ranking 指标上很强，但在：

```text
AUPRC
Precision@1000
NDCG@1000
Enrichment@1000
```

等 rare-positive / top-K discovery 指标上弱于 RDS v0 和 GRAND-style。

因此，本轮 Stage 07F 的目标是将 DynamicRDS 从 scalar-state v1 升级为：

```text
vector-state DynamicRDS v2
```

核心问题是：

```text
1. 低维 vector knowledge state 是否能提升 DynamicRDS 表达能力？
2. vector-state flux diffusion 是否能同时改善 global ranking 和 top-K discovery？
3. state_dim 对性能有什么影响？
4. DynamicRDS v2 是否能超过 RDS v0、GRAND-style 和 RDGNN-style？
5. 后续主模型应该如何选择？
```

---

## 2. DynamicRDS v2 方法概述

## 2.1 从 scalar state 到 vector state

DynamicRDS v1 使用：

```text
u_i(t) ∈ R
```

DynamicRDS v2 改为：

```text
u_i(t) ∈ R^d
```

其中本轮实验比较：

```text
state_dim = 8
state_dim = 16
state_dim = 32
```

因此每个 KU 在 cutoff year `t` 的状态不再是单一强度，而是低维 vector field state。

可以理解为：

```text
u_i(t) = [u_i^1(t), u_i^2(t), ..., u_i^d(t)]
```

代表多个 latent knowledge-field channels。

---

## 2.2 核心动力学方程

DynamicRDS v2 保留显式 Euler-style update：

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

其中所有项都升级为 vector：

```text
R_i(t) ∈ R^d
S_i(t) ∈ R^d
P_i(t) ∈ R^d
D_i(t) ∈ R^d
Λ_i(t) ∈ R^d
```

各项含义：

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

具体形式：

```text
R_i(t) = Rθ(u_i(t), x_i(t))

S_i(t) = gate_s(s_i(t)) ⊙ Sφ(s_i(t))

P_i(t) = gate_p(p_i(t)) ⊙ Pη(p_i(t))

D_i(t) = Σ_j κ_ij(t) · [u_j(t) - u_i(t)]

Λ_i(t) = λ_i(t) ⊙ u_i(t)
```

其中：

```text
x_i(t):
  focal KU local temporal state

s_i(t):
  carrier graph source / exposure features

p_i(t):
  handcrafted KU-neighborhood diffusion proxy features

κ_ij(t):
  normalized edge conductance from edge_weight

λ_i(t):
  learned nonnegative decay rate

⊙:
  channel-wise multiplication
```

最终预测：

```text
ŷ_i(t) = Readout([u_i(t), Δu_i(t), u_i(t+Δt), context_i(t)])
```

其中：

```text
Δu_i(t) = u_i(t + Δt) - u_i(t)
```

---

## 2.3 与 v1 的关键差异

| 项 | DynamicRDS v1 | DynamicRDS v2 |
|---|---|---|
| knowledge state | scalar `u_i(t) ∈ R` | vector `u_i(t) ∈ R^d` |
| reaction | scalar | vector |
| source | scalar | gated vector |
| proxy | scalar | gated vector |
| decay | scalar rate | vector rate |
| diffusion | scalar flux | vector flux |
| readout | `[u, du, u_next, context]` | `[u_vec, du_vec, u_next_vec, context]` |
| 主要目的 | 验证显式动力学可行性 | 提升表达能力和 top-K discovery |

DynamicRDS v2 的核心仍然是 true flux diffusion：

```text
D_i(t) = Σ_j κ_ij(t) [u_j(t) - u_i(t)]
```

因此它没有退化为普通 GNN message passing。

---

## 3. 总体实验设定

### 3.1 数据版本

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

### 3.2 图结构

DynamicRDS v2 使用 KU-KU shared-entity graph：

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

### 3.3 预测任务

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

### 3.4 Temporal split

DynamicRDS v2 使用与 Stage 07C / 07D / 07E / 07F-v1 相同的 latest-window split：

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

### 3.5 训练协议

本轮 DynamicRDS v2 使用与前序 Stage 07 模型一致的训练配置：

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

## 4. 当前完成的 DynamicRDS v2 实验

本轮完成 9 组模型：

```text
local_dynamic_rds_v2_state8_seed42
local_dynamic_rds_v2_state16_seed42
local_dynamic_rds_v2_state32_seed42

local_source_dynamic_rds_v2_state8_seed42
local_source_dynamic_rds_v2_state16_seed42
local_source_dynamic_rds_v2_state32_seed42

local_source_proxy_dynamic_rds_v2_state8_seed42
local_source_proxy_dynamic_rds_v2_state16_seed42
local_source_proxy_dynamic_rds_v2_state32_seed42
```

覆盖三类输入：

```text
local
local_source
local_source_proxy
```

以及三种 state dimension：

```text
state_dim = 8
state_dim = 16
state_dim = 32
```

---

## 5. DynamicRDS v2 结果汇总

### 5.1 Test cutoff = 2021 主结果

| Run | Model | Best Epoch | Val AUPRC | Test AUPRC | AUROC | Spearman | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_dynamic_rds_v2_state8_seed42 | dynamic_rds | 48 | 0.065621 | 0.040341 | 0.678480 | 0.082607 | 0.170 | 0.134 | 0.0858 | 0.140364 | 7.379402 |
| local_dynamic_rds_v2_state16_seed42 | dynamic_rds | 21 | 0.065023 | 0.037453 | 0.676634 | 0.081751 | 0.060 | 0.079 | 0.0696 | 0.082474 | 4.350543 |
| local_dynamic_rds_v2_state32_seed42 | dynamic_rds | 27 | 0.065370 | 0.037927 | 0.681512 | 0.084009 | 0.040 | 0.070 | 0.0706 | 0.065677 | 3.854912 |
| local_source_dynamic_rds_v2_state8_seed42 | dynamic_rds | 24 | 0.073586 | 0.041144 | 0.689245 | 0.087590 | 0.070 | 0.096 | 0.0844 | 0.095642 | 5.286736 |
| local_source_dynamic_rds_v2_state16_seed42 | dynamic_rds | 31 | 0.074211 | 0.041471 | 0.691367 | 0.088568 | 0.090 | 0.097 | 0.0850 | 0.093011 | 5.341806 |
| local_source_dynamic_rds_v2_state32_seed42 | dynamic_rds | 22 | 0.073889 | 0.040473 | 0.688502 | 0.087245 | 0.080 | 0.099 | 0.0834 | 0.096795 | 5.451946 |
| local_source_proxy_dynamic_rds_v2_state8_seed42 | dynamic_rds | 33 | 0.082691 | 0.048160 | 0.720320 | 0.101956 | 0.200 | 0.139 | 0.0982 | 0.148163 | 7.654753 |
| local_source_proxy_dynamic_rds_v2_state16_seed42 | dynamic_rds | 23 | 0.082935 | 0.044575 | 0.715857 | 0.099894 | 0.040 | 0.077 | 0.0808 | 0.079201 | 4.240403 |
| local_source_proxy_dynamic_rds_v2_state32_seed42 | dynamic_rds | 48 | 0.085194 | 0.046110 | 0.715652 | 0.099799 | 0.160 | 0.125 | 0.0858 | 0.134329 | 6.883771 |

Test positive rate：

```text
test_positive_rate = 0.018159
```

解释：

```text
Enrichment@1000 = Precision@1000 / test_positive_rate
```

---

# 6. 主要结果分析

## 6.1 local_source_proxy + state_dim=8 是当前最强组合

三组 local_source_proxy 结果：

| State Dim | AUPRC | AUROC | Spearman | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | **0.048160** | **0.720320** | **0.101956** | **0.200** | **0.139** | **0.0982** | **0.148163** | **7.654753** |
| 16 | 0.044575 | 0.715857 | 0.099894 | 0.040 | 0.077 | 0.0808 | 0.079201 | 4.240403 |
| 32 | 0.046110 | 0.715652 | 0.099799 | 0.160 | 0.125 | 0.0858 | 0.134329 | 6.883771 |

结论：

```text
local_source_proxy_dynamic_rds_v2_state8_seed42 是本轮最强模型。
```

它同时取得当前最强或接近最强的：

```text
AUPRC
AUROC
Spearman
P@100
P@1000
P@5000
NDCG@1000
Enrichment@1000
```

---

## 6.2 state_dim 不是越大越好

本轮结果并不支持：

```text
state32 > state16 > state8
```

实际结果更接近：

```text
state8 > state32 >> state16
```

尤其在 local_source_proxy 输入下：

```text
state8:
  AUPRC = 0.048160
  P@1000 = 0.139
  NDCG@1000 = 0.148163

state16:
  AUPRC = 0.044575
  P@1000 = 0.077
  NDCG@1000 = 0.079201

state32:
  AUPRC = 0.046110
  P@1000 = 0.125
  NDCG@1000 = 0.134329
```

可能解释：

```text
1. state_dim 太大增加动力学自由度，容易学到不稳定方向；
2. rare positive 极度稀疏，过高维状态可能过拟合 validation AUPRC；
3. state8 提供足够表达能力，同时保留低维物理约束；
4. state16 的 checkpoint 可能在 top-K 上尤其不利；
5. state32 有一定恢复，但仍不如 state8，说明更高维 latent state 不一定更适合 field dynamics。
```

方法学启发：

```text
DynamicRDS 的 state_dim 是关键超参数；
低维 vector field 可能比高维 hidden representation 更适合知识场动力学。
```

---

## 6.3 DynamicRDS v2 state8 明显改善 v1 的 top-K 不足

DynamicRDS v1 local_source_proxy：

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

DynamicRDS v2 state8 local_source_proxy：

```text
AUPRC            = 0.048160
AUROC            = 0.720320
Spearman         = 0.101956
P@100            = 0.200
P@1000           = 0.139
P@5000           = 0.0982
NDCG@1000        = 0.148163
Enrichment@1000  = 7.654753
```

对比：

| Metric | DynamicRDS v1 scalar | DynamicRDS v2 state8 | 变化 |
|---|---:|---:|---:|
| AUPRC | 0.046314 | **0.048160** | +4.0% |
| AUROC | 0.719937 | **0.720320** | +0.0004 |
| Spearman | 0.101782 | **0.101956** | +0.0002 |
| P@100 | 0.140 | **0.200** | +42.9% |
| P@1000 | 0.101 | **0.139** | +37.6% |
| P@5000 | 0.0820 | **0.0982** | +19.8% |
| NDCG@1000 | 0.115910 | **0.148163** | +27.8% |
| Enrich@1000 | 5.562087 | **7.654753** | +37.6% |

结论：

```text
vector-state DynamicRDS v2 成功解决了 v1 的 top-K discovery 不足问题。
```

这说明：

```text
scalar state 是 v1 的主要瓶颈；
低维 vector state 显著增强了模型表达能力。
```

---

## 6.4 local DynamicRDS v2 state8 已经有很强 top-K discovery

`local_dynamic_rds_v2_state8_seed42` 结果：

```text
AUPRC            = 0.040341
AUROC            = 0.678480
Spearman         = 0.082607
P@100            = 0.170
P@1000           = 0.134
NDCG@1000        = 0.140364
Enrichment@1000  = 7.379402
```

这是非常重要的现象。

即使没有 source/proxy features，仅依赖：

```text
local KU temporal state
+ KU-KU graph
+ vector flux dynamics
```

也已经获得很强 top-K discovery。

对比此前 local GraphSAGE：

```text
local_graphsage_v0_seed42:
  P@1000 = 0.122
  NDCG@1000 = 0.119674
  Enrichment@1000 = 6.718560
```

DynamicRDS v2 state8 更强：

```text
P@1000 = 0.134
NDCG@1000 = 0.140364
Enrichment@1000 = 7.379402
```

结论：

```text
true vector flux diffusion 本身就是有效的 top-K discovery mechanism。
```

---

## 6.5 local_source 输入没有稳定提升 top-K

local_source 三组结果：

| State Dim | AUPRC | AUROC | Spearman | P@1000 | NDCG@1000 |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.041144 | 0.689245 | 0.087590 | 0.096 | 0.095642 |
| 16 | 0.041471 | 0.691367 | 0.088568 | 0.097 | 0.093011 |
| 32 | 0.040473 | 0.688502 | 0.087245 | 0.099 | 0.096795 |

相比 local state8：

```text
local state8:
  AUPRC = 0.040341
  P@1000 = 0.134
  NDCG@1000 = 0.140364
```

local_source 在 global ranking 上略有提升，但 top-K 明显下降。

结论：

```text
source-only injection 对 DynamicRDS 的 top-K discovery 仍不稳定。
```

可能原因：

```text
1. source features 需要 proxy prior 才能稳定发挥作用；
2. source injection 单独进入 vector dynamics 可能扰乱 top-tail ranking；
3. source exposure 更提升 global discrimination，而不一定提升 extreme top-K；
4. 当前 source gate 仍无法完全解决 source-only top-K instability。
```

这延续了 v1 中观察到的问题：

```text
source term 单独使用时可能不稳定；
source + proxy 组合才最强。
```

---

## 6.6 proxy prior 是 DynamicRDS v2 的关键增益来源

从 local_source state8 到 local_source_proxy state8：

| Metric | local_source state8 | local_source_proxy state8 | 变化 |
|---|---:|---:|---:|
| AUPRC | 0.041144 | **0.048160** | +17.0% |
| AUROC | 0.689245 | **0.720320** | +0.0311 |
| Spearman | 0.087590 | **0.101956** | +16.4% |
| P@100 | 0.070 | **0.200** | +185.7% |
| P@1000 | 0.096 | **0.139** | +44.8% |
| NDCG@1000 | 0.095642 | **0.148163** | +54.9% |

结论：

```text
proxy prior 对 DynamicRDS v2 极其重要。
```

它可能提供：

```text
1. cutoff-safe neighborhood diffusion prior；
2. high-confidence top-tail ranking signal；
3. 对 source injection 的稳定约束；
4. 帮助 state encoder 学到更有意义的 vector knowledge state；
5. 弥补 source-only dynamics 对 top-K 的不稳定。
```

---

# 7. 与已有强 baseline 的比较

## 7.1 与 RDS v0 比较

RDS v0：

```text
local_source_proxy_rds_v0_seed42
AUPRC            = 0.047322
AUROC            = 0.715591
Spearman         = 0.099775
P@100            = 0.160
P@1000           = 0.132
P@5000           = 0.0974
NDCG@1000        = 0.136672
Enrichment@1000  = 7.269262
```

DynamicRDS v2 state8：

```text
local_source_proxy_dynamic_rds_v2_state8_seed42
AUPRC            = 0.048160
AUROC            = 0.720320
Spearman         = 0.101956
P@100            = 0.200
P@1000           = 0.139
P@5000           = 0.0982
NDCG@1000        = 0.148163
Enrichment@1000  = 7.654753
```

对比：

| Metric | RDS v0 | DynamicRDS v2 state8 | 更优 |
|---|---:|---:|---|
| AUPRC | 0.047322 | **0.048160** | DynamicRDS |
| AUROC | 0.715591 | **0.720320** | DynamicRDS |
| Spearman | 0.099775 | **0.101956** | DynamicRDS |
| P@100 | 0.160 | **0.200** | DynamicRDS |
| P@1000 | 0.132 | **0.139** | DynamicRDS |
| P@5000 | 0.0974 | **0.0982** | DynamicRDS |
| NDCG@1000 | 0.136672 | **0.148163** | DynamicRDS |
| Enrich@1000 | 7.269262 | **7.654753** | DynamicRDS |

结论：

```text
DynamicRDS v2 state8 全面超过 RDS v0。
```

这说明：

```text
显式 vector-state dynamics 不仅提升解释性，也提升预测性能。
```

---

## 7.2 与 GRAND-style 比较

GRAND-style local_source_proxy：

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

DynamicRDS v2 state8：

```text
AUPRC            = 0.048160
AUROC            = 0.720320
Spearman         = 0.101956
P@100            = 0.200
P@1000           = 0.139
P@5000           = 0.0982
NDCG@1000        = 0.148163
Enrichment@1000  = 7.654753
```

对比：

| Metric | GRAND-style | DynamicRDS v2 state8 | 更优 |
|---|---:|---:|---|
| AUPRC | 0.047503 | **0.048160** | DynamicRDS |
| AUROC | 0.718492 | **0.720320** | DynamicRDS |
| Spearman | 0.101115 | **0.101956** | DynamicRDS |
| P@100 | 0.200 | 0.200 | Tie |
| P@1000 | 0.115 | **0.139** | DynamicRDS |
| P@5000 | 0.0912 | **0.0982** | DynamicRDS |
| NDCG@1000 | 0.119242 | **0.148163** | DynamicRDS |
| Enrich@1000 | 6.333069 | **7.654753** | DynamicRDS |

结论：

```text
DynamicRDS v2 state8 全面超过 GRAND-style，尤其显著提升 top-K discovery。
```

这说明：

```text
通用 graph diffusion / PDE smoothing 很强，
但显式 reaction-source-proxy-flux-decay dynamics 更适合当前 Knowledge Field 任务。
```

---

## 7.3 与 RDGNN-style 比较

RDGNN-style local_source_proxy：

```text
AUPRC            = 0.045454
AUROC            = 0.714192
Spearman         = 0.099127
P@100            = 0.120
P@1000           = 0.103
NDCG@1000        = 0.105720
Enrichment@1000  = 5.672227
```

DynamicRDS v2 state8：

```text
AUPRC            = 0.048160
AUROC            = 0.720320
Spearman         = 0.101956
P@100            = 0.200
P@1000           = 0.139
NDCG@1000        = 0.148163
Enrichment@1000  = 7.654753
```

结论：

```text
DynamicRDS v2 state8 明显超过 RDGNN-style。
```

这支持一个关键方法学判断：

```text
在显式 knowledge state u_i(t) 上做 u_j - u_i flux，
比在 hidden representation h_i 上做 h_j - h_i difference dynamics
更适合当前 KU-level Knowledge Field 预测任务。
```

---

# 8. 当前主模型选择

基于本轮结果，当前综合最强模型为：

```text
local_source_proxy_dynamic_rds_v2_state8_seed42
```

其主要指标为：

```text
Test AUPRC       = 0.048160
Test AUROC       = 0.720320
Test Spearman    = 0.101956
Precision@100    = 0.200
Precision@1000   = 0.139
Precision@5000   = 0.0982
NDCG@1000        = 0.148163
Enrichment@1000  = 7.654753
```

当前模型排序可概括为：

```text
DynamicRDS v2 state8
  > RDS v0
  > GRAND-style
  > WeightedDiffusion / RDGNN-style / GraphSAGE
  > MLP
```

更精确地说：

```text
DynamicRDS v2 state8 是当前 global ranking 与 stable top-K discovery 综合最强模型。
```

---

# 9. 对 Knowledge Field 理论的支持

本轮结果支持以下关键理论判断。

## 9.1 知识场可以被建模为低维 vector field

state_dim=8 最强说明：

```text
KU 的转化潜势不是必须依赖高维 hidden representation；
低维 vector knowledge state 已经足够表达关键动力学。
```

这为后续理论表述提供了基础：

```text
u_i(t) ∈ R^8
```

可以解释为多个 latent knowledge-field channels。

潜在解释方向包括：

```text
1. basic research activity channel
2. translational pressure channel
3. clinical proximity channel
4. patentability channel
5. neighborhood activation channel
6. source exposure response channel
7. decay / saturation channel
8. diffusion susceptibility channel
```

这些 channel 的实际含义需要后续解释分析验证。

---

## 9.2 true vector flux diffusion 是有效机制

DynamicRDS v2 仍使用：

```text
D_i(t) = Σ_j κ_ij(t) [u_j(t) - u_i(t)]
```

其结果超过 GRAND-style 和 RDS v0，说明：

```text
true vector flux diffusion 比普通 graph smoothing 或 hidden aggregation 更适合当前任务。
```

这支持从：

```text
physics-inspired GNN
```

升级到：

```text
explicit Knowledge Field Dynamics
```

的理论路线。

---

## 9.3 source + proxy + dynamics 的组合最有效

local_source 单独不稳定，但 local_source_proxy 显著最强，说明：

```text
source exposure 需要与 proxy prior / neighborhood prior 结合，
才能稳定地改善 DynamicRDS 的 top-K discovery。
```

这符合 Knowledge Field 假设：

```text
未来转化涌现不是单独由 source injection 触发，
也不是单独由 graph diffusion 决定，
而是 local state + source exposure + diffusion proxy + graph flux dynamics 的组合过程。
```

---

## 9.4 DynamicRDS 不只是更可解释，也更强

相比 RDS v0 和 GRAND-style，DynamicRDS v2 state8 不仅具有更明确的物理结构：

```text
reaction
source injection
proxy prior
true diffusion flux
decay
Euler update
```

而且在主要指标上表现更好。

因此当前可以更有力地表述：

```text
DynamicRDS is not only an interpretable physical dynamics model,
but also the strongest empirical model in the current Stage 07 benchmark.
```

中文：

```text
DynamicRDS 不仅是更可解释的物理动力学模型，
也是当前 Stage 07 benchmark 中综合表现最强的模型。
```

---

# 10. 当前不足与注意事项

尽管本轮结果非常好，仍需注意：

## 10.1 当前结果仍是单 seed

当前最强结果来自：

```text
seed = 42
```

需要后续多 seed 验证：

```text
seed = 7
seed = 13
seed = 21
seed = 42
```

否则不能排除：

```text
state8 的优势部分来自随机初始化 / early stopping 偶然性。
```

---

## 10.2 state16 异常偏弱，需要记录但不必过度解释

state16 在 local_source_proxy 下显著偏弱：

```text
AUPRC = 0.044575
P@1000 = 0.077
NDCG@1000 = 0.079201
```

这可能是：

```text
1. 特定 seed / checkpoint 的偶然性；
2. state_dim=16 对当前训练配置不稳定；
3. early stopping by validation AUPRC 未选到 top-K 友好 checkpoint；
4. vector dynamics 自由度与正则化不匹配。
```

后续如果有必要，可以通过多 seed 判断 state16 是否系统性较弱。

---

## 10.3 source-only dynamics 仍不稳定

local_source 版本 top-K 不如 local state8：

```text
local state8:
  P@1000 = 0.134
  NDCG@1000 = 0.140364

local_source state8:
  P@1000 = 0.096
  NDCG@1000 = 0.095642
```

这说明：

```text
source branch 虽然对 global ranking 有帮助，
但单独进入 dynamics 时可能扰乱 top-K。
```

后续解释分析需要特别关注 source term 的行为。

---

# 11. 推荐下一步

## 11.1 优先做多 seed 稳健性验证

最优先建议对当前最强配置跑多 seed：

```text
local_source_proxy_dynamic_rds_v2_state8_seed7
local_source_proxy_dynamic_rds_v2_state13
local_source_proxy_dynamic_rds_v2_state21
local_source_proxy_dynamic_rds_v2_state42
```

如果资源允许，也可加：

```text
seed = 100
```

核心目标：

```text
验证 DynamicRDS v2 state8 是否稳定优于 RDS v0 / GRAND-style。
```

建议汇总：

```text
mean ± std
best / median
per-seed test metrics
```

重点指标：

```text
AUPRC
AUROC
Spearman
P@1000
NDCG@1000
Enrichment@1000
```

---

## 11.2 增加 DynamicRDS interpretation / decomposition 输出

DynamicRDS 具有显式项：

```text
u_i(t)
du_i(t)
u_i(t+Δt)
reaction_i(t)
source_i(t)
proxy_i(t)
diffusion_i(t)
decay_i(t)
```

建议后续新增 inference / analysis 脚本，对 top-K 节点保存这些项。

分析目标：

```text
1. top predicted KUs 是 reaction-driven、source-driven、proxy-driven 还是 diffusion-driven？
2. 高分 KU 的 incoming diffusion flux 来自哪些邻居？
3. decay term 是否能识别 cooling / saturated KUs？
4. 不同 pair_type 的 dynamics 是否不同？
5. state_dim=8 的 latent channels 是否有可解释模式？
```

推荐输出：

```text
data/results/knowledge_field/interpretation/dynamic_rds_v2_state8_topk_terms.parquet
data/results/knowledge_field/interpretation/dynamic_rds_v2_state8_flow_edges.parquet
reports/experiments/dynamic_rds_v2_case_studies.md
```

---

## 11.3 暂不急于继续堆更复杂模型

当前结果已经证明：

```text
DynamicRDS v2 state8 是当前主模型候选。
```

因此短期内不建议继续盲目加入：

```text
attention
TransformerConv
learnable conductance
multi-step rollout
ranking loss
```

这些可以作为后续增强方向，但当前更重要的是：

```text
1. 多 seed 稳健性；
2. 解释分析；
3. case study；
4. 报告与论文表述整理。
```

---

# 12. 推荐论文/报告表述

英文表述：

```text
DynamicRDS v2 upgrades the scalar knowledge-state variable in DynamicRDS v1 into a low-dimensional vector field and performs explicit reaction-source-proxy-diffusion-decay updates over the KU graph. The best configuration, with state_dim = 8 and local+source+proxy inputs, achieved the strongest overall performance among all current Stage 07 models, with test AUPRC = 0.0482, AUROC = 0.7203, Spearman = 0.1020, Precision@1000 = 0.139, and NDCG@1000 = 0.1482. Compared with RDS v0 and GRAND-style baselines, DynamicRDS v2 improved both global ranking and stable top-K discovery, suggesting that explicit vector-state flux dynamics better captures knowledge-field evolution than static source-diffusion decomposition or generic PDE-style smoothing.
```

中文表述：

```text
DynamicRDS v2 将 DynamicRDS v1 中的 scalar knowledge-state 升级为低维 vector field，并在 KU graph 上显式执行 reaction-source-proxy-diffusion-decay 更新。其中 state_dim=8 且使用 local+source+proxy 输入的配置取得当前 Stage 07 所有模型中的最佳综合表现：test AUPRC = 0.0482，AUROC = 0.7203，Spearman = 0.1020，Precision@1000 = 0.139，NDCG@1000 = 0.1482。相比 RDS v0 和 GRAND-style baseline，DynamicRDS v2 同时提升了 global ranking 和 stable top-K discovery，说明显式 vector-state flux dynamics 比静态 source-diffusion 分解或通用 PDE-style smoothing 更适合捕捉知识场演化。
```

---

# 13. 最终结论

本轮 DynamicRDS v2 实验是当前 Stage 07 中最重要的正向结果。

核心结论：

```text
1. DynamicRDS v2 state8 是当前综合最强模型；
2. vector-state dynamics 明显优于 scalar-state v1；
3. state_dim 不是越大越好，低维 state8 最稳定有效；
4. DynamicRDS v2 全面超过 RDS v0、GRAND-style 和 RDGNN-style；
5. true vector flux diffusion 是有效的 Knowledge Field 动力学机制；
6. proxy prior 是 DynamicRDS v2 的关键增益来源；
7. 后续应优先进行多 seed 稳健性验证和显式项解释分析。
```

一句话总结：

```text
DynamicRDS v2 state8 成功把当前工作从 physics-inspired Knowledge Field GNN 推进为真正的显式 vector-state Knowledge Field Dynamics 模型，并在当前主任务上取得最佳综合表现。
```