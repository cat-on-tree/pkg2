# Stage 07 实验总结：Temporal Knowledge Field GNN 与 Reaction-Diffusion-Source 模型

日期：2026-07-07  
阶段：Stage 07A / 07B / 07C / 07D  
主题：Temporal Knowledge Field GNN、WeightedDiffusion 与 Reaction-Diffusion-Source 模型结果总结  
用途：内部实验总结与后续 DynamicRDS / physics-style baseline 设计依据  
注意：本文档总结当前 Stage 07 v0 结果，后续 DynamicRDS、RDGNN-style、GRAND-style baseline 尚未纳入本轮结果。

---

## 1. 背景与目的

Stage 07 的核心目标是从 Stage 06 的 diagnostic baselines 进一步推进到正式的 Knowledge Field GNN 模型。

Stage 06 已经证明：

```text
1. focal KU temporal state 有预测价值；
2. carrier graph source / exposure 在 KU full 之外有独立信息；
3. KU-KU neighborhood diffusion proxy 在 local KU state 和 carrier source 之外继续提供增量；
4. 未来 translation emergence 更像是 local state + source exposure + neighborhood diffusion 的组合过程。
```

因此，Stage 07 不再重复完整的传统 tabular baseline grid，而是开始构建：

```text
Temporal Knowledge Field GNN
```

并进一步测试带有物理归纳偏置的：

```text
WeightedDiffusion GNN
Reaction-Diffusion-Source GNN
```

Stage 07 当前需要回答的问题是：

```text
1. KU-KU graph message passing 是否优于 no-graph MLP？
2. 显式 edge_weight diffusion 是否优于普通 GraphSAGE？
3. local reaction、carrier source、diffusion proxy、graph diffusion 的结构化组合是否有效？
4. 当前模型是否已经接近“知识作为物理量的动力学变化”？
5. 后续是否需要进一步开发 DynamicRDS 和 physics / PDE-style baselines？
```

---

## 2. 总体实验设定

### 2.1 数据版本

当前 Stage 07 使用 diabetes long-window formal benchmark：

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

Stage 07 使用 KU-KU shared-entity graph：

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

### 2.3 预测单位

预测单位仍然是：

```text
KU × cutoff_year
```

即每个训练样本表示：

```text
某个 Knowledge Unit 在某个 cutoff year 的状态
```

并预测其未来 3 年是否出现 translation emergence / future translation heat。

---

### 2.4 任务

当前 Stage 07 第一主任务为：

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

### 2.5 Temporal split

Stage 07 使用 latest-window split：

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

不能使用：

```text
cutoff years = 2022, 2023, 2024
```

原因是：

```text
cutoff 2022 需要 future 2023-2025
cutoff 2023 需要 future 2024-2026
cutoff 2024 需要 future 2025-2027
```

而当前数据截止到 2024。

---

### 2.6 训练协议

当前模型统一使用相同基本训练配置：

```text
hidden_dim = 128
num_layers = 2
num_neighbors = 15 10
batch_size = 16384 on A100 / 4096 on local debugging
optimizer = AdamW
lr = 1e-3
weight_decay = 1e-4
dropout = 0.1
loss = SmoothL1
monitor_metric = validation AUPRC
patience = 8
epochs = 50
seed = 42
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

## 3. 当前完成的 Stage 07 模型

当前已完成 7 组主要模型：

```text
local_source_mlp_v0_seed42
local_graphsage_v0_seed42
local_source_graphsage_v0_seed42
local_source_proxy_graphsage_v0_seed42
local_weighted_diffusion_v0_seed42
local_source_proxy_weighted_diffusion_v0_seed42
local_source_proxy_rds_v0_seed42
```

对应模型族：

```text
1. MLP no-graph baseline
2. Temporal GraphSAGE
3. WeightedDiffusion GNN
4. Reaction-Diffusion-Source GNN
```

---

## 4. 模型定义与实验目的

### 4.1 MLP no-graph baseline

MLP baseline 使用同样的 node features，但不使用 KU-KU graph。

形式：

```text
ŷ_i(t) = MLP([x_i(t), s_i(t)])
```

实验目的：

```text
判断 KU-KU graph message passing 是否提供 no-graph features 之外的增量。
```

对应 run：

```text
local_source_mlp_v0_seed42
```

---

### 4.2 Temporal GraphSAGE

GraphSAGE 使用 KU-KU graph topology 做 message passing。

形式：

```text
h_i(t) = GraphSAGEθ(feature_i(t), A)
ŷ_i(t) = Head(h_i(t))
```

其中：

```text
feature_i(t) = x_i(t)
```

或：

```text
feature_i(t) = [x_i(t), s_i(t)]
```

或：

```text
feature_i(t) = [x_i(t), s_i(t), p_i(t)]
```

实验目的：

```text
判断 learned graph diffusion / message passing 是否有效。
```

对应 runs：

```text
local_graphsage_v0_seed42
local_source_graphsage_v0_seed42
local_source_proxy_graphsage_v0_seed42
```

---

### 4.3 WeightedDiffusion GNN

WeightedDiffusion 显式使用 KU-KU graph 的 edge_weight。

普通 GraphSAGE 主要使用 graph topology，而 WeightedDiffusion 进一步使用：

```text
edge_weight
```

形式上更接近：

```text
weighted neighborhood aggregation
```

实验目的：

```text
判断 shared-entity edge_weight 是否应进入 diffusion operator。
```

对应 runs：

```text
local_weighted_diffusion_v0_seed42
local_source_proxy_weighted_diffusion_v0_seed42
```

---

### 4.4 Reaction-Diffusion-Source GNN v0

Reaction-Diffusion-Source GNN v0 将输入显式分为：

```text
local reaction branch
carrier source branch
proxy branch
graph diffusion branch
```

当前形式可以概括为：

```text
h_i(t)
=
Rθ(x_i(t))
+
Sφ(s_i(t))
+
Pη(p_i(t))
+
Dψ(N_i(t), A)
```

最终：

```text
ŷ_i(t) = Head(h_i(t))
```

其中：

```text
Rθ:
  local reaction branch

Sφ:
  carrier source / exposure branch

Pη:
  handcrafted diffusion proxy branch

Dψ:
  weighted graph diffusion / message passing branch
```

实验目的：

```text
判断 reaction + source + proxy + diffusion 的结构化组合是否优于通用 GNN。
```

对应 run：

```text
local_source_proxy_rds_v0_seed42
```

---

## 5. 当前 Stage 07 结果汇总

### 5.1 Test cutoff = 2021 主结果

| Run | Model | Best Epoch | Val AUPRC | Test AUPRC | AUROC | Spearman | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_source_graphsage_v0_seed42 | GraphSAGE | 25 | 0.076761 | 0.040983 | 0.694941 | 0.090222 | 0.090 | 0.103 | 0.0816 | 0.100399 | 5.672227 |
| local_source_mlp_v0_seed42 | MLP | 14 | 0.066106 | 0.037086 | 0.671929 | 0.079578 | 0.010 | 0.070 | 0.0786 | 0.064216 | 3.854912 |
| local_graphsage_v0_seed42 | GraphSAGE | 20 | 0.066760 | 0.040941 | 0.684940 | 0.085590 | 0.120 | 0.122 | 0.0868 | 0.119674 | 6.718560 |
| local_source_proxy_graphsage_v0_seed42 | GraphSAGE | 33 | 0.083358 | 0.045131 | 0.713521 | 0.098816 | 0.050 | 0.097 | 0.0904 | 0.090821 | 5.341806 |
| local_weighted_diffusion_v0_seed42 | WeightedDiffusion | 19 | 0.066820 | 0.040067 | 0.682789 | 0.084595 | 0.120 | 0.107 | 0.0798 | 0.105784 | 5.892508 |
| local_source_proxy_weighted_diffusion_v0_seed42 | WeightedDiffusion | 25 | 0.082561 | 0.045404 | 0.712645 | 0.098410 | 0.180 | 0.121 | 0.0898 | 0.122069 | 6.663490 |
| local_source_proxy_rds_v0_seed42 | ReactionDiffusionSource | 40 | 0.084467 | 0.047322 | 0.715591 | 0.099775 | 0.160 | 0.132 | 0.0974 | 0.136672 | 7.269262 |

Test positive rate：

```text
test_positive_rate = 0.018159
```

解释：

```text
Enrichment@1000 = Precision@1000 / test_positive_rate
```

因此，当前最佳模型：

```text
local_source_proxy_rds_v0_seed42
```

的：

```text
Precision@1000 = 0.132
Enrichment@1000 = 7.269262
```

表示模型 top 1000 候选中的 future translation emergence rate 是背景 positive rate 的约 7.27 倍。

---

## 6. 主要结果分析

## 6.1 GraphSAGE 明显优于 no-graph MLP

对比相同输入：

```text
local_source_graphsage_v0_seed42
local_source_mlp_v0_seed42
```

结果：

| Metric | MLP | GraphSAGE | 变化 |
|---|---:|---:|---:|
| Test AUPRC | 0.037086 | 0.040983 | +10.5% |
| AUROC | 0.671929 | 0.694941 | +0.0230 |
| Spearman | 0.079578 | 0.090222 | +13.4% |
| P@100 | 0.010 | 0.090 | 9.0x |
| P@1000 | 0.070 | 0.103 | +47.1% |
| NDCG@1000 | 0.064216 | 0.100399 | +56.3% |
| Enrich@1000 | 3.854912 | 5.672227 | +47.1% |

结论：

```text
KU-KU graph message passing 提供了 no-graph feature model 之外的明显增量。
```

这说明：

```text
未来 translational emergence 不只是由 focal KU 自身状态和 carrier source features 决定，
KU-KU graph neighborhood 也包含可学习的 diffusion-like signal。
```

---

## 6.2 local-only GraphSAGE 的 top-K 表现很强

`local_graphsage_v0_seed42` 在若干 top-K 指标上表现较强：

```text
P@100 = 0.120
P@1000 = 0.122
NDCG@1000 = 0.119674
Enrichment@1000 = 6.718560
```

它的整体 AUPRC / AUROC 不如 local_source_proxy GraphSAGE 或 RDS，但 top-K discovery 较强。

解释：

```text
local temporal state + KU-KU graph topology 已经能识别一批高置信 future translation candidates。
```

这与 Stage 06 中的观察一致：

```text
global discrimination 和 extreme top-K discovery 可能依赖不同信号。
```

---

## 6.3 source / proxy 特征提升整体排序，但普通 GraphSAGE 未能充分保持 top-K purity

对比：

```text
local_graphsage_v0_seed42
local_source_proxy_graphsage_v0_seed42
```

结果：

| Metric | local GraphSAGE | local_source_proxy GraphSAGE |
|---|---:|---:|
| Test AUPRC | 0.040941 | 0.045131 |
| AUROC | 0.684940 | 0.713521 |
| Spearman | 0.085590 | 0.098816 |
| P@100 | 0.120 | 0.050 |
| P@1000 | 0.122 | 0.097 |
| NDCG@1000 | 0.119674 | 0.090821 |

结论：

```text
source / proxy features 明显提升 global ranking metrics，
但普通 GraphSAGE 在 local_source_proxy 输入下没有保持最强 top-K precision。
```

可能解释：

```text
1. source / proxy 特征引入更广泛的 global propensity signal；
2. GraphSAGE 的 generic aggregation 未能将 source/proxy/diffusion 分支有效解耦；
3. top-tail discovery 需要更结构化的 source / diffusion handling。
```

这为 RDS 模型提供了动机：

```text
不要简单拼接所有 features；
而应显式分解 local reaction、source injection、proxy signal 和 graph diffusion。
```

---

## 6.4 WeightedDiffusion 证明 edge_weight 对 top-K discovery 有价值

对比：

```text
local_source_proxy_graphsage_v0_seed42
local_source_proxy_weighted_diffusion_v0_seed42
```

结果：

| Metric | GraphSAGE | WeightedDiffusion | 变化 |
|---|---:|---:|---:|
| Test AUPRC | 0.045131 | 0.045404 | +0.6% |
| AUROC | 0.713521 | 0.712645 | -0.0009 |
| Spearman | 0.098816 | 0.098410 | -0.0004 |
| P@100 | 0.050 | 0.180 | +260% |
| P@1000 | 0.097 | 0.121 | +24.7% |
| NDCG@1000 | 0.090821 | 0.122069 | +34.4% |
| Enrich@1000 | 5.341806 | 6.663490 | +24.7% |

结论：

```text
显式使用 edge_weight 对 top-K discovery 非常有帮助。
```

虽然 WeightedDiffusion 的 AUROC / Spearman 与 GraphSAGE 接近，但它显著改善：

```text
P@100
P@1000
NDCG@1000
Enrichment@1000
```

这说明：

```text
shared-entity edge_weight 不只是构图辅助信息，
而应进入 diffusion operator。
```

---

## 6.5 WeightedDiffusion 单独在 local 输入下未超过 GraphSAGE

对比：

```text
local_graphsage_v0_seed42
local_weighted_diffusion_v0_seed42
```

结果：

| Metric | local GraphSAGE | local WeightedDiffusion |
|---|---:|---:|
| Test AUPRC | 0.040941 | 0.040067 |
| AUROC | 0.684940 | 0.682789 |
| Spearman | 0.085590 | 0.084595 |
| P@100 | 0.120 | 0.120 |
| P@1000 | 0.122 | 0.107 |
| NDCG@1000 | 0.119674 | 0.105784 |

结论：

```text
edge_weight 单独使用时不一定优于 GraphSAGE；
但在 local_source_proxy 输入下，weighted diffusion 对 top-K 有明显提升。
```

这说明：

```text
edge-weight diffusion 需要与 source / proxy / local field features 结合，
才能充分发挥作用。
```

这与 Knowledge Field 假设一致：

```text
未来 translation emergence 不是单独的 graph diffusion，
而是 local reaction + carrier source + diffusion context 的组合过程。
```

---

## 6.6 Reaction-Diffusion-Source GNN 当前整体最优

当前最佳模型是：

```text
local_source_proxy_rds_v0_seed42
```

结果：

```text
Test AUPRC       = 0.047322
Test AUROC       = 0.715591
Test Spearman    = 0.099775
Precision@100    = 0.160
Precision@1000   = 0.132
Precision@5000   = 0.0974
NDCG@1000        = 0.136672
Enrichment@1000  = 7.269262
```

它同时在：

```text
global ranking metrics
top-K discovery metrics
```

上取得当前最好或接近最好表现。

---

## 7. RDS v0 与主要 baseline 的比较

### 7.1 RDS v0 vs MLP

对比：

```text
local_source_mlp_v0_seed42
local_source_proxy_rds_v0_seed42
```

| Metric | MLP | RDS v0 | 变化 |
|---|---:|---:|---:|
| Test AUPRC | 0.037086 | 0.047322 | +27.6% |
| AUROC | 0.671929 | 0.715591 | +0.0437 |
| Spearman | 0.079578 | 0.099775 | +25.4% |
| P@100 | 0.010 | 0.160 | 16.0x |
| P@1000 | 0.070 | 0.132 | +88.6% |
| NDCG@1000 | 0.064216 | 0.136672 | +112.8% |
| Enrich@1000 | 3.854912 | 7.269262 | +88.6% |

解释：

```text
从 no-graph MLP 到 RDS v0，模型不仅提升整体排序，
也显著提升 high-confidence top-K candidate discovery。
```

---

### 7.2 RDS v0 vs local_source_proxy GraphSAGE

对比：

```text
local_source_proxy_graphsage_v0_seed42
local_source_proxy_rds_v0_seed42
```

| Metric | GraphSAGE | RDS v0 | 变化 |
|---|---:|---:|---:|
| Test AUPRC | 0.045131 | 0.047322 | +4.9% |
| AUROC | 0.713521 | 0.715591 | +0.0021 |
| Spearman | 0.098816 | 0.099775 | +0.0010 |
| P@100 | 0.050 | 0.160 | +220% |
| P@1000 | 0.097 | 0.132 | +36.1% |
| P@5000 | 0.0904 | 0.0974 | +7.7% |
| NDCG@1000 | 0.090821 | 0.136672 | +50.5% |
| Enrich@1000 | 5.341806 | 7.269262 | +36.1% |

解释：

```text
RDS v0 相比普通 GraphSAGE 的主要优势集中在 top-K discovery。
```

这说明：

```text
显式 reaction-source-diffusion 分支结构
比简单 feature concatenation + generic message passing 更适合 high-confidence candidate ranking。
```

---

### 7.3 RDS v0 vs local_source_proxy WeightedDiffusion

对比：

```text
local_source_proxy_weighted_diffusion_v0_seed42
local_source_proxy_rds_v0_seed42
```

| Metric | WeightedDiffusion | RDS v0 | 变化 |
|---|---:|---:|---:|
| Test AUPRC | 0.045404 | 0.047322 | +4.2% |
| AUROC | 0.712645 | 0.715591 | +0.0029 |
| Spearman | 0.098410 | 0.099775 | +0.0014 |
| P@100 | 0.180 | 0.160 | -11.1% |
| P@1000 | 0.121 | 0.132 | +9.1% |
| P@5000 | 0.0898 | 0.0974 | +8.5% |
| NDCG@1000 | 0.122069 | 0.136672 | +12.0% |
| Enrich@1000 | 6.663490 | 7.269262 | +9.1% |

解释：

```text
WeightedDiffusion 在 P@100 上略高，
但 RDS v0 在 AUPRC、AUROC、Spearman、P@1000、P@5000、NDCG@1000 和 Enrichment@1000 上整体更优。
```

因此：

```text
WeightedDiffusion 证明 edge_weight diffusion 有效；
RDS v0 进一步证明 source / proxy / reaction / diffusion 的结构化组合更有效。
```

---

## 8. 当前模型排序

按整体 global ranking 指标：

```text
1. local_source_proxy_rds_v0_seed42
2. local_source_proxy_weighted_diffusion_v0_seed42
3. local_source_proxy_graphsage_v0_seed42
4. local_source_graphsage_v0_seed42
5. local_graphsage_v0_seed42
6. local_weighted_diffusion_v0_seed42
7. local_source_mlp_v0_seed42
```

按 top-K discovery 指标综合看：

```text
1. local_source_proxy_rds_v0_seed42
2. local_source_proxy_weighted_diffusion_v0_seed42
3. local_graphsage_v0_seed42
4. local_source_graphsage_v0_seed42 / local_weighted_diffusion_v0_seed42
5. local_source_proxy_graphsage_v0_seed42
6. local_source_mlp_v0_seed42
```

当前综合最佳模型：

```text
local_source_proxy_rds_v0_seed42
```

---

## 9. 当前结果对 Knowledge Field 假设的支持

Stage 07 当前结果支持如下 Knowledge Field 假设：

```text
未来 translational emergence 不是单纯由 focal KU 自身历史决定，
也不是单纯由普通 graph topology 决定，
而是 local state + carrier source / exposure + KU-neighborhood diffusion 的组合过程。
```

具体证据：

```text
1. MLP < GraphSAGE：
   KU-KU graph message passing 有增量。

2. GraphSAGE < WeightedDiffusion on top-K：
   edge_weight 对 high-confidence discovery 有帮助。

3. GraphSAGE / WeightedDiffusion < RDS v0：
   显式 reaction-source-diffusion 分解优于 generic graph aggregation。

4. local_source_proxy 输入整体更强：
   carrier source 和 diffusion proxy 与 learned graph diffusion 互补。

5. RDS v0 同时改善 global ranking 和 top-K：
   结构化 field model 比简单 feature concatenation 更适合该任务。
```

因此，当前 Stage 07 v0 已经从 Stage 06 的 diagnostic evidence 推进到正式模型 evidence：

```text
Knowledge Field 模型结构是有效的。
```

---

## 10. 当前 RDS v0 的理论定位

当前 RDS v0 可以称为：

```text
physics-inspired Reaction-Diffusion-Source Knowledge Field GNN
```

它已经引入：

```text
reaction
source
proxy
diffusion
```

但是，当前 RDS v0 仍然是：

```text
static supervised predictor
```

它学习的是：

```text
features at cutoff t -> future 3-year translation heat / emergence
```

而不是显式学习：

```text
u_i(t) -> u_i(t + Δt)
```

因此，当前 RDS v0 还不能称为完整的：

```text
physical graph dynamical system
```

更准确的表述是：

```text
RDS v0 introduces a reaction-source-diffusion inductive bias,
but it does not yet explicitly evolve a knowledge-state variable over time.
```

中文：

```text
RDS v0 引入了 reaction-source-diffusion 的物理归纳偏置，
但它还没有显式定义和演化知识状态变量 u_i(t)。
```

---

## 11. 当前尚未完全解决的问题

当前 Stage 07 v0 已经解决：

```text
1. KU graph message passing 是否有用；
2. edge_weight 是否有用；
3. reaction-source-diffusion 分支结构是否有效；
4. local/source/proxy/diffusion 是否存在互补信号。
```

但尚未解决：

```text
1. 如何显式定义 knowledge state u_i(t)；
2. 如何学习 u_i(t) -> u_i(t+Δt)；
3. 如何实现真正的 diffusion flux Σ_j κ_ij(u_j - u_i)；
4. 如何显式加入 decay / dissipation；
5. 如何与现有 physics / PDE-style GNN baseline 公平比较；
6. 如何解释每个 KU 的 reaction/source/diffusion/decay 贡献。
```

因此，Stage 07 后续应从：

```text
physics-inspired static predictor
```

继续推进到：

```text
explicit knowledge field dynamics model
```

---

## 12. 推荐下一步：Physics / PDE-style baselines

当前下一步不建议立即做大量普通消融，而应优先补充与理论主张最相关的 baseline：

```text
RDGNN-style reaction-diffusion baseline
GRAND-style graph diffusion baseline
```

---

### 12.1 RDGNN-style baseline

RDGNN-style baseline 表示通用 reaction-diffusion GNN 思路。

推荐形式：

```text
h_i^{l+1}
=
h_i^l
+
α_l Rθ(h_i^l, x_i(t))
+
β_l Σ_j A_ij · Φψ(h_j^l - h_i^l)
```

最终：

```text
ŷ_i(t) = Head(h_i^L)
```

其作用是回答：

```text
当前 RDS / 后续 DynamicRDS 是否优于通用 reaction-diffusion GNN？
```

推荐模型名：

```text
rdgnn_style
```

推荐首跑：

```text
local_source_proxy_rdgnn_style_v0_seed42
```

---

### 12.2 GRAND-style baseline

GRAND-style baseline 表示 graph diffusion / neural PDE 类 GNN。

推荐形式：

```text
h^{l+1}
=
h^l
+
δ · [
    -L_A h^l
    +
    Rθ(h^l)
  ]
```

或简化为：

```text
h_i^{l+1}
=
(1 - α) h_i^l
+
α Σ_j Ã_ij h_j^l
+
Rθ(h_i^l)
```

其作用是回答：

```text
简单 graph diffusion PDE-style smoothing 是否足以解释当前任务中的 diffusion signal？
```

推荐模型名：

```text
grand_style
```

推荐首跑：

```text
local_source_proxy_grand_style_v0_seed42
```

---

## 13. 推荐下一步：DynamicRDS

### 13.1 目标

DynamicRDS 的目标是把当前 RDS v0 从：

```text
physics-inspired static predictor
```

升级为：

```text
explicit knowledge field dynamical system
```

核心问题变为：

```text
learn how knowledge state u_i(t) evolves into u_i(t + Δt)
```

---

### 13.2 核心公式

DynamicRDS 应显式定义：

```text
u_i(t)
```

作为第 `i` 个 KU 在 cutoff year `t` 的知识场状态。

核心形式：

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
R_i(t): local reaction term
S_i(t): carrier source / exposure injection term
P_i(t): diffusion proxy / prior field term
D_i(t): KU-KU graph diffusion flux
Λ_i(t): decay / dissipation term
```

具体实现：

```text
R_i(t) = Rθ(u_i(t), x_i(t))
S_i(t) = Sφ(s_i(t))
P_i(t) = Pη(p_i(t))
D_i(t) = Σ_j κ_ij(t) · [u_j(t) - u_i(t)]
Λ_i(t) = λ_i(t) · u_i(t)
```

最终：

```text
ŷ_i(t) = Readout(u_i(t + Δt), Δu_i(t), context_i(t))
```

---

### 13.3 DynamicRDS 与 RDS v0 的区别

| 项 | RDS v0 | DynamicRDS |
|---|---|---|
| 模型性质 | physics-inspired static predictor | explicit graph dynamical system |
| 状态变量 | hidden representation h_i | explicit knowledge state u_i(t) |
| 时间演化 | implicit prediction | explicit u(t) -> u(t+Δt) |
| diffusion | neighbor representation aggregation | flux term Σ κ_ij(u_j-u_i) |
| source | source branch representation | source injection term S_i(t) |
| decay | absent / implicit | explicit decay Λ_i(t) |
| 物理解释 | reaction-source-diffusion inductive bias | knowledge field dynamics |
| 输出 | score_i(t) | u_next, Δu, score |

---

## 14. 推荐后续实验组合

下一阶段推荐 runs：

```text
local_source_proxy_rdgnn_style_v0_seed42
local_source_proxy_grand_style_v0_seed42
local_source_proxy_dynamic_rds_v0_seed42
```

推荐与当前已完成 runs 一起比较：

```text
local_source_mlp_v0_seed42
local_source_proxy_graphsage_v0_seed42
local_source_proxy_weighted_diffusion_v0_seed42
local_source_proxy_rds_v0_seed42
local_source_proxy_rdgnn_style_v0_seed42
local_source_proxy_grand_style_v0_seed42
local_source_proxy_dynamic_rds_v0_seed42
```

核心比较问题：

```text
1. RDGNN-style 是否优于 GraphSAGE / WeightedDiffusion？
2. GRAND-style 是否能解释 diffusion signal？
3. RDS v0 是否仍优于通用 physics / PDE-style GNN baseline？
4. DynamicRDS 是否进一步超过 RDS v0？
5. 显式 flux diffusion 和 decay 是否改善 top-K discovery？
```

---

## 15. 当前工程注意事项

Stage 07 模型在本机训练时曾遇到 CPU memory / PyG sampler memory 问题：

```text
np.concatenate ArrayMemoryError
NeighborLoader / pyg_lib index_sort MemoryError
```

主要原因：

```text
1. local_source_proxy 输入维度较高；
2. snapshot feature matrix 约 715,912 × 170；
3. PyG NeighborLoader 需要为大图构建 CPU-side CSC sampling index；
4. weighted_diffusion / RDS 额外使用 edge_weight 和更多模型分支。
```

当前建议：

```text
1. 本机调试使用 batch_size = 4096 或 2048；
2. A100 正式跑可使用 batch_size = 16384；
3. A100 作业需要足够 CPU memory；
4. 推荐优化 build_snapshot_features，避免 x_num + x_cat + concatenate 的额外 peak memory；
5. 推荐在 snapshot 后显式 gc.collect() 和 torch.cuda.empty_cache()。
```

---

## 16. 当前结论

Stage 07 当前结果是正向的。

最重要结论：

```text
GraphSAGE > MLP：
  learned KU-KU graph message passing 有价值。

WeightedDiffusion > GraphSAGE on top-K：
  edge_weight 对 high-confidence discovery 有价值。

RDS v0 > WeightedDiffusion / GraphSAGE / MLP：
  reaction + source + proxy + diffusion 的结构化 Knowledge Field 模型目前最优。
```

当前最佳模型：

```text
local_source_proxy_rds_v0_seed42
```

当前最佳综合表现：

```text
Test AUPRC       = 0.047322
Test AUROC       = 0.715591
Test Spearman    = 0.099775
Precision@1000   = 0.132
NDCG@1000        = 0.136672
Enrichment@1000  = 7.269262
```

这说明：

```text
未来 translational emergence 更像是 KU field 上的 source-driven diffusion process，
而不是单纯的 local temporal forecasting 或 generic graph aggregation。
```

---

## 17. 最终一句话

Stage 07 v0 已经证明：

```text
Reaction-Diffusion-Source Knowledge Field GNN 是当前最有效的模型结构。
```

但它仍然是：

```text
physics-inspired static predictor
```

而不是完整的：

```text
knowledge field dynamical system
```

因此，下一步最有价值的方向是：

```text
1. 补充 RDGNN-style 和 GRAND-style physics / PDE GNN baselines；
2. 开发 DynamicRDS，显式建模 u_i(t) -> u_i(t+Δt)；
3. 将当前 Knowledge Field GNN 推进为真正的 knowledge field physical dynamics model。
```