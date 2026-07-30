# Stage 07H Multi-seed Baseline Comparison 实验报告

日期：2026-07-30  
阶段：Stage 07H  
实验名称：Multi-seed fair baseline comparison for BIBM Doctoral Forum  
任务：KU-level prospective biomedical translation candidate ranking  
主任务：Future Translation Heat Forecasting  
目标会议：IEEE BIBM 2026 Doctoral Forum  

---

## 1. 实验目的

本实验的目标是为 BIBM 2026 Doctoral Forum 论文提供一组更公平、可复核的 multi-seed baseline comparison。

此前 Stage 07 已经完成：

```text
1. Temporal GraphSAGE / MLP baseline
2. WeightedDiffusion baseline
3. Reaction-Diffusion-Source model
4. GRAND-style graph diffusion baseline
5. RDGNN-style reaction-diffusion baseline
6. DynamicRDS v2 vector-state graph dynamics model
7. DynamicRDS hyperparameter sensitivity and tuned confirmation
```

但此前部分结果仍以 single-seed 或局部 confirmation 为主。  
本轮 Stage 07H 统一使用相同 train / validation / test split、相同输入数据、相同 seeds 和相近训练配置，对主要模型进行 5-seed 比较。

本实验重点回答：

```text
1. KU-KU graph context 是否明显优于 no-graph MLP？
2. graph diffusion / reaction-diffusion-style baselines 是否具有稳定优势？
3. Static RDS 与 DynamicRDS v2 的关系是什么？
4. DynamicRDS v2 是否是当前最强 top-K 模型？
5. 对 BIBM Doctoral Forum 论文应该如何严谨表述当前结果？
```

---

## 2. 数据与任务设置

### 2.1 数据来源

本实验使用 Stage 07 latest-window GNN-ready dataset：

```text
data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_proxy_latest_split/
```

对应输入特征包括：

```text
local:
  focal KU temporal state features x_i(t)

source:
  carrier graph source / exposure features s_i(t)

proxy:
  handcrafted KU-neighborhood diffusion proxy features p_i(t)

graph:
  KU-KU shared-entity graph with edge weights
```

---

### 2.2 时间切分

本实验沿用 Stage 07 latest-window split：

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

需要注意：

```text
2022-2024 是 test target window，
不是 feature cutoff years。
```

---

### 2.3 预测任务

Primary task：

```text
translation
```

Eligibility：

```text
eligible_translation_task =
  history_paper_count > 0
  AND history_patent_count == 0
  AND history_trial_count == 0
```

Target：

```text
target_future_translation_heat_3yr =
  log1p(target_future_patent_count_3yr + target_future_trial_count_3yr)
```

Binary emergence label：

```text
label_future_translation_emergence_3yr =
  1[target_future_patent_count_3yr + target_future_trial_count_3yr > 0]
```

Test set summary：

```text
eligible test KUs = 563,533
positive KUs      = 10,233
positive rate     = 0.018159 ≈ 1.82%
```

该任务属于：

```text
rare-positive top-K candidate ranking
```

因此除 AUPRC / AUROC 外，重点关注：

```text
Precision@100
Precision@1000
NDCG@1000
Enrichment@1000
```

---

## 3. 比较模型

本实验比较 7 类模型。

| CLI model name | Report name | Graph | 说明 |
|---|---|---:|---|
| `mlp` | MLP | No | no-graph neural baseline |
| `graphsage` | GraphSAGE | Yes | generic message-passing GNN |
| `weighted_diffusion` | WeightedDiffusion | Yes | edge-weighted neighbor diffusion baseline |
| `grand_style` | GRAND-style | Yes | graph diffusion / neural PDE-style baseline |
| `rdgnn_style` | RDGNN-style | Yes | generic reaction-diffusion representation baseline |
| `reaction_diffusion_source` | Static RDS | Yes | static reaction-source-proxy-diffusion predictor |
| `dynamic_rds` | DynamicRDS v2 | Yes | vector-state graph dynamical model |

---

## 4. 模型命名说明

### 4.1 Static RDS

本文中的：

```text
Static RDS
```

对应代码：

```text
--model reaction_diffusion_source
```

它是一个静态 reaction-source-proxy-diffusion predictor：

```text
score_i
=
Head(
    R(local_i)
  + S(source_i)
  + P(proxy_i)
  + D_weighted(neighbors_i)
)
```

它不显式建模：

```text
u_i(t) -> u_i(t + Δt)
```

因此不应称为 DynamicRDS v1 / v0。

---

### 4.2 DynamicRDS v2

本文中的：

```text
DynamicRDS v2
```

对应代码：

```text
--model dynamic_rds
```

它是 vector-state graph dynamical model，显式定义：

```text
u_i(t) ∈ R^d
```

并使用 Euler-style update：

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
R_i(t): local reaction
S_i(t): carrier-source injection
P_i(t): proxy / prior field
D_i(t): state-difference graph diffusion flux
Λ_i(t): decay / dissipation
```

图扩散项为：

```text
D_i(t) = Σ_j κ_ij(t) · [u_j(t) - u_i(t)]
```

因此：

```text
DynamicRDS v2 是显式动态图动力学模型；
Static RDS 是静态分支融合预测器。
```

---

## 5. 实验配置

### 5.1 Seeds

本实验使用 5 个 random seeds：

```text
1, 2, 3, 42, 2024
```

所有表格均报告：

```text
mean ± standard deviation over 5 seeds
```

---

### 5.2 统一训练配置

主要训练配置：

```text
hidden_dim = 128
num_layers = 2
batch_size = 16384
epochs = 50
patience = 8
optimizer = adamw
lr = 3e-4
weight_decay = 1e-4
dropout = 0.2
loss = smoothl1
monitor_metric = validation AUPRC
topk = 100, 1000, 5000
```

图模型使用 neighbor sampling：

```text
num_neighbors = 15 10
```

DynamicRDS v2 使用：

```text
dynamic_state_dim = 8
```

---

### 5.3 输出目录

结果目录：

```text
data/results/knowledge_field/multiseed_baseline_comparison_stage07h/
```

核心结果文件：

```text
stage07h_multiseed_runs.csv
stage07h_multiseed_summary_mean_std.csv
stage07h_best_seed_by_model.csv
stage07h_paired_tests_vs_dynamic_rds.csv
stage07h_multiseed_report.md
```

---

# 6. 主结果：Multi-seed mean ± std

## 6.1 Global ranking metrics

| Model | AUPRC | AUROC | Spearman |
|---|---:|---:|---:|
| MLP | 0.0434 ± 0.0006 | 0.7113 ± 0.0026 | 0.0978 ± 0.0012 |
| GraphSAGE | 0.0457 ± 0.0007 | 0.7156 ± 0.0026 | 0.0998 ± 0.0012 |
| WeightedDiffusion | 0.0456 ± 0.0010 | 0.7168 ± 0.0013 | 0.1003 ± 0.0006 |
| GRAND-style | 0.0466 ± 0.0007 | 0.7165 ± 0.0022 | 0.1002 ± 0.0010 |
| RDGNN-style | **0.0477 ± 0.0008** | 0.7192 ± 0.0008 | 0.1016 ± 0.0005 |
| Static RDS | **0.0477 ± 0.0006** | 0.7192 ± 0.0012 | 0.1014 ± 0.0005 |
| DynamicRDS v2 | 0.0476 ± 0.0005 | **0.7213 ± 0.0009** | **0.1024 ± 0.0004** |

### 6.1.1 Global metrics 观察

AUPRC 上最强的三个模型非常接近：

```text
RDGNN-style:
  AUPRC = 0.0477 ± 0.0008

Static RDS:
  AUPRC = 0.0477 ± 0.0006

DynamicRDS v2:
  AUPRC = 0.0476 ± 0.0005
```

DynamicRDS v2 在：

```text
AUROC
Spearman
```

上取得最高均值：

```text
AUROC    = 0.7213 ± 0.0009
Spearman = 0.1024 ± 0.0004
```

这说明：

```text
DynamicRDS v2 的 global ranking / monotonic ranking 表现具有竞争力，
并且在 AUROC / Spearman 上略优。
```

但 AUPRC 维度上，DynamicRDS v2 与 Static RDS / RDGNN-style 基本处于同一水平，而不是明显胜出。

---

## 6.2 Top-K candidate ranking metrics

| Model | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---|---:|---:|---:|---:|---:|
| MLP | 0.018 ± 0.015 | 0.0768 ± 0.0120 | 0.0776 ± 0.0017 | 0.0705 ± 0.0126 | 4.23 ± 0.66 |
| WeightedDiffusion | 0.116 ± 0.054 | 0.1000 ± 0.0146 | 0.0842 ± 0.0029 | 0.1031 ± 0.0172 | 5.51 ± 0.80 |
| GRAND-style | 0.152 ± 0.048 | 0.1114 ± 0.0119 | 0.0873 ± 0.0042 | 0.1234 ± 0.0148 | 6.13 ± 0.65 |
| GraphSAGE | 0.122 ± 0.023 | 0.1142 ± 0.0108 | 0.0884 ± 0.0049 | 0.1182 ± 0.0078 | 6.29 ± 0.60 |
| DynamicRDS v2 | 0.152 ± 0.031 | 0.1144 ± 0.0089 | 0.0931 ± 0.0023 | 0.1210 ± 0.0115 | 6.30 ± 0.49 |
| RDGNN-style | 0.136 ± 0.015 | 0.1214 ± 0.0104 | **0.0927 ± 0.0046** | 0.1283 ± 0.0117 | 6.69 ± 0.57 |
| Static RDS | **0.206 ± 0.023** | **0.1260 ± 0.0102** | 0.0917 ± 0.0058 | **0.1375 ± 0.0096** | **6.94 ± 0.56** |

### 6.2.1 Top-K 观察

当前 top-K 最强模型是：

```text
Static RDS
```

其主要指标为：

```text
P@100       = 0.206 ± 0.023
P@1000      = 0.1260 ± 0.0102
NDCG@1000   = 0.1375 ± 0.0096
Enrich@1000 = 6.94 ± 0.56
```

RDGNN-style 也很强：

```text
P@1000      = 0.1214 ± 0.0104
NDCG@1000   = 0.1283 ± 0.0117
Enrich@1000 = 6.69 ± 0.57
```

DynamicRDS v2 的 top-K 表现为：

```text
P@1000      = 0.1144 ± 0.0089
NDCG@1000   = 0.1210 ± 0.0115
Enrich@1000 = 6.30 ± 0.49
```

因此，当前不能表述为：

```text
DynamicRDS v2 achieves the best top-K performance.
```

更准确表述是：

```text
DynamicRDS v2 achieves competitive top-K performance,
but Static RDS and RDGNN-style currently achieve stronger top-K ranking.
```

---

## 6.3 Runtime and model size

| Model | Runtime / run | Best epoch | Trainable parameters |
|---|---:|---:|---:|
| MLP | 440 ± 81 s | 33.8 ± 10.9 | 22,017 |
| GraphSAGE | 1452 ± 330 s | 26.4 ± 7.7 | 93,185 |
| WeightedDiffusion | 1627 ± 374 s | 28.0 ± 8.4 | 93,185 |
| DynamicRDS v2 | 1950 ± 357 s | 36.2 ± 9.8 | 120,779 |
| GRAND-style | 2085 ± 232 s | 39.6 ± 8.1 | 105,091 |
| Static RDS | 2238 ± 214 s | 41.4 ± 7.1 | 181,762 |
| RDGNN-style | 2562 ± 9 s | 47.0 ± 1.4 | 171,141 |

### 6.3.1 Runtime 观察

MLP 最快，但性能明显较弱。

图模型中：

```text
GraphSAGE:
  runtime ≈ 1452s

WeightedDiffusion:
  runtime ≈ 1627s

DynamicRDS v2:
  runtime ≈ 1950s

Static RDS:
  runtime ≈ 2238s

RDGNN-style:
  runtime ≈ 2562s
```

DynamicRDS v2 的 runtime 低于 Static RDS 和 RDGNN-style：

```text
DynamicRDS v2:
  1950 ± 357 s

Static RDS:
  2238 ± 214 s

RDGNN-style:
  2562 ± 9 s
```

说明 DynamicRDS v2 在当前实现下不是最慢模型，且具有可接受的训练成本。

---

# 7. 主要发现

## 7.1 图模型显著优于 no-graph MLP

MLP 的 top-K 表现明显较弱：

```text
MLP:
  P@1000      = 0.0768 ± 0.0120
  NDCG@1000   = 0.0705 ± 0.0126
  Enrich@1000 = 4.23 ± 0.66
```

相比之下，所有图模型的 Enrich@1000 均明显更高：

```text
WeightedDiffusion:
  Enrich@1000 = 5.51 ± 0.80

GraphSAGE:
  Enrich@1000 = 6.29 ± 0.60

DynamicRDS v2:
  Enrich@1000 = 6.30 ± 0.49

Static RDS:
  Enrich@1000 = 6.94 ± 0.56
```

这支持：

```text
KU-KU graph context 对 rare-positive prospective translation candidate ranking 有明显增量。
```

---

## 7.2 Reaction-diffusion-source 模型族整体最强

当前表现最强的模型主要集中在 reaction-diffusion-source family：

```text
Static RDS
RDGNN-style
DynamicRDS v2
```

其中 Static RDS 在 top-K 上最强：

```text
P@1000      = 0.1260 ± 0.0102
NDCG@1000   = 0.1375 ± 0.0096
Enrich@1000 = 6.94 ± 0.56
```

RDGNN-style 次之：

```text
P@1000      = 0.1214 ± 0.0104
NDCG@1000   = 0.1283 ± 0.0117
Enrich@1000 = 6.69 ± 0.57
```

DynamicRDS v2 在 global metrics 上非常有竞争力：

```text
AUPRC    = 0.0476 ± 0.0005
AUROC    = 0.7213 ± 0.0009
Spearman = 0.1024 ± 0.0004
```

这一结果支持：

```text
reaction / diffusion / source 等结构化图动力学归纳偏置适合该任务。
```

---

## 7.3 DynamicRDS v2 不是当前 top-K 最强模型

本实验最重要的负向发现是：

```text
DynamicRDS v2 当前没有超过 Static RDS。
```

具体差异：

```text
Static RDS:
  P@1000      = 0.1260 ± 0.0102
  NDCG@1000   = 0.1375 ± 0.0096
  Enrich@1000 = 6.94 ± 0.56

DynamicRDS v2:
  P@1000      = 0.1144 ± 0.0089
  NDCG@1000   = 0.1210 ± 0.0115
  Enrich@1000 = 6.30 ± 0.49
```

因此论文中必须避免：

```text
DynamicRDS v2 outperforms all baselines.
DynamicRDS v2 is the strongest model.
```

更严谨的表述是：

```text
DynamicRDS v2 achieves competitive global ranking and top-K performance,
but Static RDS currently achieves the strongest top-K ranking.
```

---

## 7.4 DynamicRDS v2 的价值在于显式动态图动力学与机制诊断

虽然 DynamicRDS v2 当前 top-K 没有超过 Static RDS，但它仍然提供 Static RDS 不具备的建模能力：

```text
1. 显式 learned vector state u_i(t)
2. 显式 state change Δu_i(t)
3. 显式 next-state u_i(t + Δt)
4. state-difference graph diffusion flux
5. decay / dissipation term
6. term-level interpretation potential
```

这使得 DynamicRDS v2 可以支持后续分析：

```text
1. 某个 KU 的 latent translation potential 是否上升？
2. top-ranked KU 主要由 reaction、source、proxy、diffusion 还是 decay 驱动？
3. 哪些 neighbor KUs 对当前 KU 产生 incoming graph flux？
4. diffusion 与 decay 的平衡是否影响 top-K candidate ranking？
```

Static RDS 当前预测更强，但它不显式提供：

```text
u_i(t)
Δu_i(t)
u_i(t + Δt)
```

因此当前结果应解释为：

```text
Static RDS is the strongest current predictor;
DynamicRDS v2 is a competitive and more explicit graph-dynamical formulation,
whose dynamic-state calibration remains an open doctoral research problem.
```

---

# 8. 对 BIBM Doctoral Forum 写作的影响

## 8.1 推荐论文主线

本实验结果表明，BIBM Doctoral Forum 论文不应写成：

```text
We propose DynamicRDS v2 and it outperforms all baselines.
```

而应写成：

```text
We study cutoff-safe Knowledge-Unit graph dynamics for prospective biomedical translation prediction.
Reaction-diffusion-source graph dynamics is a promising model family.
Static RDS currently provides the strongest top-K predictor.
DynamicRDS v2 provides an explicit vector-state graph-dynamical formulation,
and its calibration is a key ongoing doctoral research question.
```

---

## 8.2 推荐核心 claim

推荐使用如下 claim：

```text
Preliminary five-seed results show that graph-based models substantially improve over no-graph MLP for rare-positive translation candidate ranking. Reaction-diffusion-source models form the strongest current model family. Static RDS achieves the highest current top-K scores, while DynamicRDS v2 provides competitive global ranking and a more explicit graph-dynamical formulation through learned state, state change, diffusion flux, and decay terms.
```

中文：

```text
初步 5-seed 结果显示，图模型相比 no-graph MLP 明显提升 rare-positive translation candidate ranking。Reaction-diffusion-source 模型族构成当前最强模型族。Static RDS 取得最高 top-K 分数，而 DynamicRDS v2 具有有竞争力的 global ranking，并通过 learned state、state change、diffusion flux 和 decay 项提供更显式的图动力学建模形式。
```

---

## 8.3 推荐 contribution 调整

BIBM Doctoral Forum 论文贡献应写为：

```text
1. We formulate cutoff-safe KU-level prospective translation prediction.
2. We instantiate a large-scale temporal evaluation dataset from a 2000–2024 diabetes biomedical carrier graph.
3. We study reaction-diffusion-source graph dynamics, including Static RDS and DynamicRDS v2.
4. We provide preliminary five-seed evidence that graph-based reaction-diffusion-source models are promising, while identifying dynamic-state calibration as a key open doctoral research problem.
```

不要写：

```text
DynamicRDS v2 is the best model.
```

---

# 9. 当前结果对后续研究的启发

## 9.1 Dynamic-state calibration 是下一步核心问题

Static RDS 当前 top-K 最强，而 DynamicRDS v2 没有超过它，说明：

```text
显式 vector-state dynamics 增加了建模自由度，
但也带来了状态校准难题。
```

可能原因包括：

```text
1. u_i(t), Δu_i(t), u_i(t+Δt) 的尺度尚未充分校准；
2. reaction/source/proxy/diffusion/decay 各项贡献尺度可能不平衡；
3. validation AUPRC 不一定最适合 rare-positive top-K discovery；
4. tanh du stabilization 可能压缩了 top-tail signal；
5. source/proxy gated injection 可能增强稳定性但削弱 top-K sharpness；
6. DynamicRDS v2 的显式 dynamics 对 top-K objective 需要更专门的训练或校准。
```

这可以作为 Doctoral Research Plan 的核心科学问题：

```text
How can explicit graph-dynamical state updates be calibrated to improve rare-positive top-K candidate ranking while preserving interpretability?
```

---

## 9.2 Static RDS 是强预测 baseline，而不是失败

Static RDS 的强表现说明：

```text
reaction/source/proxy/diffusion decomposition 本身是有效的。
```

这不是 DynamicRDS v2 的失败，而是说明：

```text
1. 分支式结构化归纳偏置有效；
2. 显式动态状态建模是更难的问题；
3. 后续需要把 Static RDS 的 top-K sharpness 与 DynamicRDS v2 的机制解释性结合起来。
```

---

## 9.3 BIBM Doctoral Forum 中可写的后续计划

建议后续计划包括：

```text
1. Dynamic-state calibration:
   calibrate u, Δu, u_next and term scales.

2. Top-K-aware validation:
   compare validation AUPRC vs validation NDCG@1000 / P@1000.

3. Term-scale sensitivity:
   explicitly sweep diffusion_scale, decay_scale, source_scale, proxy_scale.

4. Term-level interpretation:
   output reaction/source/proxy/diffusion/decay norms for top-ranked KUs.

5. Case-level evidence tracing:
   trace top candidates back to local evidence, carrier-source exposure, and graph-flux neighbors.

6. Temporal robustness:
   evaluate across cutoff years 2019, 2020, 2021.

7. Leakage audit and reproducibility:
   produce formal cutoff-safe audit artifacts.
```

---

# 10. Limitations

当前 Stage 07H 仍有以下限制：

## 10.1 单一 test cutoff

当前结果主要基于：

```text
test cutoff = 2021
future target window = 2022-2024
```

尚未系统评估：

```text
cutoff = 2019
cutoff = 2020
cutoff = 2021
```

因此 temporal robustness 仍需后续补充。

---

## 10.2 只使用 5 seeds

本实验使用：

```text
5 seeds
```

已经比 single-seed 更稳健，但显著性检验仍然统计功效有限。  
paired t-test / Wilcoxon 结果应作为 exploratory evidence，而不是强显著性结论。

---

## 10.3 DynamicRDS v2 尚未针对 top-K objective 专门优化

当前模型选择指标为：

```text
validation AUPRC
```

但主要应用目标是：

```text
rare-positive top-K candidate ranking
```

后续应考虑：

```text
validation NDCG@1000
validation P@1000
top-K-aware calibration
```

---

## 10.4 缺少 case-level biomedical sanity check

当前 Stage 07H 是模型级比较，尚未包含：

```text
top-ranked true positive cases
evidence tracing
entity-level biomedical interpretation
diffusion-path visualization
```

这些应作为 BIBM Doctoral Forum 后续工作或补充轻量 sanity check。

---

# 11. Recommended BIBM Doctoral Forum result statement

## 11.1 English version

```text
Preliminary five-seed results show that graph-based models substantially improve rare-positive prospective translation candidate ranking over a no-graph MLP baseline. Reaction-diffusion-source variants form the strongest current model family. Static RDS achieves the best top-K performance, with P@1000 = 0.1260 ± 0.0102 and NDCG@1000 = 0.1375 ± 0.0096. DynamicRDS v2 achieves competitive global ranking, with AUPRC = 0.0476 ± 0.0005, AUROC = 0.7213 ± 0.0009, and Spearman = 0.1024 ± 0.0004, but it does not yet outperform Static RDS on top-K metrics. This gap suggests that explicit vector-state graph dynamics introduces a nontrivial calibration problem, motivating ongoing doctoral work on dynamic-state calibration, term-scale balancing, and interpretable evidence tracing.
```

---

## 11.2 中文版

```text
初步 5-seed 结果显示，图模型相比 no-graph MLP 明显提升 rare-positive prospective translation candidate ranking。Reaction-diffusion-source 变体构成当前最强模型族。Static RDS 在 top-K 指标上表现最好，P@1000 = 0.1260 ± 0.0102，NDCG@1000 = 0.1375 ± 0.0096。DynamicRDS v2 在 global ranking 上具有竞争力，AUPRC = 0.0476 ± 0.0005，AUROC = 0.7213 ± 0.0009，Spearman = 0.1024 ± 0.0004，但当前尚未在 top-K 指标上超过 Static RDS。这一 gap 说明显式 vector-state graph dynamics 引入了非平凡的状态校准问题，也构成后续博士阶段工作的核心方向：dynamic-state calibration、term-scale balancing 和 interpretable evidence tracing。
```

---

# 12. 当前最终结论

Stage 07H 的最终结论为：

```text
1. KU-KU graph context 对未来 biomedical translation candidate ranking 有明显增量；
2. no-graph MLP 明显弱于所有图模型；
3. reaction-diffusion-source 模型族整体最强；
4. Static RDS 当前是最强 top-K predictor；
5. DynamicRDS v2 在 AUPRC / AUROC / Spearman 上具有竞争力，并提供更显式的图动力学解释；
6. DynamicRDS v2 当前没有超过 Static RDS，说明 dynamic-state calibration 是后续核心研究问题；
7. 对 BIBM Doctoral Forum，应将论文定位为 ongoing doctoral research，而不是 DynamicRDS v2 已经全面优于 baseline 的 completed full paper。
```

推荐论文主线：

```text
Toward Cutoff-Safe Knowledge-Unit Graph Dynamics for Prospective Biomedical Translation Prediction
```

推荐主 claim：

```text
Graph-based reaction-diffusion-source models are promising for rare-positive biomedical translation candidate ranking. Static RDS currently provides the strongest top-K predictor, while DynamicRDS v2 provides an explicit and interpretable vector-state graph-dynamical formulation that motivates ongoing doctoral research on dynamic-state calibration and evidence tracing.
```