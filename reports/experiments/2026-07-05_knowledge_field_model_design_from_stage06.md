# Stage 06 实验总结与 Knowledge Field 模型设计备忘录

日期：2026-07-05  
阶段：Stage 06E / 06F / 06G  
主题：从 KU temporal baselines 到 Knowledge Field model design  
用途：内部实验总结与后续正式模型设计依据  
注意：本文档是中间探索性实验总结，不作为论文正式主表结果。

---

## 1. 背景与目的

Stage 06 的核心目标不是训练最终模型，而是回答一个更基础的问题：

```text
未来 translational emergence 是否可以由 KU 自身状态、原始 carrier graph 结构暴露、以及 KU 邻域扩散状态共同解释？
```

具体来说，Stage 06 分为三个连续实验：

```text
Stage 06E:
  KU representation ablation

Stage 06F:
  Carrier graph structural readout baseline

Stage 06G:
  KU-KU diffusion proxy baseline
```

它们共同服务于后续 Knowledge Field 模型设计。

这三个阶段不是论文最终实验，而是内部 diagnostic / exploratory baselines，用于判断：

```text
1. KU abstraction 是否有预测价值？
2. 原始 carrier graph 中是否仍有 KU full 之外的结构信号？
3. KU-KU 邻域状态是否存在 diffusion-like predictive signal？
4. 后续 Knowledge Field 模型应该包含哪些组件？
```

---

## 2. 总体实验设定

所有 Stage 06 baseline 均基于 Stage 05F 冻结的 KU temporal prediction benchmark。

预测单位：

```text
KU × cutoff_year
```

历史窗口：

```text
2000 .. cutoff_year
```

预测窗口：

```text
cutoff_year + 1 .. cutoff_year + 3
```

任务：

| Task | Eligibility | Heat target | Emergence label |
|---|---|---|---|
| translation | `eligible_translation_task` | `target_future_translation_heat_3yr` | `label_future_translation_emergence_3yr` |
| patent | `eligible_patent_task` | `target_future_patent_heat_3yr` | `label_future_patent_emergence_3yr` |
| trial | `eligible_trial_task` | `target_future_trial_heat_3yr` | `label_future_trial_emergence_3yr` |

切分：

```text
train:
  cutoff_year = 2005-2016

validation:
  cutoff_year = 2017-2018

test_all:
  cutoff_year = 2019-2021

test_stable:
  cutoff_year = 2019-2020

right-edge stress cutoff:
  cutoff_year = 2021
```

主要模型：

```text
ridge
logistic
logistic_unweighted
histgb
```

训练协议：

```text
max_train_rows = 500000
```

解释：

```text
所有 trainable learners 最多使用 500,000 training rows；
evaluation 仍然使用完整 validation/test task frames。
```

---

## 3. Stage 06E：KU representation ablation

### 3.1 目的

Stage 06E 主要回答：

```text
KU full representation 中，哪些部分贡献预测信号？
```

更具体地说：

```text
1. KU temporal numeric features 是否是主要信号？
2. KU schema categorical features 是否在 temporal features 之外有增量？
3. full KU representation 是否优于 raw count-like temporal features？
```

### 3.2 比较对象

Stage 06E 中的 feature subsets 包括：

```text
ku_full / all
numeric_only
raw_counts_only
ku_schema_only
```

其中：

```text
ku_full:
  numeric temporal features + KU schema categorical features

numeric_only:
  仅 numeric temporal features

raw_counts_only:
  history / recent / growth / recency 等 raw carrier-count-like temporal summaries

ku_schema_only:
  pair_type / entity_a_type / entity_b_type 等 KU schema categorical features
```

### 3.3 主要结论

Stage 06E 的主要发现是：

```text
1. KU temporal numeric features 是最主要的预测信号；
2. KU schema categorical features 单独预测力有限，但在 temporal features 之外提供一定增量；
3. KU full representation 通常优于 raw count-like / numeric-only 变体；
4. 这说明 KU abstraction 不只是简单的 carrier count 聚合，而是形成了有预测价值的 temporal/schema representation。
```

### 3.4 对模型设计的启发

Stage 06E 支持后续模型中保留：

```text
focal KU temporal state
```

即每个 KU 自身的历史状态应作为 Knowledge Field 模型的核心输入。

可以记为：

```text
x_i(t)
```

其中：

```text
i = KU node
t = cutoff_year
x_i(t) = focal KU temporal/schema state
```

---

## 4. Stage 06F：Carrier graph structural readout baseline

### 4.1 目的

Stage 06F 回答：

```text
原始 carrier graph 中是否仍有 KU full representation 之外的结构信号？
```

换句话说，它比较：

```text
KU full representation
vs
direct structural readout from the original carrier graph
vs
KU full + carrier graph structural readout
```

### 4.2 Carrier graph structural readout 是什么？

Carrier graph structural readout 不是 GNN，也不是 learned graph embedding。

它是一个 cutoff-safe 的结构特征读出：

```text
原始 carrier graph
→ carrier node structural statistics
→ pool 到 KU × cutoff_year
```

结构特征包括：

```text
degree
weighted degree
unique neighbor count
edge-type diversity
neighbor type composition
paper / patent / trial / project / bioentity neighbor exposure
```

这些特征描述：

```text
支持某个 KU 的原始 evidence substrate 在 carrier graph 中有多活跃、多连接、多结构暴露。
```

### 4.3 比较对象

Stage 06F 主要比较三组 representation：

| Representation | 含义 | Feature count |
|---|---|---:|
| `carrier_structural_only_500k` | 仅 carrier graph structural readout | 47 |
| `ku_full_500k` | KU full representation | 61 |
| `ku_plus_structural_500k` | KU full + carrier structural | 108 |

### 4.4 主要发现

Stage 06F 的主要结论是：

```text
1. KU full 在 primary translation 和 trial 任务的整体指标上明显优于 carrier structural only；
2. carrier structural only 在 patent 任务和部分 trial top-K 指标上很强；
3. KU + carrier structural 通常获得最强或接近最强的整体表现；
4. 因此，KU abstraction 与 carrier graph structural exposure 是互补关系，而不是替代关系。
```

### 4.5 代表性结果

#### Translation / test_stable / HistGB

| Representation | AUROC | AUPRC | Spearman | Precision@1000 |
|---|---:|---:|---:|---:|
| carrier structural only | 0.6191 | 0.0463 | 0.0671 | 0.087 |
| KU full | 0.6627 | 0.0554 | 0.0916 | 0.161 |
| KU + structural | 0.6687 | 0.0605 | 0.0949 | 0.177 |

解释：

```text
在 primary translation task 上，KU full 明显强于直接 carrier graph structural readout；
但 carrier structural features 加入后仍进一步提升。
```

#### Patent / test_all / HistGB

| Representation | AUROC | AUPRC | Spearman | Precision@1000 |
|---|---:|---:|---:|---:|
| carrier structural only | 0.6462 | 0.0252 | 0.0506 | 0.108 |
| KU full | 0.6348 | 0.0251 | 0.0470 | 0.121 |
| KU + structural | 0.6482 | 0.0276 | 0.0513 | 0.149 |

解释：

```text
Patent forecasting 对 carrier graph structural exposure 更敏感；
KU full 和 carrier structural 结合后 top-K discovery 明显增强。
```

#### Trial / test_all / HistGB

| Representation | AUROC | AUPRC | Spearman | Precision@1000 |
|---|---:|---:|---:|---:|
| carrier structural only | 0.6339 | 0.0373 | 0.0598 | 0.185 |
| KU full | 0.6886 | 0.0418 | 0.0842 | 0.164 |
| KU + structural | 0.6943 | 0.0459 | 0.0867 | 0.197 |

解释：

```text
KU full 提升整体 discrimination；
carrier structural 对 extreme top-K tail 有强信号；
二者结合最好。
```

### 4.6 对模型设计的启发

Stage 06F 说明后续 Knowledge Field 模型不应只使用 KU 自身状态，还应保留：

```text
carrier graph structural exposure / source strength
```

可记为：

```text
s_i(t)
```

其中：

```text
s_i(t) = source/exposure features from original carrier graph
```

它可能对应 Knowledge Field 中的：

```text
source term
external evidence injection
structural exposure
```

---

## 5. Stage 06G：KU-KU diffusion proxy baseline

### 5.1 目的

Stage 06G 回答：

```text
KU 邻域状态是否存在 diffusion-like predictive signal？
```

也就是：

```text
如果某个 KU 周围的其他 KU 已经升温、增长、出现 patent/trial/translation evidence，
这个 KU 未来是否更可能 emergence？
```

### 5.2 KU-KU graph 构造

Stage 06G 从 KU 表构造 KU-KU graph：

```text
每个 KU 是一个节点；
两个 KU 如果共享 entity，则连边。
```

例如：

```text
KU1 = metformin - diabetes
KU2 = metformin - AMPK
```

二者共享：

```text
metformin
```

所以连边。

边权：

```text
edge_weight = sum over shared entities 1 / log(1 + entity_degree)
```

并限制 hub entities 和每个 KU 的最大邻居数：

```text
max_entity_degree = 500
top_neighbors = 30
```

说明：

```text
这是一个 shared-entity derived KU-neighborhood graph，
不是原始 carrier graph。
```

### 5.3 Diffusion proxy features

对每个：

```text
KU_i × cutoff_year
```

在 KU-KU graph 上聚合邻居 KU 的 cutoff-safe historical state：

```text
neighbor recent 3yr paper activity
neighbor recent 3yr patent activity
neighbor recent 3yr trial activity
neighbor recent 3yr translation activity
neighbor historical patent/trial/translation fractions
neighbor growth
neighbor top10 recent activity
weighted mean / mean / max / sum
```

这些构成：

```text
diffusion proxy features
```

记为：

```text
d_i(t)
```

其中：

```text
d_i(t) = neighborhood diffusion state around KU_i at cutoff t
```

### 5.4 比较对象

Stage 06G 主要比较：

| Representation | 含义 | Feature count |
|---|---|---:|
| `diffusion_proxy_only_500k` | 仅 KU-KU diffusion proxy | 55 |
| `ku_plus_diffusion_500k` | KU full + diffusion proxy | 116 |
| `ku_plus_structural_plus_diffusion_500k` | KU full + carrier structural + diffusion proxy | 163 |

并与 Stage 06F 的：

```text
carrier_structural_only_500k
ku_full_500k
ku_plus_structural_500k
```

一起比较。

### 5.5 运行状态

三组 Stage 06G baseline 均完整跑通：

| Representation | Feature count | Metric rows | Error count | Runtime seconds |
|---|---:|---:|---:|---:|
| diffusion_proxy_only_500k | 55 | 72 | 0 | 497.8 |
| ku_plus_diffusion_500k | 116 | 72 | 0 | 827.8 |
| ku_plus_carrier_structural_plus_diffusion_500k | 163 | 72 | 0 | 1138.4 |

---

## 6. Stage 06G 主要结果

### 6.1 Diffusion proxy only 单独有 global signal

`diffusion_proxy_only` 单独不如 KU full 的 top-K discovery，但在 AUROC / Spearman 上已经有明显预测力。

#### Translation / test_stable / HistGB

| Representation | Spearman | AUROC | AUPRC | Precision@1000 |
|---|---:|---:|---:|---:|
| carrier structural only | 0.0671 | 0.6191 | 0.0463 | 0.087 |
| diffusion proxy only | 0.0819 | 0.6447 | 0.0441 | 0.075 |
| KU full | 0.0916 | 0.6627 | 0.0554 | 0.161 |

解释：

```text
KU 邻域状态本身携带 future translation signal；
但单独作为 top-K selector 仍不够强。
```

---

### 6.2 KU + diffusion 明显超过 KU full

这是 Stage 06G 最重要的发现之一。

#### Translation / test_stable / HistGB

| Representation | Spearman | AUROC | AUPRC | Precision@1000 |
|---|---:|---:|---:|---:|
| KU full | 0.0916 | 0.6627 | 0.0554 | 0.161 |
| KU + diffusion | 0.1137 | 0.7016 | 0.0622 | 0.137 |

提升：

```text
Spearman: 0.0916 → 0.1137
AUROC:    0.6627 → 0.7016
AUPRC:    0.0554 → 0.0622
```

解释：

```text
KU-KU neighborhood diffusion proxy 在 focal KU temporal/schema features 之外提供独立增量。
```

#### Patent / test_stable / HistGB

| Representation | Spearman | AUROC | AUPRC | Precision@1000 |
|---|---:|---:|---:|---:|
| KU full | 0.0516 | 0.6383 | 0.0296 | 0.136 |
| KU + diffusion | 0.0765 | 0.7047 | 0.0348 | 0.133 |

解释：

```text
Patent emergence 中 KU-neighborhood diffusion signal 非常强，
尤其显著提升 AUROC / Spearman。
```

#### Trial / test_stable / HistGB

| Representation | Spearman | AUROC | AUPRC | Precision@1000 |
|---|---:|---:|---:|---:|
| KU full | 0.0902 | 0.6916 | 0.0478 | 0.174 |
| KU + diffusion | 0.1104 | 0.7320 | 0.0569 | 0.196 |

解释：

```text
Trial forecasting 也明显受 KU-neighborhood diffusion state 影响。
```

---

### 6.3 KU + structural + diffusion 通常获得最强 global metrics

#### Translation / test_stable / HistGB

| Representation | Spearman | AUROC | AUPRC | Precision@1000 |
|---|---:|---:|---:|---:|
| KU + structural | 0.0949 | 0.6687 | 0.0605 | 0.177 |
| KU + structural + diffusion | 0.1168 | 0.7070 | 0.0661 | 0.179 |

解释：

```text
在 primary translation task 上，
diffusion proxy 在 KU full 和 carrier structural exposure 之外继续提供增量。
```

#### Patent / test_all / HistGB

| Representation | Spearman | AUROC | AUPRC | Precision@1000 |
|---|---:|---:|---:|---:|
| KU + structural | 0.0513 | 0.6482 | 0.0276 | 0.149 |
| KU + structural + diffusion | 0.0718 | 0.7060 | 0.0320 | 0.157 |

解释：

```text
Patent forecasting 中，diffusion proxy 大幅增强 global discrimination，
并在 HistGB 下也提升 top-K precision。
```

#### Trial / test_stable / HistGB

| Representation | Spearman | AUROC | AUPRC | Precision@1000 |
|---|---:|---:|---:|---:|
| KU + structural | 0.0931 | 0.6978 | 0.0529 | 0.200 |
| KU + structural + diffusion | 0.1151 | 0.7434 | 0.0613 | 0.180 |

解释：

```text
diffusion proxy 显著提升 global discrimination；
但 carrier structural 对 trial top-K tail 仍更强。
```

---

## 7. Global metrics 与 top-K metrics 的差异

Stage 06G 中出现了一个重要现象：

```text
diffusion proxy 对 AUROC / AUPRC / Spearman 的提升非常稳定；
但对 NDCG@1000 / Precision@1000 / Enrichment@1000 的提升并不总是单调。
```

例如：

#### Trial / test_stable / HistGB

```text
KU + structural:
  AUROC = 0.6978
  AUPRC = 0.0529
  Precision@1000 = 0.200

KU + structural + diffusion:
  AUROC = 0.7434
  AUPRC = 0.0613
  Precision@1000 = 0.180
```

解释：

```text
diffusion proxy 改善了整体排序和 global discrimination；
但在 extreme top-K candidate discovery 上，carrier structural exposure 有时更敏感。
```

这说明：

```text
global field-level ranking
```

和：

```text
top-tail candidate discovery
```

可能需要不同机制。

对后续模型设计而言，这非常重要：

```text
1. diffusion term 用于改善全局场状态与整体 ranking；
2. source / structural exposure term 用于保持 top-K tail sensitivity；
3. final scoring function 需要同时兼顾 global discrimination 和 extreme top-K discovery。
```

---

## 8. Carrier graph 与 KU-KU graph 的区别

Stage 06F 和 06G 都使用“图”，但它们不是同一个层级。

### 8.1 Carrier graph

Carrier graph 是原始证据 / 对象层异质图：

```text
paper
patent
clinical trial
project
bioentity
drug
disease
gene
pathway
...
```

边是原始 typed relation：

```text
paper mentions gene
paper studies disease
patent claims compound
trial tests intervention
drug targets gene
project studies topic
...
```

它描述：

```text
evidence support / structural exposure / source strength
```

### 8.2 KU-KU graph

KU-KU graph 是知识单元层派生图：

```text
node = KU
edge = shared entity between two KUs
```

它描述：

```text
knowledge-neighborhood proximity / diffusion context
```

### 8.3 二者关系

可以理解为：

```text
carrier graph:
  原始 evidence/object layer

KU:
  从 evidence/entity layer 抽象出的 knowledge unit

KU-KU graph:
  在 knowledge-unit layer 上构造的 neighborhood graph
```

因此：

```text
carrier structural features 捕捉 source/exposure；
KU-KU diffusion features 捕捉 neighborhood/diffusion。
```

Stage 06F/06G 的结果显示这两类信号互补，而不是重复。

---

## 9. 从 Stage 06 到 Knowledge Field 模型

Stage 06 的三个阶段共同支持一个模型设计：

```text
Future translational emergence behaves like a field process over a KU graph,
with focal temporal state, carrier-graph source/exposure, and neighborhood diffusion.
```

中文：

```text
未来转化涌现更像是 KU 图上的场过程：
既受到 focal KU 自身历史状态影响，
也受到原始 carrier graph 的 source/exposure 影响，
还受到邻域 KU 状态扩散影响。
```

---

## 10. 建议的 Knowledge Field 模型形式

### 10.1 图与状态定义

令：

```text
G_K = (V_K, E_K)
```

其中：

```text
V_K:
  KU nodes

E_K:
  shared-entity KU-KU edges
```

每个 KU 在 cutoff 年 t 有状态：

```text
x_i(t)
```

其中包括：

```text
focal KU temporal state
KU schema features
historical activity
recent activity
growth
recency
```

Carrier graph source/exposure：

```text
s_i(t)
```

其中包括：

```text
carrier structural readout
degree exposure
neighbor type composition
edge-type diversity
evidence substrate activity
```

KU-neighborhood diffusion state：

```text
d_i(t)
```

其中包括：

```text
aggregated neighbor KU temporal states
weighted neighbor activity
neighbor patent/trial/translation history
neighbor growth
```

预测目标：

```text
y_i(t + 1 : t + 3)
```

包括：

```text
future translation heat
future patent heat
future trial heat
future emergence label
```

---

### 10.2 Conceptual model

一个自然的 Knowledge Field baseline 可以写成：

```text
score_i(t) =
  f_local(x_i(t))
  + f_source(s_i(t))
  + f_diffusion(Σ_j A_ij h(x_j(t)))
```

其中：

```text
f_local:
  focal KU temporal dynamics

f_source:
  carrier graph source/exposure

f_diffusion:
  KU-neighborhood diffusion

A_ij:
  KU-KU graph edge weight

h(x_j(t)):
  neighbor state transform
```

更接近 reaction-diffusion-source 的形式：

```text
u_i(t+1) =
  reaction(u_i(t), x_i(t))
  + diffusion(Σ_j A_ij (u_j(t) - u_i(t)))
  + source(s_i(t))
```

其中：

```text
reaction:
  focal KU self-dynamics / growth / saturation

diffusion:
  spillover from neighboring KUs

source:
  external evidence injection from carrier graph
```

---

## 11. 正式模型组件建议

基于 Stage 06 的结果，正式 Knowledge Field prototype 应至少包含以下组件。

### 11.1 Local KU state encoder

输入：

```text
x_i(t)
```

包含：

```text
KU temporal features
history counts
recent activity
growth
recency
schema categorical features
```

作用：

```text
捕捉 focal KU 自身 temporal trajectory。
```

对应 Stage 06E 证据。

---

### 11.2 Carrier source / exposure encoder

输入：

```text
s_i(t)
```

包含：

```text
carrier graph structural readout
support count
coverage
degree exposure
neighbor type composition
edge-type diversity
```

作用：

```text
捕捉原始 evidence substrate 的 source strength 和 structural exposure。
```

对应 Stage 06F 证据。

---

### 11.3 KU-neighborhood diffusion encoder

输入：

```text
{x_j(t): j ∈ N(i)}
A_ij
```

包含：

```text
neighbor state aggregation
weighted mean / max / top-k
neighbor history fractions
neighbor recent activity
neighbor growth
```

作用：

```text
捕捉 KU field 上的 local diffusion / spillover。
```

对应 Stage 06G 证据。

---

### 11.4 Final scoring head

输出：

```text
future heat score
emergence probability
```

建议多任务：

```text
translation
patent
trial
```

或者先做单任务 translation，再扩展到多任务。

---

## 12. 正式论文实验应该如何设计

Stage 06 的中间结果不建议直接作为论文主表。正式论文实验应整理为更干净的模型与消融。

### 12.1 Main model

```text
Knowledge Field Model
```

或者：

```text
KU Field Model
Reaction-Diffusion-Source KU Model
```

### 12.2 Baselines

建议包括：

```text
1. Temporal activity baseline
2. KU tabular baseline
3. Carrier structural baseline
4. KU-neighborhood diffusion baseline
5. Standard ML baselines
6. Full Knowledge Field model
```

### 12.3 Ablations

正式消融建议围绕模型组件，而不是 Stage 06 中间目录名。

例如：

| Ablation | 含义 |
|---|---|
| Full model | local + source + diffusion |
| w/o local state | 去掉 focal KU temporal state |
| w/o source | 去掉 carrier structural exposure |
| w/o diffusion | 去掉 KU-neighborhood diffusion |
| local only | 只用 focal KU state |
| source only | 只用 carrier structural exposure |
| diffusion only | 只用 KU-neighborhood diffusion |
| local + source | 对应 KU + carrier structural |
| local + diffusion | 对应 KU + diffusion |
| source + diffusion | 可选 |

这样更适合论文表述。

---

## 13. 方法学备注

Stage 06G 因 feature set 较宽、evaluation frame 很大，使用了低内存评估实现。

其中：

```text
Spearman:
  deterministic sampled estimate for large groups

AUROC / AUPRC:
  binned low-memory approximation

Top-K metrics:
  argpartition top-K computation
```

由于 Stage 06F/06G 是探索性 representation diagnostics，而非论文正式 benchmark，本报告主要将结果解释为：

```text
directional model-design evidence
```

而不是最终论文数值。

正式论文实验中应统一：

```text
evaluation implementation
random seeds
training protocol
model selection
baselines
ablation definitions
```

---

## 14. Stage 06 的总体结论

Stage 06E / 06F / 06G 共同给出如下结论：

```text
1. KU temporal/schema representation 有明显预测价值；
2. 原始 carrier graph structural exposure 在 KU full 之外提供独立信息；
3. KU-KU neighborhood diffusion proxy 在 KU full 和 carrier structural 之外继续提供增量；
4. diffusion proxy 主要提升 global discrimination；
5. carrier structural exposure 对 top-K tail discovery 仍然很重要；
6. 最合理的后续模型应同时包含 local KU state、carrier source/exposure、和 KU-neighborhood diffusion。
```

更简洁地说：

```text
未来转化涌现不是单纯由 focal KU 自身历史决定，
也不是单纯由原始 graph topology 决定，
而是 local state + source exposure + neighborhood diffusion 的组合过程。
```

---

## 15. 推荐下一步：Stage 07 Knowledge Field prototype

基于 Stage 06，下一步建议进入：

```text
Stage 07: Knowledge Field prototype
```

### 15.1 第一版不建议过度复杂

不建议一开始直接上复杂 GNN 或 full reaction-diffusion neural system。

建议先做一个可解释、可控的 prototype：

```text
Local + Source + Diffusion tabular field model
```

即把三类 feature 显式分块：

```text
local_features
source_features
diffusion_features
```

然后训练：

```text
HistGB
regularized logistic
ridge
possibly LightGBM / XGBoost if available
```

并做模块化消融：

```text
local only
source only
diffusion only
local + source
local + diffusion
local + source + diffusion
```

这会把 Stage 06 的探索性结论整理成正式模型框架。

---

### 15.2 第二版再考虑图模型

如果 prototype 证明模块组合稳定有效，再考虑：

```text
GraphSAGE over KU-KU graph
temporal GNN
reaction-diffusion neural update
learned diffusion weights
typed KU-KU edges
relation-aware diffusion
```

但这应是 Stage 08 或更后面的内容。

---

## 16. 最终一句话

Stage 06 的意义在于：

```text
它证明了 Knowledge Field model 的三个核心组件都有独立预测信号：
focal KU temporal state、carrier graph source/exposure、以及 KU-neighborhood diffusion。
```

因此，后续正式模型应从：

```text
local KU forecasting
```

升级为：

```text
source-driven diffusion over a KU field
```

即：

```text
Knowledge Field Model = Local KU State + Carrier Source/Exposure + KU-KU Diffusion
```