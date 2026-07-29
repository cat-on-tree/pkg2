# Stage 07G 实验总结：DynamicRDS v2 分层评价与 Top-K 组成分析

日期：2026-07-10  
阶段：Stage 07G  
主题：DynamicRDS v2 state8 的 stratified evaluation / top-K composition analysis  
用途：解释 DynamicRDS v2 在不同 KU 子群体中的表现，验证模型优势是否来自单一 shortcut  
注意：本文档不是新的训练实验，而是基于已训练模型 prediction 的 post-hoc 分层评价。

---

## 1. 背景与目的

前序 Stage 07F 已经完成 DynamicRDS v2 vector-state knowledge field dynamics 实验，并表明当前最强模型为：

```text
local_source_proxy_dynamic_rds_v2_state8_seed42
```

在主任务：

```text
Future Translation Heat Forecasting
```

上，DynamicRDS v2 state8 已经在整体指标上超过了主要 baseline，包括：

```text
RDS v0
GRAND-style
RDGNN-style
WeightedDiffusion
GraphSAGE
MLP
```

其中最关键的整体结果为：

```text
Test AUPRC       = 0.048160
Test AUROC       = 0.720320
Test Spearman    = 0.101956
Precision@1000   = 0.139
NDCG@1000        = 0.148163
Enrichment@1000  = 7.654753
```

因此，本阶段实验的目的不是再次证明整体最强，而是进一步回答：

```text
DynamicRDS v2 为什么强？
它在哪些类型的 KU 上强？
它是否只是依赖某个简单 shortcut？
```

具体来说，本次分层评价重点回答以下问题：

```text
1. DynamicRDS 的优势是否只来自 mature / high-paper-count KU？
2. DynamicRDS 是否主要依赖近期 paper activity？
3. DynamicRDS 是否只是按照 source exposure 排序？
4. DynamicRDS 是否只是依赖 handcrafted diffusion proxy？
5. 不同 pair_type 上模型表现是否不同？
6. 模型全局 top-1000 候选主要由哪些类型 KU 组成？
```

---

## 2. 实验设定

### 2.1 数据版本

本实验使用 Stage 07 latest-window GNN input：

```text
data/datasets/knowledge_field/diabetes_2000_2024_v1_gnn_local_source_proxy_latest_split/
```

该数据包含：

```text
local KU temporal features
carrier source / exposure features
KU-neighborhood diffusion proxy features
KU-KU shared-entity graph
```

---

### 2.2 任务

本实验仍然使用 Stage 07 主任务：

```text
Future Translation Heat Forecasting
```

Eligibility：

```text
history_paper_count > 0
AND history_patent_count == 0
AND history_trial_count == 0
```

Target：

```text
target_future_translation_heat_3yr
=
log1p(target_future_patent_count_3yr + target_future_trial_count_3yr)
```

Binary label：

```text
label_future_translation_emergence_3yr
=
1[target_future_patent_count_3yr + target_future_trial_count_3yr > 0]
```

Test cutoff：

```text
cutoff_year = 2021
```

Target window：

```text
2022-2024
```

Test set size：

```text
n = 563,533
positive_count = 10,233
positive_rate = 0.018159
```

也就是说，随机选择一个 eligible KU，其未来 3 年发生 patent / trial 转化的概率约为：

```text
1.82%
```

---

### 2.3 比较模型

本次分层评价比较三个模型：

```text
dynamic_rds_v2_state8
grand_style
rdgnn_style
```

对应 run：

```text
local_source_proxy_dynamic_rds_v2_state8_seed42
local_source_proxy_grand_style_v0_seed42
local_source_proxy_rdgnn_style_v0_seed42
```

这三个模型均使用：

```text
local_source_proxy
```

输入，以保证 feature set 一致。

---

## 3. 评价方法

本实验包含两类评价。

---

### 3.1 Within-stratum evaluation

Within-stratum evaluation 的含义是：

```text
在某个 KU 子群体内部重新排序并计算指标。
```

例如：

```text
只看 early-stage KU，
在 early-stage KU 内部根据模型分数排序，
计算 AUPRC / AUROC / P@1000 / NDCG@1000 / Enrichment@1000。
```

它回答的问题是：

```text
如果我们只关注某一类 KU，模型在这一类内部是否仍然有效？
```

---

### 3.2 Global top-K composition

Global top-K composition 的含义是：

```text
先在全部 test eligible KUs 上按模型分数全局排序，
取 top-K，
再统计这些 top-K 候选来自哪些 strata。
```

例如：

```text
DynamicRDS top1000 中有多少 seed-stage KU？
有多少 early-stage KU？
这些 KU 中有多少未来真的转化？
```

它回答的问题是：

```text
模型最终推荐的 top candidates 是由哪些类型 KU 组成的？
```

这与 within-stratum evaluation 不同。

一个 stratum 内部表现强，不代表它一定大量进入全局 top-K；  
反过来，一个 stratum 在全局 top-K 中占比高，也不代表模型在该 stratum 内部的所有排序都最好。

---

## 4. 分层定义

本次评价包含以下 strata。

### 4.1 历史论文数量分层

根据：

```text
history_paper_count
```

划分为：

```text
seed_1_2:
  1-2 篇历史论文

early_3_9:
  3-9 篇历史论文

established_10_49:
  10-49 篇历史论文

mature_50_plus:
  50 篇及以上历史论文
```

该分层用于判断：

```text
模型是否只依赖 mature KU，还是也能发现 seed / early KU。
```

---

### 4.2 近期论文活跃度分层

根据：

```text
recent_3yr_paper_count
```

划分为：

```text
dormant_recent0:
  最近 3 年没有论文

low_recent_1_2:
  最近 3 年 1-2 篇论文

active_recent_3_9:
  最近 3 年 3-9 篇论文

very_active_recent_10_plus:
  最近 3 年 10 篇及以上论文
```

该分层用于判断：

```text
模型是否只是依赖 recent paper activity。
```

---

### 4.3 paper growth 分层

根据 paper growth score 的分位数划分为：

```text
low_growth_q0_25
mid_growth_q25_75
high_growth_q75_100
```

用于判断：

```text
模型是否主要依赖近期增长趋势。
```

---

### 4.4 source score 分层

根据 carrier source / exposure features 构造启发式：

```text
source_score
```

并按分位数划分为：

```text
low_source_q0_25
mid_source_q25_75
high_source_q75_100
```

注意：当前 `source_score` 是基于 feature name 自动推断的 source feature group，并做 z-score 平均得到的启发式分数。它适合作为分层分析，不等价于 DynamicRDS 内部的 source term。

该分层用于判断：

```text
模型是否只是按照 source exposure 排序。
```

---

### 4.5 proxy score 分层

根据 diffusion proxy features 构造启发式：

```text
proxy_score
```

并按分位数划分为：

```text
low_proxy_q0_25
mid_proxy_q25_75
high_proxy_q75_100
```

同样，当前 `proxy_score` 是基于 feature name 自动推断的 proxy feature group 的启发式聚合，不等价于 DynamicRDS 内部的 proxy term。

该分层用于判断：

```text
模型是否只是依赖 handcrafted diffusion proxy。
```

---

### 4.6 pair_type 分层

根据 KU 类型划分为：

```text
chemical-disease
chemical-gene
gene-disease
```

用于判断：

```text
不同知识单元类型的预测难度和模型优势是否不同。
```

---

## 5. Overall 结果

整体结果如下：

| Model | AUPRC | AUROC | Spearman | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DynamicRDS v2 state8 | **0.048172** | **0.720318** | **0.101955** | 0.190 | **0.139** | **0.0980** | **0.148235** | **7.654753** |
| GRAND-style | 0.047502 | 0.718462 | 0.101101 | **0.200** | 0.117 | 0.0904 | 0.120904 | 6.443210 |
| RDGNN-style | 0.045362 | 0.713884 | 0.098985 | 0.100 | 0.106 | 0.0902 | 0.107654 | 5.837440 |

结论：

```text
DynamicRDS v2 state8 在整体 AUPRC、AUROC、Spearman、P@1000、P@5000、NDCG@1000 和 Enrichment@1000 上均为最强。
```

GRAND-style 在 P@100 上略高：

```text
GRAND-style P@100 = 0.200
DynamicRDS P@100 = 0.190
```

但 P@100 的差异只对应：

```text
20/100 vs 19/100
```

而 DynamicRDS 在更稳定的 top-1000 指标上明显更强：

```text
DynamicRDS P@1000 = 0.139
GRAND-style P@1000 = 0.117
RDGNN-style P@1000 = 0.106
```

这意味着：

```text
DynamicRDS top1000 中约有 139 个未来转化 KU；
GRAND-style 约有 117 个；
RDGNN-style 约有 106 个。
```

因此，整体评价确认：

```text
DynamicRDS v2 state8 是当前综合最强模型。
```

---

## 6. 历史论文数量分层结果

### 6.1 Within-stratum 结果

| Stratum | Model | Positive rate | AUPRC | AUROC | P@1000 | NDCG@1000 | Enrich@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| early_3_9 | DynamicRDS | 0.0115 | **0.03059** | **0.71230** | **0.083** | **0.09169** | **7.22** |
| early_3_9 | GRAND | 0.0115 | 0.03012 | 0.71154 | 0.079 | 0.08029 | 6.87 |
| early_3_9 | RDGNN | 0.0115 | 0.02963 | 0.70654 | 0.076 | 0.08102 | 6.61 |
| seed_1_2 | DynamicRDS | 0.0292 | **0.05463** | 0.65907 | **0.096** | **0.09955** | **3.28** |
| seed_1_2 | GRAND | 0.0292 | 0.05444 | **0.65962** | **0.096** | 0.09608 | **3.28** |
| seed_1_2 | RDGNN | 0.0292 | 0.05325 | 0.65643 | 0.084 | 0.08460 | 2.87 |

主要观察：

```text
1. early_3_9 是最大 KU 群体，n = 364,617，但 positive rate 只有 1.15%；
2. DynamicRDS 在 early_3_9 内部取得最高 AUPRC、AUROC、P@1000 和 NDCG@1000；
3. 在 seed_1_2 中，DynamicRDS 与 GRAND 的 P@1000 持平，但 DynamicRDS 的 AUPRC 和 NDCG@1000 略高；
4. mature_50_plus 的 AUPRC 较高，但该组样本量较小，且正样本率本身更高，因此不应过度解读。
```

---

### 6.2 Top-1000 composition

DynamicRDS top1000 按历史阶段组成：

| Stratum | Count in top1000 | Positive in top1000 | Precision |
|---|---:|---:|---:|
| seed_1_2 | 195 | 39 | 0.200 |
| early_3_9 | 633 | 72 | 0.114 |
| established_10_49 | 158 | 27 | 0.171 |
| mature_50_plus | 14 | 1 | 0.071 |

合计：

```text
seed + early count = 195 + 633 = 828
```

即：

```text
DynamicRDS top1000 中 82.8% 是 seed 或 early-stage KU。
```

top1000 中真实 positives：

```text
total positives = 139
seed + early positives = 39 + 72 = 111
```

即：

```text
DynamicRDS top1000 中约 79.9% 的真实命中来自 seed / early-stage KU。
```

这说明：

```text
DynamicRDS 不是主要依靠 mature KU 取胜；
它的 top candidates 大部分来自 seed / early-stage KU，
并且这些 early candidates 确实有较高未来转化命中率。
```

这是本次分层评价最重要的结论之一：

```text
DynamicRDS v2 具有 early translational discovery 能力。
```

---

## 7. 近期论文活跃度分层结果

### 7.1 Within-stratum 结果

DynamicRDS 在 `dormant_recent0` 中：

```text
positive_rate = 0.01445
AUPRC = 0.04049
AUROC = 0.72704
P@1000 = 0.080
Enrichment@1000 = 5.53
```

这说明：

```text
即使在最近 3 年没有 paper activity 的 KU 中，
DynamicRDS 仍能将未来转化候选富集到随机的 5.5 倍。
```

DynamicRDS 在 `very_active_recent_10_plus` 中：

```text
positive_rate = 0.05632
AUPRC = 0.10277
P@1000 = 0.138
```

该组正样本率本身较高，因此 AUPRC 和 P@1000 较高是合理的。

---

### 7.2 Top-1000 composition

DynamicRDS top1000 按近期论文活跃度组成：

| Stratum | Count in top1000 | Positive in top1000 | Precision |
|---|---:|---:|---:|
| dormant_recent0 | 392 | 57 | 0.145 |
| low_recent_1_2 | 436 | 60 | 0.138 |
| active_recent_3_9 | 152 | 21 | 0.138 |
| very_active_recent_10_plus | 20 | 1 | 0.050 |

可以看到：

```text
dormant + low_recent = 392 + 436 = 828
```

即：

```text
DynamicRDS top1000 中 82.8% 来自近期论文不活跃或低活跃 KU。
```

这说明：

```text
DynamicRDS 并不是简单地把 recent paper activity 高的 KU 排到前面。
```

相反，模型能够在近期论文不活跃或低活跃的 KU 中，结合 source / proxy / graph flux dynamics 发现未来转化候选。

---

## 8. Paper growth 分层结果

在 paper growth 分层中，三个模型均在 `mid_growth_q25_75` 中表现较强。

DynamicRDS：

```text
mid_growth_q25_75:
  AUPRC = 0.06190
  AUROC = 0.75445
  P@1000 = 0.137
  Enrichment@1000 = 7.76
```

GRAND-style：

```text
mid_growth_q25_75:
  AUPRC = 0.06331
  P@1000 = 0.154
  Enrichment@1000 = 8.72
```

RDGNN-style：

```text
mid_growth_q25_75:
  AUPRC = 0.06390
  P@1000 = 0.164
  Enrichment@1000 = 9.29
```

该结果说明：

```text
在 mid-growth 子群体内部，GRAND-style / RDGNN-style 的 top-K 排序也很强。
```

因此，DynamicRDS 的优势不是在每一个单独 stratum 上都绝对最大，而是在整体和多个关键 discovery 维度上综合最强。

Top1000 composition 中，DynamicRDS 候选分布为：

```text
high_growth: 310 candidates, 43 positives
mid_growth:  437 candidates, 61 positives
low_growth:  253 candidates, 35 positives
```

这说明：

```text
DynamicRDS top candidates 并不只来自 high-growth KU。
```

---

## 9. Source score 分层结果

### 9.1 Within-stratum 结果

DynamicRDS 在 high-source 组中表现最好：

```text
high_source_q75_100:
  AUPRC = 0.05657
  AUROC = 0.73561
  Spearman = 0.11050
  P@1000 = 0.127
  NDCG@1000 = 0.13866
  Enrichment@1000 = 6.81
```

对应 GRAND-style：

```text
P@1000 = 0.121
NDCG@1000 = 0.12354
```

对应 RDGNN-style：

```text
P@1000 = 0.108
NDCG@1000 = 0.10984
```

这说明：

```text
DynamicRDS 在 source-rich KU 中最能利用外部 evidence exposure。
```

该结果支持 DynamicRDS 中 source injection 设计的必要性。

---

### 9.2 Top-1000 composition

DynamicRDS top1000 按 source strata 组成：

| Source stratum | Count in top1000 | Positive in top1000 | Precision |
|---|---:|---:|---:|
| high_source | 235 | 26 | 0.111 |
| mid_source | 512 | 79 | 0.154 |
| low_source | 253 | 34 | 0.134 |

如果模型只是简单按照 source score 排序，则 top1000 应该主要来自 high-source 组。

但实际结果显示：

```text
mid_source 占 top1000 的 51.2%，并贡献了 56.8% 的 top1000 positives。
```

因此可以得出：

```text
DynamicRDS 能利用 source exposure，
但并不是简单的 source-score shortcut。
```

模型 top candidates 来自 source、local state、proxy prior 和 graph diffusion 的综合作用。

---

## 10. Proxy score 分层结果

### 10.1 Within-stratum 结果

DynamicRDS 在 proxy 分层中的结果如下：

| Proxy stratum | Positive rate | AUPRC | AUROC | P@1000 | NDCG@1000 | Enrichment |
|---|---:|---:|---:|---:|---:|---:|
| high_proxy | 0.02530 | 0.05671 | 0.69697 | 0.108 | 0.11783 | 4.27 |
| mid_proxy | 0.01535 | 0.03955 | 0.71534 | 0.097 | 0.10507 | 6.32 |
| low_proxy | 0.01664 | 0.05031 | 0.72903 | 0.113 | 0.11791 | 6.79 |

一个重要观察是：

```text
low_proxy 组的 P@1000 和 AUROC 不低，甚至 P@1000 高于 high_proxy。
```

这说明：

```text
DynamicRDS 不是简单依赖 high diffusion proxy 排序。
```

---

### 10.2 Top-1000 composition

DynamicRDS top1000 按 proxy strata 组成：

| Proxy stratum | Count in top1000 | Positive in top1000 | Precision |
|---|---:|---:|---:|
| high_proxy | 241 | 33 | 0.137 |
| mid_proxy | 509 | 69 | 0.136 |
| low_proxy | 250 | 37 | 0.148 |

可以看到：

```text
top1000 并没有被 high_proxy 垄断；
low_proxy 组的 top1000 内部 precision 反而最高。
```

该结果非常重要，因为它说明：

```text
DynamicRDS 并不是简单地利用 handcrafted proxy shortcut。
```

Proxy prior 对模型有帮助，但模型最终排序并不等同于 proxy_score 排序。

---

## 11. Pair type 分层结果

### 11.1 Chemical-disease

DynamicRDS 在 chemical-disease 上优势最明显：

```text
DynamicRDS:
  AUPRC = 0.05382
  AUROC = 0.69986
  P@1000 = 0.115
  NDCG@1000 = 0.12164
```

GRAND-style：

```text
P@1000 = 0.103
NDCG@1000 = 0.10837
```

RDGNN-style：

```text
P@1000 = 0.098
NDCG@1000 = 0.09738
```

结论：

```text
DynamicRDS 对 chemical-disease 转化预测特别有效。
```

这符合生物医学转化直觉，因为 chemical-disease pair 更接近 drug-disease / treatment-related translation。

---

### 11.2 Chemical-gene

Chemical-gene 的整体 positive rate 较低：

```text
positive_rate = 0.01108
```

但 DynamicRDS 的 top-K enrichment 很高：

```text
P@1000 = 0.120
Enrichment@1000 = 10.83
```

说明：

```text
chemical-gene 虽然整体转化较少，
但模型排在前面的 chemical-gene KUs 具有很高候选价值。
```

在该 stratum 中，DynamicRDS 与 GRAND-style 表现接近。

---

### 11.3 Gene-disease

在 gene-disease 中，GRAND-style 略高于 DynamicRDS：

```text
GRAND-style:
  AUPRC = 0.04570
  P@1000 = 0.102

DynamicRDS:
  AUPRC = 0.04488
  P@1000 = 0.098
```

差距不大。

该结果提示：

```text
gene-disease 可能更适合 generic graph smoothing；
chemical-disease 更受益于 DynamicRDS 的 source/proxy/flux decomposition。
```

这一点可作为后续 case study 的方向，但当前不宜过度解释。

---

## 12. 主要结论

本次分层评价得到以下核心结论。

---

### 12.1 DynamicRDS v2 整体仍然最强

在全部 test eligible KU 上：

```text
DynamicRDS v2 state8 在 AUPRC、AUROC、Spearman、P@1000、NDCG@1000 和 Enrichment@1000 上均优于 GRAND-style 和 RDGNN-style。
```

这与 Stage 07F 的主表结论一致。

---

### 12.2 DynamicRDS 的优势不是来自 mature KU

DynamicRDS top1000 中：

```text
82.8% 是 seed / early-stage KU；
约 79.9% 的 top1000 positives 来自 seed / early-stage KU。
```

这说明：

```text
DynamicRDS 不是简单地把成熟高论文数 KU 排在前面，
而是具有 early translational discovery 能力。
```

---

### 12.3 DynamicRDS 不是 recent activity shortcut

DynamicRDS top1000 中：

```text
dormant_recent0 + low_recent_1_2 = 82.8%
```

也就是说，top candidates 主要不是 recent paper activity 很高的 KU。

同时，这些 dormant / low-recent KUs 仍然具有较高未来转化命中率。

因此：

```text
DynamicRDS 并不是只依赖近期论文热度。
```

---

### 12.4 DynamicRDS 能利用 source，但不是 source shortcut

DynamicRDS 在 high-source 组内表现最好，说明：

```text
source exposure 对模型有帮助。
```

但 DynamicRDS top1000 中，贡献最多 positives 的是：

```text
mid_source
```

而不是 high_source。

因此：

```text
DynamicRDS 并不是简单按照 source exposure 排序。
```

---

### 12.5 DynamicRDS 不是 proxy shortcut

如果模型只是依赖 diffusion proxy，top1000 应主要来自 high_proxy 组。

但实际结果显示：

```text
top1000 分布在 low / mid / high proxy；
low_proxy 组的 top1000 precision 甚至最高。
```

因此：

```text
DynamicRDS 不是简单依赖 handcrafted proxy。
```

这支持 DynamicRDS 的核心设计假设：

```text
模型通过 local state、source exposure、proxy prior 和 graph flux diffusion 的组合进行预测。
```

---

### 12.6 DynamicRDS 对 chemical-disease 特别有效

DynamicRDS 在 chemical-disease stratum 上明显优于 GRAND-style 和 RDGNN-style。

这符合 translational discovery 任务的直觉：

```text
chemical-disease KUs 更接近 drug-disease / treatment-related translation，
因此更容易受益于 source-proxy-diffusion dynamics。
```

---

## 13. 与 Stage 07 主结果的关系

Stage 07F 主结果已经说明：

```text
DynamicRDS v2 state8 是当前综合最强模型。
```

本次 Stage 07G 分层评价进一步说明：

```text
DynamicRDS 为什么强，以及它的优势是否可靠。
```

具体来说，Stage 07G 证明：

```text
1. DynamicRDS 的优势不只来自 mature KUs；
2. DynamicRDS 的 top candidates 多数是 seed / early KUs；
3. DynamicRDS 不是 recent paper activity shortcut；
4. DynamicRDS 不是 source shortcut；
5. DynamicRDS 不是 proxy shortcut；
6. DynamicRDS 对 chemical-disease KUs 特别有效。
```

因此，本次实验不是替代 Stage 07F 的 overall comparison，而是对 Stage 07F 的模型行为解释和分层验证。

---

## 14. 局限性

### 14.1 当前仍是单 seed 分析

本次分析基于：

```text
seed = 42
```

后续多 seed 稳健性验证完成后，应对主要分层结论进行复核。

---

### 14.2 source_score / proxy_score 是启发式分层变量

当前 `source_score` 和 `proxy_score` 是根据 feature names 自动推断 feature group 后，进行 z-score 平均得到的启发式分数。

它们适合进行初步分层分析，但不能替代 DynamicRDS 内部项：

```text
source_i^*(t)
proxy_i^*(t)
diffusion_i(t)
reaction_i(t)
decay_i(t)
```

后续 Stage 08 仍需要导出模型内部动力学项进行更严格的机制解释。

---

### 14.3 Within-stratum P@1000 受 stratum size 和 positive rate 影响

不同 strata 的样本量和 positive rate 差异较大，因此不能只看 P@1000 或 AUPRC 单一指标。

解释时应同时参考：

```text
positive_rate
AUPRC
AUROC
P@1000
NDCG@1000
Enrichment@1000
top-K composition
```

---

## 15. 推荐后续工作

### 15.1 将分层评价纳入 Stage 07 总报告

建议在 Stage 07 总报告中新增：

```text
Stage 07G: Stratified Evaluation and Top-K Composition Analysis
```

重点展示：

```text
1. Overall comparison
2. History paper stage stratification
3. Recent paper activity stratification
4. Source/proxy stratification
5. Pair-type stratification
6. Global top-1000 composition
```

---

### 15.2 进入 Stage 08 解释分析

下一步建议导出 DynamicRDS v2 的显式动力学项：

```text
u_i(t)
Δu_i(t)
u_i(t+Δt)
reaction_i(t)
source_i^*(t)
proxy_i^*(t)
diffusion_i(t)
decay_i(t)
```

并分析：

```text
1. top predicted KUs 是 reaction-driven、source-driven、proxy-driven 还是 diffusion-driven？
2. high incoming flux 的邻居是什么？
3. source 高但 proxy 低的 KU 表现如何？
4. proxy 低但被模型高分预测的 KU 是否对应 early discovery？
5. state_dim=8 的 latent channels 是否存在可解释模式？
6. different pair_types 是否具有不同 field dynamics？
```

---

## 16. 当前推荐表述

### 中文表述

```text
分层评价显示，DynamicRDS v2 state8 不仅在整体 test set 上优于 GRAND-style 和 RDGNN-style，而且其优势并非来自单一 shortcut。模型的 top-1000 候选中，82.8% 为 seed 或 early-stage KUs，并贡献了约 79.9% 的真实未来转化正例，说明 DynamicRDS 具有 early translational discovery 能力。进一步地，top-1000 候选并未被 high-source 或 high-proxy strata 垄断；low-proxy KUs 仍具有较高 top-K precision，说明 DynamicRDS 并不是简单依赖 handcrafted proxy，而是综合 local state、source exposure、proxy prior 和 graph flux diffusion 进行预测。
```

---

### 英文表述

```text
The stratified evaluation shows that DynamicRDS v2 state8 not only outperforms GRAND-style and RDGNN-style baselines on the overall test set, but also that its advantage is not driven by a single shortcut. In the global top-1000 candidates, 82.8% are seed or early-stage KUs, contributing approximately 79.9% of all true future translation positives in the top-1000. This indicates that DynamicRDS is effective for early translational discovery rather than merely ranking mature high-activity KUs. Moreover, the top-1000 candidates are not dominated by high-source or high-proxy strata; low-proxy KUs still achieve strong top-K precision, suggesting that DynamicRDS integrates local state, source exposure, proxy prior, and graph flux dynamics rather than simply following handcrafted proxy signals.
```

---

## 17. 最终结论

本次 Stage 07G 分层评价表明：

```text
1. DynamicRDS v2 state8 在整体评价中仍然最强；
2. DynamicRDS 的 top-K 优势主要来自 seed / early-stage KUs，而不是 mature KUs；
3. DynamicRDS 不是 recent activity shortcut；
4. DynamicRDS 能利用 source exposure，但不是 source-score shortcut；
5. DynamicRDS 能利用 proxy prior，但不是 proxy shortcut；
6. DynamicRDS 对 chemical-disease KUs 表现尤其强；
7. 分层结果进一步支持 DynamicRDS 作为显式 Knowledge Field Dynamics 模型的合理性。
```

一句话总结：

```text
DynamicRDS v2 的优势不仅体现在整体跑分上，更体现在它能够在 seed / early-stage、低近期活跃度、不同 source/proxy 条件和不同 pair_type 中稳定发现未来转化候选，说明它捕捉到的是综合知识场动力学，而不是单一特征 shortcut。
```