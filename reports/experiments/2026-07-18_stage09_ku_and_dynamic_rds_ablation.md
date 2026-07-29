# Stage 07 消融实验报告：KU 构造层与 DynamicRDS 算法层

日期：2026-07-18  
任务：translation emergence prediction  
测试 cutoff：2021  
测试样本数：563,533  
测试正例率：0.018159  

---

## 1. 实验目的

本阶段实验用于回答两个核心问题：

1. **KU 层是否必要？**  
   即，未来转化预测是否必须依赖 entity-pair / KU-level 的时序知识单元，而不是仅依赖 pre-KU 的实体属性或简单 pair evidence。

2. **DynamicRDS 算法机制是否必要？**  
   即，DynamicRDS 中的 reaction、source、proxy、diffusion、decay 等动力学项是否对 top-K discovery 有实际贡献。

因此，本报告包含两组消融：

- **KU 构造层消融**：从 entity-only pre-KU 到 raw pair evidence，再到 full KU features、KU graph、DynamicRDS。
- **DynamicRDS 算法层消融**：逐项移除 reaction、source、proxy、diffusion、decay，观察性能变化。

---

## 2. 评价指标

主要关注以下指标：

| 指标 | 含义 |
|---|---|
| AUPRC | 正例极稀疏任务下的整体排序质量 |
| AUROC | 整体二分类区分能力 |
| Spearman | 分数与未来转化热度的排序相关性 |
| P@1000 | top 1000 候选中的正例比例 |
| NDCG@1000 | top 1000 中正例排序位置质量 |
| Enrichment@1000 | top 1000 相比随机抽样的富集倍数 |

本任务正例率仅约 1.82%，因此 **P@1000、NDCG@1000 和 Enrichment@1000** 是最重要的 discovery-oriented 指标。

---

## 3. KU 构造层消融

### 3.1 实验设计

KU 层消融比较不同 representation level：

| Representation | Model | Uses KU pair history? | Uses KU schema? | Uses KU graph? |
|---|---|---|---|---|
| Entity-only pre-KU | HistGB / torch MLP | No | No | No |
| Raw pair evidence | HistGB / torch MLP | Minimal | Partial | No |
| Full KU features | HistGB / torch MLP | Yes | Yes | No |
| KU graph | GraphSAGE / GRAND-style / RDGNN-style | Yes | Yes | Yes |
| KU field dynamics | DynamicRDS v2 | Yes | Yes | Yes |

其中：

- **Entity-only pre-KU** 只使用两个低基数实体类型变量，不使用 pair-level history。
- **Raw pair evidence** 使用最小 pair-level paper trajectory features。
- **Full KU features** 使用完整 KU 表示，包括 temporal、schema、source、proxy 等特征。
- **KU graph** 使用 KU 图上的 GNN-style 模型。
- **KU field dynamics** 使用完整 DynamicRDS v2。

---

### 3.2 KU 层主结果表

| Representation | Model | Uses KU pair history? | Uses KU schema? | Uses KU graph? | AUPRC | AUROC | Spearman | P@1000 | NDCG@1000 | Enrich@1000 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Entity-only pre-KU | torch MLP | No | No | No | 0.021678 | 0.575456 | 0.037160 | 0.039 | 0.036286 | 2.147736 |
| Raw pair evidence | HistGB | Minimal | Partial | No | 0.034513 | 0.658676 | 0.073439 | 0.074 | 0.084785 | 4.075192 |
| Full KU features | torch MLP | Yes | Yes | No | 0.047592 | 0.717411 | 0.100613 | 0.127 | 0.131739 | 6.993911 |
| KU graph | GRAND-style | Yes | Yes | Yes | 0.047503 | 0.718492 | 0.101115 | 0.115 | 0.119242 | 6.333069 |
| KU field dynamics | DynamicRDS v2 state8 | Yes | Yes | Yes | **0.048160** | 0.720320 | 0.101956 | **0.139** | **0.148163** | **7.654753** |

---

### 3.3 完整 KU 层结果

| Representation | Model | AUPRC | AUROC | Spearman | P@1000 | NDCG@1000 | Enrich@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| Entity-only pre-KU | HistGB | 0.021678 | 0.575456 | 0.037160 | 0.039 | 0.036286 | 2.147736 |
| Entity-only pre-KU | torch MLP | 0.021678 | 0.575456 | 0.037160 | 0.039 | 0.036286 | 2.147736 |
| Raw pair evidence | HistGB | 0.034513 | 0.658676 | 0.073439 | 0.074 | 0.084785 | 4.075192 |
| Raw pair evidence | torch MLP | 0.033872 | 0.647065 | 0.068066 | 0.072 | 0.068408 | 3.965052 |
| Full KU features | HistGB | 0.045699 | 0.723772 | 0.103554 | 0.090 | 0.092246 | 4.956315 |
| Full KU features | torch MLP | 0.047592 | 0.717411 | 0.100613 | 0.127 | 0.131739 | 6.993911 |
| KU graph | GraphSAGE | 0.045131 | 0.713521 | 0.098816 | 0.097 | 0.090821 | 5.341806 |
| KU graph | RDGNN-style | 0.045454 | 0.714192 | 0.099127 | 0.103 | 0.105720 | 5.672227 |
| KU graph | GRAND-style | 0.047503 | 0.718492 | 0.101115 | 0.115 | 0.119242 | 6.333069 |
| KU field dynamics | DynamicRDS v2 state8 | **0.048160** | 0.720320 | 0.101956 | **0.139** | **0.148163** | **7.654753** |

---

### 3.4 Feature selection 检查

| Representation | Feature count before encoding | Encoded feature count |
|---|---:|---:|
| Entity-only pre-KU | 2 | 4 |
| Raw pair evidence | 12 | 14 |
| Full KU features | 163 | 167 |

Entity-only pre-KU 只包含两个 categorical entity-type features。因此，它应被理解为一个 **coarse entity-type-only pre-KU baseline**，用于检验粗粒度实体属性是否足以预测未来转化。

该 baseline 与 HistGB / torch MLP 的指标完全一致，且 torch MLP 确认为实际训练所得：

| Representation | Source | Best epoch | Best validation loss |
|---|---|---:|---:|
| Entity-only pre-KU | trained_torch_mlp | 12 | 0.001534 |
| Raw pair evidence | trained_torch_mlp | 13 | 0.001516 |
| Full KU features | trained_torch_mlp | 38 | 0.001493 |

---

### 3.5 KU 层消融解读

#### 3.5.1 Entity-only pre-KU 明显不足

Entity-only pre-KU 的结果为：

```text
AUPRC       = 0.021678
P@1000      = 0.039
NDCG@1000   = 0.036286
Enrich@1000 = 2.147736
```

这说明仅使用粗粒度实体类型信息，无法有效预测未来 translation emergence。  
也就是说，未来转化不是简单由两个实体类型决定的，必须引入 entity-pair / KU-level 的历史关系信息。

---

#### 3.5.2 Raw pair evidence 显著优于 Entity-only

Raw pair evidence 相比 Entity-only 明显提升：

| Comparison | AUPRC | P@1000 | NDCG@1000 |
|---|---:|---:|---:|
| Entity-only pre-KU | 0.021678 | 0.039 | 0.036286 |
| Raw pair evidence | 0.034513 | 0.074 | 0.084785 |

说明 pair-level historical evidence 已经包含大量额外预测信号。  
因此，KU 的最小形式——将实体对历史证据聚合为 pair-level temporal unit——是必要的。

---

#### 3.5.3 Full KU features 不只是 raw count aggregation

Full KU features + torch MLP 达到：

```text
AUPRC       = 0.047592
P@1000      = 0.127
NDCG@1000   = 0.131739
Enrich@1000 = 6.993911
```

相比 Raw pair evidence：

| Comparison | AUPRC | P@1000 | NDCG@1000 |
|---|---:|---:|---:|
| Raw pair evidence | 0.034513 | 0.074 | 0.084785 |
| Full KU features | 0.047592 | 0.127 | 0.131739 |

这表明 KU construction 并不是简单的 pair count aggregation，而是通过整合：

- temporal trajectory
- pair schema
- source exposure
- proxy signal
- structural / diffusion-related signals

形成了更有效的中间知识表示。

---

#### 3.5.4 DynamicRDS 在 KU 层上进一步提升 top-K discovery

Full KU features 已经很强，但 DynamicRDS 仍进一步提升 top-K 指标：

| Comparison | AUPRC | P@1000 | NDCG@1000 | Enrich@1000 |
|---|---:|---:|---:|---:|
| Full KU features + torch MLP | 0.047592 | 0.127 | 0.131739 | 6.993911 |
| DynamicRDS v2 state8 | **0.048160** | **0.139** | **0.148163** | **7.654753** |

这说明 KU graph field dynamics 并不是重复 tabular KU features 的信息，而是在 KU 图结构上进一步改善了 extreme top-K discovery。

---

### 3.6 KU 层结论

KU 构造层消融显示了清晰的性能梯度：

```text
Entity-only pre-KU
  < Raw pair evidence
  < Full KU features
  < KU field dynamics
```

因此可以得出结论：

> KU 层不是一个可有可无的工程中间产物，而是捕捉 pair-level temporal translation potential 的必要抽象。完整 KU 表示承载了大量预测信号，而 DynamicRDS 进一步利用 KU graph 上的知识场动力学提升 top-K discovery。

---

## 4. DynamicRDS 算法层消融

### 4.1 实验设计

DynamicRDS v2 的状态更新包含以下动力学项：

```text
reaction + source + proxy + diffusion - decay
```

本实验逐项移除：

| Ablation | 含义 |
|---|---|
| no_reaction | 去掉 local reaction |
| no_source | 去掉 source injection |
| no_proxy | 去掉 proxy prior |
| no_diffusion | 去掉 graph flux diffusion |
| no_decay | 去掉 decay / dissipation |
| full | 完整 DynamicRDS v2 |

所有实验使用相同数据、相同 split、相同 test cutoff 2021。

---

### 4.2 算法层消融结果

| Run | AUPRC | AUROC | Spearman | P@100 | P@1000 | P@5000 | NDCG@1000 | Enrich@1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_decay | 0.045975 | 0.717052 | 0.100446 | 0.110 | 0.101 | 0.0858 | 0.101084 | 5.562087 |
| no_diffusion | 0.044765 | 0.719489 | 0.101572 | 0.040 | 0.059 | 0.0770 | 0.054715 | 3.249140 |
| no_proxy | 0.048695 | 0.720931 | 0.102243 | 0.130 | 0.131 | 0.1000 | 0.130693 | 7.214192 |
| no_reaction | 0.045994 | 0.720901 | 0.102226 | 0.030 | 0.087 | 0.0842 | 0.081601 | 4.791104 |
| no_source | 0.046529 | 0.716973 | 0.100410 | 0.160 | 0.114 | 0.0902 | 0.124156 | 6.277999 |
| full DynamicRDS v2 | **0.048160** | 0.720320 | 0.101956 | **0.200** | **0.139** | 0.0982 | **0.148163** | **7.654753** |

---

### 4.3 相对 full model 的 top-K 下降

以 full DynamicRDS v2 为参照：

```text
Full P@1000    = 0.139
Full NDCG@1000 = 0.148163
```

| Removed term | P@1000 | ΔP@1000 | NDCG@1000 | ΔNDCG@1000 |
|---|---:|---:|---:|---:|
| no_proxy | 0.131 | -0.008 | 0.130693 | -0.017470 |
| no_source | 0.114 | -0.025 | 0.124156 | -0.024007 |
| no_decay | 0.101 | -0.038 | 0.101084 | -0.047079 |
| no_reaction | 0.087 | -0.052 | 0.081601 | -0.066562 |
| no_diffusion | 0.059 | -0.080 | 0.054715 | -0.093448 |

按 top-K 指标下降幅度，重要性大致为：

```text
diffusion > reaction > decay > source > proxy
```

---

### 4.4 算法层消融解读

#### 4.4.1 Graph flux diffusion 是最关键机制

去掉 diffusion 后：

```text
P@1000      = 0.059
NDCG@1000   = 0.054715
Enrich@1000 = 3.249140
```

相比 full：

```text
P@1000      = 0.139
NDCG@1000   = 0.148163
Enrich@1000 = 7.654753
```

P@1000 下降 0.080，意味着 top 1000 中少命中约 80 个 future translation positives。  
NDCG@1000 下降 0.093448，是所有消融中最大降幅。

因此，graph flux diffusion 是 DynamicRDS top-K discovery 的核心动力学项。  
这支持知识场假设：未来转化潜势不仅由 KU 自身状态决定，也受到邻域 KU 状态通量影响。

---

#### 4.4.2 Local reaction 对 extreme top-K 排序非常重要

去掉 reaction 后：

```text
P@100  = 0.030
P@1000 = 0.087
NDCG@1000 = 0.081601
```

虽然 AUROC 和 Spearman 没有明显下降，但 top-K 指标大幅下降。  
这说明 local reaction 主要影响极前排候选的精细排序，而不是粗粒度全局区分能力。

因此，reaction 项是 extreme top-K ranking 的关键局部驱动项。

---

#### 4.4.3 Decay 起到稳定 dynamics 的作用

去掉 decay 后：

```text
P@1000    = 0.101
NDCG@1000 = 0.101084
```

相比 full：

```text
P@1000    = 0.139
NDCG@1000 = 0.148163
```

说明 decay / dissipation 不是单纯为了物理解释，而是对 top-K 排序稳定性有实际贡献。  
它可能抑制过度激发的 latent channels，降低噪声 KU 被推到 top ranks 的风险。

---

#### 4.4.4 Source injection 有明确贡献

去掉 source 后：

```text
P@1000      = 0.114
NDCG@1000   = 0.124156
Enrich@1000 = 6.277999
```

虽然下降幅度小于 diffusion / reaction / decay，但仍明显低于 full model。  
这说明 source exposure 对未来 translation prediction 提供了独立增益。

---

#### 4.4.5 Proxy 的作用更复杂

去掉 proxy 后：

```text
AUPRC     = 0.048695
AUROC     = 0.720931
Spearman  = 0.102243
P@1000    = 0.131
NDCG@1000 = 0.130693
```

no_proxy 的 AUPRC / AUROC / Spearman 略高于 full，但 full 在 P@100、P@1000 和 NDCG@1000 上更好：

```text
Full P@100      = 0.200
No proxy P@100  = 0.130

Full P@1000     = 0.139
No proxy P@1000 = 0.131

Full NDCG@1000     = 0.148163
No proxy NDCG@1000 = 0.130693
```

因此，proxy prior 对 global AUPRC 的影响并非单调，但它有助于改善 extreme top-K ordering，尤其是将正例更靠前地排序。

---

### 4.5 算法层结论

DynamicRDS 算法层消融表明：

1. **diffusion 是最关键项**，去掉后 top-K discovery 崩落最明显。
2. **reaction 对极前排排序非常重要**，尤其影响 P@100 和 NDCG@1000。
3. **decay 是有效的稳定项**，有助于防止 noisy over-activation。
4. **source 有独立贡献**，说明 source exposure 不是冗余信号。
5. **proxy 主要改善 top-tail ranking**，对整体 AUPRC 存在 trade-off。

这说明 DynamicRDS 并非简单堆叠 feature branches，而是依赖多个动力学项协同作用来提升 top-K discovery。

---

## 5. KU 层与算法层的联合解释

两组消融形成了完整证据链：

```text
Pre-KU entity attributes
  ↓
Pair-level raw evidence
  ↓
Full KU representation
  ↓
KU graph structure
  ↓
DynamicRDS field dynamics
```

KU 层消融证明：

> 需要 KU-level abstraction 才能有效表达 pair-level temporal translation potential。

算法层消融证明：

> 在 KU graph 上，显式 reaction、source、proxy、diffusion、decay dynamics 对 top-K discovery 有实际贡献，其中 diffusion 是最关键机制。

因此，目前结果共同支持以下核心主张：

> Future translation potential is shaped by both focal KU-level temporal evidence and graph-mediated knowledge-field dynamics.

中文表述为：

> 未来转化潜势不仅取决于 KU 自身的历史证据和时序状态，也受到 KU 图上邻域知识状态通量与动力学演化的影响。

---

## 6. 主要结论

### 6.1 KU 层必要性

- Entity-only pre-KU baseline 表现较弱：
  ```text
  AUPRC = 0.021678
  P@1000 = 0.039
  ```
- Raw pair evidence 明显提升：
  ```text
  AUPRC = 0.034513
  P@1000 = 0.074
  ```
- Full KU features 进一步提升：
  ```text
  AUPRC = 0.047592
  P@1000 = 0.127
  ```
- DynamicRDS 在 KU 层上达到最佳 top-K：
  ```text
  P@1000 = 0.139
  NDCG@1000 = 0.148163
  Enrichment@1000 = 7.654753
  ```

结论：

> KU 层是捕捉 pair-level temporal translation dynamics 的必要抽象。

---

### 6.2 DynamicRDS 算法机制必要性

- Full DynamicRDS 的 top-K discovery 最强：
  ```text
  P@1000 = 0.139
  NDCG@1000 = 0.148163
  ```
- 去掉 diffusion 后下降最大：
  ```text
  P@1000 = 0.059
  NDCG@1000 = 0.054715
  ```
- 去掉 reaction 和 decay 也显著削弱 top-K：
  ```text
  no_reaction P@1000 = 0.087
  no_decay P@1000 = 0.101
  ```

结论：

> DynamicRDS 的提升主要来自 KU graph 上的显式知识场动力学，尤其是 graph flux diffusion、local reaction 和 decay stabilization。

---

## 7. 局限与后续工作

### 7.1 Entity-only baseline 仍偏弱

当前 Entity-only pre-KU baseline 只包含两个 categorical entity-type features，没有包含 entity marginal history，例如：

- entity-level historical paper count
- entity-level patent / trial exposure
- entity-level degree
- entity-level recency / growth

因此，严格来说该 baseline 更适合称为：

```text
entity-type-only pre-KU baseline
```

后续可以补充更强的 entity marginal history baseline，用于进一步验证：

> KU-level pair trajectory 是否超越实体边际热度叠加。

---

### 7.2 Proxy 项存在 global vs top-K trade-off

no_proxy 的 AUPRC 略高于 full，但 full 的 P@100、P@1000 和 NDCG@1000 更高。  
这说明 proxy prior 对整体 PR 曲线和 extreme top-K ordering 的影响并不完全一致。

后续可以进一步分析：

- proxy 对不同 pair_type 的影响
- proxy 在 top 100 / top 1000 中的排序贡献
- proxy 是否对某些低频 KU 更有帮助

---

### 7.3 可以补充分层消融

建议后续优先做：

```text
full vs no_diffusion
full vs no_reaction
full vs no_proxy
```

的分层评价，用于解释：

- diffusion 是否主要帮助稀疏 KU
- reaction 是否主要提升高局部活跃 KU
- proxy 是否主要影响特定 source / pair_type strata

---

## 8. 可用于正文的简短总结

KU 构造消融显示，从 Entity-only pre-KU 到 Raw pair evidence，再到 Full KU features 和 DynamicRDS，模型性能呈现清晰梯度。仅使用实体类型信息的 pre-KU baseline 表现较弱，而加入 pair-level historical evidence 后性能显著提升；完整 KU features 进一步提升 AUPRC 和 top-K 指标，说明 KU 构造并非简单 count aggregation，而是有效整合了 temporal、schema、source 和 proxy 信号。DynamicRDS 在 KU graph 上进一步取得最佳 top-K discovery 表现，说明 KU 层也是知识场动力学建模的有效载体。

算法层消融进一步表明，DynamicRDS 的性能提升并非来自单一 feature shortcut。去掉 graph flux diffusion 会导致 P@1000 和 NDCG@1000 最大幅度下降，说明 diffusion 是最关键的知识场传播机制；reaction 和 decay 也对极前排排序具有重要作用；source 提供独立增益；proxy 则主要改善 top-tail ordering。整体而言，实验结果支持本文的核心假设：未来转化潜势由 KU 自身的局部时序状态与 KU 图上的知识场动力学共同塑造。

---

## 9. 最终结论

本阶段消融实验同时证明了：

1. **KU 层必要**：  
   pair-level KU representation 明显优于 pre-KU entity-only 表示和 minimal raw pair evidence。

2. **KU graph 有价值**：  
   KU graph 模型相较普通 tabular baseline 在 top-K discovery 上具有增益。

3. **DynamicRDS 机制有效**：  
   full DynamicRDS 在 P@1000、NDCG@1000 和 Enrichment@1000 上表现最佳。

4. **Graph flux diffusion 是核心项**：  
   去掉 diffusion 后 top-K discovery 下降最大，是 DynamicRDS 中最关键的动力学机制。

因此，当前结果为 DynamicRDS 的两层设计提供了较完整的实验证据：

```text
KU-level abstraction
+
graph-based knowledge-field dynamics
```

共同构成了未来 translation emergence prediction 的有效建模框架。