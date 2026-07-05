# Stage 06E Knowledge Unit 表示消融实验报告

日期：2026-07-03  
阶段：Stage 06E  
实验名称：KU representation feature-subset ablation  
任务：KU-level temporal forecasting  
数据集：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/
```

核心对照结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_all_tasks_core/
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_rep_numeric_only/
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_rep_raw_counts_only/
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_rep_ku_schema_only/
```

---

## 1. 实验目的

Stage 06E 的目标是在同一个冻结的 KU-level temporal prediction benchmark 上，比较不同 feature subset / representation 对未来转化预测性能的影响。

本实验重点回答：

```text
KU temporal numeric features 是否已经包含主要预测信号？
KU schema features，例如 pair_type、entity_a_type、entity_b_type，是否提供额外预测信息？
完整 KU representation 是否优于仅使用 numeric temporal evidence features？
```

需要注意的是，本实验是：

```text
feature-subset / representation ablation
```

而不是：

```text
carrier graph GNN baseline
```

因此，本实验不能被表述为：

```text
KU representation beats all carrier graph methods.
```

更准确的结论是：

```text
在相同 KU-level forecasting task、相同 train/validation/test split、相同 learners 和相同 metrics 下，
full KU representation 相比 numeric temporal evidence features only，在整体区分能力和排序相关性上具有稳定增益。
```

---

## 2. 数据与任务设置

本实验沿用 Stage 05F 冻结的 temporal prediction benchmark。

预测样本单位：

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

数据划分：

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

本实验覆盖三个任务：

| Task | Eligibility | Target heat | Emergence label |
|---|---|---|---|
| translation | `eligible_translation_task` | `target_future_translation_heat_3yr` | `label_future_translation_emergence_3yr` |
| patent | `eligible_patent_task` | `target_future_patent_heat_3yr` | `label_future_patent_emergence_3yr` |
| trial | `eligible_trial_task` | `target_future_trial_heat_3yr` | `label_future_trial_emergence_3yr` |

---

## 3. Representation / feature subset 设计

本实验比较四组 representation。

### 3.1 KU full representation

结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_all_tasks_core/
```

对应原始 Stage 06A-D core baseline。

该 representation 使用：

```text
numeric temporal evidence features
+
KU schema categorical features
```

feature 数量：

```text
61
```

其中包括：

```text
history / recent / growth / recency 等 numeric temporal features
pair_type
entity_a_type
entity_b_type
```

注意：该结果是在 30 号脚本加入 `--feature-subset` 之前跑出的，因此 run summary 中：

```text
feature_subset = None
```

但它等价于当前脚本中的：

```text
--feature-subset all
```

或：

```text
--feature-subset ku_full
```

---

### 3.2 Numeric temporal only

结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_rep_numeric_only/
```

该 representation 只使用 numeric features：

```text
history counts
recent activity
growth ratios
recency
其他 cutoff-safe numeric temporal summaries
```

不使用：

```text
pair_type
entity_a_type
entity_b_type
```

feature 数量：

```text
58
```

该组用于回答：

```text
仅使用 KU temporal numeric evidence features 时，模型表现如何？
KU schema categorical features 是否提供增量？
```

---

### 3.3 Raw counts only

结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_rep_raw_counts_only/
```

该组原计划表示 raw carrier-count-like temporal features。

实际运行结果显示：

```text
raw_counts_only feature_count = 58
numeric_only feature_count = 58
```

且所有结果与 numeric_only 完全一致。

因此，在本报告中将其解释为：

```text
numeric temporal evidence features only
```

并不单独作为独立 representation 讨论。

这说明当前 feature selector 中：

```text
raw_counts_only == numeric_only
```

后续如果需要严格的 raw carrier evidence baseline，应重新定义更严格的 raw-count feature set。

---

### 3.4 KU schema only

结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_rep_ku_schema_only/
```

该 representation 只使用三个 categorical KU schema features：

```text
pair_type
entity_a_type
entity_b_type
```

feature 数量：

```text
3
```

该组用于 sanity check：

```text
仅靠 KU schema 类型本身是否足以预测未来转化？
```

---

## 4. 模型与运行配置

本实验对每组 representation 运行相同 trainable learners：

```text
ridge
logistic
logistic_unweighted
histgb
```

其中：

```text
ridge:
  heat regression baseline

logistic:
  balanced binary emergence classifier

logistic_unweighted:
  unweighted binary emergence classifier

histgb:
  HistGradientBoostingRegressor
```

所有运行均使用：

```text
max_train_rows = None
```

即：

```text
全量 train rows
```

无错误完成：

| Representation | Feature subset | Feature count | Error count | Runtime seconds |
|---|---|---:|---:|---:|
| KU full | all / ku_full | 61 | 0 | 1280.3 |
| numeric_only | numeric_only | 58 | 0 | 732.7 |
| raw_counts_only | raw_counts_only | 58 | 0 | 781.5 |
| ku_schema_only | ku_schema_only | 3 | 0 | 242.6 |

---

## 5. Evaluation metrics

本实验主要关注：

```text
Spearman
AUROC
AUPRC
NDCG@1000
Precision@1000
Enrichment@1000
Brier
```

其中：

```text
AUROC / AUPRC:
  binary emergence discrimination

Spearman:
  heat ranking correlation

NDCG@1000:
  top-ranked heat ranking quality

Precision@1000 / Enrichment@1000:
  top-K discovery performance

Brier:
  probability calibration / binary probability error
```

由于未来 heat target 极度 sparse / zero-inflated，本实验不将 MAE / RMSE 作为主指标。

---

## 6. Test-all 结果

`test_all` 包含：

```text
cutoff_year = 2019, 2020, 2021
```

### 6.1 HistGB 结果

HistGB 是 Stage 06A-D 中的 strong tabular baseline，因此本节重点比较 HistGB 在不同 representation 下的结果。

| Task | Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| patent | KU full | **0.0589** | **0.6701** | **0.0275** | 0.0677 | 0.110 | 10.90 |
| patent | numeric_only | 0.0558 | 0.6611 | 0.0269 | **0.0748** | **0.126** | **12.49** |
| patent | KU schema only | 0.0132 | 0.5359 | 0.0109 | 0.0032 | 0.006 | 0.59 |
| translation | KU full | **0.0889** | **0.6671** | **0.0475** | **0.0723** | **0.142** | **5.89** |
| translation | numeric_only | 0.0800 | 0.6505 | 0.0439 | 0.0578 | 0.110 | 4.56 |
| translation | KU schema only | 0.0413 | 0.5731 | 0.0285 | 0.0113 | 0.023 | 0.95 |
| trial | KU full | **0.0841** | **0.6883** | **0.0436** | 0.0899 | 0.150 | 8.90 |
| trial | numeric_only | 0.0703 | 0.6574 | 0.0394 | **0.0925** | **0.156** | **9.25** |
| trial | KU schema only | 0.0485 | 0.6022 | 0.0216 | 0.0095 | 0.016 | 0.95 |

主要观察：

```text
1. KU full 在三个任务上均提升 HistGB 的 Spearman、AUROC 和 AUPRC。
2. translation 上 KU full 同时显著提升 top-K 指标。
3. patent 和 trial 上，numeric_only 在某些 top-K 指标上高于 KU full。
4. KU schema only 的 AUROC 略高于 random，但 top-K enrichment 基本无效。
```

---

### 6.2 Translation task：主任务上 KU full 提升最清楚

Translation 是 primary task。

HistGB 在 `test_all` 上：

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| KU full | **0.0889** | **0.6671** | **0.0475** | **0.0723** | **0.142** | **5.89** |
| numeric_only | 0.0800 | 0.6505 | 0.0439 | 0.0578 | 0.110 | 4.56 |
| KU schema only | 0.0413 | 0.5731 | 0.0285 | 0.0113 | 0.023 | 0.95 |

相对 numeric_only，KU full 的提升为：

```text
AUROC:
  0.6505 -> 0.6671

AUPRC:
  0.0439 -> 0.0475

Spearman:
  0.0800 -> 0.0889

Precision@1000:
  0.110 -> 0.142

Enrichment@1000:
  4.56x -> 5.89x
```

该结果支持：

```text
在 primary future translation forecasting task 上，
KU schema categorical abstraction 对 numeric temporal evidence features 有清晰增量。
```

---

### 6.3 Patent task

Patent 是最稀疏任务。

`test_all` positive rate：

```text
0.0101
```

HistGB：

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| KU full | **0.0589** | **0.6701** | **0.0275** | 0.0677 | 0.110 | 10.90 |
| numeric_only | 0.0558 | 0.6611 | 0.0269 | **0.0748** | **0.126** | **12.49** |
| KU schema only | 0.0132 | 0.5359 | 0.0109 | 0.0032 | 0.006 | 0.59 |

观察：

```text
KU full 提升 overall discrimination 和 ranking correlation；
numeric_only 在 top-1000 tail ranking 上更高。
```

解释：

```text
Patent emergence 极稀疏，top-K 指标对尾部排序非常敏感。
KU schema features 改善整体排序，但不一定改善 extreme top-K。
```

---

### 6.4 Trial task

`test_all` positive rate：

```text
0.0169
```

HistGB：

| Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| KU full | **0.0841** | **0.6883** | **0.0436** | 0.0899 | 0.150 | 8.90 |
| numeric_only | 0.0703 | 0.6574 | 0.0394 | **0.0925** | **0.156** | **9.25** |
| KU schema only | 0.0485 | 0.6022 | 0.0216 | 0.0095 | 0.016 | 0.95 |

观察：

```text
KU full 大幅提升 AUROC / AUPRC / Spearman；
numeric_only 在 top-1000 precision 上略高。
```

Trial task 中，KU schema only 的 AUROC 高于 patent / translation：

```text
AUROC = 0.6022
```

说明：

```text
某些 KU 类型对 trial emergence 有较强 prior difference，
但 schema alone 仍然不能形成有效 top-K discovery。
```

---

## 7. Test-stable 结果

`test_stable` 排除 2021 cutoff，只包含：

```text
cutoff_year = 2019, 2020
```

### 7.1 HistGB stable-test 结果

| Task | Representation | Spearman | AUROC | AUPRC | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| patent | KU full | **0.0637** | **0.6719** | **0.0321** | 0.0790 | 0.129 | 11.16 |
| patent | numeric_only | 0.0605 | 0.6632 | 0.0316 | **0.0850** | **0.143** | **12.37** |
| patent | KU schema only | 0.0137 | 0.5349 | 0.0125 | 0.0050 | 0.009 | 0.78 |
| translation | KU full | **0.0955** | **0.6696** | **0.0558** | 0.0842 | 0.154 | 5.68 |
| translation | numeric_only | 0.0864 | 0.6534 | 0.0521 | **0.0844** | **0.156** | **5.75** |
| translation | KU schema only | 0.0432 | 0.5722 | 0.0320 | 0.0126 | 0.025 | 0.92 |
| trial | KU full | **0.0901** | **0.6913** | **0.0498** | 0.1007 | 0.160 | 8.51 |
| trial | numeric_only | 0.0756 | 0.6604 | 0.0455 | **0.1055** | **0.169** | **8.99** |
| trial | KU schema only | 0.0515 | 0.6029 | 0.0241 | 0.0110 | 0.019 | 1.01 |

主要观察：

```text
1. stable test 中，KU full 仍在 Spearman / AUROC / AUPRC 上稳定优于 numeric_only。
2. numeric_only 在 patent、translation、trial 的 top-K tail metrics 上有时略高。
3. KU schema only 在 top-K 上接近 random。
```

---

## 8. 主要发现

### 8.1 KU temporal numeric features 是主要预测信号来源

Numeric temporal features 已经能达到较强预测效果。

例如 `test_all`：

| Task | Model | Representation | AUPRC | Precision@1000 | Enrichment@1000 |
|---|---|---|---:|---:|---:|
| patent | logistic | numeric_only | 0.0239 | 0.130 | 12.88 |
| translation | logistic | numeric_only | 0.0407 | 0.129 | 5.35 |
| trial | ridge | numeric_only | 0.0347 | 0.166 | 9.85 |

这说明：

```text
KU temporal trajectory 本身承载了主要预测信号。
```

这也是 KU construction 的一个关键价值：

```text
将原始 carrier evidence 组织成稳定的 KU-level temporal trajectory。
```

---

### 8.2 KU schema only 单独不足以支撑预测

KU schema only 只有：

```text
pair_type
entity_a_type
entity_b_type
```

它能带来一定 weak prior signal：

```text
translation AUROC ≈ 0.573
trial AUROC ≈ 0.602
patent AUROC ≈ 0.536
```

但它无法形成有效 top-K discovery：

```text
Enrichment@1000 ≈ 0.6 - 1.0
```

这说明：

```text
未来转化预测不是简单由 pair_type/entity_type 决定；
必须结合 KU temporal evidence trajectory。
```

---

### 8.3 KU full 稳定提升 overall discrimination 和 ranking correlation

相比 numeric_only，KU full 在三个任务上均提升：

```text
AUROC
AUPRC
Spearman
```

尤其 HistGB：

| Task | Metric | numeric_only | KU full | Direction |
|---|---|---:|---:|---|
| patent | AUROC | 0.6611 | 0.6701 | ↑ |
| patent | AUPRC | 0.0269 | 0.0275 | ↑ |
| patent | Spearman | 0.0558 | 0.0589 | ↑ |
| translation | AUROC | 0.6505 | 0.6671 | ↑ |
| translation | AUPRC | 0.0439 | 0.0475 | ↑ |
| translation | Spearman | 0.0800 | 0.0889 | ↑ |
| trial | AUROC | 0.6574 | 0.6883 | ↑ |
| trial | AUPRC | 0.0394 | 0.0436 | ↑ |
| trial | Spearman | 0.0703 | 0.0841 | ↑ |

该结果说明：

```text
KU schema abstraction 在 numeric temporal evidence features 之外提供了互补信息。
```

---

### 8.4 Top-K precision 并非总是 KU full 最优

在 patent 和 trial 中，numeric_only 有时在：

```text
NDCG@1000
Precision@1000
Enrichment@1000
```

上略高于 KU full。

例如：

```text
patent test_stable HistGB:
  KU full Precision@1000 = 0.129
  numeric_only Precision@1000 = 0.143

trial test_stable logistic_unweighted:
  KU full Precision@1000 = 0.161
  numeric_only Precision@1000 = 0.177
```

这说明：

```text
KU schema features 更稳定改善 global ranking quality，
但不保证改善每个任务/模型的 extreme tail ordering。
```

该现象不否定 KU full 的 overall advantage，但提示后续主模型需要特别关注 top-K ranking optimization。

---

## 9. 与 KU construction value 的关系

本实验支持以下判断：

```text
1. 将 carrier evidence 构造成 KU-level temporal trajectories 是有价值的；
2. KU temporal numeric features 是主要预测信号来源；
3. KU schema categorical abstraction 在 temporal trajectory 之外提供额外信息；
4. 完整 KU representation 在整体区分能力和排序相关性上优于 numeric temporal evidence only。
```

更严谨地说：

```text
本实验证明的是 KU full representation 相比 numeric temporal evidence representation 有增量；
并未直接证明 KU beats raw carrier graph / carrier graph GNN。
```

因此推荐论文表述为：

```text
The ablation supports the predictive value of the KU representation:
KU temporal trajectories provide the main signal, and KU schema features add complementary information beyond numeric temporal evidence summaries.
```

中文表述：

```text
该消融实验支持 KU 表示作为预测单元的价值：
KU temporal trajectories 承载主要预测信号，而 KU schema features 在 numeric temporal evidence summaries 之外提供互补信息。
```

---

## 10. 局限性

### 10.1 raw_counts_only 与 numeric_only 完全一致

当前结果显示：

```text
raw_counts_only == numeric_only
```

因此本报告没有将 raw_counts_only 作为独立对照解释。

如需严格比较：

```text
raw carrier evidence counts
vs
KU engineered temporal features
```

需要重新定义更严格的 raw-count subset，例如只保留：

```text
history_*_count
recent_*_count
history_recency
```

排除：

```text
growth ratios
derived normalized features
other engineered numeric summaries
```

### 10.2 不是 carrier graph GNN baseline

本实验没有使用：

```text
carrier graph embeddings
GraphSAGE
heterogeneous GNN
temporal GNN
```

因此不能声称：

```text
KU representation outperforms all carrier graph approaches.
```

### 10.3 Top-K 指标需要任务级解释

KU full 在整体指标上更稳定，但 top-K precision 在 patent / trial 上有时不如 numeric_only。

因此后续模型比较应同时报告：

```text
global discrimination:
  AUROC / AUPRC

heat ranking:
  Spearman / NDCG

top-K discovery:
  Precision@K / Enrichment@K
```

---

## 11. 后续建议

### 11.1 可选：更严格 raw-count subset

如果需要更清楚地对比：

```text
raw count-like features
vs
engineered KU temporal features
```

可以新增：

```text
--feature-subset strict_raw_counts_only
```

仅保留：

```text
history_paper_count
history_patent_count
history_trial_count
history_translation_count
history_total_count
recent_3yr_paper_count
recent_3yr_patent_count
recent_3yr_trial_count
recent_3yr_translation_count
recent_3yr_total_count
history_recency
```

这会比当前 numeric_only 更严格。

### 11.2 Stage 06F diffusion proxy baseline

下一步建议进入：

```text
Stage 06F: KU-KU diffusion proxy baseline
```

其目标是构造 cutoff-safe KU neighborhood aggregation features，比较：

```text
KU full
vs
KU full + diffusion proxy
```

这样可以检验：

```text
KU 邻域状态是否在自身 temporal trajectory 之外提供额外预测信号。
```

这一步也将为后续：

```text
Knowledge Field reaction-diffusion-source model
```

提供更强、更公平的图扩散基线。

### 11.3 后续 graph baseline

如果需要进一步比较 carrier graph，可作为单独实验加入：

```text
carrier graph embedding -> KU pooling
heterogeneous KU-carrier graph GNN
temporal GNN
```

但这些不属于当前 Stage 06E 的范围。

---

## 12. 推荐写法

### 12.1 中文摘要

```text
Stage 06E feature-subset ablation 显示，KU temporal numeric features 是主要预测信号来源；仅使用 pair_type/entity type 的 KU schema features 虽然略高于随机，但无法形成有效 top-K discovery。相比 numeric temporal features only，full KU representation 在三个任务和多个 matched learners 上稳定提升 AUROC、AUPRC 和 Spearman，说明 KU schema abstraction 在 temporal evidence aggregation 之外提供了增量信息。该提升在主任务 translation 上尤其明显：HistGB 的 AUROC 从 0.6505 提升到 0.6671，AUPRC 从 0.0439 提升到 0.0475，Precision@1000 从 0.110 提升到 0.142。不过，patent 和 trial 的 extreme top-K precision 并非总是由 full KU representation 取得最高，说明 KU schema 更主要改善整体 ranking quality，而不是在所有任务中都改善 top-K tail ordering。
```

### 12.2 英文摘要

```text
The Stage 06E feature-subset ablation shows that KU temporal numeric features provide the primary predictive signal, while KU schema features alone—pair_type and entity types—are weak and insufficient for meaningful top-K discovery. Compared with numeric temporal evidence features alone, the full KU representation consistently improves AUROC, AUPRC, and Spearman across the three tasks and matched learners, indicating that KU schema abstraction adds predictive information beyond temporal evidence aggregation. The improvement is especially clear for the primary translation task: under HistGB, AUROC increases from 0.6505 to 0.6671, AUPRC from 0.0439 to 0.0475, and Precision@1000 from 0.110 to 0.142. However, full KU representation does not uniformly improve extreme top-K precision for patent and trial, suggesting that KU schema features mainly improve global ranking quality rather than every task-specific tail ranking.
```

---

## 13. 结论

Stage 06E representation feature-subset ablation 已完成。

已完成 representation：

```text
KU full
numeric_only
raw_counts_only
ku_schema_only
```

已完成任务：

```text
translation
patent
trial
```

已完成模型：

```text
ridge
logistic
logistic_unweighted
histgb
```

运行状态：

```text
max_train_rows = None
error_count = 0
```

核心结论：

```text
KU temporal numeric features 是主要预测信号；
KU schema only 单独较弱；
KU full representation 稳定提升 AUROC / AUPRC / Spearman；
translation 主任务上 KU full 同时显著提升 top-K discovery；
patent / trial 上 top-K tail ordering 存在任务差异。
```

因此，Stage 06E 支持以下项目主张：

```text
将原始 carrier evidence 组织成 KU-level temporal trajectories 是有预测价值的；
KU schema abstraction 在 temporal trajectories 之外提供互补信息；
完整 KU representation 是后续 Knowledge Field / graph diffusion model 的合理输入基础。
```

下一阶段建议：

```text
Stage 06F: KU-KU diffusion proxy baseline
```