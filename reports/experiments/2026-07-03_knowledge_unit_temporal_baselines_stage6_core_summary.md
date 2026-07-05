# Knowledge Unit Temporal Baselines Stage 6 Core 实验总结报告

日期：2026-07-03  
任务：KU-level temporal heat forecasting and emergence prediction  
阶段：Stage 06A-D core temporal baselines  
数据集：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/
```

实验结果目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_all_tasks_core/
```

核心结果文件：

```text
baseline_metrics.csv
baseline_metrics_sorted.csv
baseline_comparison.csv
baseline_comparison_test_all.csv
baseline_comparison_validation.csv
baseline_comparison_test_stable.csv
baseline_metrics_by_cutoff.csv
baseline_comparison_report.md
baseline_run_summary.json
baseline_errors.csv
baseline_manifest.csv
```

---

## 1. 实验目的

Stage 06 的目标是在 Stage 05F 冻结的 KU temporal prediction benchmark 上建立一组稳定、可复现、无泄漏的 baseline。

Stage 05F 已经完成：

```text
KU × cutoff_year prediction examples
future heat targets
binary emergence labels
train / validation / test splits
leakage audit
```

Stage 06 core baseline 的目标是回答：

```text
在只使用 cutoff 年及之前的 KU 历史特征时，
简单 heuristic、线性模型、logistic 模型和强 tabular 模型
能在多大程度上预测未来 KU translation / patent / trial heat？
```

本阶段不是最终主模型贡献，而是为后续 Stage 06E representation ablation 和 Stage 07 Knowledge Field reaction-diffusion-source model 提供强参照。

---

## 2. 输入数据与任务设置

### 2.1 输入数据

输入目录：

```text
data/datasets/knowledge_units/diabetes_2000_2024_v1_temporal_prediction/
```

核心输入文件：

```text
prediction_feature_table.parquet
feature_columns.json
target_columns.json
```

Stage 05F benchmark 采用：

```text
example unit = KU × cutoff_year
history = 2000 .. cutoff_year
future = cutoff_year + 1 .. cutoff_year + 3
horizon = 3 years
```

预测 cutoff years：

```text
2005-2021
```

数据划分：

```text
train cutoffs:
  2005-2016

validation cutoffs:
  2017-2018

test cutoffs:
  2019-2021
```

其中：

```text
test_stable = 2019, 2020
right-edge stress cutoff = 2021
```

### 2.2 三个任务

本阶段覆盖三个 KU-level future heat forecasting tasks。

| Task | Eligibility | Heat target | Emergence label |
|---|---|---|---|
| translation | `eligible_translation_task` | `target_future_translation_heat_3yr` | `label_future_translation_emergence_3yr` |
| patent | `eligible_patent_task` | `target_future_patent_heat_3yr` | `label_future_patent_emergence_3yr` |
| trial | `eligible_trial_task` | `target_future_trial_heat_3yr` | `label_future_trial_emergence_3yr` |

### 2.3 Task definitions

#### Future Translation Heat Forecasting

```text
target_future_translation_count_3yr =
  target_future_patent_count_3yr
  + target_future_trial_count_3yr

target_future_translation_heat_3yr =
  log1p(target_future_translation_count_3yr)

label_future_translation_emergence_3yr =
  1[target_future_translation_count_3yr > 0]
```

Eligibility：

```text
history_paper_count > 0
AND history_patent_count == 0
AND history_trial_count == 0
```

#### Future Patent Heat Forecasting

```text
target_future_patent_heat_3yr =
  log1p(target_future_patent_count_3yr)

label_future_patent_emergence_3yr =
  1[target_future_patent_count_3yr > 0]
```

Eligibility：

```text
history_paper_count > 0
AND history_patent_count == 0
```

#### Future Trial Heat Forecasting

```text
target_future_trial_heat_3yr =
  log1p(target_future_trial_count_3yr)

label_future_trial_emergence_3yr =
  1[target_future_trial_count_3yr > 0]
```

Eligibility：

```text
history_paper_count > 0
AND history_trial_count == 0
```

---

## 3. Baseline families

本阶段实现并评估以下 baseline families：

| Family | Models | 说明 |
|---|---|---|
| Constant baseline | `mean` | 预测 train heat mean / train label prevalence |
| Recency heuristic | `recency` | 使用 `history_recency` 排序 |
| Activity heuristic | `activity_paper`, `activity_total` | 使用历史 paper / total activity |
| Recent activity heuristic | `recent_3yr_paper`, `recent_3yr_total` | 使用近期 3 年 activity |
| Growth heuristic | `growth_paper_3yr_ratio`, `growth_total_3yr_ratio` | 使用近期增长 ratio |
| Linear regression | `ridge` | 预测 future log heat |
| Logistic classification | `logistic`, `logistic_unweighted` | 预测 future emergence probability |
| Strong tabular model | `histgb` | sklearn HistGradientBoostingRegressor |

### 3.1 Logistic variants

当前包含两个 logistic variants：

```text
logistic:
  class_weight = "balanced"

logistic_unweighted:
  class_weight = None
```

区别：

```text
balanced logistic:
  ranking / AUPRC 通常更强，但 probability calibration 较差

unweighted logistic:
  Brier score 更好，概率更接近 calibrated probability，但 ranking 可能略弱
```

### 3.2 HistGB

`histgb` 使用：

```text
HistGradientBoostingRegressor
```

它预测 heat target，预测值同时作为 ranking score。

HistGB 是本阶段的强 tabular baseline，用于估计非线性 tabular features 的可达性能上限。

---

## 4. 运行配置

执行任务：

```text
tasks = translation, patent, trial
```

执行模型：

```text
models =
  mean
  recency
  activity_paper
  activity_total
  recent_3yr_paper
  recent_3yr_total
  growth_paper_3yr_ratio
  growth_total_3yr_ratio
  ridge
  logistic
  logistic_unweighted
  histgb
```

训练方式：

```text
max_train_rows = None
```

即：

```text
所有 trainable baselines 使用全量 train rows 训练
```

运行状态：

```text
error_count = 0
runtime_seconds = 1280.27
```

输出目录：

```text
data/results/knowledge_units/temporal_baselines/diabetes_2000_2024_v1_all_tasks_core/
```

---

## 5. Evaluation protocol

### 5.1 Evaluation groups

本阶段报告：

```text
validation:
  cutoffs = 2017, 2018

test_all:
  cutoffs = 2019, 2020, 2021

test_stable:
  cutoffs = 2019, 2020

test_by_cutoff:
  cutoff = 2019
  cutoff = 2020
  cutoff = 2021
```

其中：

```text
cutoff 2021 = right-edge stress test
future window = 2022-2024
```

### 5.2 Metrics

Heat / regression metrics：

```text
MAE
RMSE
Spearman
Pearson
```

Classification metrics：

```text
AUROC
AUPRC
Brier
```

Ranking metrics：

```text
NDCG@100
NDCG@1000
Precision@1000
Recall@1000
Enrichment@1000
```

### 5.3 Metric interpretation

由于 future heat target 极度 sparse / zero-inflated：

```text
大多数 eligible KU 的 future heat = 0
```

因此：

```text
MAE / RMSE 容易偏向接近 0 的保守预测
```

本阶段更重视：

```text
Spearman
AUPRC
NDCG@1000
Precision@1000
Enrichment@1000
```

这些指标更能反映 future translation discovery / ranking 能力。

---

## 6. Test-all results

`test_all` 包含：

```text
cutoffs = 2019, 2020, 2021
```

### 6.1 Translation task

| Model | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| histgb | **0.6671** | **0.0475** | **0.0889** | **0.0723** | **0.142** | **5.89** |
| logistic | 0.6170 | 0.0432 | 0.0622 | 0.0692 | 0.128 | 5.31 |
| logistic_unweighted | 0.6138 | 0.0420 | 0.0605 | 0.0662 | 0.122 | 5.06 |
| ridge | 0.6005 | 0.0392 | 0.0534 | 0.0517 | 0.094 | 3.90 |
| recent_3yr_total | 0.5934 | 0.0384 | 0.0523 | 0.0227 | 0.048 | 1.99 |
| recency | 0.5670 | 0.0289 | 0.0363 | 0.0113 | 0.022 | 0.91 |
| mean | 0.5000 | 0.0241 | NA | NA | NA | NA |

Translation task positive rate：

```text
positive_rate = 0.0241
```

Key finding：

```text
HistGB is the strongest overall baseline on translation test_all.
```

HistGB achieves:

```text
Precision@1000 = 0.142
Enrichment@1000 = 5.89
```

This means:

```text
Among the top 1000 KUs ranked by HistGB,
14.2% show future translation evidence,
which is 5.89x higher than random selection.
```

---

### 6.2 Patent task

| Model | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| histgb | **0.6701** | **0.0275** | **0.0589** | 0.0677 | 0.110 | 10.90 |
| logistic | 0.6154 | 0.0240 | 0.0400 | 0.0762 | **0.127** | **12.58** |
| logistic_unweighted | 0.6138 | 0.0228 | 0.0394 | **0.0768** | 0.120 | 11.89 |
| ridge | 0.6157 | 0.0242 | 0.0400 | 0.0690 | 0.125 | 12.39 |
| recent_3yr_total | 0.6087 | 0.0215 | 0.0394 | 0.0374 | 0.069 | 6.84 |
| activity_total | 0.5653 | 0.0192 | 0.0228 | 0.0226 | 0.045 | 4.46 |
| mean | 0.5000 | 0.0101 | NA | NA | NA | NA |

Patent task positive rate：

```text
positive_rate = 0.0101
```

Key finding：

```text
Patent task is the sparsest task but achieves the highest top-K enrichment.
```

Although HistGB has the best overall AUROC / AUPRC / Spearman, logistic and ridge provide very strong top-1000 ranking:

```text
logistic Precision@1000 = 0.127
logistic Enrichment@1000 = 12.58
```

This means:

```text
The top-1000 ranked patent candidates are enriched by more than 12x over random selection.
```

---

### 6.3 Trial task

| Model | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| histgb | **0.6883** | **0.0436** | **0.0841** | 0.0899 | 0.150 | 8.90 |
| ridge | 0.6426 | 0.0368 | 0.0637 | **0.0977** | **0.167** | **9.91** |
| logistic | 0.6618 | 0.0392 | 0.0722 | 0.0926 | 0.157 | 9.31 |
| logistic_unweighted | 0.6570 | 0.0376 | 0.0701 | 0.0844 | 0.142 | 8.42 |
| recent_3yr_total | 0.6237 | 0.0316 | 0.0581 | 0.0258 | 0.048 | 2.85 |
| activity_total | 0.5574 | 0.0269 | 0.0259 | 0.0140 | 0.027 | 1.60 |
| mean | 0.5000 | 0.0169 | NA | NA | NA | NA |

Trial task positive rate：

```text
positive_rate = 0.0169
```

Key finding：

```text
Trial task has strong predictable signal.
```

HistGB has the best overall AUROC / AUPRC / Spearman, while ridge achieves the highest test_all top-1000 precision:

```text
ridge Precision@1000 = 0.167
ridge Enrichment@1000 = 9.91
```

---

## 7. Test-stable results

`test_stable` excludes cutoff 2021:

```text
cutoffs = 2019, 2020
```

This subset is used to reduce right-edge observation incompleteness.

### 7.1 Translation stable test

| Model | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| histgb | **0.6696** | **0.0558** | **0.0955** | 0.0842 | 0.154 | 5.68 |
| logistic | 0.6193 | 0.0509 | 0.0671 | **0.0917** | **0.167** | **6.16** |
| logistic_unweighted | 0.6154 | 0.0493 | 0.0650 | 0.0881 | 0.161 | 5.94 |
| ridge | 0.6025 | 0.0458 | 0.0577 | 0.0707 | 0.128 | 4.72 |

Key finding：

```text
HistGB is the strongest overall model, while balanced logistic is strongest for stable-test top-K translation discovery.
```

---

### 7.2 Patent stable test

| Model | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| histgb | **0.6719** | **0.0321** | **0.0637** | 0.0790 | 0.129 | 11.16 |
| ridge | 0.6158 | 0.0281 | 0.0429 | 0.0778 | **0.137** | **11.85** |
| logistic | 0.6168 | 0.0280 | 0.0432 | 0.0803 | 0.132 | 11.42 |
| logistic_unweighted | 0.6140 | 0.0269 | 0.0422 | **0.0817** | 0.124 | 10.72 |

Key finding：

```text
HistGB remains strongest overall, while ridge / logistic achieve the highest stable-test patent top-K precision.
```

---

### 7.3 Trial stable test

| Model | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---|---:|---:|---:|---:|---:|---:|
| histgb | **0.6913** | **0.0498** | **0.0901** | **0.1007** | 0.160 | 8.51 |
| logistic_unweighted | 0.6603 | 0.0441 | 0.0755 | 0.0982 | **0.161** | **8.56** |
| logistic | 0.6659 | 0.0460 | 0.0781 | 0.0991 | 0.159 | 8.46 |
| ridge | 0.6468 | 0.0426 | 0.0692 | 0.0961 | 0.157 | 8.35 |

Key finding：

```text
Trial stable-test results are strong and stable.
HistGB is strongest overall, while top-K precision is very close among HistGB, logistic, logistic_unweighted, and ridge.
```

---

## 8. By-cutoff temporal robustness

### 8.1 Translation

HistGB by cutoff:

| Cutoff | Positive rate | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.0289 | 0.6678 | 0.0591 | 0.0974 | 0.0885 | 0.151 | 5.23 |
| 2020 | 0.0253 | 0.6726 | 0.0525 | 0.0940 | 0.0812 | 0.122 | 4.81 |
| 2021 | 0.0182 | 0.6670 | 0.0338 | 0.0773 | 0.0418 | 0.055 | 3.03 |

Observation：

```text
Translation task shows the strongest right-edge degradation in 2021.
```

Although AUROC remains stable, top-K precision and NDCG drop substantially in 2021.

---

### 8.2 Patent

HistGB by cutoff:

| Cutoff | Positive rate | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.0127 | 0.6732 | 0.0356 | 0.0673 | 0.0817 | 0.130 | 10.21 |
| 2020 | 0.0104 | 0.6712 | 0.0287 | 0.0602 | 0.0788 | 0.110 | 10.57 |
| 2021 | 0.0072 | 0.6696 | 0.0195 | 0.0497 | 0.0671 | 0.077 | 10.70 |

Observation：

```text
Patent positive rate declines in 2021, but enrichment remains high.
```

The task becomes sparser, but the model still concentrates positives in the top-ranked candidates.

---

### 8.3 Trial

HistGB by cutoff:

| Cutoff | Positive rate | AUROC | AUPRC | Spearman | NDCG@1000 | Precision@1000 | Enrichment@1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.0196 | 0.6873 | 0.0508 | 0.0901 | 0.1047 | 0.164 | 8.36 |
| 2020 | 0.0180 | 0.6966 | 0.0491 | 0.0906 | 0.1240 | 0.165 | 9.17 |
| 2021 | 0.0130 | 0.6874 | 0.0323 | 0.0736 | 0.0934 | 0.121 | 9.29 |

Observation：

```text
Trial task also shows a 2021 decline in positive rate and AUPRC,
but top-K enrichment remains relatively stable.
```

Trial is less affected by right-edge degradation than translation.

---

## 9. Main findings

### 9.1 Stage 06 core baselines are successfully established

All three tasks completed full-train baseline evaluation:

```text
translation
patent
trial
```

All models ran successfully:

```text
error_count = 0
```

No training subsampling was used:

```text
max_train_rows = None
```

### 9.2 HistGB is the strongest overall tabular baseline

Across tasks, HistGB consistently achieves the best or near-best:

```text
AUROC
AUPRC
Spearman
Pearson
```

This establishes HistGB as the current strongest non-graph tabular baseline.

### 9.3 Top-K best model depends on the task

For top-K discovery:

```text
translation:
  logistic is strongest on test_stable top-K

patent:
  logistic / ridge are strongest for top-K enrichment

trial:
  HistGB / logistic / ridge are all close
```

Thus, Stage 06 should not select a single model only by one metric. Instead, results should be reported by metric family:

```text
overall discrimination:
  AUROC / AUPRC

heat ranking:
  Spearman / NDCG

top-K discovery:
  Precision@K / Enrichment@K
```

### 9.4 Patent has the highest enrichment

Patent task is most sparse:

```text
test_all positive_rate = 0.0101
```

But top-K enrichment is highest:

```text
logistic Enrichment@1000 = 12.58
ridge Enrichment@1000 = 12.39
histgb Enrichment@1000 = 10.90
```

This suggests that future patent evidence is rare but relatively concentrated among high-ranked candidates.

### 9.5 Translation is most affected by right-edge sparsity

Translation 2021 cutoff shows clear degradation:

```text
HistGB Precision@1000:
  2019 = 0.151
  2020 = 0.122
  2021 = 0.055
```

This supports the interpretation that:

```text
2021 -> 2022-2024
```

is a right-edge stress test.

### 9.6 MAE/RMSE should be interpreted cautiously

Mean baseline often has competitive MAE/RMSE because most future heat targets are zero.

Therefore:

```text
MAE / RMSE are not primary ranking metrics.
```

The primary reporting focus should remain:

```text
AUPRC
Spearman
NDCG@1000
Precision@1000
Enrichment@1000
```

---

## 10. Comparison to Stage 03

Stage 03 and Stage 06 are not directly comparable.

Stage 03 task:

```text
carrier-level Patent-Paper link prediction
```

Stage 06 task:

```text
KU-level future heat forecasting
```

They differ in:

```text
prediction unit
target definition
temporal split
label sparsity
evaluation protocol
```

Therefore, Stage 03 GraphSAGE AUPRC cannot be directly compared with Stage 06 KU-level AUPRC.

Current interpretation:

```text
Stage 03:
  carrier context graph contains strong link prediction signal

Stage 06:
  KU history features contain meaningful but challenging future forecasting signal
```

A fair comparison between carrier graph and KU abstraction should be added later as a same-task representation ablation:

```text
Stage 06E:
  KU-only
  carrier-context-only projected to KU × cutoff
  KU + carrier-context
  KU + diffusion proxy
```

---

## 11. Current limitations

### 11.1 No graph / diffusion features yet

Current Stage 06A-D baselines use only cutoff-safe tabular KU history features.

They do not use:

```text
KU-KU graph
carrier graph neighborhood
project / funding source graph
entity co-neighborhood diffusion
temporal graph propagation
```

Therefore, these baselines should be interpreted as:

```text
strong non-graph tabular baselines
```

not as graph baselines.

### 11.2 No carrier-context representation ablation yet

Current experiments do not yet answer:

```text
Does the KU abstraction outperform carrier-level context on the same future forecasting task?
```

That requires a same-task ablation using Stage 05F examples.

### 11.3 No sequence model yet

Current baselines use aggregate history features, not full temporal sequences.

Temporal sequence baselines such as:

```text
GRU
Temporal CNN
Transformer encoder
```

remain future Stage 06 extensions.

---

## 12. Recommended next steps

### 12.1 Stage 06E: Representation abstraction ablation

Next recommended experiment:

```text
KU-only vs carrier-context-only vs KU + carrier-context
```

Proposed feature sets:

```text
KU-only:
  current feature_columns.json

carrier-context-only:
  carrier graph features aggregated to KU × cutoff

KU + carrier-context:
  union of both feature sets
```

The goal is to answer:

```text
Does carrier graph context provide information beyond KU temporal history?
```

### 12.2 Stage 06F: Diffusion proxy baseline

Add simple cutoff-safe KU neighborhood features:

```text
neighbor_recent_3yr_translation_count_mean
neighbor_recent_3yr_translation_count_max
neighbor_prior_translation_fraction
neighbor_recent_patent_count_mean
neighbor_recent_trial_count_mean
neighbor_growth_total_mean
```

These features would provide a lightweight test of the diffusion hypothesis before Stage 07.

### 12.3 Stage 07 preparation

After Stage 06E / 06F:

```text
Stage 07 Knowledge Field reaction-diffusion-source model
```

should compare against:

```text
HistGB
logistic
ridge
KU + carrier-context
KU + diffusion proxy
```

---

## 13. Current recommended result statement

### Chinese

```text
Stage 06 core baselines 在 translation、patent 和 trial 三个 KU-level temporal forecasting tasks 上完成了全量训练与评估。结果显示，HistGradientBoosting 是整体最强的非图 tabular baseline，在三个任务中均取得最高或接近最高的 AUROC、AUPRC 和 Spearman。对于 top-K discovery，不同任务中最优模型略有差异：translation stable test 上 balanced logistic 的 top-1000 precision 最高；patent task 上 logistic / ridge 的 top-K enrichment 最强；trial task 上 HistGB、logistic 和 ridge 的 top-K 表现接近。所有任务均显著优于随机选择，其中 test_all top-1000 enrichment 在 patent、trial、translation 上分别最高达到约 12.6x、9.9x 和 5.9x。2021 cutoff 持续表现出 right-edge label sparsity，尤其影响 translation task，因此后续报告应同时保留 test_all、test_stable 和 by-cutoff metrics。
```

### English

```text
Stage 06 core baselines were trained and evaluated on all three KU-level temporal forecasting tasks: future translation, patent, and trial heat forecasting. HistGradientBoosting is the strongest overall non-graph tabular baseline, achieving the best or near-best AUROC, AUPRC, and Spearman across tasks. For top-K discovery, the best model is task-dependent: balanced logistic performs best for stable-test translation top-K ranking, logistic/ridge are highly competitive for patent top-K enrichment, and HistGB/logistic/ridge are close on trial top-K discovery. All tasks show substantial enrichment over random selection, with test_all top-1000 enrichment reaching approximately 12.6x for patent, 9.9x for trial, and 5.9x for translation. The 2021 cutoff consistently shows right-edge label sparsity, especially for the translation task, motivating separate reporting of test_all, test_stable, and by-cutoff metrics.
```

---

## 14. Conclusion

Stage 06A-D core baselines are complete.

Completed components:

```text
Stage 06A: evaluation harness ✅
Stage 06B: heuristic baselines ✅
Stage 06C: linear / logistic baselines ✅
Stage 06D: strong tabular baseline HistGB ✅
```

Core benchmark results:

```text
tasks:
  translation
  patent
  trial

models:
  mean
  recency
  activity_paper
  activity_total
  recent_3yr_paper
  recent_3yr_total
  growth_paper_3yr_ratio
  growth_total_3yr_ratio
  ridge
  logistic
  logistic_unweighted
  histgb

training:
  full train rows
  no subsampling

status:
  error_count = 0
```

Current strongest baseline family:

```text
HistGradientBoosting
```

Current strongest top-K baselines:

```text
translation:
  balanced logistic / HistGB

patent:
  logistic / ridge / HistGB

trial:
  HistGB / logistic / ridge
```

Next stage:

```text
Stage 06E: representation abstraction ablation
  KU-only
  carrier-context-only
  KU + carrier-context

Stage 06F: diffusion proxy baseline

Stage 07: Knowledge Field reaction-diffusion-source model
```