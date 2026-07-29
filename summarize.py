from pathlib import Path
import json
import pandas as pd

base = Path("data/results/knowledge_field/temporal_gnn")

runs = [
    "local_source_proxy_dynamic_rds_v2_state8_no_decay_seed42",
    "local_source_proxy_dynamic_rds_v2_state8_no_diffusion_seed42",
    "local_source_proxy_dynamic_rds_v2_state8_no_proxy_seed42",
    "local_source_proxy_dynamic_rds_v2_state8_no_reaction_seed42",
    "local_source_proxy_dynamic_rds_v2_state8_no_source_seed42",
    "local_source_proxy_dynamic_rds_v2_state8_seed42"
    ]

rows = []

for run in runs:
    run_dir = base / run
    summary_path = run_dir / "run_summary.json"
    test_path = run_dir / "test_metrics_by_cutoff.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test = pd.read_csv(test_path)
    t = test.iloc[0].to_dict()

    rows.append({
        "run": run,
        "model": summary.get("model_name"),
        "best_epoch": summary.get("best_epoch"),
        "best_val_metric": summary.get("best_monitor_metric"),
        "best_val_value": summary.get("best_monitor_value"),
        "test_cutoff": t.get("cutoff_year"),
        "test_n": t.get("n"),
        "test_positive_rate": t.get("positive_rate"),
        "test_auprc": t.get("auprc"),
        "test_auroc": t.get("auroc"),
        "test_spearman": t.get("spearman"),
        "test_precision_at_100": t.get("precision_at_100"),
        "test_precision_at_1000": t.get("precision_at_1000"),
        "test_precision_at_5000": t.get("precision_at_5000"),
        "test_recall_at_1000": t.get("recall_at_1000"),
        "test_ndcg_at_1000": t.get("ndcg_at_1000"),
        "test_enrichment_at_1000": t.get("enrichment_at_1000"),
    })

df = pd.DataFrame(rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
print(df.to_string(index=False))