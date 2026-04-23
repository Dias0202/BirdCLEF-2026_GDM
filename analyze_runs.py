"""
Utility script: query and display MLflow training runs.

Run from the project root directory:
    python analyze_runs.py
"""
import os

import mlflow
import pandas as pd

db_path = os.path.abspath("mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# ── List all experiments ───────────────────────────────────────────────────
experiments = mlflow.search_experiments()
print("Available experiments:")
for exp in experiments:
    print(f"  - {exp.name}  (ID: {exp.experiment_id})")

# ── Query the main training experiment ────────────────────────────────────
experiment_name = "BirdCLEF_2026_Real_Training"
runs = mlflow.search_runs(experiment_names=[experiment_name])

if runs.empty:
    print(f"\nNo runs found for experiment '{experiment_name}'.")
else:
    print(f"\nRuns for '{experiment_name}':")
    for i, run in runs.iterrows():
        start = pd.to_datetime(run["start_time"], unit="ms")
        print(f"  [{i}] run_id={run['run_id']}  status={run['status']}  started={start}")

    print("\nAnalysing the most recent run:")
    last_run = runs.iloc[0]

    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    param_cols  = [c for c in runs.columns if c.startswith("params.")]

    print("\n  Parameters:")
    for p in param_cols:
        print(f"    {p.replace('params.', '')}: {last_run[p]}")

    print("\n  Final Metrics:")
    for m in metric_cols:
        print(f"    {m.replace('metrics.', '')}: {last_run[m]}")

    # Show per-epoch metric history for the last 5 epochs.
    run_id = last_run["run_id"]
    client = mlflow.tracking.MlflowClient()

    print("\n  Metric history (last 5 epochs):")
    for metric_name in ["train_loss", "val_loss", "val_auc"]:
        history = client.get_metric_history(run_id, metric_name)
        values  = [m.value for m in history]
        if metric_name == "train_loss":
            print(f"    Total epochs recorded: {len(values)}")
        print(f"    {metric_name}: {values[-5:] if len(values) > 5 else values}")
