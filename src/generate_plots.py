import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os

db_path = os.path.abspath("mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

run_id = "ea5dc1d893f04329b0bbaa021ef783e1"
client = mlflow.tracking.MlflowClient()

def plot_metric(run_id, metric_name, title, filename):
    history = client.get_metric_history(run_id, metric_name)
    steps = [m.step for m in history]
    values = [m.value for m in history]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, marker='o', linestyle='-', label=metric_name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

# Generate Plots
plot_metric(run_id, "train_loss", "Training Loss per Epoch", "training_loss.png")
plot_metric(run_id, "val_loss", "Validation Loss per Epoch", "validation_loss.png")
plot_metric(run_id, "val_auc", "Validation ROC-AUC per Epoch", "validation_auc.png")

# Combined Loss Plot
history_train = client.get_metric_history(run_id, "train_loss")
history_val = client.get_metric_history(run_id, "val_loss")

plt.figure(figsize=(10, 5))
plt.plot([m.step for m in history_train], [m.value for m in history_train], label='Train Loss')
plt.plot([m.step for m in history_val], [m.value for m in history_val], label='Val Loss')
plt.title('Train vs Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_comparison.png")
plt.close()
print("Saved: loss_comparison.png")
