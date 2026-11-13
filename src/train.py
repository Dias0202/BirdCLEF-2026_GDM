import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import BATCH_SIZE, BEST_MODEL_PATH
from dataset import get_dataloader
from eda import run_eda
from metrics import macro_roc_auc
from model import SEDModel
from validation import train_val_split


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.5,
    device: torch.device = None,
) -> tuple:
    """
    Apply MixUp data augmentation to a batch of waveforms and labels.

    Randomly blends pairs of samples: mixed = λ·a + (1−λ)·b,
    where λ is drawn from a Beta(alpha, alpha) distribution.

    This regularises the model against overfitting and improves calibration
    on recordings that contain overlapping species vocalisations.

    Args:
        x:      Batch of audio waveforms, shape (batch, 1, audio_length).
        y:      Corresponding one-hot label vectors, shape (batch, num_classes).
        alpha:  Beta distribution concentration parameter.  Higher values
                produce more aggressive mixing (0.5 is a common default).
        device: Device on which x and y reside, used to place the shuffle index.

    Returns:
        tuple: (mixed_x, mixed_y) with the same shapes as the inputs.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixup_alpha: float = 0.5,
) -> float:
    """
    Run one full training epoch with Automatic Mixed Precision and MixUp.

    AMP (fp16 forward/backward) keeps GPU memory usage within the 4 GB budget
    of the GTX 1650 Super while maintaining numerical stability via GradScaler.

    Returns:
        float: Mean training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)

    for batch_idx, (waveforms, targets) in pbar:
        waveforms = waveforms.to(device)
        targets   = targets.to(device)

        waveforms, targets = mixup_data(waveforms, targets, alpha=mixup_alpha, device=device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            logits = model(waveforms)
            loss   = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"loss": running_loss / (batch_idx + 1)})

    return running_loss / len(dataloader)


def val_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run one full validation epoch without gradient computation.

    Collects predictions from all batches, then computes the competition
    metric (macro ROC-AUC over classes with true positives) on the full
    validation fold.

    Returns:
        tuple: (val_loss, val_auc) — mean BCE loss and macro ROC-AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets  = []
    all_preds    = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, (waveforms, targets) in pbar:
            waveforms = waveforms.to(device)
            targets   = targets.to(device)

            with autocast(device_type=device.type):
                logits = model(waveforms)
                loss   = criterion(logits, targets)

            running_loss += loss.item()
            all_targets.append(targets)
            all_preds.append(torch.sigmoid(logits))

    all_targets = torch.cat(all_targets)
    all_preds   = torch.cat(all_preds)

    val_loss = running_loss / len(dataloader)
    val_auc  = macro_roc_auc(all_targets.float(), all_preds.float())
    return val_loss, val_auc


def main():
    print("=== BirdCLEF 2026 — Full Training Run ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── MLflow experiment tracking ─────────────────────────────────────────
    # The MLflow database lives at the project root, one level above src/.
    db_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
    )
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("BirdCLEF_2026_Real_Training")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading metadata...")
    train_df, _ = run_eda()
    if train_df is None:
        return

    # Fold 0 of 5, grouped by recordist and stratified by species (see validation.py).
    train_split, val_split = train_val_split(train_df, fold=0, n_splits=5, seed=42)
    print(f"Train: {len(train_split)} samples  |  Val: {len(val_split)} samples")

    train_loader = get_dataloader(train_split, is_train=True)
    val_loader   = get_dataloader(val_split,   is_train=False)

    # ── Model, loss, optimiser, AMP scaler ────────────────────────────────
    model     = SEDModel(num_classes=234).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler    = GradScaler(device.type)

    num_epochs = 30
    best_auc   = 0.0

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")
        mlflow.log_params({
            "model"           : "EfficientNet-B0_AttentionPooling",
            "batch_size"      : BATCH_SIZE,
            "optimizer"       : "Adam",
            "learning_rate"   : 1e-3,
            "mixed_precision" : True,
            "mixup_alpha"     : 0.5,
            "spec_augment"    : True,
            "validation"      : "StratifiedGroupKFold(5, groups=author) fold 0",
            "val_metric"      : "macro ROC-AUC over classes with positives",
        })

        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch + 1}/{num_epochs}]")

            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )
            val_loss, val_auc = val_epoch(model, val_loader, criterion, device)

            print(
                f"  train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f}"
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc},
                step=epoch,
            )

            # Save checkpoint whenever validation AUC improves.
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                mlflow.log_artifact(str(BEST_MODEL_PATH), "models")
                print(f"  -> New best model saved (AUC: {best_auc:.4f})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
