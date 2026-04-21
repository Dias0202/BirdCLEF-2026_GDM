import torch
import mlflow
import os
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm

from model import SEDModel
from dataset import get_dataloader
from eda import run_eda
from metrics import padded_cmap
from config import BATCH_SIZE

import numpy as np

def mixup_data(x, y, alpha=0.5, device='cuda'):
    """Returns mixed inputs and targets for MixUp"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, mixup_alpha=0.5):
    model.train()
    running_loss = 0.0
    
    # We will limit the batches for testing the pipeline if this is just a setup phase
    # But let's build the full loop logic here.
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    
    for batch_idx, (waveforms, targets) in pbar:
        waveforms = waveforms.to(device)
        targets = targets.to(device)
        
        # Apply MixUp
        waveforms, targets = mixup_data(waveforms, targets, alpha=mixup_alpha, device=device)
        
        optimizer.zero_grad()
        
        # Mixed Precision forward pass
        with autocast():
            logits = model(waveforms)
            loss = criterion(logits, targets)
            
        # Backward pass using scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
            
    return running_loss / len(dataloader)

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (waveforms, targets) in pbar:
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            
            with autocast():
                logits = model(waveforms)
                loss = criterion(logits, targets)
                
            running_loss += loss.item()
            
            all_targets.append(targets)
            all_preds.append(torch.sigmoid(logits))
                
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)
    
    val_loss = running_loss / len(dataloader)
    val_auc = padded_cmap(all_targets, all_preds)
    
    return val_loss, val_auc

def main():
    print("=== Orquestrador Final: Treinamento Completo ===")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Utilizando dispositivo: {device}")
    
    # Setup MLflow
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db'))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("BirdCLEF_2026_Treinamento_Real")
    
    # Load Data (we use a sample for this pipeline test)
    print("-> Carregando metadados...")
    train_df, _ = run_eda()
    
    if train_df is None:
        return
        
    # Utilizando o dataset completo (todas as amostras) para o treinamento final.
    
    # Split mock (80% train, 20% val)
    train_split = train_df.sample(frac=0.8, random_state=42)
    val_split = train_df.drop(train_split.index)
    
    print(f"-> Train samples: {len(train_split)}, Val samples: {len(val_split)}")
    
    # Dataloaders
    train_loader = get_dataloader(train_split, is_train=True)
    val_loader = get_dataloader(val_split, is_train=False)
    
    # Model
    model = SEDModel(num_classes=234).to(device)
    
    # Optimizer, Loss, Scaler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    
    num_epochs = 30 # Treinamento longo e intensivo para a noite
    best_auc = 0.0
    
    with mlflow.start_run() as run:
        print(f"-> MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("model", "EfficientNet-B0_AttentionPooling")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("mixed_precision", "True")
        mlflow.log_param("mixup_alpha", 0.5)
        mlflow.log_param("spec_augment", "True")
        
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, mixup_alpha=0.5)
            print(f"-> Train Loss: {train_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            
            # Validation
            val_loss, val_auc = val_epoch(model, val_loader, criterion, device)
            print(f"-> Val Loss: {val_loss:.4f}, Val ROC-AUC: {val_auc:.4f}")
            
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            
            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), "best_model.pth")
                mlflow.log_artifact("best_model.pth", "models")
                print(f"-> Novo melhor modelo salvo! (AUC: {best_auc:.4f})")

    print("\n-> Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()
