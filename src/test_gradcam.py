import os
import torch
import numpy as np
import mlflow
import cv2

from model import SEDModel
from dataset import get_dataloader
from eda import run_eda
from gradcam import GradCAM, plot_gradcam
from config import BATCH_SIZE

def test_gradcam():
    print("=== Fase 4: Teste de Explicabilidade com Grad-CAM ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Dispositivo: {device}")
    
    # 1. Carregar Metadados
    train_df, _ = run_eda()
    if train_df is None: return
    
    # Usar um pequeno subconjunto para teste
    sample_df = train_df.head(5)
    dataloader = get_dataloader(sample_df, is_train=False)
    
    # 2. Inicializar Modelo
    model = SEDModel(num_classes=234).to(device)
    
    # 3. Inicializar Grad-CAM no último layer convolucional da EfficientNet
    target_layer = model.backbone.conv_head
    grad_cam = GradCAM(model, target_layer)
    
    # 4. Processar uma amostra
    waveforms, targets = next(iter(dataloader))
    input_tensor = waveforms[0:1].to(device) # Pega o 1º áudio do batch
    true_label_idx = torch.argmax(targets[0]).item()
    
    print(f"-> Processando áudio... Classe Alvo: {true_label_idx}")
    
    # Gerar Spectrograma Real para plot (mesmo processamento do forward pass)
    with torch.no_grad():
        mel_spec = model.mel_spectrogram(input_tensor)
        mel_db = model.amplitude_to_db(mel_spec)
        spectrogram_np = mel_db[0, 0].cpu().numpy() # [Mel_Bins, Time_Frames]
    
    # 5. Gerar o Mapa Grad-CAM
    cam_map, pred_class = grad_cam.generate(input_tensor, target_class=true_label_idx)
    
    print(f"-> Grad-CAM gerado. Map Shape: {cam_map.shape}")
    
    # 6. Salvar e Logar no MLflow
    mlflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlruns'))
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
    mlflow.set_experiment("BirdCLEF_2026_Fase4_GradCAM")
    
    with mlflow.start_run() as run:
        print(f"-> MLflow Run ID: {run.info.run_id}")
        
        # Plotar e salvar a imagem
        plot_path = "gradcam_sample.png"
        title = f"Grad-CAM (Pred: {pred_class} | True: {true_label_idx})"
        plot_gradcam(spectrogram_np, cam_map, title=title, save_path=plot_path)
        
        # Logar o artefato no MLflow
        mlflow.log_artifact(plot_path, "gradcam_plots")
        print(f"-> Imagem salva localmente em '{plot_path}' e logada no MLflow (Artifacts).")
        
    print("\n-> Teste Grad-CAM concluído com sucesso!")

if __name__ == "__main__":
    test_gradcam()
