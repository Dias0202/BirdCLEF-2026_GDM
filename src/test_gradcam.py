import os

import mlflow
import torch

from config import BATCH_SIZE
from dataset import get_dataloader
from eda import run_eda
from gradcam import GradCAM, plot_gradcam
from model import SEDModel


def test_gradcam():
    """
    Smoke test for the Grad-CAM pipeline.

    Loads a tiny subset of training data, runs a forward + backward pass through
    the model to produce a Grad-CAM heatmap, and logs the resulting plot to a
    dedicated MLflow experiment for visual inspection.
    """
    print("=== Grad-CAM Smoke Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load a small data sample ────────────────────────────────────────────
    train_df, _ = run_eda()
    if train_df is None:
        return

    dataloader = get_dataloader(train_df.head(5), is_train=False)

    # ── Initialise model and Grad-CAM ───────────────────────────────────────
    model = SEDModel(num_classes=234).to(device)

    # conv_head is the last convolutional layer of EfficientNet-B0,
    # providing rich semantic features while retaining spatial resolution.
    target_layer = model.backbone.conv_head
    grad_cam     = GradCAM(model, target_layer)

    # ── Generate one Grad-CAM heatmap ───────────────────────────────────────
    waveforms, targets = next(iter(dataloader))
    input_tensor    = waveforms[0:1].to(device)
    true_label_idx  = torch.argmax(targets[0]).item()
    print(f"Target class index: {true_label_idx}")

    # Extract the spectrogram for plotting (no gradient needed).
    with torch.no_grad():
        mel_spec    = model.mel_spectrogram(input_tensor)
        mel_db      = model.amplitude_to_db(mel_spec)
        spec_np     = mel_db[0, 0].cpu().numpy()   # (n_mels, time_frames)

    cam_map, pred_class = grad_cam.generate(input_tensor, target_class=true_label_idx)
    print(f"Grad-CAM map shape: {cam_map.shape}  |  predicted class: {pred_class}")

    # ── Log the result to MLflow ────────────────────────────────────────────
    mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    mlflow.set_experiment("BirdCLEF_2026_Phase4_GradCAM")

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        save_path = "gradcam_sample.png"
        title     = f"Grad-CAM (pred={pred_class} | true={true_label_idx})"
        plot_gradcam(spec_np, cam_map, title=title, save_path=save_path)

        mlflow.log_artifact(save_path, "gradcam_plots")
        print(f"Plot saved to '{save_path}' and logged to MLflow.")

    print("Grad-CAM smoke test complete.")


if __name__ == "__main__":
    test_gradcam()
