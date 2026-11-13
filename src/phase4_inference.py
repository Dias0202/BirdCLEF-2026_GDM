import os

import cv2
import matplotlib.pyplot as plt
import mlflow
import torch
from torch.utils.data import DataLoader

from config import BEST_MODEL_PATH, TRAIN_AUDIO_DIR
from dataset import BirdCLEFDataset
from eda import run_eda
from gradcam import GradCAM
from model import SEDModel
from validation import train_val_split


def main():
    """
    Phase 4 — Explainability: Generate and log Grad-CAM visualisations.

    For the first 10 samples of the validation split, this script produces a
    three-panel figure per sample:
        Panel 1 — Original log-Mel spectrogram
        Panel 2 — Spectrogram with Grad-CAM overlay (model attention)
        Panel 3 — Standalone Grad-CAM heatmap (activation hotspots)

    All figures are logged as MLflow artifacts under the 'gradcam_results'
    subdirectory of the most recent training run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── MLflow setup ────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path    = os.path.abspath(os.path.join(script_dir, "..", "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("BirdCLEF_2026_Real_Training")

    # ── Load trained model ──────────────────────────────────────────────────
    model = SEDModel(num_classes=234).to(device)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print("Model loaded successfully.")

    # ── Prepare validation data ─────────────────────────────────────────────
    # Reproduce the validation fold used during training so that the
    # Grad-CAM visualisations are generated on held-out (unseen) recordists.
    train_df, taxonomy_df = run_eda()
    if train_df is None:
        return

    _, val_split = train_val_split(train_df, fold=0, n_splits=5, seed=42)

    # Build a label-to-common-name mapping for readable plot titles.
    label_to_name = dict(
        zip(taxonomy_df["primary_label"].astype(str), taxonomy_df["common_name"])
    )

    val_dataset = BirdCLEFDataset(val_split.head(10), audio_dir=TRAIN_AUDIO_DIR, is_train=False)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ── Grad-CAM setup ──────────────────────────────────────────────────────
    # conv_head is the last convolutional layer of EfficientNet-B0 and captures
    # high-level semantic features before global pooling — an ideal target for CAM.
    target_layer = model.backbone.conv_head
    print(f"Grad-CAM target layer: {target_layer}")
    cam_gen = GradCAM(model, target_layer)

    # ── Find the most recent training run ───────────────────────────────────
    experiment = mlflow.get_experiment_by_name("BirdCLEF_2026_Real_Training")
    if experiment is None:
        print("ERROR: Experiment not found. Run train.py first.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        print("ERROR: No runs found for this experiment.")
        return

    target_run_id = runs.iloc[0].run_id
    print(f"Logging artifacts to run: {target_run_id}")

    # ── Generate Grad-CAM figures ───────────────────────────────────────────
    with mlflow.start_run(run_id=target_run_id):
        for count, (waveforms, _) in enumerate(val_loader):
            waveforms = waveforms.to(device)

            # Get the true species name from the validation split metadata.
            primary_label = val_split.iloc[count]["primary_label"]
            species_name  = label_to_name.get(str(primary_label), "Unknown Species")

            # Compute the spectrogram for visualisation (no gradient needed here).
            with torch.no_grad():
                spec    = model.mel_spectrogram(waveforms)
                spec_db = model.amplitude_to_db(spec)
                spec_np = spec_db[0, 0].cpu().numpy()  # (n_mels, time_frames)

            # Grad-CAM requires gradients, so we enable them on the input.
            input_tensor = waveforms[0:1].detach().requires_grad_(True)
            cam, pred_class = cam_gen.generate(input_tensor)

            # Resize the coarse CAM to the full spectrogram resolution.
            cam_resized = cv2.resize(cam, (spec_np.shape[1], spec_np.shape[0]))

            # Three-panel figure: original | overlay | heatmap only.
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].imshow(spec_np, aspect="auto", origin="lower", cmap="magma")
            axes[0].set_title(f"Original Spectrogram\n({species_name})")
            axes[0].set_xlabel("Time Frames")
            axes[0].set_ylabel("Mel Bins")

            axes[1].imshow(spec_np, aspect="auto", origin="lower", cmap="magma")
            axes[1].imshow(cam_resized, aspect="auto", origin="lower", cmap="jet", alpha=0.4)
            axes[1].set_title(f"Model Attention (Grad-CAM)\nSpecies: {species_name}")
            axes[1].set_xlabel("Time Frames")

            im3 = axes[2].imshow(cam_resized, aspect="auto", origin="lower", cmap="jet")
            axes[2].set_title("Activation Hotspots")
            axes[2].set_xlabel("Time Frames")
            plt.colorbar(im3, ax=axes[2])

            plt.suptitle(
                f"Phase 4 — Model Explainability: {species_name}  "
                f"(predicted class {pred_class})",
                fontsize=14,
            )
            plt.tight_layout()

            save_path = f"gradcam_didactic_{count}.png"
            plt.savefig(save_path)
            plt.close()

            mlflow.log_artifact(save_path, "gradcam_results")
            print(f"  [{count}] Grad-CAM saved for: {species_name}")

    print("\nPhase 4 complete — Grad-CAM artifacts logged to MLflow.")


if __name__ == "__main__":
    main()
