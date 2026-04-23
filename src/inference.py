import soundfile as sf
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import (
    BEST_MODEL_PATH,
    CHUNK_LENGTH,
    SAMPLE_RATE,
    SUBMISSION_CSV_PATH,
    TAXONOMY_CSV_PATH,
    TEST_SOUNDSCAPES_DIR,
)
from model import SEDModel


def generate_submission():
    """
    Generate submission.csv for the Kaggle competition.

    Each test soundscape is split into N × 5-second chunks (N = floor(duration / 5)).
    The model predicts species-presence probabilities for each chunk, and
    the results are written as rows in submission.csv using the format:

        row_id = <soundscape_stem>_<end_time_seconds>

    For example, the 00:15–00:20 window of file
    BC2026_Test_0001_S05_20250227_010002.ogg becomes row_id
    BC2026_Test_0001_S05_20250227_010002_20.

    Competition constraints:
      - Submissions must run as CPU-only Kaggle notebooks (<= 90 minutes).
        GPU notebook submissions are limited to 1 minute of runtime.
      - Internet access is disabled during scoring.
      - The model is loaded with pretrained=False to prevent any attempt to
        download ImageNet weights, which would fail in the offline environment.
    """
    print("=== Generating submission.csv (CPU, offline) ===")

    # Force CPU: Kaggle scores CPU notebooks; GPU runtime is limited to 1 min.
    device = torch.device("cpu")
    print(f"Inference device: {device}")
    INFERENCE_BATCH = 12  # sub-batch size for memory control

    # ── Load taxonomy to determine class names and their correct order ──────
    try:
        taxonomy_df = pd.read_csv(TAXONOMY_CSV_PATH)
        classes = sorted(taxonomy_df["primary_label"].unique())
    except FileNotFoundError:
        print(f"WARNING: {TAXONOMY_CSV_PATH} not found — using placeholder class names.")
        classes = [f"class_{i}" for i in range(234)]

    num_classes = len(classes)

    # ── Load model ──────────────────────────────────────────────────────────
    # pretrained=False prevents any network call in the offline Kaggle environment.
    print("Loading model (pretrained=False to ensure offline compatibility)...")
    model = SEDModel(num_classes=num_classes, pretrained=False).to(device)

    if BEST_MODEL_PATH.exists():
        model.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        )
        print(f"Weights loaded from {BEST_MODEL_PATH}")
    else:
        print(
            f"WARNING: {BEST_MODEL_PATH} not found — using random weights for demonstration."
        )

    model.eval()

    # ── Locate test soundscapes ─────────────────────────────────────────────
    if not TEST_SOUNDSCAPES_DIR.exists():
        print(f"Test directory {TEST_SOUNDSCAPES_DIR} not found.")
        print("This is expected when running outside Kaggle (no test files available).")
        test_files = []
    else:
        test_files = sorted(TEST_SOUNDSCAPES_DIR.glob("*.ogg"))

    print(f"Found {len(test_files)} test soundscapes in {TEST_SOUNDSCAPES_DIR}")

    # ── Process each soundscape ─────────────────────────────────────────────
    submission_rows = []

    with torch.no_grad():
        for file_path in tqdm(test_files, desc="Processing soundscapes"):
            file_stem = file_path.stem  # e.g. "BC2026_Test_0001_S05_20250227_010002"

            try:
                waveform_np, sr = sf.read(file_path, dtype="float32", always_2d=True)
                # soundfile returns (frames, channels); transpose to (channels, frames).
                waveform = torch.from_numpy(waveform_np).T

                if sr != SAMPLE_RATE:
                    waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

            except Exception as exc:
                print(f"Error reading {file_path}: {exc} — using silence.")
                waveform = torch.zeros((1, SAMPLE_RATE * 60))  # 60-second silent fallback

            # Divide soundscape into 5-second chunks (supports variable-length files).
            duration_samples = waveform.shape[1]
            n_chunks  = max(1, duration_samples // CHUNK_LENGTH)
            chunks    = []
            end_times = []

            for chunk_idx in range(n_chunks):
                start = chunk_idx * CHUNK_LENGTH
                chunk = waveform[:, start : start + CHUNK_LENGTH]

                # Zero-pad if the final chunk is shorter than CHUNK_LENGTH.
                if chunk.shape[1] < CHUNK_LENGTH:
                    pad   = CHUNK_LENGTH - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad))

                chunks.append(chunk)
                end_times.append((chunk_idx + 1) * 5)  # 5, 10, 15, …

            # Process in sub-batches to keep RAM usage bounded.
            import numpy as np
            all_probs = []
            for b in range(0, len(chunks), INFERENCE_BATCH):
                sub   = torch.stack(chunks[b : b + INFERENCE_BATCH], dim=0).to(device)
                logits = model(sub)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_probs = np.concatenate(all_probs, axis=0)  # (n_chunks, num_classes)

            for i, end_time in enumerate(end_times):
                row = {"row_id": f"{file_stem}_{end_time}"}
                for cls_idx, cls_name in enumerate(classes):
                    row[cls_name] = float(all_probs[i, cls_idx])
                submission_rows.append(row)

    # ── Write submission.csv ────────────────────────────────────────────────
    if submission_rows:
        submission_df = pd.DataFrame(submission_rows)
        # Ensure row_id is the first column, followed by classes in sorted order.
        submission_df = submission_df[["row_id"] + classes]
        submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)
        print(f"submission.csv saved — shape: {submission_df.shape}")
    else:
        print("No test files processed — submission.csv not written.")


if __name__ == "__main__":
    generate_submission()
