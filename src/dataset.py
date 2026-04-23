import random
import soundfile as sf
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config import (
    TRAIN_AUDIO_DIR,
    SAMPLE_RATE,
    CHUNK_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    TAXONOMY_CSV_PATH,
)


class BirdCLEFDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF 2026 audio clips.

    Each sample returns:
      waveform – float32 tensor of shape (1, CHUNK_LENGTH)
      target   – float32 one-hot vector of shape (NUM_CLASSES,)

    Chunking strategy:
      Training  : a random 5-second window is extracted so the model
                  sees different parts of each recording every epoch.
      Validation: the first 5-second window is always used, ensuring
                  reproducible evaluation across runs.

    Files shorter than CHUNK_LENGTH are zero-padded on the right.
    Corrupt or missing files are silently replaced with a silent chunk
    so that one bad file does not abort an entire training run.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir,
        chunk_length: int = CHUNK_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.chunk_length = chunk_length
        self.sample_rate = sample_rate
        self.is_train = is_train

        # Build a deterministic class index from the taxonomy file so that
        # label indices are stable regardless of the order rows appear in df.
        taxonomy = pd.read_csv(TAXONOMY_CSV_PATH)
        self.classes = sorted(taxonomy["primary_label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = str(self.audio_dir / row["filename"])

        try:
            # Read only the required window instead of the whole file.
            # sf.info() is cheap (reads the header only) and lets us know
            # how many frames exist before committing to a full read.
            info = sf.info(audio_path)
            num_frames = info.frames
            orig_sr = info.samplerate

            if num_frames > self.chunk_length:
                start_frame = (
                    random.randint(0, num_frames - self.chunk_length)
                    if self.is_train
                    else 0
                )
                waveform_np, sr = sf.read(
                    audio_path,
                    start=start_frame,
                    frames=self.chunk_length,
                    dtype="float32",
                    always_2d=True,
                )
            else:
                waveform_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)

            # soundfile returns shape (frames, channels); transpose to (channels, frames).
            waveform = torch.from_numpy(waveform_np).T

            # Resample only when the source rate differs from the target rate.
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

            # Collapse multi-channel audio to mono by averaging all channels.
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Zero-pad clips that are shorter than the required chunk length.
            if waveform.shape[1] < self.chunk_length:
                pad_length = self.chunk_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        except Exception as exc:
            # Return silence so a single corrupt file does not crash training.
            print(f"[Dataset] Failed to load {audio_path}: {exc}")
            waveform = torch.zeros((1, self.chunk_length))

        # Build a one-hot target vector from the primary species label.
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        label = row["primary_label"]
        if label in self.class_to_idx:
            target[self.class_to_idx[label]] = 1.0

        return waveform, target


def get_dataloader(df: pd.DataFrame, is_train: bool = True) -> DataLoader:
    """Return a DataLoader for the given metadata DataFrame."""
    dataset = BirdCLEFDataset(df, TRAIN_AUDIO_DIR, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_train,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # speeds up host-to-GPU transfer
    )


if __name__ == "__main__":
    from eda import run_eda

    train_df, _ = run_eda()
    if train_df is not None:
        print("\n=== DataLoader smoke test ===")
        loader = get_dataloader(train_df.head(10), is_train=True)
        for waveforms, targets in loader:
            print(f"Waveforms : {waveforms.shape}")
            print(f"Targets   : {targets.shape}")
            break
