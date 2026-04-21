import os
import random
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_AUDIO_DIR, SAMPLE_RATE, CHUNK_LENGTH, BATCH_SIZE, NUM_WORKERS, TAXONOMY_CSV_PATH

class BirdCLEFDataset(Dataset):
    def __init__(self, df, audio_dir, chunk_length=CHUNK_LENGTH, sample_rate=SAMPLE_RATE, is_train=True):
        self.df = df
        self.audio_dir = audio_dir
        self.chunk_length = chunk_length
        self.sample_rate = sample_rate
        self.is_train = is_train
        
        # Load taxonomy to get all 234 classes mapping
        taxonomy = pd.read_csv(TAXONOMY_CSV_PATH)
        self.classes = sorted(taxonomy['primary_label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        audio_path = os.path.join(self.audio_dir, filename)
        
        try:
            import soundfile as sf
            
            # Get audio metadata to know the number of frames
            info = sf.info(audio_path)
            num_frames = info.frames
            orig_sr = info.samplerate
            
            # If audio is shorter than 5 seconds, pad it later. 
            # If longer, take a random chunk if training, or first chunk if not.
            if num_frames > self.chunk_length:
                if self.is_train:
                    start_frame = random.randint(0, num_frames - self.chunk_length)
                else:
                    start_frame = 0
                waveform_np, sr = sf.read(audio_path, start=start_frame, frames=self.chunk_length, dtype='float32', always_2d=True)
            else:
                waveform_np, sr = sf.read(audio_path, dtype='float32', always_2d=True)
                
            waveform = torch.from_numpy(waveform_np).T # [channels, time]
                
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # Convert to mono if it has multiple channels
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # Padding if shorter than chunk_length
            if waveform.shape[1] < self.chunk_length:
                pad_length = self.chunk_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
                
        except Exception as e:
            # If there's an error reading, return a zero tensor
            print(f"Error loading {audio_path}: {e}")
            waveform = torch.zeros((1, self.chunk_length))
            
        # Creating multi-hot labels based on primary_label
        label = row['primary_label']
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        if label in self.class_to_idx:
            target[self.class_to_idx[label]] = 1.0
            
        return waveform, target

def get_dataloader(df, is_train=True):
    dataset = BirdCLEFDataset(df, TRAIN_AUDIO_DIR, is_train=is_train)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_train,
        num_workers=NUM_WORKERS,
        pin_memory=True # speeds up transfer to GPU
    )
    return dataloader

if __name__ == "__main__":
    from eda import run_eda
    train_df, _ = run_eda()
    if train_df is not None:
        print("\n=== Testando o DataLoader ===")
        # Testing with a small sample
        test_loader = get_dataloader(train_df.head(10), is_train=True)
        for batch_idx, (waveforms, targets) in enumerate(test_loader):
            print(f"Batch {batch_idx + 1}")
            print(f"Waveforms shape: {waveforms.shape}")
            print(f"Targets shape: {targets.shape}")
            break
