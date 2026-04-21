import os
from pathlib import Path

# Paths
DATA_DIR = Path(r"C:\Users\<user>\.cache\kagglehub\competitions\birdclef-2026")
TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"
TRAIN_CSV_PATH = DATA_DIR / "train.csv"
TAXONOMY_CSV_PATH = DATA_DIR / "taxonomy.csv"

# Hyperparameters (Optimized for GTX 1650 Super - 4GB VRAM)
SAMPLE_RATE = 32000
CHUNK_DURATION = 5  # seconds
CHUNK_LENGTH = SAMPLE_RATE * CHUNK_DURATION

BATCH_SIZE = 16
NUM_WORKERS = 4  # Good for i7-10700F

# Mel Spectrogram Parameters
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 50
F_MAX = 14000
