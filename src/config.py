import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Data Paths
# Competition data is downloaded via kagglehub and cached locally.
# Falls back to a local "data/" directory when the cache is absent
# (e.g., during CI or early development without Kaggle credentials).
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\<user>\.cache\kagglehub\competitions\birdclef-2026")
if not DATA_DIR.exists():
    DATA_DIR = Path("data")

TRAIN_AUDIO_DIR            = DATA_DIR / "train_audio"
TRAIN_CSV_PATH             = DATA_DIR / "train.csv"
TAXONOMY_CSV_PATH          = DATA_DIR / "taxonomy.csv"
TRAIN_SOUNDSCAPES_DIR      = DATA_DIR / "train_soundscapes"
TRAIN_SOUNDSCAPES_LABELS_CSV_PATH = DATA_DIR / "train_soundscapes_labels.csv"
TEST_SOUNDSCAPES_DIR       = DATA_DIR / "test_soundscapes"
SUBMISSION_CSV_PATH        = Path("submission.csv")

# Best model checkpoint is stored at the project root (one level above src/).
# Using an absolute path anchored to this file avoids silent CWD-dependent
# save/load mismatches when scripts are launched from different directories.
BEST_MODEL_PATH = Path(__file__).resolve().parent.parent / "best_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Audio Parameters
# The competition mandates 32 kHz audio; all recordings are provided at this
# sample rate.  Each prediction window covers exactly 5 seconds of audio.
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 32_000                      # Hz  — mandated by competition
CHUNK_DURATION = 5                           # seconds per prediction window
CHUNK_LENGTH   = SAMPLE_RATE * CHUNK_DURATION  # samples per window

# 234 species columns in taxonomy.csv / submission.csv
NUM_CLASSES = 234

# ─────────────────────────────────────────────────────────────────────────────
# Mel-Spectrogram Parameters
# These values are tuned for wildlife vocalisations in the Pantanal.
# F_MIN / F_MAX bracket the vocal range of target species while excluding
# low-frequency wind noise and ultrasonic components outside the recording
# equipment's useful range.
# ─────────────────────────────────────────────────────────────────────────────
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 512
F_MIN      =    50   # Hz — low-frequency cutoff
F_MAX      = 14_000  # Hz — high-frequency cutoff

# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Parameters
# Tuned for the development machine: Intel i7-10700F + GTX 1650 Super (4 GB).
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 16
NUM_WORKERS = 4
