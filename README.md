# 🦜 BirdCLEF 2026 - Pantanal Ecosystem Soundscape Detection

![BirdCLEF 2026](https://img.shields.io/badge/Competition-BirdCLEF_2026-blue?style=for-the-badge&logo=kaggle)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM-brightgreen?style=for-the-badge)

## Overview

The Pantanal wetlands span over 150,000 km² and are home to unparalleled biodiversity. Monitoring this vast, remote landscape manually is practically impossible. The **BirdCLEF 2026** challenge asks us to listen to the ecosystem: identifying wildlife species from passive acoustic monitoring (PAM) recordings.

Our solution provides a **robust, lightweight, and highly explainable Machine Learning pipeline** capable of analyzing continuous audio data to automatically identify 234 different species of birds, amphibians, mammals, and insects.

---

## Why Our Solution is Innovative

### 1. Explainable AI (XAI) for Conservation
A major hurdle in conservation technology is the "black box" nature of deep learning. Biologists and ecologists need to *trust* the model. We implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize exactly what part of the audio the model is paying attention to when it identifies a species. 

By analyzing the generated Grad-CAM heatmaps overlaid on the Mel-Spectrograms (e.g., `gradcam_didactic_0.png`), researchers can visually verify if the model is keying in on actual bird calls or just background noise.

*Check out our generated Grad-CAM artifacts in the repository to see the model's "attention" hotspots in action!*

### 2. High Precision & Robustness
Messy, field-collected data requires robust training. Our pipeline utilizes:
- **SpecAugment** (Frequency & Time masking) to simulate missing data and background noise.
- **MixUp Augmentation** to create robust decision boundaries between the 234 overlapping species classes.
- **Attention Pooling** on top of the CNN backbone to intelligently aggregate features over the time dimension.

### 3. Extreme Computational Economy
Kaggle enforces a strict rule: **Inference must run offline and on CPU in under 90 minutes.** 
- We chose **EfficientNet-B0**, a highly optimized, lightweight architecture. 
- Audio preprocessing (Mel-Spectrogram generation) is pushed directly into the PyTorch computational graph via `torchaudio`, drastically reducing CPU bottlenecking and data-transfer overhead.
- Our `inference.py` script chunks 1-minute audio files and processes them dynamically as a single batch, minimizing read/write operations and fully saturating the CPU cores.

---

## Project Structure & Pipeline

```text
BirdCLEF+2026/
├── src/
│   ├── config.py           # Centralized hyperparameters & paths
│   ├── dataset.py          # PyTorch Dataset for chunking and multi-hot encoding
│   ├── eda.py              # Data sanity checks & Taxonomy loading
│   ├── model.py            # EfficientNet-B0 + Attention Pooling SED Model
│   ├── train.py            # Full training loop with AMP and MLflow tracking
│   ├── gradcam.py          # Explainability logic (Grad-CAM generation)
│   ├── metrics.py          # Kaggle's custom padded macro ROC-AUC metric
│   └── inference.py        # Offline CPU-optimized submission generator
├── mlruns/                 # Local MLflow tracking data (Loss, AUC, Artifacts)
├── create_dummy_test.py    # Script to simulate Kaggle's hidden test set
└── README.md               # You are here!
```

---

## Kaggle Rule Compliance

We have strictly engineered this repository to adhere to the BirdCLEF 2026 rules:
1. **No Internet Access during Inference:** The `model.py` architecture dynamically disables downloading pre-trained weights (`pretrained=False`) during the inference phase, ensuring the notebook does not crash in Kaggle's offline environment. Weights must be loaded locally.
2. **CPU Inference Time Limit:** The `inference.py` script chunks files and batches them natively, easily handling ~600 soundscapes within the 90-minute CPU window.
3. **Evaluation Metric:** We implemented `padded_cmap` in `metrics.py`, exactly mirroring Kaggle's macro-averaged ROC-AUC calculation (which pads classes without true positives to prevent zero-division errors).

---

## How to Run

### 1. Training the Model
To start training from scratch with Mixed Precision and MLflow tracking:
```bash
python src/train.py
```
*You can monitor the training progress, loss curves, and best models by running `mlflow ui` in the terminal.*

### 2. Explainability (Grad-CAM)
To test the model's vision and generate didactic heatmaps:
```bash
python src/phase4_inference.py
```
*This will generate `gradcam_didactic_*.png` images in your workspace.*

### 3. Generating the Kaggle Submission
To run the offline, CPU-only inference pipeline that iterates over `test_soundscapes` and generates `submission.csv`:
```bash
# Optional: generate some dummy test files first if you don't have the real test set downloaded
python create_dummy_test.py 

# Generate the submission
python src/inference.py
```

---

## Real-World Impact

Beyond winning competitions, this model is designed to be deployed directly into the Pantanal on low-power edge devices (like Raspberry Pi or Coral TPUs). Its lightweight footprint and interpretable outputs make it an ideal engine for a real-time, 24/7 acoustic monitoring grid, helping conservationists track species migration, population health, and ecosystem responses to climate change—without needing an expensive cloud infrastructure.
