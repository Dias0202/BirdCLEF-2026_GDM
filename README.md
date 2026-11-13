# BirdCLEF+ 2026 — Pantanal acoustic species identification

![Kaggle](https://img.shields.io/badge/Competition-BirdCLEF%2B_2026-blue?style=flat-square&logo=kaggle)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)

Sound event detection (SED) pipeline for 234 Pantanal species (birds, amphibians, mammals, reptiles and insect
sonotypes) in passive acoustic monitoring recordings, built for the
[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition. Trained on a consumer GPU
(GTX 1650 Super, 4 GB) with mixed precision.

## Validation status

The competition metric is a macro-averaged ROC-AUC that **skips classes with no true positives**
([`References/kaggle - overview.md`](References/kaggle%20-%20overview.md)). Validation now follows that rule and a
split that does not leak between near-duplicate recordings:

| | Before | Now |
|---|---|---|
| Metric | `padded_cmap`: adds all-positive padding rows, so classes with no positives in the fold score ≈ 1.0 and inflate the mean | `macro_roc_auc` (`src/metrics.py`): mean per-class ROC-AUC over classes with at least one positive (and one negative) |
| Split | random 80/20 by row | `StratifiedGroupKFold(5)` grouped by recordist, stratified by species (`src/validation.py`); recordings with unknown author are their own group |

Each `train.csv` row is already one file, so grouping by file alone would change nothing. Recordists upload
several recordings from the same session and site, and those near-duplicates are what a row split puts on both sides.

**The validation score under the corrected protocol has not been recomputed yet.** Earlier runs reported a
padded ROC-AUC around 0.98–0.99. That number is not a valid estimate of competition performance and is not reported
here. Validation also uses the first 5 s of focal `train_audio` recordings, while the competition scores 5 s windows
of continuous soundscapes. Expect a domain gap.

## Approach

```
Raw waveform  (1 × 160,000 samples @ 32 kHz, 5 s window)
      |
  MelSpectrogram (128 mels, 50–14,000 Hz) + AmplitudeToDB     torchaudio, on GPU
      |
  SpecAugment: FrequencyMasking(24) + TimeMasking(64)         training only
      |
  Per-batch normalisation
      |
  EfficientNet-B0 (timm, ImageNet-pretrained, in_chans=1)
      |
  Mean over frequency  →  AttentionPooling over time (2-layer MLP + softmax)
      |
  Linear (1280 → 234)  →  sigmoid  →  per-species probabilities
```

| Component | Detail |
|---|---|
| Loss | `BCEWithLogitsLoss` (multi-label); targets from `primary_label` |
| Optimiser | Adam, lr 1e-3, 30 epochs, batch 16 |
| Mixed precision | `torch.amp` autocast + `GradScaler` |
| Augmentation | MixUp (α = 0.5) on waveforms and labels; SpecAugment |
| Validation | fold 0 of 5, grouped by recordist; competition-style macro ROC-AUC |
| Tracking | MLflow (SQLite backend): params, losses, validation AUC, best checkpoint |
| Explainability | Grad-CAM on `backbone.conv_head`, projected onto the log-Mel spectrogram |
| Inference | `src/inference.py` splits each soundscape into 5 s windows; `kaggle_submission.ipynb` is the self-contained notebook |

Attention pooling replaces global average pooling so a short call in a mostly silent window is not diluted.

### Grad-CAM examples

Three panels per validation clip: log-Mel spectrogram, Grad-CAM overlay, isolated activation map. These are
qualitative examples, not an evaluation.

<table>
  <tr><td><img src="gradcam_didactic_0.png" alt="Grad-CAM example" width="560"/></td></tr>
  <tr><td align="center"><i>Sustained harmonic call: activation follows the harmonic structure</i></td></tr>
  <tr><td><img src="gradcam_didactic_5.png" alt="Grad-CAM example" width="560"/></td></tr>
  <tr><td align="center"><i>Periodic pulse train: activation on the pulses, background frames suppressed</i></td></tr>
</table>

## Project structure

```
src/
  config.py              audio/mel parameters and data paths (BIRDCLEF_DATA_DIR or the kagglehub cache)
  dataset.py             PyTorch Dataset: 5 s windows (random in training, first window in validation), resampling
  model.py               SEDModel: EfficientNet-B0 + AttentionPooling + SpecAugment
  metrics.py             competition-style macro ROC-AUC (no PyTorch dependency)
  validation.py          recordist-grouped, species-stratified folds
  train.py               training loop: AMP, MixUp, MLflow logging, checkpointing
  inference.py           soundscape → submission.csv
  gradcam.py             Grad-CAM; phase4_inference.py renders validation examples
tests/                   unit tests for the metric and the folds
main.py                  downloads the competition data with kagglehub (needs KAGGLE_API_TOKEN in .env)
kaggle_submission.ipynb  Kaggle inference notebook
docs/                    phase reports (Portuguese); AUC numbers in them use the old metric
```

Model weights, `mlruns/` and `mlflow.db` are not versioned. Competition data is never committed (Kaggle rules).

## Reproducing

```bash
pip install -r requirements.txt            # CUDA 11.8 wheels for PyTorch; adjust the index URL for your setup
python main.py                             # download competition data (KAGGLE_API_TOKEN in .env)
python src/train.py                        # train; inspect with `mlflow ui --backend-store-uri sqlite:///mlflow.db`
python src/phase4_inference.py             # Grad-CAM figures on the validation fold
python create_dummy_test.py && python src/inference.py   # local inference smoke test
```

Unit tests (no GPU or data needed):

```bash
pip install -r requirements-test.txt
pytest
```

## Next steps

1. Re-run training with the grouped folds and report the per-fold macro ROC-AUC with its spread.
2. Validate on the expert-labelled `train_soundscapes` segments, which match the test domain better than focal recordings.
3. Baselines to beat: embeddings from a pretrained bioacoustic model (BirdNET or Perch) + logistic regression per species.
4. Use `secondary_labels` as additional positives and compare.
5. Report the public leaderboard score of a submitted notebook alongside the local estimate.

## License

Code: [MIT](LICENSE). Competition data and the copied competition pages in `References/` belong to their owners
and follow the competition rules.
