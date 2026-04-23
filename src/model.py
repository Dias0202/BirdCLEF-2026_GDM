import torch
import torch.nn as nn
import torchaudio
import timm

from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX


class AttentionPooling(nn.Module):
    """
    Learned attention-weighted pooling over the time dimension.

    Given a sequence of frame-level feature vectors, this module produces a
    single fixed-length clip-level embedding by computing a weighted sum.
    The weights are produced by a two-layer MLP with Tanh activation followed
    by a Softmax so that they sum to 1 across all time steps.

    Input  shape: (batch, time_steps, in_features)
    Output shape: (batch, in_features)
    """

    def __init__(self, in_features: int, hidden_dim: int = 512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),  # weights sum to 1 over the time axis
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x        : (batch, time_steps, in_features)
        # weights  : (batch, time_steps, 1)
        # output   : (batch, in_features)
        weights = self.attention(x)
        return torch.sum(x * weights, dim=1)


class SEDModel(nn.Module):
    """
    Sound Event Detection (SED) model for BirdCLEF 2026.

    The model processes raw audio waveforms end-to-end on the GPU.

    Pipeline:
        1. Waveform → Log-Mel Spectrogram          (torchaudio MelSpectrogram)
        2. SpecAugment frequency + time masking    (training only)
        3. Per-batch zero-mean unit-variance norm
        4. EfficientNet-B0 feature extraction      (timm, ImageNet pretrained, 1-channel input)
        5. Mean pooling over the frequency axis
        6. AttentionPooling over the time axis
        7. Linear classifier → raw logits for NUM_CLASSES species

    Outputs raw logits.  Apply torch.sigmoid() at inference time to obtain
    species-presence probabilities in the range [0, 1].
    """

    def __init__(
        self,
        num_classes: int = 234,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Frontend: waveform → log-Mel spectrogram ──────────────────────
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # ── SpecAugment data augmentation ──────────────────────────────────
        # Applied only during training (guarded by self.training flag in forward).
        # FrequencyMasking hides up to 24 mel bins; TimeMasking hides up to 64 frames.
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=64)

        # ── Backbone: EfficientNet-B0 ──────────────────────────────────────
        # in_chans=1: the spectrogram is treated as a single-channel (greyscale) image.
        # pretrained=True: uses ImageNet weights for faster convergence, which is
        # standard practice even for non-image domains (transfer learning).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1
        )
        in_features = self.backbone.num_features

        # Remove the backbone's built-in global pooling and classification head
        # so that we can apply our own temporal aggregation strategy.
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        # ── Temporal aggregation ───────────────────────────────────────────
        self.pool = AttentionPooling(in_features)

        # ── Final classifier ───────────────────────────────────────────────
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform, shape (batch, 1, audio_length).

        Returns:
            logits: Shape (batch, num_classes).
        """
        # Step 1 – Convert waveform to log-Mel spectrogram.
        x = self.mel_spectrogram(x)   # (batch, 1, n_mels, time_frames)
        x = self.amplitude_to_db(x)

        # Step 2 – Apply SpecAugment only when the model is in training mode.
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # Step 3 – Normalise to approximately zero mean, unit variance.
        # This is computed per batch and stabilises training across varying
        # recording conditions (room acoustics, mic sensitivity, etc.).
        x = (x - x.mean()) / (x.std() + 1e-6)

        # Step 4 – CNN feature extraction.
        # EfficientNet returns (batch, channels, freq_bins, time_steps).
        x = self.backbone.forward_features(x)

        # Step 5 – Collapse the frequency axis with mean pooling.
        # Shape after: (batch, channels, time_steps)
        x = torch.mean(x, dim=2)

        # Step 6 – Rearrange for AttentionPooling: (batch, time_steps, channels).
        x = x.transpose(1, 2)

        # Step 7 – Attend over the time axis to get a clip-level embedding.
        x = self.pool(x)    # (batch, channels)

        # Step 8 – Linear classification.
        return self.classifier(x)   # (batch, num_classes)


if __name__ == "__main__":
    print("=== SEDModel smoke test (EfficientNet-B0 + Attention Pooling) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SEDModel(num_classes=234).to(device)
    dummy_audio = torch.randn(2, 1, 5 * 32_000).to(device)
    logits = model(dummy_audio)

    print(f"Input  : {dummy_audio.shape}")
    print(f"Output : {logits.shape}")
    assert logits.shape == (2, 234), "Unexpected output shape!"
    print("Smoke test passed.")
