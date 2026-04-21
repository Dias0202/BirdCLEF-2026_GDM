import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import torchaudio
import timm
from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX

class AttentionPooling(torch_nn.Module):
    def __init__(self, in_features, hidden_dim=512):
        super().__init__()
        self.attention = torch_nn.Sequential(
            torch_nn.Linear(in_features, hidden_dim),
            torch_nn.Tanh(),
            torch_nn.Linear(hidden_dim, 1),
            torch_nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [batch_size, time_steps, in_features]
        attn_weights = self.attention(x) # [batch_size, time_steps, 1]
        pooled_x = torch.sum(x * attn_weights, dim=1) # [batch_size, in_features]
        return pooled_x

class SEDModel(torch_nn.Module):
    def __init__(self, num_classes=234, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        
        # Audio to Mel Spectrogram transformation (on-the-fly, inside the model, on GPU)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # SpecAugment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=64)
        
        # Backbone CNN (timm)
        # in_chans=1 because spectrogram is 1 channel (grayscale)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        
        # Get the number of features outputted by the backbone
        in_features = self.backbone.num_features
        
        # Remove the classification head from the backbone
        self.backbone.classifier = torch_nn.Identity()
        self.backbone.global_pool = torch_nn.Identity()
        
        # Attention Pooling head
        self.pool = AttentionPooling(in_features)
        
        # Final classification layer
        self.classifier = torch_nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, 1, audio_length]
        
        # 1. Frontend: Audio to Log-Mel Spectrogram
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)
        
        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        
        # Normalization (roughly standardize image around 0)
        x = (x - x.mean()) / (x.std() + 1e-6)
        
        # 2. Backbone: Feature extraction
        # EfficientNet outputs [batch_size, channels, freq_bins, time_steps]
        x = self.backbone.forward_features(x)
        
        # 3. Pooling over frequency dimension
        # Shape becomes [batch_size, channels, time_steps]
        x = torch.mean(x, dim=2) 
        
        # Transpose for Attention Pooling: [batch_size, time_steps, channels]
        x = x.transpose(1, 2)
        
        # 4. Attention Pooling over time
        x = self.pool(x)
        
        # 5. Classification
        logits = self.classifier(x)
        
        return logits

if __name__ == "__main__":
    print("=== Testando SEDModel (EfficientNet-B0 + Attention Pooling) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Instantiate model
    model = SEDModel(num_classes=234).to(device)
    
    # Create dummy waveform tensor: [batch_size, channels, length]
    # Example: 2 audios, 1 channel, 5 seconds at 32kHz
    dummy_audio = torch.randn(2, 1, 5 * 32000).to(device)
    
    # Forward pass
    logits = model(dummy_audio)
    
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape (logits): {logits.shape}")
    print("Sucesso! Modelo consegue processar o áudio cru até os logits finais.")
