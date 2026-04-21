import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)
        
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # Backward pass for the target class
        target = logits[0, target_class]
        target.backward()
        
        # Calculate Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # [batch, channels, 1, 1]
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True) # [batch, 1, H, W]
        cam = F.relu(cam) # ReLU
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam[0, 0].cpu().numpy(), target_class

def plot_gradcam(spectrogram, cam, title="Grad-CAM", save_path=None):
    """
    spectrogram: 2D numpy array
    cam: 2D numpy array
    """
    import cv2
    
    # Resize CAM to spectrogram dimensions
    cam_resized = cv2.resize(cam, (spectrogram.shape[1], spectrogram.shape[0]))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    # Plot Spectrogram
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    
    # Overlay CAM
    ax.imshow(cam_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Bins')
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig
