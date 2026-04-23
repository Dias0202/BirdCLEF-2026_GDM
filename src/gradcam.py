import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for the SEDModel.

    Grad-CAM uses the gradients of the target class score flowing into the
    chosen convolutional layer to produce a coarse localisation map that
    highlights the spectrogram regions most influential for the prediction.

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017.

    Usage:
        cam_gen = GradCAM(model, model.backbone.conv_head)
        heatmap, pred_class = cam_gen.generate(input_tensor)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients    = None

        # Register hooks to capture activations and gradients during
        # the forward and backward passes respectively.
        target_layer.register_forward_hook(self._save_feature_maps)
        target_layer.register_full_backward_hook(self._save_gradients)

    # ── Hook callbacks ─────────────────────────────────────────────────────

    def _save_feature_maps(self, module, input, output):
        """Store the feature maps produced by the target layer on the forward pass."""
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Store the gradients flowing into the target layer on the backward pass."""
        self.gradients = grad_output[0].detach()

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor:  Waveform batch of shape (1, 1, audio_length).
                           Must have requires_grad=True or be wrapped before
                           calling this method.
            target_class:  Species class index to explain.  If None, the
                           predicted (argmax) class is used automatically.

        Returns:
            tuple: (cam_np, target_class)
                   cam_np       – 2-D numpy array in [0, 1], shape (H, W).
                   target_class – The class index that was explained.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass.
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # Backward pass for the target class score only.
        logits[0, target_class].backward()

        # Compute per-channel importance weights as the global-average-pooled gradients.
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps followed by ReLU (keep only positive activations).
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalise to [0, 1] so the heatmap can be displayed as a probability map.
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0, 0].cpu().numpy(), target_class


def plot_gradcam(
    spectrogram: np.ndarray,
    cam: np.ndarray,
    title: str = "Grad-CAM",
    save_path: str = None,
):
    """
    Overlay a Grad-CAM heatmap on top of a log-Mel spectrogram.

    Args:
        spectrogram: 2-D numpy array, shape (n_mels, time_frames).
        cam:         2-D numpy array in [0, 1]; will be resized to match the
                     spectrogram dimensions before plotting.
        title:       Figure title string.
        save_path:   If provided, save the figure to this path and close it.
                     If None, return the Figure object for further use.

    Returns:
        matplotlib.figure.Figure or None.
    """
    # Resize the coarse CAM to the full spectrogram resolution.
    cam_resized = cv2.resize(cam, (spectrogram.shape[1], spectrogram.shape[0]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Draw the spectrogram as the base layer.
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")

    # Overlay the CAM with transparency so both layers are visible.
    ax.imshow(cam_resized, aspect="auto", origin="lower", cmap="jet", alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bins")
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    return fig
