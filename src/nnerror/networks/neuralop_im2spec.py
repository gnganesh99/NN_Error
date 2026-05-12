

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from neuralop.models import FNO


class FNOEncoder(nn.Module):
    """
    Encode a 2D image into a latent vector using an FNO backbone.

    Architecture:
        (B, H, W)                              raw image
            │
            │  unsqueeze(1)                    add channel dim
            ▼
        (B, 1, H, W)
            │
            │  FNO 2D backbone                 spectral feature extraction
            ▼
        (B, hidden_channels, H, W)
            │
            │  AdaptiveAvgPool2d((1, 1))       collapse spatial dims
            ▼
        (B, hidden_channels, 1, 1)
            │
            │  flatten                         drop trivial dims
            ▼
        (B, hidden_channels)
            │
            │  MLP head                        project to latent size
            ▼
        (B, latent_dim)                        final latent vector

    Why this design:
        - FNO backbone gives resolution-invariant feature extraction:
          you can train on one image size and evaluate on another.
        - Global average pooling absorbs the spatial dimensions, so the
          head's input size doesn't depend on H or W.
        - The MLP head maps to a fixed-size latent regardless of input.

    Args:
        latent_dim       (int): Size N of the output latent vector.
        in_channels      (int): Channels in the input image (1 for grayscale,
                                3 for RGB, etc.). Default 1.
        hidden_channels  (int): Width of the FNO backbone (channels inside
                                each Fourier layer). Default 32.
        n_modes          (tuple): Number of Fourier modes kept per spatial
                                  axis. Length must be 2 for 2D. Default (16, 16).
        n_layers         (int): Number of stacked Fourier blocks. Default 4.
        mlp_hidden       (int): Width of the MLP head's hidden layer.
                                Default: same as hidden_channels.
    """

    def __init__(
        self,
        latent_dim,
        in_channels=1,
        hidden_channels=32,
        n_modes=(16, 16),
        n_layers=4,
        mlp_hidden=None,
    ):
        super().__init__()

        if mlp_hidden is None:
            mlp_hidden = hidden_channels

        # FNO 2D backbone: maps (B, in_channels, H, W) to (B, hidden_channels, H, W).
        # Setting out_channels=hidden_channels gives a richer feature map than 1.
        self.fno = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

        # Global average pooling over the spatial dims.
        # AdaptiveAvgPool2d((1, 1)) outputs (B, C, 1, 1) for ANY input H, W,
        # which is what gives this whole module its resolution invariance.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP head: projects the pooled feature vector to the desired latent_dim.
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, latent_dim),
        )

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, H, W) -- grayscale images.
               If you have multi-channel images already in (B, C, H, W) form,
               either remove the unsqueeze below or pass in_channels=C at init.

        Returns:
            Latent vector of shape (B, latent_dim).
        """
        # Add the channel dim that FNO expects: (B, H, W) -> (B, 1, H, W).
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # FNO backbone -> (B, hidden_channels, H, W)
        x = self.fno(x)

        # Pool to (B, hidden_channels, 1, 1), then flatten to (B, hidden_channels)
        x = self.pool(x).flatten(1)

        # Project to (B, latent_dim)
        return self.head(x)



























class FNO_im2spec(nn.Module):
    """
    im2spec-style encoder-decoder with an FNO encoder.

    The 2D image is encoded into a latent vector using a Fourier Neural
    Operator backbone (resolution-invariant), then decoded into a 1D
    spectrum by the same MLP decoder used in im2spec.
    """
    def __init__(self,
                 target_size: int,
                 latent_dim: int,
                 in_channels: int = 1,
                 hidden_channels: int = 32,
                 n_modes: tuple = (16, 16),
                 n_layers: int = 4,
                 mlp_hidden: int = None) -> None:

        super().__init__()

        self.ts = target_size

        # FNO backbone -> latent_dim vector
        self.fno_encoder = FNOEncoder(
            latent_dim=latent_dim,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=n_layers,
            mlp_hidden=mlp_hidden,
        )

        # Wrap the encoder function into an nn.Module (mirrors im2spec)
        self.encoder = Encoder_Wrapper(self._encoder)

        # Decoder: identical to im2spec
        self.dec_fc1 = nn.Linear(latent_dim, self.ts // 4)
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts // 4 * 2)
        self.dec_fc3 = nn.Linear(self.ts // 4 * 2, self.ts // 4 * 3)
        self.dec_fc4 = nn.Linear(self.ts // 4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        self.dec_fc6 = nn.Linear(self.ts, self.ts)

    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeds the input image into a latent vector via the FNO backbone.
        """
        return self.fno_encoder(features)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """Generate 1D signal from the embedded features."""
        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))
        x = F.relu(self.dec_fc5(x))
        return self.dec_fc6(x).reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""
        with torch.no_grad():
            return self.forward(x)
        


    
class Encoder_Wrapper(nn.Module):
    
    def __init__(self, encoder_fn):
        super().__init__()
        
        self.encoder_fn = encoder_fn
        
    def forward(self, x):
    
        return self.encoder_fn(x)