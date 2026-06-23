import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from atomai.nets.blocks import ConvBlock as conv_block
from nnerror.networks.nn_combiners import Encoder_Wrapper
from nnerror.networks.neuralop_im2spec import FNOEncoder



class Attn_Block(nn.Module):
    def __init__(self, feature_dims, embed_dim = 32, num_heads = 4):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.feature_dims = feature_dims
        self.embed_dim = embed_dim

        self.token_projection = nn.Linear(1, embed_dim)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, feature_dims, embed_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.out_projection = nn.Linear(embed_dim, 1)

        self.attn_weights = None

    def forward(self, x, residual = True):
        """
        x: [B, D]

        B = batch size
        D = number of feature tokens
        embed_dim = embedding size of each token
        """

        x = x.unsqueeze(-1)                       # [B, D, 1]

        x = self.token_projection(x)              # [B, D, embed_dim]

        x = x + self.pos_embedding                # [B, D, embed_dim]

        attn_output, attn_weights = self.attn(
            x, x, x,
            need_weights=True
        )                                         # [B, D, embed_dim]

        self.attn_weights = attn_weights.detach()       # [B, D, D]

        if residual:
            x = self.attn_norm(x + attn_output)       # [B, D, embed_dim]
        else:
            x = self.attn_norm(attn_output)           # [B, D, embed_dim]

        x = self.out_projection(x).squeeze(-1)    # [B, D]

        return x, self.attn_weights                                  # [B, D]

    def get_attn_weights(self):
        return self.attn_weights # [B, D, D]





class im2spec_attn(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int,
                 latent_dim: int,
                 nb_filters_enc: int = 128,
                 nb_filters_dec: int = 64) -> None:

        """Initialize im2spec_attn."""
        super(im2spec_attn, self).__init__()

        self.n, self.m = feature_size

        self.ts = target_size

        self.e_filt = nb_filters_enc

        self.d_filt = nb_filters_dec


        # Encoder params
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, batch_norm=True, dropout_ = 0.5)

        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        self.attn_block = Attn_Block(feature_dims=latent_dim, embed_dim=32, num_heads=4)
        self.attn_weights = None


        self.encoder = Encoder_Wrapper(self._encoder)


        # Decoder params

        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts //4 * 3 )
        self.dec_fc4 = nn.Linear(self.ts //4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        self.dec_fc6 = nn.Linear(self.ts, self.ts)


    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        B, C, H, W = features.shape

        x = self.enc_conv(features)
        x = x.reshape(B, self.e_filt * self.m * self.n)
        x = F.relu(self.enc_fc(x))
        x, self.attn_weights = self.attn_block(x)

        return x

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """

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

    def get_attn_weights(self):
        return self.attn_weights.detach() if self.attn_weights is not None else None






class FNO_im2spec_attn(nn.Module):
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

        """Initialize FNO_im2spec_attn."""
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
        self.attn_block = Attn_Block(feature_dims=latent_dim, embed_dim=32, num_heads=4)
        self.attn_weights = None

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
        x = self.fno_encoder(features)
        x, self.attn_weights = self.attn_block(x)
        return x


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
        
        if x.ndim == 3:
            x = x.unsqueeze(1) # Add channel dimension if missing.

        encoded = self.encoder(x)
        return self.decoder(encoded)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""
        with torch.no_grad():
            return self.forward(x)

    def get_attn_weights(self):
        return self.attn_weights.detach() if self.attn_weights is not None else None
























# This uses the entire image as feature tokens to calculate the attention --> Could result in a memory issue.
class attn_im2spec(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int,
                 latent_dim: int,
                 embed_dim: int = 32,
                 num_heads: int = 4
                 ) -> None:

        """Initialize attn_im2spec."""
        super(attn_im2spec, self).__init__()

        self.n, self.m = feature_size

        self.ts = target_size

        # Encoder params

        self.enc_attn = Attn_Block(feature_dims=self.n*self.m, embed_dim=embed_dim, num_heads=num_heads)
        self.attn_weights = None
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.n * self.m, latent_dim *2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )

        # Decoder params


#         #Wrap the encoder function into an `nn.Module`
#         self.encoder = nn.Module()
#         self.encoder.forward = lambda x: self._encoder(x)
        self.encoder = Encoder_Wrapper(self._encoder)



        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts //4 * 3 )
        self.dec_fc4 = nn.Linear(self.ts //4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        self.dec_fc6 = nn.Linear(self.ts, self.ts)


    def _encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        B, C, H, W = features.shape
        x = features.reshape(B, self.n * self.m) # [B, n*m]
        x, self.attn_weights = self.enc_attn(x)  # [B, n*m]; [B, n*m, n*m]
        x = self.enc_mlp(x)                      # [B, latent_dim]

        return x


    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """

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


    def get_attn_weights(self):
        return self.attn_weights.detach() if self.attn_weights is not None else None






