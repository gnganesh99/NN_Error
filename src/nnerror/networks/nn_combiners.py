
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from atomai.nets.blocks import ConvBlock as conv_block



        
        
class Swa_Ensemble(nn.Module):

    def __init__(self, model_list):
        super().__init__()

        self.models = model_list
        
    def forward(self, x):
        pred = [model(x) for model in self.models]
        return pred

    def train(self):
        [model.train() for model in self.models]
    
    def eval(self):
        [model.eval() for model in self.models]

    def predict(self, x):
        
        [model.eval() for model in self.models]

        with torch.no_grad():
        
            return self.forward(x)
    
    def to(self, device):
        
        [model.to(device) for model in self.models]
    
    
    
class Encoder_Wrapper(nn.Module):
    
    def __init__(self, encoder_fn):
        super().__init__()
        
        self.encoder_fn = encoder_fn
        
    def forward(self, x):
    
        return self.encoder_fn(x)
    
    
    
    
# class error_model(nn.Module):

#     def __init__(self, in_dim):
#         super(error_model, self).__init__()
#         self.encoder = im2spec(in_dim, target_size=1, latent_dim = 64).encoder
#         self.fc1 = nn.Linear(64, 32)
#         self.fc2 = nn.Linear(32, 1)
        
#     def forward(self, x):

#         x = x.unsqueeze(1)
#         x = self.encoder(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         return x.reshape(-1)
    
#     def to(self, device):
#         for param in self.parameters():
#             param.data = param.data.to(device)
#             if param.grad is not None:
#                 param.grad.data = param.grad.data.to(device)



        
class error_model(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size,
                 target_size: int = 1,
                 latent_dim: int = 32,
                 nb_filters_enc: int = 128,
                 nb_filters_dec: int = 64) -> None:
        
        super().__init__()
        
        self.n, self.m = feature_size
        
        self.ts = target_size
        
        self.e_filt = nb_filters_enc
        
        self.d_filt = nb_filters_dec
        # Encoder params
        
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, batch_norm=True, dropout_ = 0.1)
        
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        # Decoder params
        
        self.dec_fc1 = nn.Linear(latent_dim, 32 )
        self.dec_fc2 = nn.Linear(32, 16 )
        self.dec_fc3 = nn.Linear(16, 8)
        self.dec_fc4 = nn.Linear(8, 1) 
        
 
        
       
    def encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.m * self.n)
        return self.enc_fc(x)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """

        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))  
        
        
        return x.reshape(-1, self.ts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)

        
        

class ensemble_error_model(nn.Module):

    def __init__(self, in_dim, n_models, model = error_model):
        super().__init__()

        self.models = [model(in_dim) for i in range(n_models)]
        
    def forward(self, x):

        ensemble_error = [model(x) for model in self.models]

        return ensemble_error
    
    def predict(self, x):

        [model.eval() for model in self.models]

        with torch.no_grad():
            return self.forward(x)
        
    def train(self):
        
        [model.train() for model in self.models]
    
    def eval(self):

        [model.eval() for model in self.models]

    def to(self, device):
        
        [model.to(device) for model in self.models]
        
        
        


class DecoderModule(nn.Module):
    def __init__(self, embed_dim, target_size=1):
        
        super(DecoderModule, self).__init__()
        
        self.ts = target_size
        
        self.dec_fc1 = nn.Linear(embed_dim, 32)
        self.dec_fc2 = nn.Linear(32, 16)
        self.dec_fc3 = nn.Linear(16, 8)
        self.dec_fc4 = nn.Linear(8, target_size)

    def forward(self, embedding):
        
        x = F.relu(self.dec_fc1(embedding))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = self.dec_fc4(x)
        
        return x.reshape(-1, self.ts)
            
            
        
class CustomDecoder(nn.Module):
    def __init__(self, encoder, embed_dim, target_size=1):
        super(CustomDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = DecoderModule(embed_dim, target_size)  # Now a module

    def forward(self, x):
        x = x.unsqueeze(1)
        embedding = self.encoder(x)
        return self.decoder(embedding)

    def train_only_decoder(self):
        
        # Freeze encoder params
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True
            
    def predict(self, x: torch.Tensor):
        """Predict spectra from image"""

        with torch.no_grad():
            return self.forward(x)


