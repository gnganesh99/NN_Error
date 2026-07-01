import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


#from im2spec.im2spec.models import conv_block, dilated_block

import atomai
from atomai.nets.blocks import ResBlock, ResModule
from atomai.nets.blocks import ConvBlock as conv_block
from atomai.nets.blocks import DilatedBlock as dilated_block

class custom_nn(nn.Module):
    def __init__(self, input_channels, output_spectra_length, latent_dim, img_size=(32, 32)):
        super(custom_nn, self).__init__()
        self.output_spectra_length = output_spectra_length
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Encoder part
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Added Batch Normalization
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # Added Batch Normalization
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Added another conv layer
        self.bn3 = nn.BatchNorm2d(128) # Added Batch Normalization
        self.pool3 = nn.MaxPool2d(2, 2) # Added another pooling layer

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Calculate flattened_size dynamically
        # Start with input image dimensions
        h, w = img_size
        # After pool1 (halves dimensions)
        h, w = h // 2, w // 2
        # After pool2 (halves dimensions again)
        h, w = h // 2, w // 2
        # After pool3 (halves dimensions again)
        h, w = h // 2, w // 2
        self.flattened_size = 128 * h * w # 128 is out_channels of conv3

        self.fc_encoder = nn.Linear(self.flattened_size, latent_dim)

        #encoder module
        self.encoder = nn.Module() 
        self.encoder.forward = lambda x: self._encode(x)

        # Decoder part
        self.dec_fc1 = nn.Linear(latent_dim, output_spectra_length // 4)
        self.dec_fc2 = nn.Linear(output_spectra_length // 4, output_spectra_length // 2)
        self.dec_fc3 = nn.Linear(output_spectra_length // 2, output_spectra_length)


    def _encode(self, x):
        # Ensure input has channel dimension for Conv2d
        if x.dim() == 3: # If input is (batch_size, H, W)
            x = x.unsqueeze(1) # Make it (batch_size, 1, H, W)

        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1)) 


        # Flatten the output for the fully connected layers
        x = x.view(-1, self.flattened_size)

        encoded = self.fc_encoder(x)
        return encoded

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """
        x = F.leaky_relu(self.dec_fc1(encoded), negative_slope=0.1)
        x = F.leaky_relu(self.dec_fc2(x), negative_slope=0.1)
        output = self.dec_fc3(x)
        return output.reshape(-1, self.output_spectra_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict spectra from image"""
        with torch.no_grad():
            return self.forward(x)
