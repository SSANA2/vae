#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import Module
import torchgan.models as models
import numpy as np

from src.vae import VAE

class AAE(VAE):
    def __init__(self, input_dim, channels, num_z):
        super(AAE, self).__init__(input_dim=input_dim,
                                   channels=channels,
                                   num_z=num_z)
    def encode(self, x):
        h = self.encoder(x)
        return self.z_mu(h), self.z_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x_bar = self.decoder(z)
        return torch.sigmoid(x_bar)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_batch = self.decode(z)
        
        return recon_batch, mu, logvar
    
    @torch.no_grad()
    def reconstruct(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_batch = self.decode(z)
        return recon_batch

class Discriminator(VAE):
    def __init__(self,input_dim, channels,num_z):
        super(Discriminator, self).__init__(input_dim = input_dim,
                                            channels=channels,
                                            num_z=num_z)
        self.discriminator = nn.Sequential(
            nn.Linear(num_z, num_z//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_z//2, num_z//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_z//4, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.discriminator(z)
        return validity        
