import torch
import torch.nn as nn
from src.models.swin_hierarchical_VAE import Swin_VAE_decoder, Swin_VAE_encoder


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)



    #TODO ADD a skip connection at input resolution you dumbass
    def forward(self, condition, target):

        hidden_states, condition_latent = self.encoder(condition, target)



