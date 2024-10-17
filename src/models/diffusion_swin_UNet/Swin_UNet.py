import torch
import torch.nn as nn

from src.models.diffusion_swin_UNet import Swin_decoder, Swin_encoder


class U_NET_Swin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Swin_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_decoder.Swin_VAE_decoder(config)

    def forward(self, condition, noisy_data, t_emb):

        input = torch.cat([condition, noisy_data], dim=1)
        skip_connections = self.encoder(input)

        skip_connections = list(reversed(skip_connections))
        skip_connections[0] = torch.cat([skip_connections[0], t_emb], dim=1)

        predicted_noise = self.decoder(skip_connections)

        return predicted_noise






