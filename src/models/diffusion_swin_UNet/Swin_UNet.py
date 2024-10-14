import torch
import torch.nn as nn
import math

from src.models.diffusion_swin_UNet import Swin_decoder, Swin_encoder


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.encoder = Swin_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_decoder.Swin_VAE_decoder(config)

    def sinusoidal_embedding(self, timesteps, dim):
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embeddings)
        embeddings = embeddings.to("cuda:0")
        timesteps = timesteps.to("cuda:0")
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(self, condition, target, t):
        B, C, H, W = target.shape
        t_emb = self.sinusoidal_embedding(t, math.prod(self.config.swin_encoder.skip_connection_shape[-1]))
        t_emb = t_emb.view(B, *self.config.swin_encoder.skip_connection_shape[-1])

        input = torch.cat([condition, target], dim=1)
        skip_connections = self.encoder(input)

        skip_connections = list(reversed(skip_connections))
        skip_connections[0] = torch.cat([skip_connections[0], t_emb], dim=1)

        prediction = self.decoder(skip_connections)

        return prediction






