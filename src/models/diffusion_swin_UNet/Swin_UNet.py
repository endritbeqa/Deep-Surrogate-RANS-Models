import math

import torch
import torch.nn as nn

from src.models.diffusion_swin_UNet import Swin_decoder, Swin_encoder, Noise_scheduler


class U_NET_Swin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Swin_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_decoder.Swin_VAE_decoder(config)
        self.noise_scheduler = Noise_scheduler.get_noise_scheduler(self.config)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")


    def noise_step(self, x_0, t):
        alpha_bar_t = self.noise_scheduler.get_alpha_bar(t).to(self.device)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0).to(self.device)
        noisy_data = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_data, noise

    def sinusoidal_embedding(self, timesteps, dim):
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embeddings)
        embeddings = embeddings.to(self.device)
        timesteps = timesteps.to(self.device)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(self, condition, noisy_data, t_emb):

        input = torch.cat([condition, noisy_data], dim=1)
        skip_connections = self.encoder(input)

        skip_connections = list(reversed(skip_connections))
        skip_connections[0] = torch.cat([skip_connections[0], t_emb], dim=1)

        predicted_noise = self.decoder(skip_connections)

        return predicted_noise

    def sample(self, condition, eta=1.0):
        B, C, H, W = condition.shape
        x = torch.randn(condition.shape)
        time_steps = torch.linspace(self.config.timesteps - 1, 0, self.config.timesteps).long()

        for t in time_steps:
            t_batch = torch.full((x.size(0),), t)
            t_emb = self.sinusoidal_embedding(t_batch, math.prod(self.config.swin_decoder.time_embedding))
            t_emb = t_emb.view(B, *self.config.swin_decoder.time_embedding)
            noise_pred = self(condition, x, t_emb)
            beta_t = self.noise_scheduler.betas[t]
            alpha_t = 1.0 - beta_t
            x = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = eta * beta_t.sqrt()
                x += sigma_t * noise

        return x




