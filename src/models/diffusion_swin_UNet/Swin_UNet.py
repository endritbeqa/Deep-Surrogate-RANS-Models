import math

import torch
import torch.nn as nn

from src.models.diffusion_swin_UNet import Swin_decoder, Swin_encoder, Noise_scheduler
from src.models import Time_embedding


class U_NET_Swin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Swin_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_decoder.Swin_VAE_decoder(config)
        self.noise_scheduler = Noise_scheduler.get_noise_scheduler(self.config)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")


    def noise_step(self, x_0, t):
        noise = torch.randn_like(x_0).to(self.device)
        noisy_data = self.noise_scheduler.sqrt_alphas_bar[t] * x_0 + self.noise_scheduler.sqrt_one_minus_alphas_bar[t] * noise
        return noisy_data, noise

    def sinusoidal_embedding(self, timesteps, dim):
        half_dim = dim // 2
        embeddings = math.log(10000)
        embeddings /= (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embeddings)
        embeddings = embeddings.to(self.device)
        timesteps = timesteps.to(self.device)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(self, condition, noisy_data, t_emb):

        input = torch.cat([condition, noisy_data], dim=1)
        skip_connections = self.encoder(input, t_emb)
        skip_connections = list(reversed(skip_connections))
        predicted_noise = self.decoder(skip_connections, t_emb)

        return predicted_noise


    def sample(self, condition, num_samples,eta=1.0):
        condition = condition.unsqueeze(0).repeat(num_samples, 1, 1, 1)

        B, C, H, W = condition.shape
        x_t = torch.randn_like(condition).to(self.device)
        time_steps = torch.linspace(self.config.timesteps - 1, 0, self.config.timesteps).long()

        t = torch.tensor([self.noise_scheduler.steps], device=self.device).repeat(x_t.shape[0])
        t_pre = t - 1

        for t in time_steps:
            print(t)
            t_batch = torch.full((x_t.size(0),), t)
            t_emb = self.sinusoidal_embedding(t_batch, 100)
            noise_pred = self.forward(condition, x_t, t_emb)

            coef1 = 1 / self.noise_scheduler.sqrt_alphas[t]
            coef2 = self.noise_scheduler.betas[t] / self.noise_scheduler.sqrt_one_minus_alphas_bar[t]
            sig = torch.sqrt(self.noise_scheduler.betas[t]) * self.noise_scheduler.sqrt_one_minus_alphas_bar[t_pre] / self.noise_scheduler.sqrt_one_minus_alphas_bar[t]
            x_t = coef1 * (x_t - coef2 * noise_pred) + sig * torch.randn_like(x_t)

            t = t_pre
            t_pre = t_pre - 1


        return x_t




