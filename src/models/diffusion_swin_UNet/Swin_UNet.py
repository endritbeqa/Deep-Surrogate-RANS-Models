import torch
import torch.nn as nn
import math

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


    def sample(self, condition):
        B, C, H, W = condition.shape
        device = condition.device

        x_t = torch.randn((B, C, H, W), device=device)

        for t in reversed(range(self.config.timesteps)):
            t_tensor = torch.tensor([t] * B, device=device)
            t_emb = self.sinusoidal_embedding(t_tensor, math.prod(self.config.swin_decoder.time_embedding))
            t_emb = t_emb.view(B, *self.config.swin_decoder.time_embedding)

            input = torch.cat([condition, x_t], dim=1)
            skip_connections = self.encoder(input)
            skip_connections = list(reversed(skip_connections))
            skip_connections[0] = torch.cat([skip_connections[0], t_emb], dim=1)

            predicted_noise = self.decoder(skip_connections)

            alpha_bar_t = self.noise_scheduler.get_alpha_bar(t_tensor).view(-1, 1, 1, 1)
            alpha_bar_t_minus_1 = self.noise_scheduler.get_alpha_bar(torch.tensor([t-1], device=device)).view(-1, 1, 1, 1)

            if t > 0:
                noise = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
                x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                x_t = x_t + torch.sqrt(1 - alpha_bar_t_minus_1) * noise
            else:
                x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        return x_t



