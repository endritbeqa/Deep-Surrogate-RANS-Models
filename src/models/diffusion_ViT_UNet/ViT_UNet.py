import math

import torch
import torch.nn as nn

from src.models.diffusion_ViT_UNet import Noise_scheduler
from src.models.diffusion_ViT_UNet.layers import ViTBlock
from src.models.diffusion_ViT_UNet.ViT_Encoder import Encoder
from src.models.diffusion_ViT_UNet.ViT_Decoder import Decoder
from src.models.Time_embedding import TimeEmbedding



class Final_layer(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.time_embedding = TimeEmbedding(100, dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim//2, dim // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim // 2, output_dim, kernel_size=1)
        self.layerNorm1 = nn.LayerNorm([dim//2, 32, 32])
        self.layerNorm2 = nn.LayerNorm([dim // 2, 32, 32])

    def forward(self, x, t):
        b, l, c = x.shape
        h = w = int(math.sqrt(l))
        x = x.permute(0,2,1)
        x = x.view(b,c,h,w)

        x = self.time_embedding(x, t)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.layerNorm1(x)
        x = self.conv2(x)
        x = self.layerNorm2(x)
        x = self.conv3(x)

        return x


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.middle_block = ViTBlock(config.init_dim * (2 ** (len(config.depths)-1)), config.num_heads[-1], config.mlp_ratio)
        self.decoder = Decoder(config)
        self.final_proj = Final_layer(64, 3)
        self.noise_scheduler = Noise_scheduler.get_noise_scheduler(self.config)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

    def noise_step(self, x_0, t):
        noise = torch.randn_like(x_0).to(self.device)
        noisy_data = self.noise_scheduler.sqrt_alphas_bar[t] * x_0 + self.noise_scheduler.sqrt_one_minus_alphas_bar[
            t] * noise
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

    def forward(self, x, condition, t):

        x = torch.cat([condition, x], dim=1)

        x, skip_connections = self.encoder(x, t)
        skip_connections = list(reversed(skip_connections))
        x = self.middle_block(x, t)
        x = self.decoder(x, t, skip_connections)
        return self.final_proj(x, t)


    def sample(self, condition, num_samples,eta=1.0):
        condition = condition.unsqueeze(0).repeat(num_samples, 1, 1, 1)

        B, C, H, W = condition.shape
        x_t = torch.randn_like(condition).to(self.device)
        time_steps = torch.linspace(self.config.timesteps - 1, 1, self.config.timesteps-1).long()

        t = torch.tensor([self.noise_scheduler.steps], device=self.device).repeat(x_t.shape[0])
        t_pre = t - 1

        for t in time_steps:
            #print(t)
            t_batch = torch.full((x_t.size(0),), t)
            t_emb = self.sinusoidal_embedding(t_batch, 100)
            noise_pred = self.forward(condition, x_t, t_emb)

            coef1 = 1 / self.noise_scheduler.sqrt_alphas[t]
            coef2 = self.noise_scheduler.betas[t] / self.noise_scheduler.sqrt_one_minus_alphas_bar[t]
            sig = torch.sqrt(self.noise_scheduler.betas[t]) * self.noise_scheduler.sqrt_one_minus_alphas_bar[t_pre] / self.noise_scheduler.sqrt_one_minus_alphas_bar[t]
            x_t = coef1 * (x_t - coef2 * noise_pred) + sig * torch.randn_like(x_t)

            #if torch.isnan(x_t).any():
            #    print("NaN detected")

            t = t_pre
            t_pre = t_pre - 1


        return x_t
