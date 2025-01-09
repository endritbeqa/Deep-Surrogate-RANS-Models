import math
import torch
import torch.nn as nn
from src.models.diffusion.Time_embedding import TimeEmbedding

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.time_embedding = TimeEmbedding(100, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, time):
        x = self.time_embedding(x, time)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Conv_layer(nn.Module):
    def __init__(self, input_channels, kernel_size, hidden_dim, output_channels):
        super().__init__()
        self.time_embedding = TimeEmbedding(100, input_channels)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)
        self.skip_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.non_linearity = nn.GELU()
        self.norm1 = nn.GroupNorm(num_groups=hidden_dim // 4, num_channels=hidden_dim)
        self.norm2 = nn.GroupNorm(num_groups=hidden_dim // 4, num_channels=hidden_dim)

    def forward(self, x, t):
        x_initial = x
        x = self.time_embedding(x, t)
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.non_linearity(x)
        x = self.norm2(x)
        x = self.conv3(x)

        x_initial = self.skip_conv(x_initial)
        x = x + x_initial

        return x


class Upsample(nn.Module):
    def __init__(self, dim, output_res):
        super().__init__()
        self.upsample = nn.Upsample(size=output_res, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=dim//4, num_channels=dim)

    def forward(self, x, reshape=True):
        if reshape:
            b, l, c = x.shape
            h = w = int(math.sqrt(l))
            x = x.permute(0, 2, 1)
            x = x.view(b, c, h, w)

        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1) #turn into B, L, C
        return x



'''
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.time_embedding = TimeEmbedding(100, 3)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        condition = x[:, 0:3, :, :]
        y = x[:, 3:6, :, :]
        y = self.time_embedding(y)

        x = torch.cat([condition, y], dim=1)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1) #turn into B, L, C
        return x
'''