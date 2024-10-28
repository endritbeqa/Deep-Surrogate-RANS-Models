import torch.nn as nn
from src.models.Time_embedding import TimeEmbedding

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(ConvBlock, self).__init__()
        self.time_embedding = TimeEmbedding(100, hidden_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.LayerNorm([hidden_channels, 1, 1])
        self.norm2 = nn.LayerNorm([hidden_channels, 1, 1])
        self.norm3 = nn.LayerNorm([out_channels, 1, 1])

        self.relu = nn.ReLU()

        self.skip_conv = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.time_embedding(out, t)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        skip = self.skip_conv(x)
        out += skip
        out = self.relu(out)

        return out