import torch.nn as nn
from einops import rearrange
from src.models.diffusion_ViT_UNet.layers import ViTBlock

class PatchMerging(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        b, l, c = x.shape
        x = x.view(b, l // 4, 4, c)
        x = x.flatten(2)
        x = self.proj(x)
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

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_blocks = len(config.depths)
        self.input_dim = config.input_dim
        self.init_dim = config.init_dim
        self.depths = config.depths
        self.num_heads = config.num_heads
        self.mlp_ratio = config.mlp_ratio
        self.patch_size = config.patch_size
        self.patch_embed = PatchEmbedding(self.input_dim, self.init_dim, self.patch_size)
        self.layers = nn.ModuleList()
        dim = self.init_dim
        for i, num_blocks in enumerate(self.depths):
            blocks = nn.ModuleList()
            for j in range(num_blocks):
                blocks.append(ViTBlock(dim, self.num_heads[i], self.mlp_ratio))
            if i < self.num_blocks-1:
                blocks.append(PatchMerging(4*dim, 2*dim))
            else: blocks.append(None)
            dim *= 2
            self.layers.append(blocks)

    def forward(self, x, t):
        x = self.patch_embed(x)
        skip_connections = []
        for layer in self.layers:
            vit_blocks = layer[:-1]
            downsample = layer[-1]
            for vit_block in vit_blocks:
                x = vit_block(x, t)
            skip_connections.append(x)
            if downsample is not None:
                x = downsample(x)
        return x, skip_connections
