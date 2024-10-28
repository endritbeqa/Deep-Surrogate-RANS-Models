import math

import torch
import torch.nn as nn
from src.models.diffusion_ViT_UNet.layers import ViTBlock



class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1)

    def forward(self, x):
        b, l, c = x.shape
        h = w = int(math.sqrt(l))
        x = x.permute(0,2,1)
        x = x.view(b,c,h,w)

        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(2)
        x = x.permute(0,2,1)

        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_blocks = len(config.depths)
        self.depths = config.depths
        self.num_heads = config.num_heads
        self.mlp_ratio = config.mlp_ratio
        self.layers = nn.ModuleList()
        dim = config.init_dim * (2 ** (self.num_blocks-1))
        for i, num_blocks in enumerate(self.depths):
            blocks = nn.ModuleList()
            blocks.append(Upsample(2*dim))
            for j in range(num_blocks):
                blocks.append(ViTBlock(dim//2, self.num_heads[i], self.mlp_ratio))
            self.layers.append(blocks)
            dim = dim//2


    def forward(self, x, t, skip_connections):
        for i, layer in enumerate(self.layers):
            x = torch.cat([x, skip_connections[i]], dim=2)
            upsample = layer[0]
            vit_blocks = layer[1:]
            x = upsample(x)
            for block in vit_blocks:
                x = block(x, t)
        return x