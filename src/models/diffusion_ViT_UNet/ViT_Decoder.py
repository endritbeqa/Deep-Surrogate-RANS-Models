import math

import torch
import torch.nn as nn
from src.models.diffusion_ViT_UNet.layers import ViTBlock, Upsample, Conv_layer



class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.last_layer = Conv_layer(config.init_dim//2, config.output_dim, (config.image_size, config.image_size))
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
        self.final_layer = Conv_layer(dim, config.output_dim, (config.image_size, config.image_size))


    def forward(self, x, t, skip_connections):
        for i, layer in enumerate(self.layers):
            x = torch.cat([x, skip_connections[i]], dim=2)
            upsample = layer[0]
            vit_blocks = layer[1:]
            x = upsample(x)
            for block in vit_blocks:
                x = block(x, t)
        x = self.last_layer(x, t)
        return x