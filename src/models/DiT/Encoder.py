import torch.nn as nn
from src.models.ViT_UNet.layers import ViTBlock, PatchEmbedding


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
            self.layers.append(blocks)

    def forward(self, x, t):
        x = self.patch_embed(x)
        for layer in self.layers:
            for vit_block in layer[:]:
                x = vit_block(x, t)

        return x
