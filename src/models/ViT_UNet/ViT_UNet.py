import torch
import torch.nn as nn

from src.models.ViT_UNet.layers import ViTBlock
from src.models.ViT_UNet.ViT_Encoder import Encoder
from src.models.ViT_UNet.ViT_Decoder import Decoder


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.middle_block = ViTBlock(config.init_dim * (2 ** (len(config.depths)-1)), config.num_heads[-1], config.mlp_ratio)
        self.decoder = Decoder(config)

    def forward(self, x, condition):

        x = torch.cat([condition, x], dim=1)
        x, skip_connections = self.encoder(x)
        skip_connections = list(reversed(skip_connections))
        x = self.middle_block(x)
        x = self.decoder(x, skip_connections)
        return x
