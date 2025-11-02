import torch
import torch.nn as nn

from src.models.uncertainty.diffusion.Swin_UNet.Decoder import Decoder
from src.models.uncertainty.diffusion.Swin_UNet.Encoder import Encoder
from src.models.uncertainty.diffusion.Swin_UNet.Middle_Block import Middle_Block


class Swin_UNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
        self.middel_block = Middle_Block(config.middle_block)

    def forward(self, conditions, x, t):
        x = torch.cat([conditions, x], dim=1)
        skip_connections = self.encoder(x, t)
        x = skip_connections[-1]
        x = self.middel_block(x, t)
        skip_connections = list(reversed(skip_connections))
        x = self.decoder(x, skip_connections, t)

        return x
