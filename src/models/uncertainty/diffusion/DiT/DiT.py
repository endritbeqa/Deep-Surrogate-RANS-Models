import torch
import torch.nn as nn

from src.models.uncertainty.diffusion.DiT.Decoder import CNN_decoder, MLP_decoder
from src.models.uncertainty.diffusion.DiT.Encoder import Encoder


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        if config.decoder == 'MLP':
            self.decoder = MLP_decoder(config.MLP_decoder)
        elif config.decoder == "CNN":
            self.decoder = CNN_decoder(config.CNN_decoder)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

    def forward(self, x, condition, t):

        x = torch.cat([condition, x], dim=1)
        x = self.encoder(x, t)
        x = self.decoder(x, t)
        return x
