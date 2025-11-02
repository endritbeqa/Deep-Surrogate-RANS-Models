import torch
import torch.nn as nn

from src.models.uncertainty.diffusion.Swin.Decoder import CNN_decoder, MLP_decoder
from src.models.uncertainty.diffusion.Swin.Encoder import Encoder


class Swin(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(config.encoder)
        if config.decoder == 'MLP':
            self.decoder = MLP_decoder(config.MLP_decoder)
        elif config.decoder == "CNN":
            self.decoder = CNN_decoder(config.CNN_decoder)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

    def forward(self, conditions, x, t):
        x = torch.cat([conditions, x], dim=1)
        x = self.encoder(x, t)[0]
        x = self.decoder(x, t)

        return x
