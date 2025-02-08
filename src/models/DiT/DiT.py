import torch
import torch.nn as nn

from src.models.DiT.Encoder import Encoder
from src.models.DiT.Decoder import MLP_decoder, CNN_decoder


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

    def forward(self, x, condition):

        x = torch.cat([condition, x], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
