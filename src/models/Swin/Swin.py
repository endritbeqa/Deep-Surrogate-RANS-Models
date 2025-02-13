import torch
import torch.nn as nn
from src.models.Swin.Encoder import Encoder
from src.models.Swin.Decoder import MLP_decoder, CNN_decoder


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

    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.decoder(x)

        return x
