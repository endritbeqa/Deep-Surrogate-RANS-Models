import torch.nn as nn
from src.models.swin import Swin_VAE_encoder, Swin_VAE_decoder


class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)


    def forward(self, input):
        swin_hidden_states = self.encoder(input)
        swin_hidden_states = list(reversed(swin_hidden_states))
        prediction = self.decoder(swin_hidden_states)

        return prediction


