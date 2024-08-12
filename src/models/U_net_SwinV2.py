import torch.nn as nn
from src.models import Swin_VAE_encoder, Swin_VAE_decoder



class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)


    def forward(self, condition, target):
        z, mu, log_var = self.encoder(condition, target)
        prediction = self.decoder(z)

        return prediction , mu, log_var


