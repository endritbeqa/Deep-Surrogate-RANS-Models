import torch.nn as nn
from src.models.swin import Swin_VAE_encoder, Swin_VAE_decoder



class U_NET_Swin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)


    def forward(self, input):
        Swin_encoder_output, shape = self.encoder(input)
        skip_connections = Swin_encoder_output.hidden_states
        prediction = self.decoder(Swin_encoder_output.last_hidden_state, skip_connections, shape)

        return prediction


