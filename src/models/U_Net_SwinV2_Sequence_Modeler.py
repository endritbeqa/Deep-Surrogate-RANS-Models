import torch
import torch.nn as nn
from src.models import Swin_VAE_encoder, Swin_VAE_decoder, Transformer_decoder




class U_NET_Swin_Sequence_Modeler(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Swin_VAE_encoder.Swin_VAE_encoder(config)
        self.decoder = Swin_VAE_decoder.Swin_VAE_decoder(config)
        skip_connection_decoders = []
        for i in range(config.sequence_modeler.depth):
            skip_connection_decoders.append(
                Transformer_decoder.TransformerDecoder(config.sequence_modeler)
            )
        self.sequence_model = nn.ModuleList(skip_connection_decoders)


    def forward(self, inputs):
        encoder_output = self.encoder(inputs)

        latent_predictions = []
        for i, skip_connection in enumerate(encoder_output):
            b, t, c, h, w = skip_connection.shape
            skip_connection.view(b, t, c*h*w)
            latent_predictions = self.sequence_model[i](skip_connection)
            latent_predictions = latent_predictions.view(b, t, c, h, w)
            latent_predictions.append(latent_predictions)

        prediction = self.decoder(latent_predictions)

        return prediction