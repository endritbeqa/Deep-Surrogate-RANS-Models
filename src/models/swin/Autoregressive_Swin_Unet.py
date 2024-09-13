import torch.nn as nn
from src.models.swin import Swin_encoder, Swin_decoder, Transformer_decoder


class U_NET_Swin_Sequence_Modeler(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Swin_encoder.Swin_encoder(config)
        self.decoder = Swin_decoder.Swin_decoder(config)
        skip_connection_decoders = []
        for i in range(len(config.sequence_modeler.depths)):
            skip_connection_decoders.append(
                Transformer_decoder.Decoder(config.sequence_modeler, i)
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