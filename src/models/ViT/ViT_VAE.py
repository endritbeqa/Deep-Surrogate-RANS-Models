import torch
import torch.nn as nn
from src.models.ViT import Config_ViT_VAE,encoder, decoder, Z_cell

class VisionTransformerVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = encoder.VisionTransformerEncoder(config.encoder)
        self.bottleneck = Z_cell.VAEBottleneck(config.bottleneck)
        self.decoder = decoder.VisionTransformerDecoder(config.decoder)

    def forward(self, x):

        encoder_output = self.encoder(x)
        cls_token = encoder_output[:, 0, :]
        z, mu, logvar = self.bottleneck(cls_token)
        z = z.unsqueeze(1).repeat(1, encoder_output.size(1), 1)
        reconstructed = self.decoder(z, encoder_output)

        return reconstructed, mu, logvar