import torch
import torch.nn as nn
from src.models.ViT import Config_ViT_VAE, encoder, decoder, Z_cell


class AutoregressiveImageTransformer(nn.Module):
    def __init__(self, config ,img_size=32, patch_size=4, in_channels=3, embed_dim=32, num_heads=4, num_layers=2):
        super(AutoregressiveImageTransformer, self).__init__()


        self.encoder = encoder.Encoder(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
                               num_layers=num_layers, num_heads=num_heads)

        self.decoder = decoder.Decoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, num_layers=num_layers,
                               num_heads=num_heads)

    def forward(self,inputs ,x):

        encoded_patches = self.encoder(x)
        output_image = self.decoder(encoded_patches)

        return output_image, [345,454], [567, 45]