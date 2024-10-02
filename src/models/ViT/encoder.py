import torch.nn as nn
from src.models.ViT.layers import PositionalEncoding, PatchEmbedding

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.patch_embedding = PatchEmbedding(img_size=config.img_size, patch_size=config.patch_size, in_channels=config.in_channels,
                                              embed_dim=config.embed_dim)

        self.pos_encoding = PositionalEncoding(self.patch_embedding.n_patches, config.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    def forward(self, x):
        patches = self.patch_embedding(x)
        patches = self.pos_encoding(patches)
        encoded_patches = self.encoder(patches)

        return encoded_patches
