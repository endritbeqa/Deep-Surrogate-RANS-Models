import torch
import torch.nn as nn
from src.models.ViT.layers import PositionalEncoding, PatchEmbedding

class Encoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=64, num_layers=4, num_heads=4):
        
        super(Encoder, self).__init__()

        # Patch embedding
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                              embed_dim=embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.patch_embedding.n_patches, embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Get patch embeddings
        patches = self.patch_embedding(x)

        # Add positional encoding
        patches = self.pos_encoding(patches)

        # Encode patches using transformer encoder
        encoded_patches = self.encoder(patches)

        return encoded_patches  # Output is a sequence of encoded patch embeddings
