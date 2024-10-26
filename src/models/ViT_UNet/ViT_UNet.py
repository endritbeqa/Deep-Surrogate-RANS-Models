import torch
import torch.nn as nn
from ml_collections import ConfigDict

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Convert to patches
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

# ViT Block
class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(ViTBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, time_embedding):
        # Time embedding injection
        x = x + time_embedding
        # Transformer block
        x_attn = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x

# Patch to Image (Decoder)
class PatchToImage(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size):
        super(PatchToImage, self).__init__()
        self.patch_size = patch_size
        self.unprojection = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, h, w):
        x = x.transpose(1, 2).view(-1, x.size(-1), h // self.patch_size, w // self.patch_size)
        return self.unprojection(x)

# UNet with ViT Blocks
class UNetViT(nn.Module):
    def __init__(self, config):
        super(UNetViT, self).__init__()

        # Encoder: Downsample path
        self.encoder_blocks = nn.ModuleList()
        for i in range(config.num_layers):
            self.encoder_blocks.append(
                ViTBlock(
                    config.embed_dim, config.num_heads, config.mlp_dim, config.dropout
                )
            )

        # Path embedding
        self.patch_embedding = PatchEmbedding(config.in_channels, config.embed_dim, config.patch_size)

        # Decoder: Upsample path
        self.decoder_blocks = nn.ModuleList()
        for i in range(config.num_layers):
            self.decoder_blocks.append(
                ViTBlock(
                    config.embed_dim, config.num_heads, config.mlp_dim, config.dropout
                )
            )

        # Patch to Image
        self.patch_to_image = PatchToImage(config.embed_dim, config.out_channels, config.patch_size)

        # Time embedding MLP
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

    def forward(self, x, time):
        # Compute time embedding
        time_embedding = self.time_embedding_mlp(time).unsqueeze(1)

        # Encoder
        patches = self.patch_embedding(x)
        skip_connections = []
        for block in self.encoder_blocks:
            patches = block(patches, time_embedding)
            skip_connections.append(patches)

        # Decoder
        for block in self.decoder_blocks:
            patches = block(patches, time_embedding)

        # Reconstruct image from patches
        h, w = x.shape[2], x.shape[3]
        out = self.patch_to_image(patches, h, w)

        return out
