import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection for patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x is [B, C, H, W] where B is batch size, C is channels, H and W are height and width
        patches = self.proj(x)  # [B, embed_dim, H/P, W/P]
        patches = patches.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return patches


# Positional Encoding as before
class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, embed_dim))

    def forward(self, x):
        B, P, C = x.shape
        return x + self.pos_embedding[:,:P,:]