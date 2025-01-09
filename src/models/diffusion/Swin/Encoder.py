import torch.nn as nn
from src.models.diffusion.Swin.layers import Swin_Encoder, Swinv2PatchEmbeddings


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = Swinv2PatchEmbeddings(config)
        self.swin_encoder = Swin_Encoder(config)


    def forward(self, x, t):
        patches, grid_size = self.patch_embedding(x, t)
        skip_connections = self.swin_encoder(patches, t, grid_size)


        return skip_connections



