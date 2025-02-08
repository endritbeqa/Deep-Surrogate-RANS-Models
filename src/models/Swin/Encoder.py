import torch.nn as nn
from src.models.Swin.layers import Swin_Encoder, Swinv2PatchEmbeddings


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = Swinv2PatchEmbeddings(config)
        self.swin_encoder = Swin_Encoder(config)


    def forward(self, x):
        patches, grid_size = self.patch_embedding(x)
        skip_connections = self.swin_encoder(patches, grid_size)

        return skip_connections



