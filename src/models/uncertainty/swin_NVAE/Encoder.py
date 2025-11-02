import torch.nn as nn

from src.models.uncertainty.modeling_swinV2 import Swinv2PatchEmbeddings
from src.models.uncertainty.swin_NVAE.layers import Swin_Encoder


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        #self.conv_layer = Conv_layer(train_config.input_dim, train_config.conv_output_dim, (train_config.image_size, train_config.image_size))
        self.patch_embedding = Swinv2PatchEmbeddings(config)
        self.swin_encoder = Swin_Encoder(config)


    def forward(self, x):
        #x = self.conv_layer(x, reshape=False)
        patches, grid_size = self.patch_embedding(x)
        skip_connections = self.swin_encoder(patches, grid_size)
        #x = x.flatten(2)
        #x = x.permute(0, 2, 1)
        #skip_connections.insert(0, x)

        return skip_connections



