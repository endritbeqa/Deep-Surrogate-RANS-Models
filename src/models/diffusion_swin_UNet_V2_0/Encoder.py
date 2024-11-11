import torch.nn as nn
from src.models.diffusion_swin_UNet_V2_0.layers import Swin_Encoder, Conv_layer, Swinv2PatchEmbeddings


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.input_res_skip:
            self.conv_layer = Conv_layer(config.conv_input_dim, config.conv_input_dim*2, config.conv_output_dim, (config.image_size, config.image_size))
        self.patch_embedding = Swinv2PatchEmbeddings(config)
        self.swin_encoder = Swin_Encoder(config)


    def forward(self, x, t):
        if self.config.input_res_skip:
            x = self.conv_layer(x, t, reshape=False)

        patches, grid_size = self.patch_embedding(x, t)
        skip_connections = self.swin_encoder(patches, t, grid_size)

        if self.config.input_res_skip:
            x = x.flatten(2)
            x = x.permute(0, 2, 1)
            skip_connections.insert(0, x)

        return skip_connections



