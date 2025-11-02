import math

import torch
import torch.nn as nn
from src.models.steady_state.Swin_UNet.layers import Swin_Decoder, Conv_layer

class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.input_res_skip:
            self.second_last_layer = Conv_layer(config.embed_dim * 2, config.embed_dim, config.embed_dim//2, (config.image_size, config.image_size))
            self.last_layer = Conv_layer(config.embed_dim//2+config.conv_skip_dim, config.embed_dim//2, config.num_output_channels,(config.image_size, config.image_size))
        else:
            self.last_layer = Conv_layer(config.embed_dim*2, config.embed_dim, config.num_output_channels,(config.image_size, config.image_size))
        self.decoder = Swin_Decoder(config)


    def forward(self, x, skip_connections):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = self.decoder(x, skip_connections, (H, W))
        if self.config.input_res_skip:
            x = self.second_last_layer(x)
            last_skip = skip_connections[-1]
            last_skip = last_skip.permute(0, 2, 1)
            b, c, l = last_skip.shape
            h = w = int(math.sqrt(l))
            last_skip = last_skip.view(b, c, h, w)
            x = torch.cat([x, last_skip], dim=1)
            x = self.last_layer(x, reshape=False)
        else:
            x = self.last_layer(x, reshape=True)
        return x