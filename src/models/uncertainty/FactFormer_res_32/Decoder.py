import math

import torch.nn as nn

from src.models.uncertainty.diffusion.FactFormer.layers import (Conv_layer, TimeEmbedding,
                                                                Upsample)

'''
class MLP_decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.latent_dim_1 = config.latent_dim_1
        self.latent_dim_2 = config.latent_dim_2
        self.output_dim = config.output_dim

        self.linear_1 = nn.Linear(self.encoder_dim, self.latent_dim_1)
        self.linear_2 = nn.Linear(self.latent_dim_1, math.prod(self.output_dim))
        self.non_linearity = nn.GELU()
        self.blocks = nn.ModuleList([self.linear_1, self.non_linearity,self.linear_2])


    def forward(self, x):
        B, L, C = x.shape
        x = x.flatten(1)
        for layer in self.blocks:
            x = layer(x)
        C, W, H = self.output_dim
        x = x.reshape(B, C, H, W)

        return x


class CNN_decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.upsample = Upsample(config.input_dim[0], config.upsample_res)
        layers = []
        for i in range(len(config.input_dim)):
            layers.append(Conv_layer(config.input_dim[i], config.kernel_size[i], config.hidden_dim[i], config.output_dim[i]))
        self.conv_layers = nn.ModuleList(layers)


    def forward(self, x, t):
        x = self.upsample(x, reshape=False)
        for layer in self.conv_layers:
            x = layer(x, t)

        return x







'''
class MLP_decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.latent_dim_1 = config.latent_dim_1
        self.latent_dim_2 = config.latent_dim_2
        self.output_dim = config.output_dim

        self.time_embedding = TimeEmbedding(self.encoder_dim)
        self.linear_1 = nn.Linear(self.encoder_dim, self.latent_dim_1)
        self.linear_2 = nn.Linear(self.latent_dim_1, math.prod(self.output_dim))
        self.non_linearity = nn.GELU()
        self.blocks = nn.ModuleList([self.linear_1, self.non_linearity, self.linear_2])

    def forward(self, x, t):
        B, L, C = x.shape
        x = self.time_embedding(x)
        x = x.flatten(1)
        for layer in self.blocks:
            x = layer(x)
        C, W, H = self.output_dim
        x = x.reshape(B, C, H, W)

        return x


class CNN_decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv_1 = Conv_layer(config.input_dim, config.input_dim, config.input_dim, config.input_size)
        self.upsample = Upsample(config.input_dim)
        self.conv_2 = Conv_layer(config.input_dim, config.input_dim // 4, config.output_dim[0],
                                 config.output_dim[1:3])

    def forward(self, x, t):
        x = self.conv_1(x, t, reshape=False)
        #x = self.upsample(x, reshape=False)
        x = self.conv_2(x, t, reshape=False)

        return x

