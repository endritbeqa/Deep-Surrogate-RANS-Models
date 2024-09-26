import math
import torch.nn as nn
import torch
from transformers import AutoConfig
from transformers import Swinv2Model


def load_swin_transformer(config_dict: dict) -> nn.Module:
    custom_config = AutoConfig.for_model('swinv2', **config_dict)
    model = Swinv2Model(custom_config)

    return model


class Conv_Block(nn.Module):
    def __init__(self, res, input_dim, hidden_layer_dim, out_dim):
        super(Conv_Block, self).__init__()
        self.upsample = nn.Upsample(size=(res+2, res+2), mode='bilinear')
        self.conv1 = nn.Conv2d(input_dim, hidden_layer_dim, kernel_size=3)
        self.layerNorm1 = nn.LayerNorm([hidden_layer_dim, res, res])
        self.non_linearity = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=hidden_layer_dim,
                              out_channels=out_dim,
                              kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.layerNorm1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)

        return x

class Swin_VAE_encoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = Conv_Block(config.conv_block.image_size,
                                     config.conv_block.num_channels,
                                     config.conv_block.embed_dim,
                                     config.conv_block.output_dim)
        self.conv_block_condition = Conv_Block(config.conv_block.image_size,
                                     config.conv_block.num_channels_condition,
                                     config.conv_block.embed_dim,
                                     config.conv_block.output_dim)
        self.encoder = load_swin_transformer(config.swin_encoder)
        self.condition_encoder = load_swin_transformer(config.swin_encoder)
        self.non_linearity = nn.ReLU()
        self.fc_condition = nn.ModuleList([nn.Linear(math.prod(config.swin_encoder.skip_connection_shape[i]),
                                                     config.condition_latent_dim[i])
                                           for i in range(len(config.swin_encoder.skip_connection_shape))])
        self.fc_mu = nn.ModuleList([nn.Linear(math.prod(config.swin_encoder.skip_connection_shape[i]),
                                              config.latent_dim[i])
                                    for i in range(len(config.swin_encoder.skip_connection_shape))])
        self.fc_logvar = nn.ModuleList([nn.Linear(math.prod(config.swin_encoder.skip_connection_shape[i]),
                                                  config.latent_dim[i])
                                        for i in range(len(config.swin_encoder.skip_connection_shape))])
        self.condition_layerNorm = nn.ModuleList([nn.LayerNorm(config.condition_latent_dim[i])
                                                  for i in range(len(config.condition_latent_dim))])
        self.mu_layerNorm = nn.ModuleList([nn.LayerNorm(config.latent_dim[i])
                                                  for i in range(len(config.latent_dim))])
        self.logvar_layerNorm = nn.ModuleList([nn.LayerNorm(config.latent_dim[i])
                                                  for i in range(len(config.latent_dim))])


    def forward(self, condition, target):
        target = torch.cat([condition, target], dim=1)
        conv_block_output = self.conv_block(target)
        condition_conv_block_output = self.conv_block_condition(condition)
        swin_encoder_output = self.encoder(conv_block_output, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2]
        first_hidden = torch.permute(torch.flatten(conv_block_output, start_dim=2, end_dim=3), dims=(0,2,1))
        hidden_states.insert(0, first_hidden)
        hidden_states.append(last_hidden_state)

        condition_output = self.condition_encoder(condition_conv_block_output, output_hidden_states=True)
        condition_hidden_states = condition_output.hidden_states
        condition_hidden_states = list(condition_hidden_states)[:-2]
        first_condition = torch.permute(torch.flatten(condition_conv_block_output, start_dim=2, end_dim=3), dims=(0,2,1))
        condition_hidden_states.insert(0, first_condition)
        condition_hidden_states.append(condition_output.last_hidden_state)

        condition_latent = []
        for i, condition in enumerate(condition_hidden_states):
            condition = torch.flatten(condition, start_dim=1, end_dim=2)
            condition = self.fc_condition[i](condition)
            condition = self.condition_layerNorm[i](condition)
            condition = self.non_linearity(condition)
            condition_latent.append(condition)

        z = []
        mu = []
        logvar = []

        for i, skip in enumerate(hidden_states):
            skip = torch.flatten(skip, start_dim=1, end_dim=2)
            mu_i = self.fc_mu[i](skip)
            mu_i = self.mu_layerNorm[i](mu_i)
            mu_i = self.non_linearity(mu_i)
            logvar_i = self.fc_logvar[i](skip)
            logvar_i = self.logvar_layerNorm[i](logvar_i)
            logvar_i = self.non_linearity(logvar_i)
            std = torch.exp(0.5 * logvar_i)
            eps = torch.randn_like(std)
            z_i = mu_i + eps * std
            z_i = torch.cat([z_i, condition_latent[i]], dim=1)
            z.append(z_i)
            mu.append(mu_i)
            logvar.append(logvar_i)

        return z, mu, logvar
