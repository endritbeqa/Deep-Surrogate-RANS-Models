import math
import torch.nn as nn
import torch
from transformers import AutoConfig
from transformers import Swinv2Model


def load_swin_transformer(config_dict: dict) -> nn.Module:
    custom_config = AutoConfig.for_model('swinv2', **config_dict)
    model = Swinv2Model(custom_config)

    return model


class Swin_VAE_encoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = load_swin_transformer(config.swin_encoder)
        self.condition_encoder = load_swin_transformer(config.swin_encoder)
        self.fc_condition = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_mu = []
        self.fc_logvar = []

        for i in range(len(config.image_sizes)):
            self.fc_mu.append(nn.Linear(config.swin_encoder.image_sizes[i][0] * config.swin_encoder.image_sizes[i][1] *
                                        config.swin_encoder.skip_channels[i], config.latent_dim))
            self.fc_logvar.append(
                nn.Linear(config.swin_encoder.image_sizes[i][0] * config.swin_encoder.image_sizes[i][1] *
                          config.swin_encoder.skip_channels[i], config.latent_dim))

    def forward(self, condition, target):
        Swin_encoder_output = self.encoder(target, output_hidden_states=True)
        last_hidden_state, pooler_output, hidden_states, attentions, reshaped_hidden_states = Swin_encoder_output
        skip_connections = list(hidden_states)[:-2]
        skip_connections.append(last_hidden_state)
        skip_connections.reverse()

        # condition is the freestream velocities and binary mask of the case
        condition = self.condition_encoder(condition, output_hidden_states=False)
        condition = condition.last_hidden_state
        condition = torch.flatten(condition, start_dim=1, end_dim=2)
        condition_latent = self.fc_condition(condition)

        z = []
        mu = []
        logvar = []

        for i, skip in enumerate(skip_connections):
            skip = torch.flatten(skip, start_dim=1, end_dim=2)
            mu_i = self.fc_mu[i](skip)
            logvar_i = self.fc_logvar[i](skip)
            std = torch.exp(0.5 * logvar_i)
            eps = torch.randn_like(std)
            z_i = mu + eps * std
            z_i = torch.cat([z_i,condition_latent], dim=1)
            z.append(z_i)
            mu.append(mu_i)
            logvar.append(logvar_i)


        return z, mu, logvar
