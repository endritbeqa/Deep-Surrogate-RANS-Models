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
        self.fc_mu = nn.ModuleList([nn.Linear(config.swin_encoder.image_sizes[i][0] *
                                               config.swin_encoder.image_sizes[i][1] *
                                               config.swin_encoder.skip_channels[i], config.latent_dim) for i in
                                    range(len(config.swin_encoder.image_sizes))])
        self.fc_logvar = nn.ModuleList([nn.Linear(config.swin_encoder.image_sizes[i][0] *
                                                   config.swin_encoder.image_sizes[i][1] *
                                                   config.swin_encoder.skip_channels[i], config.latent_dim) for i in
                                        range(len(config.swin_encoder.image_sizes))])

    def forward(self, condition, target):
        Swin_encoder_output = self.encoder(target, output_hidden_states=True)
        last_hidden_state = Swin_encoder_output.last_hidden_state
        hidden_states = Swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2]
        hidden_states.append(last_hidden_state)
        hidden_states.reverse()


        # condition is the freestream velocities and binary mask of the case
        condition = self.condition_encoder(condition, output_hidden_states=False)
        condition = condition.last_hidden_state
        condition = torch.flatten(condition, start_dim=1, end_dim=2)
        condition_latent = self.fc_condition(condition)

        z = []
        mu = []
        logvar = []

        for i, skip in enumerate(hidden_states):
            skip = torch.flatten(skip, start_dim=1, end_dim=2)
            mu_i = self.fc_mu[-(i+1)](skip)
            logvar_i = self.fc_logvar[-(i+1)](skip)
            std = torch.exp(0.5 * logvar_i)
            eps = torch.randn_like(std)
            z_i = mu_i + eps * std
            z_i = torch.cat([z_i, condition_latent], dim=1)
            z.append(z_i)
            mu.append(mu_i)
            logvar.append(logvar_i)

        return z, mu, logvar
