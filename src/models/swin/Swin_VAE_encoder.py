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
        self.fc_condition = nn.ModuleList([nn.Linear(math.prod(config.swin_encoder.skip_connection_shape[i]),
                                                     config.condition_latent_dim)
                                           for i in range(len(config.swin_encoder.skip_connection_shape))])
        self.fc_mu = nn.ModuleList([nn.Linear(math.prod(config.swin_encoder.skip_connection_shape[i]),
                                              config.latent_dim)
                                    for i in range(len(config.swin_encoder.skip_connection_shape))])
        self.fc_logvar = nn.ModuleList([nn.Linear(math.prod(config.swin_encoder.skip_connection_shape[i]),
                                                  config.latent_dim)
                                        for i in range(len(config.swin_encoder.skip_connection_shape))])

    def forward(self, condition, target):
        swin_encoder_output = self.encoder(target, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2]
        hidden_states.append(last_hidden_state)

        condition_output = self.condition_encoder(condition, output_hidden_states=True)
        condition_hidden_states = condition_output.hidden_states
        condition_hidden_states = list(condition_hidden_states)[:-2]
        condition_hidden_states.append(condition_output.last_hidden_state)

        condition_latent = []
        for i, condition in enumerate(condition_hidden_states):
            condition = torch.flatten(condition, start_dim=1, end_dim=2)
            condition_latent.append(self.fc_condition[i](condition))

        z = []
        mu = []
        logvar = []

        for i, skip in enumerate(hidden_states):
            skip = torch.flatten(skip, start_dim=1, end_dim=2)
            mu_i = self.fc_mu[i](skip)
            logvar_i = self.fc_logvar[i](skip)
            std = torch.exp(0.5 * logvar_i)
            eps = torch.randn_like(std)
            z_i = mu_i + eps * std
            z_i = torch.cat([z_i, condition_latent[i]], dim=1)
            z.append(z_i)
            mu.append(mu_i)
            logvar.append(logvar_i)

        return z, mu, logvar
