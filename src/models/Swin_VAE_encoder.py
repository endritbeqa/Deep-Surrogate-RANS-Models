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
        #self.condition_encoder = load_swin_transformer(config.swin_encoder)
        #self.fc_condition = nn.Linear(config.hidden_dim, config.latent_dim)
        #self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        #self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)


    def forward(self, condition, target):
        Swin_encoder_output = self.encoder(target, output_hidden_states = True)
        #condition_latent = self.condition_encoder(condition, output_hidden_states = False)
        #condition_latent = condition_latent.last_hidden_state


        #Swin_encoder_last_hidden_state_flattened = torch.flatten(Swin_encoder_output.last_hidden_state, start_dim=1, end_dim=2)
        #condition_latent_flattened = torch.flatten(condition_latent, start_dim=1, end_dim=2)

        #condition_latent_flattened = self.fc_condition(condition_latent_flattened)
        #mu = self.fc_mu(Swin_encoder_last_hidden_state_flattened)
        #logvar = self.fc_logvar(Swin_encoder_last_hidden_state_flattened)

        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std)
        #z = mu + eps * std
        #z = torch.cat([z,condition_latent_flattened], dim=1)

        return Swin_encoder_output,Swin_encoder_output.last_hidden_state.shape
        #return Swin_encoder_output, z, mu, logvar, Swin_encoder_output.last_hidden_state.shape

