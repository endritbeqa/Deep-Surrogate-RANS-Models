import math
import torch.nn as nn
import torch
from transformers import AutoConfig
from transformers import Swinv2Model


def load_swin_transformer(config_dict: dict) -> nn.Module:
    custom_config = AutoConfig.for_model('swinv2', **config_dict)
    model = Swinv2Model(custom_config)

    return model


class Swin_encoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = load_swin_transformer(config.swin_encoder)
        self.condition_encoder = load_swin_transformer(config.swin_encoder)
        self.non_linearity = nn.ReLU()
        self.fc_bottleneck = nn.ModuleList([nn.Linear(math.prod(config.swin_encoder.skip_connection_shape[i]),
                                              config.latent_dim[i])
                                    for i in range(len(config.swin_encoder.skip_connection_shape))])
        self.bottleneck_layerNorm = nn.ModuleList([nn.LayerNorm(config.latent_dim[i])
                                                  for i in range(len(config.latent_dim))])


    def forward(self, target):
        swin_encoder_output = self.encoder(target, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2]
        hidden_states.append(last_hidden_state)

        latents = []

        for i, skip in enumerate(hidden_states):
            skip = torch.flatten(skip, start_dim=1, end_dim=2)
            latent = self.fc_bottleneck[i](skip)
            latent = self.bottleneck_layerNorm[i](latent)
            latent = self.non_linearity(latent)
            latents.append(latent)

        return latents
