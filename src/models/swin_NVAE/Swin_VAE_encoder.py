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
        self.config = config.swin_encoder
        self.condition_config = config.swin_encoder.copy_and_resolve_references()
        self.condition_config.num_channels = 3
        self.encoder = load_swin_transformer(config.swin_encoder)
        self.condition_encoder = load_swin_transformer(self.condition_config)



    def forward(self, condition, target):
        B, _, _, _ = target.shape

        target = torch.cat([condition, target], dim=1)

        swin_encoder_output = self.encoder(target, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2]
        hidden_states.append(last_hidden_state)

        condition_output = self.condition_encoder(condition, output_hidden_states=True)
        condition_hidden_states = condition_output.hidden_states
        condition_hidden_states = list(condition_hidden_states)[:-2]
        condition_hidden_states.append(condition_output.last_hidden_state)

        for i, hidden_state in enumerate(hidden_states):
            hidden_states[i] = hidden_states[i].permute(dims=(0, 2, 1))
            hidden_states[i] = hidden_states[i].view(B,*self.config.skip_connection_shape[i])

        for i, condition_hidden_state in enumerate(condition_hidden_states):
            condition_hidden_states[i] = condition_hidden_states[i].permute(dims=(0, 2, 1))
            condition_hidden_states[i] = condition_hidden_states[i].view(B,*self.config.skip_connection_shape[i])


        return hidden_states, condition_hidden_states
