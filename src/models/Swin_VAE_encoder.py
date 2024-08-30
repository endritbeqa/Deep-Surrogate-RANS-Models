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

    def forward(self, input):
        b, t, h, w, c = input.shape
        input.reshape(b*t, h, w, c)#TODO check channel order
        swin_encoder_output = self.encoder(input, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        #TODO complete the forward


        return