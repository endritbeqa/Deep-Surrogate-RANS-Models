import torch.nn as nn
from transformers import AutoConfig
from transformers import Swinv2Model


def load_swin_transformer(config_dict: dict) -> nn.Module:
    custom_config = AutoConfig.for_model('swinv2', **config_dict)
    model = Swinv2Model(custom_config)

    return model


class Swin_VAE_encoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoders = load_swin_transformer(config.swin_encoder)

    def forward(self, input):
        B, T, C, H, W = input.shape
        input = input.view(B*T, C, H, W)
        Swin_encoder_output = self.encoder(input, output_hidden_states=True)
        last_hidden_state = Swin_encoder_output.last_hidden_state
        hidden_states = Swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2]
        hidden_states.append(last_hidden_state)
        hidden_states.reverse()
        for i, _ in enumerate(hidden_states):
            b_t, c, h, w = hidden_states[i].shape
            hidden_states[i]= hidden_states[i].view(B, T, c, h, w)

        return hidden_states