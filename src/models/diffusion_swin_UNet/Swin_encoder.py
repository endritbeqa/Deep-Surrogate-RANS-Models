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
        self.config = config.swin_encoder
        self.encoder = load_swin_transformer(config.swin_encoder)

        #self.condition_config = config.swin_encoder.copy_and_resolve_references()
        #self.condition_config.num_channels = 3
        #self.condition_encoder = load_swin_transformer(self.condition_config)



    def forward(self,target):
        B, _, _, _ = target.shape

        swin_encoder_output = self.encoder(target, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2] # removes the redundant info from the swinV2 transformer output
        hidden_states.append(last_hidden_state)

        for i, hidden_state in enumerate(hidden_states):
            hidden_states[i] = hidden_states[i].permute(dims=(0, 2, 1))
            hidden_states[i] = hidden_states[i].view(B, *self.config.skip_connection_shape[i])

        return hidden_states
