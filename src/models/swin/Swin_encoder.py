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
class Swin_encoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = Conv_Block(config.conv_block.image_size,
                                     config.conv_block.num_channels,
                                     config.conv_block.embed_dim,
                                     config.conv_block.output_dim)
        self.encoder = load_swin_transformer(config.swin_encoder)

    def forward(self, input):
        conv_block_output = self.conv_block(input)
        swin_encoder_output = self.encoder(conv_block_output, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2]
        first_hidden = torch.permute(torch.flatten(conv_block_output, start_dim=2, end_dim=3), dims=(0, 2, 1))
        hidden_states.insert(0, first_hidden)
        hidden_states.append(last_hidden_state)

        return hidden_states
