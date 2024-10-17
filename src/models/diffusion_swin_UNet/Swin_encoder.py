import torch.nn as nn
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




class Swin_VAE_encoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config.swin_encoder
        self.encoder = load_swin_transformer(config.swin_encoder)
        self.conv_block = Conv_Block(config.encoder_conv_block.image_size,
                                     config.encoder_conv_block.num_channels,
                                     config.encoder_conv_block.embed_dim,
                                     config.encoder_conv_block.output_dim)


    def forward(self,target):
        B, _, _, _ = target.shape

        conv_block_output = self.conv_block(target)
        swin_encoder_output = self.encoder(conv_block_output, output_hidden_states=True)
        last_hidden_state = swin_encoder_output.last_hidden_state
        hidden_states = swin_encoder_output.hidden_states

        hidden_states = list(hidden_states)[:-2] # removes the redundant info from the swinV2 transformer output
        hidden_states.append(last_hidden_state)
        hidden_states.insert(0, conv_block_output)

        for i, hidden_state in enumerate(hidden_states):
            if i==0:
                continue
            hidden_states[i] = hidden_states[i].permute(dims=(0, 2, 1))
            hidden_states[i] = hidden_states[i].view(B, *self.config.skip_connection_shape[i])

        return hidden_states
