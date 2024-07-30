import torch.nn as nn
import torch
from transformers import AutoConfig
from transformers import Swinv2Model


def load_swin_transformer(config_dict: dict) -> nn.Module:
    custom_config = AutoConfig.for_model('swinv2', **config_dict)
    model = Swinv2Model(custom_config)

    return model

class CNNDecoder(nn.Module):
    def __init__(self, config):
        super(CNNDecoder, self).__init__()
        self.output_image_size = config.output_image_size
        self.output_image_channels = config.output_image_channels
        self.decoder_channels = config.decoder_channels
        self.encoder_channels = config.encoder_channels
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels
        self.decoder_stages = nn.ModuleList()
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels

        for i in range(len(self.encoder_channels)-1):
            self.decoder_stages.append(self.expanding_block(in_channels=self.input_channels[i], out_channels=self.decoder_channels[i+1]))

        self.decoder_stages.append(nn.Sequential(

                nn.Conv2d(self.input_channels[-1], self.output_channels[-1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.output_channels[-1], self.output_channels[-1], kernel_size=3, padding=1),
        ))


    def expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        return block


    ## reshaped_hidden_states(output of Stage1, output Stage2,... last two elements are used to calculate the final
    # output(push them trough layer norm) right now idk what they capture exactly so just toss them in the trash)
    def forward(self, output: torch.Tensor, reshaped_hidden_states: tuple):
        reshaped_hidden_states = list(reshaped_hidden_states)[:-2]
        reshaped_hidden_states.reverse()

        output = output.permute(0, 2, 1)
        b, c, h_w = output.shape
        l = int(h_w ** 0.5)
        assert l ** 2 == h_w
        output = output.view(b, c, l, l)

        x = self.decoder_stages[0](output)

        for i, reshaped_hidden_states in enumerate(reshaped_hidden_states):
            x = torch.cat((x,reshaped_hidden_states), dim=1)
            x = self.decoder_stages[i+1](x)
        x = self.decoder_stages[-1](x)

        return x

    def _get_activation_function(self, activation_name):
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")


class U_NET_Swin_CNN(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_hidden_states = config.swin_encoder.output_hidden_states
        self.encoder = load_swin_transformer(config.swin_encoder)
        self.decoder = CNNDecoder(config.CNN_decoder)

    def forward(self, x):
        x = self.encoder(x, output_hidden_states = self.output_hidden_states)
        output, hidden_states = x.last_hidden_state, x.reshaped_hidden_states
        y = self.decoder(output, hidden_states)
        return y

