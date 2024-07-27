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

        self.num_layers = config.num_layers
        self.embedding_dim = config.embedding_dim
        self.output_size = config.output_size
        self.output_channels = config.output_channels
        self.decoder_channels = config.decoder_channels
        self.encoder_channels = config.encoder_channels
        self.input_channels = config.input_channels
        self.scale_factor = config.upsample_ratios
        self.decoder_stages = nn.ModuleList()

        for i in range (self.num_layers):
            self.decoder_stages.append(self.expanding_block(in_channels=self.input_channels[i], out_channels=self.decoder_channels[i+1],scale_factor=self.scale_factor[i]))

        self.decoder_stages.append(nn.Sequential(

                nn.Conv2d(self.decoder_channels[-1], self.output_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1),

        ))


    def expanding_block(self, in_channels, out_channels, scale_factor):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )
        return block

    def forward(self, features):
        features_reshaped = []

        for i in range(len(features)):
            feature = features.hidden_states[-(i+1)].permute(0, 2, 1)
            b, c, h_w = feature.shape
            l = int(h_w ** 0.5)
            assert l ** 2 == h_w
            feature = feature.view(b, c, l, l)
            features_reshaped.append(feature)

        x = features_reshaped[0]
        x = self.decoder_stages[0](x)
        for i in range(1,len(features_reshaped)):
            x = torch.cat((x, features_reshaped[i]), dim=1)
            x = self.decoder_stages[i](x)
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
        y = self.decoder(x)
        return y

