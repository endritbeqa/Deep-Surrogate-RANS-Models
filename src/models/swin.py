import torch.nn as nn
from transformers import AutoConfig, AutoModel


def load_swin_transformer(config_dict: dict ) -> nn.Module:

    custom_config = AutoConfig.for_model('swinv2', **config_dict)
    model = AutoModel.from_config(custom_config)

    return model


class CNNDecoder(nn.Module):
    def __init__(self, config):
        super(CNNDecoder, self).__init__()

        self.num_layers = config.num_layers
        self.embedding_dim = config.embedding_dim
        self.output_size = config.output_size
        self.num_channels = config.num_channels
        self.activation_fns = config.activation_fns
        self.kernel_sizes = config.kernel_sizes
        self.strides = config.strides

        # Initialize the decoder layers
        layers = []
        in_channels = self.embedding_dim
        current_size = 1  # Initial spatial dimension of the embedding

        # Adding intermediate layers
        for i in range(self.num_layers):
            out_channels = self.num_channels[i]
            kernel_size = self.kernel_sizes[i]
            stride = self.strides[i]
            activation_fn = self._get_activation_function(self.activation_fns[i])

            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn)

            current_size = (current_size - 1) * stride + kernel_size - 2  # Calculate new spatial dimension
            in_channels = out_channels

        # Final layer to match the output size and channels
        final_kernel_size = self.output_size - current_size + self.kernel_sizes[-1] - 2
        layers.append(nn.ConvTranspose2d(in_channels, self.output_size, final_kernel_size))
        #layers.append(nn.Sigmoid())  # Assuming the output image pixel values are normalized between 0 and 1

        self.decoder = nn.Sequential(*layers)

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

    def forward(self, x):
        # Reshape input to the appropriate shape for ConvTranspose2d
        x = x.view(x.size(0), self.embedding_dim, 1, 1)
        x = self.decoder(x)
        return x





class Swin_CNN(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = load_swin_transformer(config.swin_encoder)
        self.decoder = CNNDecoder(config.CNN_decoder)


    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y