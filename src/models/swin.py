import ml_collections
import torch.nn as nn
import huggingface_hub
from transformers import AutoConfig, AutoModel


def load_swin_transformer(model_name: str, config_dict: dict ) -> nn.Module:

    custom_config = AutoConfig.for_model('swinv2', **config_dict)
    model = AutoModel.from_config(custom_config)

    return model


class CNNDecoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, output_size, num_channels, activation_fns, kernel_sizes, strides):
        super(CNNDecoder, self).__init__()

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.num_channels = num_channels
        self.activation_fns = activation_fns
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Initialize the decoder layers
        layers = []
        in_channels = embedding_dim
        current_size = 1  # Initial spatial dimension of the embedding

        # Adding intermediate layers
        for i in range(num_layers):
            out_channels = num_channels[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            activation_fn = self._get_activation_function(activation_fns[i])

            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn)

            current_size = (current_size - 1) * stride + kernel_size - 2  # Calculate new spatial dimension
            in_channels = out_channels

        # Final layer to match the output size and channels
        final_kernel_size = output_size - current_size + kernel_sizes[-1] - 2
        layers.append(nn.ConvTranspose2d(in_channels, output_size, final_kernel_size))
        layers.append(nn.Sigmoid())  # Assuming the output image pixel values are normalized between 0 and 1

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


# Example usage:
num_layers = 4
embedding_dim = 128
output_size = 64
num_channels = [128, 64, 32, 16]
activation_fns = ['relu', 'relu', 'relu', 'relu']
kernel_sizes = [4, 4, 4, 4]
strides = [2, 2, 2, 2]

decoder = CNNDecoder(num_layers, embedding_dim, output_size, num_channels, activation_fns, kernel_sizes, strides)
print(decoder)

# Dummy input
embedding = torch.randn(8, embedding_dim)
output_image = decoder(embedding)
print(output_image.shape)  # Should be (8, output_size, output_size)



def create_model() -> nn.Module:


    encoder = load_swin_transformer()