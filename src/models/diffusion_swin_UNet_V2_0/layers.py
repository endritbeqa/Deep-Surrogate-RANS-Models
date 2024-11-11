import collections
import math
from typing import Tuple, Optional, Union, List

import torch
import torch.nn as nn
from src.models.Time_embedding import TimeEmbedding
from src.models.modeling_swinV2 import Swinv2Layer, Swinv2PatchMerging


class Swinv2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(num_samples, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(num_samples, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.time_embedding = TimeEmbedding(100, num_channels)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor], t) -> Tuple[torch.Tensor, Tuple[int]]:
        pixel_values = self.time_embedding(pixel_values, t)
        _, num_channels, height, width = pixel_values.shape
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


class Upsample(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=dim//(2*4), num_channels=dim//2)
        #self.norm1 = nn.LayerNorm([dim//2, input_resolution[0], input_resolution[1]])
        self.non_linearity = nn.GELU()

    def forward(self, x, place_holder):
        b, l, c = x.shape
        h = w = int(math.sqrt(l))
        x = x.permute(0,2,1)
        x = x.view(b,c,h,w)

        x = self.upsample(x)
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = x.flatten(2)
        x = x.permute(0,2,1)

        return x


class Conv_layer(nn.Module):
    def __init__(self, input_channels, hidden_dim,output_channels, output_size):
        super().__init__()
        self.time_embedding = TimeEmbedding(100, input_channels)
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)
        self.skip_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        #self.norm1 = nn.LayerNorm([hidden_dim, output_size[0], output_size[1]])
        #self.norm2 = nn.LayerNorm([hidden_dim, output_size[0], output_size[1]])
        self.non_linearity = nn.GELU()
        self.norm1 = nn.GroupNorm(num_groups=hidden_dim//4, num_channels=hidden_dim)
        self.norm2 = nn.GroupNorm(num_groups=hidden_dim//4, num_channels=hidden_dim)

    def forward(self, x, t, reshape=True):
        if reshape:
            b, l, c = x.shape
            h = w = int(math.sqrt(l))
            x = x.permute(0, 2, 1)
            x = x.view(b, c, h, w)

        x = self.upsample(x)
        x_initial = x
        x = self.time_embedding(x, t)
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.non_linearity(x)
        x = self.norm2(x)
        x = self.conv3(x)

        x_initial = self.skip_conv(x_initial)
        x = x + x_initial

        return x


class Swinv2Stage(nn.Module):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, downsample, isDownsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.time_embedding = TimeEmbedding(100, dim)
        blocks = []
        for i in range(depth):
            block = Swinv2Layer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if downsample is not None:
            if isDownsample:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
            else:
                self.downsample = downsample(input_resolution, dim=dim * 4, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None



    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep,
        input_dimensions: Tuple[int, int],
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions

        if isinstance(self.downsample, Upsample):
            hidden_states_before_downsampling = hidden_states
            height_upsampled, width_upsampled = height * 2, width * 2
            output_dimensions = (height, width, height_upsampled, width_upsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
            input_dimensions = (height_upsampled, width_upsampled)

        hidden_states = self.time_embedding(hidden_states, timestep)

        for i, layer_module in enumerate(self.blocks):
            layer_outputs = layer_module(hidden_states, input_dimensions)
            hidden_states = layer_outputs[0]

        if isinstance(self.downsample, Swinv2PatchMerging):
            hidden_states_before_downsampling = hidden_states
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)

        if self.downsample is None:
            hidden_states_before_downsampling = hidden_states
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        return stage_outputs



class Swin_Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid_size = (int(config.image_size/config.patch_size),int(config.image_size/config.patch_size))
        self.num_layers = len(config.depths)
        layers = []
        for i_layer in range(self.num_layers):
            stage = Swinv2Stage(
                config=config,
                dim=int(config.embed_dim * 2 ** i_layer),
                input_resolution=(self.grid_size[0] // (2 ** i_layer), self.grid_size[1] // (2 ** i_layer)),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                downsample=Swinv2PatchMerging, #if (i_layer < self.num_layers - 1) else None
                isDownsample=True
            )
            layers.append(stage)
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, timestep, input_dimensions: Tuple[int, int]) -> List[torch.Tensor]:

        batch_size, _, hidden_size = hidden_states.shape
        all_hidden_states = [hidden_states]# these are after patch embedding

        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, timestep, input_dimensions)
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

        return all_hidden_states



class Swin_Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = len(config.depths)
        self.grid_size = (int(config.image_size/config.patch_size),int(config.image_size/config.patch_size))
        self.grid_size = (self.grid_size[0] // (2 ** (self.num_layers-1)),self.grid_size[0] // (2 ** (self.num_layers-1)))
        layers = []
        for i_layer in range(self.num_layers):
            stage = Swinv2Stage(
                config=config,
                dim=int(config.embed_dim * 2 ** (self.num_layers - i_layer - 1)),
                input_resolution=(self.grid_size[0] * (2 ** (i_layer+1)), self.grid_size[1] * (2 ** (i_layer+1))),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                downsample=Upsample,#if (i_layer < self.num_layers - 1) else None,
                isDownsample= False
            )
            layers.append(stage)
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, skip_connections, timestep, input_dimensions: Tuple[int, int]):

        batch_size, _, hidden_size = hidden_states.shape

        for i in range(len(self.layers)+1):
            hidden_states = torch.cat([hidden_states, skip_connections[i]], dim=2) #TODO check dim
            if i == len(self.layers):
                continue
            layer_outputs = self.layers[i](hidden_states, timestep, input_dimensions)
            hidden_states = layer_outputs[0]
            input_dimensions = (input_dimensions[0]*2, input_dimensions[1]*2)

        return hidden_states
