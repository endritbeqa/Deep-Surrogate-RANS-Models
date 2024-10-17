from typing import Tuple, Optional
import torch.nn as nn
import torch
from transformers.models.swinv2.modeling_swinv2 import Swinv2Layer


class Conv_Block(nn.Module):
    def __init__(self, config):
        super(Conv_Block, self).__init__()
        self.upsample = nn.Upsample(size=(config.res + 8, config.res + 8), mode='bilinear')
        self.conv1 = nn.Conv2d(config.input_dim, config.hidden_dim_1, kernel_size=7)
        self.layerNorm1 = nn.LayerNorm([config.hidden_dim_1, config.res + 2, config.res + 2])
        self.conv2 = nn.Conv2d(config.hidden_dim_1, config.hidden_dim_2, kernel_size=3)
        self.layerNorm2 = nn.LayerNorm([config.hidden_dim_2, config.res, config.res])
        self.non_linearity = nn.GELU()
        self.conv3 = nn.Conv2d(in_channels=config.hidden_dim_2,
                               out_channels=config.out_dim,
                               kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.layerNorm1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)
        x = self.layerNorm2(x)
        x = self.non_linearity(x)
        x = self.conv3(x)

        return x


class SwinUpsample(nn.Module):
    #TODO refactor this reshape arg
    def __init__(self, res, in_channels, reshape=True):
        super(SwinUpsample, self).__init__()
        self.reshape = reshape
        self.upsample = nn.Upsample(size=(res, res), mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3)
        self.non_linearity = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)

    def forward(self, x, input_dimensions):
        if self.reshape:
            height, width = input_dimensions
            batch_size, dim, num_channels = x.shape
            x = x.permute(0, 2, 1)
            x = x.view(batch_size, num_channels, height, width)

        x = self.upsample(x)
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)

        return x


class Swinv2DecoderStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, upsample=None, pretrained_window_size=0):
        super().__init__()
        self.config = config
        self.upsample = upsample
        blocks = []
        for i in range(depth):
            block = Swinv2Layer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                pretrained_window_size=pretrained_window_size,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_dimensions: Tuple[int, int],
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
    ):

        B, C, H, W = hidden_states.shape
        hidden_states = hidden_states.view((B, C, H * W))
        hidden_states = hidden_states.permute(0, 2, 1)

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states, input_dimensions)

        return hidden_states


class Swin_VAE_decoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.num_layers = len(config.swin_decoder.depths)
        self.last_upsample = nn.Upsample(size=(config.swin_decoder.image_size, config.swin_decoder.image_size), mode='bilinear')
        self.last_layer = Conv_Block(self.config.conv_block)  # conv layer to go from latent dim to output dim

        layers = []
        for i_layer in range(self.num_layers):
            C, H, W = config.swin_decoder.skip_connection_shape[i_layer]
            stage = Swinv2DecoderStage(
                config=config.swin_decoder,
                dim=int(C),
                input_resolution=(H, W),
                depth=config.swin_decoder.depths[i_layer],
                num_heads=config.swin_decoder.num_heads[i_layer],
                ## TODO this 2*H is ugly, maybe refactor it
                upsample=SwinUpsample(res=(2 * H + 2), in_channels=int(C))
            )
            layers.append(stage)

        self.layers = nn.ModuleList(layers)

    def forward(self, skip_connections):
        hidden_state = skip_connections[0]

        for i in range(len(self.layers)):
            if i != 0:
                hidden_state = torch.cat([hidden_state, skip_connections[i]], dim=1)
            shape = self.config.swin_decoder.skip_connection_shape_pre_cat[i]
            input_dimension = shape[1:3]
            hidden_state = self.layers[i](hidden_state, input_dimension)

        hidden_state = self.last_upsample(hidden_state)
        hidden_state = torch.cat([hidden_state, skip_connections[-1]], dim=1)
        output = self.last_layer(hidden_state)

        return output
