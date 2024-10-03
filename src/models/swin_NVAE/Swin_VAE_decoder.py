from typing import Tuple, Optional
import torch.nn as nn
import torch
from transformers.models.swinv2.modeling_swinv2 import Swinv2Layer




class Conv_Block(nn.Module):
    def __init__(self, res, input_dim, out_dim):
        super(Conv_Block, self).__init__()
        self.upsample = nn.Upsample(size=(res+2, res+2), mode='bilinear')
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3)
        self.layerNorm1 = nn.LayerNorm([input_dim, res, res])
        self.non_linearity = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=input_dim,
                              out_channels=out_dim,
                              kernel_size=1)

    def forward(self, x, tuple):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.layerNorm1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)

        return x


class SwinUpsample(nn.Module):
    def __init__(self, res, in_channels, reshape=True):
        super(SwinUpsample, self).__init__()
        self.reshape = reshape
        self.upsample = nn.Upsample(size=(res,res), mode='bilinear', align_corners=True)
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
        self.dim = dim
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
        self.upsample = upsample

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ):

        B, C, H, W = hidden_states.shape
        hidden_states = hidden_states.view((B, C, H*W))
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

        self.conv_block = Conv_Block(config.conv_block.image_size,
                                     config.conv_block.num_channels,
                                     config.conv_block.embed_dim,
                                     config.conv_block.output_dim)

        self.config = config
        self.num_layers = len(config.swin_decoder.depths)

        layers = []
        for i_layer in range(self.num_layers):
            stage = Swinv2DecoderStage(
                config=config.swin_decoder,
                dim=int(config.swin_decoder.skip_connection_shape[i_layer][2]),
                input_resolution=(config.swin_decoder.skip_connection_shape[i_layer][0], config.swin_decoder.skip_connection_shape[i_layer][1]),
                depth=config.swin_decoder.depths[i_layer],
                num_heads=config.swin_decoder.num_heads[i_layer],
                upsample=SwinUpsample(res=(config.swin_decoder.skip_connection_shape[i_layer + 1][0] + 2),
                                      in_channels=int(config.swin_decoder.skip_connection_shape[i_layer][2]))
            )
            layers.append(stage)

        self.last_layer = Conv_Block(config.swin_decoder.skip_connection_shape[-1][0],config.swin_decoder.skip_connection_shape[-1][2],config.swin_decoder.num_channels)

        layers.append(self.last_layer)
        self.layers = nn.ModuleList(layers)

        










