import math
from typing import Tuple, Optional, Union
import torch.nn as nn
import torch
from transformers.models.swinv2.modeling_swinv2 import Swinv2Layer, Swinv2EncoderOutput


class SwinV2Final_DecoderBlock(nn.Module):
    def __init__(self,res, in_channels, out_channels=3, kernel_size=1, padding=0):
        super(SwinV2Final_DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(size=(res,res), mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3)
        self.non_linearity = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x):
        batch_size, dim, num_channels = x.shape
        height = int(math.sqrt(dim))
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, num_channels, height, height)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)

        return x

class SwinUpsample(nn.Module):
    def __init__(self, res,in_channels):
        super(SwinUpsample, self).__init__()
        self.upsample = nn.Upsample(size=(res,res), mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3)
        self.non_linearity = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)

    def forward(self, x, input_dimensions):
        height, width = input_dimensions
        batch_size, dim, num_channels = x.shape
        x = x.permute(0,2,1)
        x = x.view(batch_size, num_channels ,height, width)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)
        batch_size, num_channels, height , width = x.shape
        x = x.view(batch_size, num_channels, height * width)
        x = x.permute(0,2,1)
        return x


class Swinv2DecoderStage(nn.Module):
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, upsample, pretrained_window_size=0
    ):
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

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample
        else:
            self.upsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_upsampling = hidden_states
        if self.upsample is not None:
            height_upsampled, width_upsampled = (height) * 2, (width) * 2
            output_dimensions = (height, width, height_upsampled, width_upsampled)
            hidden_states = self.upsample(hidden_states_before_upsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_upsampling, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs




class Swinv2Decoder(nn.Module):
    def __init__(self, config, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        self.final_layer = SwinV2Final_DecoderBlock(config.image_size+2,config.skip_connection_shape[-1][2])

        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        layers = []
        for i_layer in range(self.num_layers):
            stage = Swinv2DecoderStage(
                config=config,
                dim=int(config.skip_connection_shape[i_layer][2]),
                input_resolution=(config.skip_connection_shape[i_layer][0], config.skip_connection_shape[i_layer][1]),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                drop_path=dpr[sum(config.depths[:i_layer]): sum(config.depths[: i_layer + 1])],
                upsample=SwinUpsample(res=(config.skip_connection_shape[i_layer+1][0] + 2),
                                      in_channels=int(config.skip_connection_shape[i_layer][2]))
                                      if (i_layer < self.num_layers - 1) else None
                #pretrained_window_size=pretrained_window_sizes[i_layer],
            )
            layers.append(stage)

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        skip_connections: Optional[torch.Tensor],
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if i !=0:
                hidden_states = torch.cat((hidden_states, skip_connections[i-1]), dim=2)
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_reshaped_hidden_states]
                if v is not None
            )

        output = self.final_layer(hidden_states)

        return output




class Swin_decoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.decoder = Swinv2Decoder(config.swin_decoder)

    def forward(self, skip_connections):
        y = self.decoder(skip_connections[0], skip_connections[1:], tuple(self.config.swin_decoder.skip_connection_shape[0][0:2]))
        return y

