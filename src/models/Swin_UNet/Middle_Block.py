import torch.nn as nn
from src.models.Swin_UNet.layers import Swinv2Stage


class Middle_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_res = config.input_res
        self.stage = Swinv2Stage(
            config=config,
            dim=int(config.dim),
            input_resolution=config.input_res,
            depth=config.depth,
            num_heads=config.num_heads,
            downsample=None,  # if (i_layer < self.num_layers - 1) else None,
            isDownsample=False
        )

    def forward(self, x):
        return self.stage(x,  self.input_res)[0]