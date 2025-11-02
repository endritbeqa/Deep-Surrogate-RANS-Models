import einops
import torch.nn as nn

from src.models.steady_state.FactFormer.Decoder import MLP_decoder, CNN_decoder
from src.models.steady_state.FactFormer.Encoder import FactorizedTransformer

class FactFormer(nn.Module):
    def __init__(self,
                 config
                 ):
        super().__init__()
        self.config = config
        self.device = config.device

        self.in_dim = config.in_dim
        self.out_dim = config.out_dim

        self.dim = config.dim
        self.depth = config.depth
        self.dim_head = config.dim_head
        #self.reducer = config.reducer
        self.resolution = config.resolution

        self.heads = config.heads

        self.patch_size = config.patch_size
        self.pos_in_dim = config.pos_in_dim
        self.pos_out_dim = config.pos_out_dim
        self.positional_embedding = config.positional_embedding
        self.kernel_multiplier = config.kernel_multiplier

        self.to_in = nn.Linear(self.in_dim, self.dim, bias=True)

        self.encoder = FactorizedTransformer(self.patch_size, self.in_dim, self.dim, self.dim_head, self.heads, self.dim, self.depth,self.device,
                                             kernel_multiplier=self.kernel_multiplier)

        if config.decoder == 'MLP':
            self.decoder = MLP_decoder(config.MLP_decoder)
        elif config.decoder == "CNN":
            self.decoder = CNN_decoder(config.CNN_decoder)

    def forward(self, x):

        x = self.encoder(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        x = self.decoder(x)

        return x

