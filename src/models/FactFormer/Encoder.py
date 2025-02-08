import einops
import torch
import torch.nn as nn

from src.models.FactFormer.layers import  PatchEmbedding
from src.models.FactFormer.libs.factorization_module import FABlock2D
from src.models.FactFormer.libs.positional_encoding_module import GaussianFourierFeatureTransform


class FactorizedTransformer(nn.Module):
    def __init__(self,
                 patch_size,
                 in_dim,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 device,
                 **kwargs):
        super().__init__()


        self.layers = nn.ModuleList([])
        self.device = device
        self.patch_embed = PatchEmbedding(in_dim, dim, patch_size)
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(2, dim // 2, 1),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True,
                                   **kwargs))
            self.layers.append(layer)

    def forward(self, u):
        #u = self.to_in(u)
        u = self.patch_embed(u)
        u = einops.rearrange(u, 'b c h w -> b h w c')
        b, nx, ny, c = u.shape
        pos_x = torch.linspace(0, 1, nx).float().to(self.device).unsqueeze(-1)
        pos_y = torch.linspace(0, 1, ny).float().to(self.device).unsqueeze(-1)
        pos_lst = [pos_x, pos_y]
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)]), dim=-1)
        for pos_enc, attn_layer in self.layers:
            pos_encoding = pos_enc(pos).view(1, nx, ny, -1)
            u += pos_encoding
            attn = attn_layer(u, pos_lst)
            b, l, c = attn.shape
            h = w = int(l**0.5)
            attn = attn.reshape(b, h, w, c)
            u = attn + u
        return u
