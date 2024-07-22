import torch
import torch.nn as nn
from src.models.swinV2_CNN import swin
from src.models.sequence_modeling import config


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.patch_embed = swin.load_swin_transformer(config.swin_encoder)
        self.decoder = swin.CNNDecoder(config.CNN_decoder)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.embed_dim))
        self.dropout = nn.Dropout(config.emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mlp_head[1].weight, std=0.02)
        if self.mlp_head[1].bias is not None:
            nn.init.zeros_(self.mlp_head[1].bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = x[:, 0]

        x = self.mlp_head(x)
        return x


if __name__ == "__main__":
    config = config.get_config()
    model = VisionTransformer(config)
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)
    output = model(dummy_input)
    print(output.shape)