import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=config.in_channels,
                              out_channels=config.embed_dim,
                              kernel_size=config.patch_size,
                              stride=config.patch_size)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = self.proj(x)  # [batch_size, embed_dim, num_patches ** 0.5, num_patches ** 0.5]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == config.embed_dim, "Embedding dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(config)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1 + num_patches, embed_dim]

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return x
