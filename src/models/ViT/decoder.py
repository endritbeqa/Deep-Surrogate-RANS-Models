import torch
import torch.nn as nn

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
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x, encoder_output):
        batch_size, num_tokens, embed_dim = x.shape

        q = self.q_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(encoder_output).reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(encoder_output).reshape(batch_size, -1, self.num_heads, self.head_dim)

        q = q.permute(2, 0, 1, 3)  # [num_heads, batch_size, num_tokens, head_dim]
        k = k.permute(2, 0, 1, 3)  # [num_heads, batch_size, encoder_tokens, head_dim]
        v = v.permute(2, 0, 1, 3)  # [num_heads, batch_size, encoder_tokens, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(1, 2, 0, 3).reshape(batch_size, num_tokens, embed_dim)
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


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(config.embed_dim)
        self.cross_attn_norm = nn.LayerNorm(config.embed_dim)
        self.mlp_norm = nn.LayerNorm(config.embed_dim)

        self.self_attn = MultiHeadSelfAttention(config)
        self.cross_attn = CrossAttention(config)
        self.mlp = MLP(config)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, encoder_output):
        x = x + self.dropout(self.self_attn(self.self_attn_norm(x)))
        x = x + self.dropout(self.cross_attn(self.cross_attn_norm(x), encoder_output))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class VisionTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout_rate)

        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.num_layers)
        ])

        self.reconstruction = nn.Linear(config.embed_dim, config.patch_size ** 2 * config.out_channels)

    def forward(self, x, encoder_output):
        batch_size, num_tokens, _ = x.size()

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.decoder_blocks:
            x = block(x, encoder_output)

        # Reconstruct image patches
        x = self.reconstruction(x)
        x = x.view(batch_size, num_tokens, self.config.patch_size, self.config.patch_size, self.config.out_channels)
        return x.permute(0, 4, 1, 2, 3).contiguous()