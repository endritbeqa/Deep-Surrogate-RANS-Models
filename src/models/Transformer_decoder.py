import torch
import torch.nn as nn
from ml_collections import ConfigDict

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MaskedMultiHeadSelfAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.causal_mask = config.causal_mask
        self.use_bias = config.use_bias
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            bias=self.use_bias,
            batch_first=True,
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Input embedding dimension mismatch"
        if self.causal_mask:
            causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device) * float('-inf'), diagonal=1)
        else:
            causal_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            combined_mask = causal_mask + attention_mask if causal_mask is not None else attention_mask
        else:
            combined_mask = causal_mask
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=combined_mask)
        return attn_output

class AttentionAddNorm(nn.Module):
    def __init__(self, config):
        super(AttentionAddNorm, self).__init__()
        self.attention = MaskedMultiHeadSelfAttention(config)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask)
        x = x + self.dropout(attn_output)
        output = self.norm(x)
        return output

class Decoder(nn.Module):
    def __init__(self, config, num_layers):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([AttentionAddNorm(config) for _ in range(self.num_layers)])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x
