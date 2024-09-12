import torch
import torch.nn as nn
from ml_collections import ConfigDict

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, config, i):
        super(MaskedMultiHeadSelfAttention, self).__init__()
        self.context_size = config.context_size
        self.embed_dim = config.embed_dim[i]
        self.num_heads = config.num_heads[i]
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
        assert seq_length == self.context_size, "Input context size mismatch"
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
    def __init__(self, config, i):
        super(AttentionAddNorm, self).__init__()
        self.attention = MaskedMultiHeadSelfAttention(config, i)
        self.norm = nn.LayerNorm(config.embed_dim[i])
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask)
        x = x + self.dropout(attn_output)
        output = self.norm(x)
        return output

class Decoder(nn.Module):
    def __init__(self, config, i):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU
        self.FC_projection = nn.Linear(config.skip_connection_dim[i], config.embed_dim[i])
        self.layers = nn.ModuleList([AttentionAddNorm(config, i) for _ in range(config.depths[i])])
        self.FC_expansion = nn.Linear(config.embed_dim[i], config.skip_connection_dim[i])

    def forward(self, x, attention_mask=None):
        x = self.FC_projection(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.FC_expansion(x)
        x = self.relu(x)
        return x
