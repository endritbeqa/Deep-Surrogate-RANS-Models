import torch
import torch.nn as nn
import torchvision.transforms as T


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, ff_hidden_dim, num_layers, num_classes,
                 dropout_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape  # (batch_size, seq_len, channels, height, width)
        x = x.view(B * T, C, H, W)  # Flatten temporal dimension
        x = self.patch_embed(x)  # (B * T, n_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # (B * T, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B * T, 1 + n_patches, embed_dim)
        x = x + self.pos_embed[:, :(1 + self.patch_embed.n_patches)]
        x = self.dropout(x)

        for blk in self.transformer:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]  # (B * T, embed_dim)
        cls_token_final = cls_token_final.view(B, T, -1)  # (B, T, embed_dim)
        cls_token_final = cls_token_final.mean(dim=1)  # (B, embed_dim)

        return self.fc(cls_token_final)


# Hyperparameters
img_size = 224
patch_size = 16
in_channels = 3
embed_dim = 768
num_heads = 12
ff_hidden_dim = 3072
num_layers = 12
num_classes = 1000
dropout_rate = 0.1

# Model
vit_model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout_rate=dropout_rate
)

# Example input
x = torch.randn(8, 10, 3, 224, 224)  # (batch_size, seq_len, channels, height, width)
logits = vit_model(x)
print(logits.shape)  # (batch_size, num_classes)