import torch
import torch.nn as nn
from src.models.ViT.layers import PositionalEncoding, PatchEmbedding


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config
        self.pos_encoding = PositionalEncoding((config.img_size // config.patch_size) ** 2, config.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.embed_dim, nhead=config.num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        self.to_patch = nn.Linear(config.embed_dim, config.out_channels * config.patch_size * config.patch_size)


        self.img_size = config.img_size
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.n_patches = (config.img_size // config.patch_size) ** 2

    def forward(self, encoded_patches, x=None):
        # Optionally use autoregressive input `x`
        if x is None:
            x = torch.zeros_like(encoded_patches)

        output_patches = []
        for i in range(1, self.n_patches + 1):
            # Only use the patches seen so far for autoregressive generation
            input_seq = encoded_patches[:, :i, :]

            # Apply the positional encoding
            input_seq = self.pos_encoding(input_seq)

            # Causal mask to prevent future patches from leaking into the past
            tgt_mask = self._generate_square_subsequent_mask(i).to(input_seq.device)

            # Transformer decoder
            decoded_patch = self.decoder(input_seq, encoded_patches, tgt_mask=tgt_mask)

            # Predict the next patch
            next_patch = self.to_patch(decoded_patch[:, -1, :])
            output_patches.append(next_patch)

        # Stack the patches together to form the output image
        output = torch.stack(output_patches, dim=1)
        output = output.view(-1, self.img_size, self.img_size, self.out_channels).permute(0, 3, 1, 2)  # [B, C, H, W]
        return output

    def _generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))