import torch
import torch.nn as nn
from src.models.ViT import encoder, decoder, prior_select


class AutoregressiveImageTransformer(nn.Module):
    def __init__(self, config):
        super(AutoregressiveImageTransformer, self).__init__()
        self.config = config
        self.encoder = encoder.Encoder(config.encoder)
        self.condition_encoder = encoder.Encoder(config.condition_encoder)
        self.z_cell, self.z_cell_config = prior_select.get_Z_Cell(config.prior, config)
        self.decoder = decoder.Decoder(config.decoder)

    def forward(self, condition, targets):

        input = torch.cat([condition, targets], dim=1)

        encoded_patches = self.encoder(input)
        encoded_condition_patches = self.condition_encoder(condition)
        patch_shape = encoded_patches.shape

        encoded_patches = torch.flatten(encoded_patches, start_dim=1, end_dim=-1)
        encoded_condition_patches = torch.flatten(encoded_condition_patches, start_dim=1, end_dim=-1)

        z, mu, logvar = self.z_cell(encoded_patches, encoded_condition_patches)
        z = torch.reshape(z, patch_shape)

        output_image = self.decoder(z)

        return output_image, mu, logvar



    def sample(self, condition):

        encoded_condition_patches = self.condition_encoder(condition)
        patch_shape = encoded_condition_patches.shape


        encoded_condition_patches = torch.flatten(encoded_condition_patches, start_dim=1, end_dim=-1)

        noise = torch.unsqueeze(torch.randn(self.z_cell_config.latent_dim), dim=0)

        condition_latent = self.z_cell.fc_condition(encoded_condition_patches)

        z = torch.cat((noise, condition_latent), dim=1)
        z = self.z_cell.fc_z(z)
        z = torch.reshape(z, patch_shape)


        output = self.decoder(z)

        return output



