import math
import torch.nn as nn
import torch




class Z_cell(nn.Module):

    def __init__(self, config, i):
        super().__init__()
        self.i = i
        self.config = config
        if i==0:
            self.parameter = nn.Parameter(torch.randn(config.Z_cells.encoder_shapes[i]))

        self.fc_mu = nn.Linear(2* math.prod(config.Z_cells.encoder_shapes[i]), config.latent_dim[i])
        self.fc_logvar = nn.Linear(2* math.prod(config.Z_cells.encoder_shapes[i]), config.latent_dim[i])
        self.fc_condition = nn.Linear(math.prod(config.Z_cells.encoder_shapes[i]), config.latent_dim[i])
        self.fc_z = nn.Linear(config.Z_cells.latent_dim[i] + config.Z_cells.condition_latent_dim[i], config.Z_cells.encoder_shapes[i])

    def forward(self, encoder_input, condition_input, decoder_input):

        B, H, W, C = encoder_input.shape

        if self.i == 0:
            decoder_input = self.parameter

        #input = torch.cat((encoder_input, decoder_input), dim=1)
        input = decoder_input + decoder_input
        #combine H and W dimension and move C to last index
        input = torch.permute(torch.flatten(input, start_dim=2, end_dim=3), dims=(0,2,1))
        input = torch.flatten(input)

        # combine H and W dimension and move C to last index
        condition_input = torch.permute(torch.flatten(condition_input, start_dim=2, end_dim=3), dims=(0, 2, 1))
        condition_input = torch.flatten(condition_input)
        condition_latent = self.fc_condition(condition_input)

        mu = self.fc_mu(input)
        #mu = self.mu_layerNorm(mu)
        #mu = self.non_linearity(mu)
        logvar = self.fc_logvar(input)
        #logvar = self.logvar_layerNorm(logvar)
        #logvar = self.non_linearity(logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = torch.cat([z, condition_latent], dim=1)

        output = self.fc_z(z)
        # TODO look into first reshaping then permuting then reshaping again
        output = output.view(B, C, H, W)

        return output, mu, logvar








