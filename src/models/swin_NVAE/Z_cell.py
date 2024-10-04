import torch
import torch.nn as nn


class VAEBottleneck(nn.Module):
    def __init__(self, config, i_layer):
        super().__init__()
        self.i_layer = i_layer

        if i_layer == 0:
            self.H = nn.Parameter(torch.randn(config.previous_dim[i_layer]))

        self.latent_dim = config.latent_dim[i_layer]
        self.hidden_dim = config.hidden_dim[i_layer]
        self.prev_dim = config.previous_dim[i_layer]

        self.fc_mu = nn.Linear(self.hidden_dim + self.prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim + self.prev_dim, self.latent_dim)

        self.fc_condition = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_prev = nn.Linear(self.prev_dim, self.latent_dim)

        self.fc_z = nn.Linear(3*self.latent_dim, self.hidden_dim)

    def forward(self, encoder_input, previous, condition):
        B, _ = encoder_input.shape
        if self.i_layer == 0:
            previous = self.H.repeat(B,1)

        x = torch.cat((encoder_input, previous), dim=-1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        condition = self.fc_condition(condition)
        previous = self.fc_prev(previous)

        z = torch.cat((z, previous, condition), dim=1)
        z = self.fc_z(z)

        return z, mu, logvar


