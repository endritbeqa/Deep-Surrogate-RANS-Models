import torch
import torch.nn as nn


class VAEBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim

        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.fc_condition = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_z = nn.Linear(2*self.latent_dim, self.hidden_dim)

    def forward(self, x, condition):

        condition = self.fc_condition(condition)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z = torch.cat((z, condition), dim=1)
        z = self.fc_z(z)

        return z, mu, logvar


