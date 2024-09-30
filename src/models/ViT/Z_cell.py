import torch
import torch.nn as nn


class VAEBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim

        # Layers for mean and log-variance
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        # Get the mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)  # Standard deviation from log-variance
        eps = torch.randn_like(std)  # Random noise for reparameterization

        # Latent vector
        z = mu + eps * std

        return z, mu, logvar


