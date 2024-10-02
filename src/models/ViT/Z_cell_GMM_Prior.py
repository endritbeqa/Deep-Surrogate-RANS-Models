import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMVAEBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.num_components = config.num_components

        # Parameters for the GMM (num_components Gaussian mixtures)
        # Each Gaussian component has its own mean and log variance
        self.fc_mu = nn.Linear(self.hidden_dim, self.num_components * self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.num_components * self.latent_dim)

        # Mixing coefficients (logits for the categorical distribution)
        self.fc_mix_logits = nn.Linear(self.hidden_dim, self.num_components)

        self.means = nn.Parameter(torch.randn(self.num_components, self.L) * multiplier)
        self.logvars = nn.Parameter(torch.randn(num_components, self.L))

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

    def reparameterize(self, mu, logvar, mixture_index):
        """Reparameterization trick to sample from the selected Gaussian."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Compute the means and log variances for all components
        mu = self.fc_mu(x)  # Shape: [batch_size, num_components * latent_dim]
        logvar = self.fc_logvar(x)  # Shape: [batch_size, num_components * latent_dim]
        mixture_logits = self.fc_mix_logits(x)  # Shape: [batch_size, num_components]

        # Reshape to [batch_size, num_components, latent_dim]
        mu = mu.view(-1, self.num_components, self.latent_dim)
        logvar = logvar.view(-1, self.num_components, self.latent_dim)

        # Compute the mixing probabilities (softmax over the logits)
        mixture_probs = F.softmax(mixture_logits, dim=-1)  # Shape: [batch_size, num_components]

        # Sample a component from the categorical distribution (Gumbel-softmax or hard sampling)
        mixture_index = torch.multinomial(mixture_probs, 1).squeeze(-1)  # Shape: [batch_size]

        # Gather the corresponding mu and logvar for the selected mixture component
        selected_mu = mu[torch.arange(mu.size(0)), mixture_index]  # Shape: [batch_size, latent_dim]
        selected_logvar = logvar[torch.arange(logvar.size(0)), mixture_index]  # Shape: [batch_size, latent_dim]

        # Reparameterize to get latent vector z
        z = self.reparameterize(selected_mu, selected_logvar, mixture_index)

        return z, selected_mu, selected_logvar, mixture_probs