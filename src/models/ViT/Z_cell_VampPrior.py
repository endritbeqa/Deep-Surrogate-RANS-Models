import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict

class VampPriorVAEBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.num_pseudo_inputs = config.num_pseudo_inputs

        # Mean and log variance for the latent representation
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Pseudo-inputs: These are learned parameters
        self.pseudo_inputs = nn.Parameter(torch.randn(self.num_pseudo_inputs, self.hidden_dim))

        # Encoder to generate q(z|x) for pseudo-inputs
        self.pseudo_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2)  # mu and logvar for each pseudo-input
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Step 1: Get the mean and log variance for the input (x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick for the latent vector
        z = self.reparameterize(mu, logvar)

        # Step 2: VampPrior - compute q(z|pseudo-inputs)
        # Run the pseudo-inputs through the pseudo-encoder to get mean and logvar
        pseudo_out = self.pseudo_encoder(self.pseudo_inputs)
        pseudo_mu, pseudo_logvar = torch.chunk(pseudo_out, 2, dim=-1)  # Split into mu and logvar

        # Compute p(z) as the mixture of posteriors
        # Prior will be the average of q(z|pseudo-inputs)
        log_pz = self.compute_log_pz(z, pseudo_mu, pseudo_logvar)

        return z, mu, logvar, log_pz

    def compute_log_pz(self, z, pseudo_mu, pseudo_logvar):
        # This function computes log p(z) as the mixture of Gaussians from the pseudo-inputs
        batch_size, latent_dim = z.shape
        num_pseudo_inputs = self.num_pseudo_inputs

        # Reshape z to compute distance with all pseudo-inputs
        z = z.unsqueeze(1)  # Shape: (batch_size, 1, latent_dim)
        pseudo_mu = pseudo_mu.unsqueeze(0)  # Shape: (1, num_pseudo_inputs, latent_dim)
        pseudo_logvar = pseudo_logvar.unsqueeze(0)  # Shape: (1, num_pseudo_inputs, latent_dim)

        # Compute the log likelihood for each pseudo-input
        log_pseudo = -0.5 * torch.sum(
            ((z - pseudo_mu) ** 2) / pseudo_logvar.exp() + pseudo_logvar + torch.log(2 * torch.pi),
            dim=-1
        )  # Shape: (batch_size, num_pseudo_inputs)

        # Compute the average likelihood over all pseudo-inputs (i.e., the mixture)
        log_pz = torch.logsumexp(log_pseudo - torch.log(torch.tensor(num_pseudo_inputs, dtype=torch.float)), dim=-1)

        return log_pz
