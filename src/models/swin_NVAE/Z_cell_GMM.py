import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent, MultivariateNormal


class GMM_VAEBottleneck(nn.Module):
    def __init__(self, config, i_layer):
        super().__init__()
        self.i_layer = i_layer

        self.latent_dim = config.latent_dim[i_layer]
        self.num_components = config.num_components[i_layer]
        self.means = nn.Parameter(torch.randn(config.num_components[i_layer], config.latent_dim[i_layer]))
        self.logvars = nn.Parameter(torch.randn(config.num_components[i_layer], config.latent_dim[i_layer]))
        self.mixture_weights = nn.Parameter(torch.zeros(config.num_components[i_layer], 1, 1))


        if i_layer == 0:
            self.H = nn.Parameter(torch.randn(config.previous_dim[i_layer]))

        self.latent_dim = config.latent_dim[i_layer]
        self.hidden_dim = config.hidden_dim[i_layer]
        self.prev_dim = config.previous_dim[i_layer]

        self.fc_mu = nn.Linear(self.hidden_dim + self.prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim + self.prev_dim, self.latent_dim)

        self.fc_condition = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_prev = nn.Linear(self.prev_dim, self.latent_dim)

        self.fc_z = nn.Linear(3 * self.latent_dim, self.hidden_dim)

    def gmm_loss(self, z):
        batch_size = z.size(0)
        means = self.means.unsqueeze(0).expand(batch_size, self.num_components,-1)
        logvars = self.logvars.unsqueeze(0).expand(batch_size, self.num_components,-1)

        multi_normal_dist = MultivariateNormal(means, torch.diag_embed(torch.exp(0.5 * logvars)))

        mixture_weights = self.mixture_weights.unsqueeze(0).expand(batch_size,self.num_components,-1,-1)
        mixture_weights = mixture_weights.squeeze()
        mixture_dist = Categorical(logits=mixture_weights)

        gmm = MixtureSameFamily(mixture_dist, multi_normal_dist)

        log_prob = gmm.log_prob(z)
        return -log_prob.mean()


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

        gmm_loss = self.gmm_loss(z)

        condition = self.fc_condition(condition)
        previous = self.fc_prev(previous)

        z = torch.cat((z, previous, condition), dim=1)
        z = self.fc_z(z)

        return z, mu, logvar, gmm_loss

    def sample(self, num_samples=1):

        component_dist = Categorical(logits=self.mixture_weights)
        component_samples = component_dist.sample((num_samples,))

        means = self.means[component_samples]
        logvars = self.logvars[component_samples]

        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        samples = means + eps * std

        return samples

