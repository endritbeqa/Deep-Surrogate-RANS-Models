import torch
import torch.nn as nn
import torch.nn.functional as F


class GMM_VAEBottleneck(nn.Module):
    def __init__(self, config, i_layer):
        super().__init__()

        self.device = config.device
        self.latent_dim = config.latent_dim[i_layer]
        self.hidden_dim = config.hidden_dim[i_layer]
        self.num_components = config.num_components[i_layer]

        self.means = nn.Parameter(torch.randn(self.num_components, self.latent_dim))
        self.logvars = nn.Parameter(torch.randn(self.num_components, self.latent_dim))
        self.mixture_weights = nn.Parameter(torch.rand(self.num_components, 1, 1))

        self.fc_mu = nn.Linear(self.hidden_dim + self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim + self.hidden_dim, self.latent_dim)

        self.fc_condition = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_prev = nn.Linear(self.hidden_dim, self.latent_dim)

        self.fc_z = nn.Linear(3 * self.latent_dim, self.hidden_dim)

    def log_normal_diag(self, x, mu, log_var, dim=None):
        D = x.shape[1]
        log_p = (
            -0.5 * D * torch.log(torch.tensor(2) * torch.pi)
            - 0.5 * log_var
            - 0.5 * torch.exp(-log_var) * (x - mu) ** 2.0
        )

        return log_p

    def gmm_loss(self, z, mean, logvar):

        encoder_log_prob = self.log_normal_diag(z, mean, logvar)

        w = F.softmax(self.mixture_weights, dim=0)
        z = z.unsqueeze(0)  # 1 x B x L
        prior_means = self.means.unsqueeze(1)  # K x 1 x L
        prior_logvars = self.logvars.unsqueeze(1)  # K x 1 x L

        prior_log_p = self.log_normal_diag(z, prior_means, prior_logvars) + torch.log(
            w
        )  # K x B x L
        prior_log_prob = torch.logsumexp(prior_log_p, dim=0, keepdim=False)  # B x L

        return torch.mean(
            (torch.exp(prior_log_prob) * (prior_log_prob - encoder_log_prob)).mean(-1)
        )

    def forward(self, encoder_input, previous, condition):
        B, _ = encoder_input.shape

        x = torch.cat((encoder_input, previous), dim=-1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        gmm_loss = self.gmm_loss(z, mu, logvar)

        condition = self.fc_condition(condition)
        previous = self.fc_prev(previous)

        z = torch.cat((z, previous, condition), dim=1)
        z = self.fc_z(z)

        return z, gmm_loss

    def sample(self, num_samples):

        w = F.softmax(self.mixture_weights, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, num_samples, replacement=True)

        eps = torch.randn(num_samples, self.latent_dim)
        eps = eps.to(self.device)
        for i in range(num_samples):
            indx = indexes[i]
            if i == 0:
                z = self.means[[indx]] + eps[[i]] * torch.exp(self.logvars[[indx]])
            else:
                z = torch.cat(
                    (
                        z,
                        self.means[[indx]] + eps[[i]] * torch.exp(self.logvars[[indx]]),
                    ),
                    0,
                )
        return z


"""
    def sample(self, num_samples=1):

        component_dist = Categorical(logits=self.mixture_weights)
        component_samples = component_dist.sample((num_samples,))

        means = self.means[component_samples]
        logvars = self.logvars[component_samples]

        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        samples = means + eps * std

        return samples

"""
