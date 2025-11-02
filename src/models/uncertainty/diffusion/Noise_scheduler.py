import torch
import torch.nn as nn


class LinearNoiseScheduler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.start_beta = config.start_beta
        self.end_beta = config.end_beta
        self.steps = config.timesteps
        self.device = config.device
        self.betas = torch.linspace(self.start_beta, self.end_beta, self.steps)
        self.betas = self.betas.view(self.steps, 1, 1, 1)
        self.betas = self.betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)


class CosineNoiseScheduler:
    def __init__(self, config, s=0.008):
        self.steps = config.timesteps
        t_list = torch.arange(1, self.steps + 1, 1)
        temp1 = torch.cos((t_list / config.timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        temp2 = (
            torch.cos(((t_list - 1) / config.timesteps + s) / (1 + s) * torch.pi / 2)
            ** 2
        )
        self.beta_source = 1 - (temp1 / temp2)
        self.beta_source[self.beta_source > 0.999] = 0.999
        self.betas = self.beta_source
        self.betas = self.betas.view(self.steps, 1, 1, 1)
        self.betas = self.betas.to(config.device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)


def get_noise_scheduler(config):
    if config.noise_scheduler == "linear":
        return LinearNoiseScheduler(config)
    elif config.noise_scheduler == "cosine":
        return CosineNoiseScheduler(config)
    else:
        raise ValueError(f"Unknown noise scheduler: {config.noise_scheduler}")
