import torch
import torch.nn as nn
import numpy as np

#TODO introduce a parent class Scheduler
#TODO look into start_beta and end_beta which needs to be bigger??????


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



class CosineNoiseScheduler(nn.Module):
    def __init__(self, config, s=0.008):
        super().__init__()
        self.steps = config.timesteps
        self.device = config.device
        t_list = torch.arange(1, self.steps + 1, 1)
        temp1 = torch.cos((t_list / config.timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        temp2 = torch.cos(((t_list-1) / config.timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        self.beta_source = 1 - (temp1 / temp2)
        self.beta_source[self.beta_source > 0.999] = 0.999
        self.betas = torch.cat((torch.tensor([0]), self.beta_source), dim=0)
        self.betas = self.betas.to(self.device)
        self.betas = self.betas.view(self.steps + 1, 1, 1, 1)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_bar = 1 - self.alphas_bar
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)



# Quadratic noise scheduler
class QuadraticNoiseScheduler:
    def __init__(self, config):
        self.start_beta = config.start_beta
        self.end_beta = config.end_beta
        self.timesteps = config.timesteps
        self.betas = torch.linspace(self.start_beta ** 0.5, self.end_beta ** 0.5, self.timesteps) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


# Sigmoid noise scheduler
class SigmoidNoiseScheduler:
    def __init__(self, config):
        self.start_beta = config.start_beta
        self.end_beta = config.end_beta
        self.timesteps = config.timesteps
        self.sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        self.betas = self.sigmoid(torch.linspace(-6, 6, self.timesteps)) * (self.end_beta - self.start_beta) + self.start_beta
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


# Polynomial noise scheduler
class PolynomialNoiseScheduler:
    def __init__(self, config):
        self.start_beta = config.start_beta
        self.end_beta = config.end_beta
        self.timesteps = config.timesteps
        self.power = config.power
        self.betas = torch.linspace(self.start_beta, self.end_beta, self.timesteps) ** self.power
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


# Inverse quadratic noise scheduler
class InverseQuadraticNoiseScheduler:
    def __init__(self, config):
        self.start_beta = config.start_beta
        self.end_beta = config.end_beta
        self.timesteps = config.timesteps
        self.betas = torch.linspace(self.start_beta ** 0.5, self.end_beta ** 0.5, self.timesteps)[::-1] ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


# Logarithmic noise scheduler
class LogarithmicNoiseScheduler:
    def __init__(self, config):
        self.start_beta = config.start_beta
        self.end_beta = config.end_beta
        self.timesteps = config.timesteps
        self.betas = torch.logspace(torch.log10(torch.tensor(self.start_beta)),
                                    torch.log10(torch.tensor(self.end_beta)),
                                    steps=self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


def get_noise_scheduler(config):
    if config.noise_scheduler == 'linear':
        return LinearNoiseScheduler(config)
    elif config.noise_scheduler == 'cosine':
        return CosineNoiseScheduler(config)
    elif config.noise_scheduler == 'quadratic':
        return QuadraticNoiseScheduler(config)
    elif config.noise_scheduler == 'sigmoid':
        return SigmoidNoiseScheduler(config)
    elif config.noise_scheduler == 'polynomial':
        return PolynomialNoiseScheduler(config)
    elif config.noise_scheduler == 'inverse_quadratic':
        return InverseQuadraticNoiseScheduler(config)
    elif config.noise_scheduler == 'logarithmic':
        return LogarithmicNoiseScheduler(config)
    else:
        raise ValueError(f"Unknown noise scheduler: {config.noise_scheduler}")