import torch
import numpy as np

#TODO introduce a parent class Scheduler


class LinearNoiseScheduler:
    def __init__(self, config):
        self.start_beta = config.start_beta
        self.end_beta = config.end_beta
        self.timesteps = config.timesteps
        self.betas = torch.linspace(self.start_beta, self.end_beta, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]

class CosineNoiseScheduler:
    def __init__(self, config):
        self.timesteps = config.timesteps
        self.alphas = torch.linspace(0, np.pi / 2, config.timesteps)
        self.alphas = torch.cos(self.alphas) ** 2
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


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