import torch.nn.functional as F
import torch
import torch.nn as nn


def gradient(tensor, dim):
    grad = torch.zeros_like(tensor)
    # Central difference
    grad[:, :, 1:-1] = (tensor[:, :, 2:] - tensor[:, :, :-2]) / 2
    # Forward difference
    grad[:, :, 0] = tensor[:, :, 1] - tensor[:, :, 0]
    # Backward difference
    grad[:, :, -1] = tensor[:, :, -1] - tensor[:, :, -2]
    return grad


def con_of_mass(x_velocity, y_velocity):
    # Compute the gradients
    grad_x_velocity_x = gradient(x_velocity, dim=2)
    grad_y_velocity_y = gradient(y_velocity, dim=1)

    # Compute the divergence of the velocity field
    divergence = grad_x_velocity_x + grad_y_velocity_y

    # Conservation of mass loss: we want divergence to be close to zero
    loss = torch.mean(divergence ** 2)
    return loss


def mean_relative_loss_function(input, target):
    epsilon = 1e-6
    absolute_difference = torch.abs(input - target)
    absolute_target = torch.abs(target)

    relative_difference = absolute_difference / torch.max(absolute_target, torch.tensor(epsilon, dtype=target.dtype,
                                                                                        device=target.device))

    loss = torch.mean(relative_difference)
    return loss

def KLD(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.1*KLD + 4*MSE


def get_loss_function(loss_name: list):
    loss_functions = {
        'mse': F.mse_loss,
        'mae': F.l1_loss,
        'huber_loss': F.smooth_l1_loss,
        'mean_relative_loss': mean_relative_loss_function,
        'con_of_mass': con_of_mass
    }

    # Convert loss_name to lowercase for case-insensitive comparison
    loss_name_lower = [loss.lower() for loss in loss_name]
    losses = []

    for loss in loss_name_lower:
        if loss in loss_functions:
            losses.append(loss_functions[loss])
        else:
            raise ValueError(f"Loss function '{loss}' not supported. Supported losses are: "
                             f"Mean squared error, Mean absolute error, Huber loss, Mean relative loss, Conservation of mass loss.")

    assert len(losses) != 0, "No losses were provided."

    def loss(input, target):
        total_loss = 0
        for loss in losses:
            total_loss += loss(input, target)
        return total_loss

    return loss
