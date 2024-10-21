import torch
import torch.nn.functional as F


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


def get_loss_function(loss: str):

    loss_functions = {
        'mse': F.mse_loss,
        'mae': F.l1_loss,
        'huber_loss': F.smooth_l1_loss,
        'mrl': mean_relative_loss_function,
        'con_of_mass': con_of_mass,
    }

    if loss not in loss_functions:
        raise ValueError(f"Loss function '{loss}' not supported. Supported losses are:\n"
                             f"Mean squared error, Mean absolute error, Huber loss, Mean relative loss, Conservation of mass loss, KL Divergence with following names.\n"
                             f"mse, mae, hubber_loss, mrl, con_of_mass, beta_KLD")


    return loss_functions[loss]
