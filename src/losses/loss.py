import torch
import torch.nn.functional as F


def get_loss_function(loss_name):
    loss_functions = {
        'mse': F.mse_loss,
        'mae': F.l1_loss,
        'huber_loss': F.smooth_l1_loss,
        'mean_relative_loss': mean_relative_loss_function,
    }

    # Convert loss_name to lowercase for case-insensitive comparison
    loss_name_lower = loss_name.lower()

    # Check if the loss_name exists in the dictionary
    if loss_name_lower in loss_functions:
        return loss_functions[loss_name_lower]
    else:
        raise ValueError(f"Loss function '{loss_name}' not supported. Supported losses are: "
                         f"Mean squared error, Mean absolute error, Huber loss, Mean relative loss.")



def mean_relative_loss_function(input, target):

    epsilon = 1e-8
    absolute_difference = torch.abs(input - target)
    absolute_target = torch.abs(target)

    relative_difference = absolute_difference / torch.max(absolute_target, torch.tensor(epsilon, dtype=target.dtype,
                                                                                        device=target.device))

    loss = torch.mean(relative_difference)
    return loss