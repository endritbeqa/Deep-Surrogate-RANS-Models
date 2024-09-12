import torch.nn.functional as F
import torch


#TODO check and fix the gradient loss
def gradient(tensor):
    grad = torch.zeros_like(tensor)
    # Central difference
    grad[:, :, :, 1:-1] = (tensor[:, :, :, 2:] - tensor[:, :, :-2]) / 2
    # Forward difference
    grad[:, :, :, 0] = tensor[:, :, :, 1] - tensor[:, :, 0]
    # Backward difference
    grad[:, :, :, -1] = tensor[:, :, :, -1] - tensor[:, :, -2]
    return grad


def con_of_mass(predictions, targets):
    b, c, h, w = predictions.shape

    #TODO check the indexes again
    grad_x_velocity_x = gradient(predictions[:, 0, :, :])
    grad_y_velocity_y = gradient(predictions[:, 1, :, :])
    divergence = grad_x_velocity_x + grad_y_velocity_y
    loss = torch.mean(divergence ** 2)

    return loss


def mean_relative_loss_function(prediction, target):
    epsilon = 1e-6
    absolute_difference = torch.abs(prediction - target)
    absolute_target = torch.abs(target)
    relative_difference = absolute_difference / torch.max(absolute_target, torch.tensor(epsilon, dtype=target.dtype,
                                                                                        device=target.device))
    loss = torch.mean(relative_difference)
    return loss

def beta_KLD(mu, logvar, beta):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta*KLD

def get_loss_function(losses: list):
    assert len(losses) != 0, "No losses were provided."

    loss_functions = {
        'mse': F.mse_loss,
        'mae': F.l1_loss,
        'huber_loss': F.smooth_l1_loss,
        'mrl': mean_relative_loss_function,
        'con_of_mass': con_of_mass,
        'beta_KLD': beta_KLD
    }

    for loss in losses:
        if loss not in loss_functions:
            raise ValueError(f"Loss function '{loss}' not supported. Supported losses are:\n"
                             f"Mean squared error, Mean absolute error, Huber loss, Mean relative loss, Conservation of mass loss, KL Divergence with following names.\n"
                             f"mse, mae, hubber_loss, mrl, con_of_mass, beta_KLD")

    def loss(prediction, target, mu=None, logvar=None, beta=None):
        total_loss = 0
        for loss in losses:
            if loss == 'beta_KLD':
                if any((mu==None, logvar==None, beta==None)):
                    raise ValueError("Beta KLD loss requires mu, logvar and beta arguments.")
                total_loss += loss_functions[loss](mu, logvar, beta)
                print(0)
            else:
                total_loss += loss_functions[loss](prediction, target)
        return total_loss

    return loss


### quick and dirty testing
if __name__ == '__main__':
    losses = ['mse', 'beta_KLD']
    loss = get_loss_function(losses)
    x = torch.zeros((3,64,64))
    y = torch.ones((3,64,64))
    mu = torch.ones((3,64,64))
    logvar = torch.ones((3,64,64))

    a = loss(x,y, mu, logvar, 4)
    b = loss(x,y, mu, logvar, 6)
    print(a)

