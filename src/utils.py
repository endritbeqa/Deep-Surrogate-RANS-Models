import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_loss, validation_loss, xlabel='Epoch', ylabel='Loss', title='Train/Val loss curves', label1='Training loss', label2='Validation loss'):

    if len(train_loss) != len(validation_loss):
        raise ValueError("The two arrays must have the same length.")

    x_values = np.arange(len(train_loss))
    plt.plot(x_values, train_loss, label=label1, color='blue')
    plt.plot(x_values, validation_loss, label=label2, color='orange')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    return plt