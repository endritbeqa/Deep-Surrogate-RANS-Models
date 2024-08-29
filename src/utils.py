import numpy as np
import os
from PIL import Image
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_losses(train_loss, validation_loss, xlabel='Epoch', ylabel='Loss', title='Train/Val loss curves',
                label1='Training loss', label2='Validation loss'):
    if len(train_loss) != len(validation_loss):
        raise ValueError("The two arrays must have the same length.")

    x_values = np.arange(len(train_loss))
    plt.plot(x_values, train_loss, label=label1, color='blue')
    plt.plot(x_values, validation_loss, label=label2, color='orange')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    return plt


def save_images(outputs, output_dir, mode , epoch):
    try:
        if outputs.is_cuda:
            outputs = outputs.cpu()
            outputs = outputs.numpy()
    except:
        print()

    os.makedirs("{}/images".format(output_dir),exist_ok=True)
    os.makedirs("{}/images/{}/{}".format(output_dir,mode , epoch))

    b, c, h, w = outputs.shape
    labels = ['pressure', "vel_x", "vel_y"]
    for i in range(min(40,b)):
        for j in range(c):
            field = np.copy(outputs[i, j])
            field = np.flipud(field.transpose())

            min_value = np.min(field)
            max_value = np.max(field)
            field -= min_value
            max_value -= min_value
            field /= max_value

            im = Image.fromarray(cm.magma(field, bytes=True))
            im = im.resize((h, w))
            file_path = "{}/images/{}/{}/{}_{}.png".format(output_dir, mode,epoch, labels[j], i)
            im.save(file_path)


