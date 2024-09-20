import numpy as np
import os
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt


def plot_losses(train_loss, validation_loss, xlabel='Epoch', ylabel='Loss', title='Train/Val loss curves',
                label1='Training loss', label2='Validation loss'):
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


def save_images(outputs, output_dir,mode , epoch):
    if outputs.is_cuda:
        outputs = outputs.cpu()
    outputs = outputs.numpy()

    os.makedirs("{}/images/{}/{}".format(output_dir,mode , epoch))

    b, c, h, w = outputs.shape
    labels = ['pressure', "vel_x", "vel_y"]
    for i in range(min(5,b)):
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


def plot_comparison(targets, predictions, output_dir, file_name):
    if targets.shape != predictions.shape or targets.shape[0] != 3:
        raise ValueError("Input arrays must have shape (3, H, W)")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    column_labels = ['P', 'Ux', 'Uy']
    row_labels = ['Target', 'Prediction']

    for i in range(3):
        im = axes[0, i].imshow(targets[i], cmap=cm.magma)
        fig.colorbar(im, ax=axes[0, i])
        axes[0, i].set_title(column_labels[i])

    for i in range(3):
        im = axes[1, i].imshow(predictions[i], cmap=cm.magma)
        fig.colorbar(im, ax=axes[1, i])

    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row_label, rotation=90, size='large')

    plt.tight_layout()

    save_path = os.path.join(output_dir, file_name+".png")
    plt.savefig(save_path)
    plt.close()