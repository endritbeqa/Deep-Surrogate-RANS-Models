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

def plot_recon_vs_KLD(train_recon, train_KLD,val_recon,val_KLD, xlabel='Epoch', ylabel='Loss', title='Train/Val loss curves',
                label1='Training Reconstruction', label2='Training KLD', label3='Validation Reconstruction', label4='Validation KLD'):

    x_values = np.arange(len(train_recon))
    plt.plot(x_values, train_recon, label=label1, color='blue')
    plt.plot(x_values, train_KLD, label=label2, color='pink')
    plt.plot(x_values, val_recon, label=label3, color='orange')
    plt.plot(x_values, val_KLD, label=label4, color='red')
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

            #grayscale_field = ((field * 127) + 128).astype(np.uint8)  # Grayscale mapping
            #rgb_field = np.stack([grayscale_field] * 3, axis=-1)  # Convert to RGB (R=G=B for grayscale)
            #im = Image.fromarray(rgb_field)

            im = Image.fromarray(cm.magma(field, bytes=True))
            im = im.resize((h, w))
            file_path = "{}/images/{}/{}/{}_{}.png".format(output_dir, mode,epoch, labels[j], i)
            im.save(file_path)


def save_samples(samples, output_dir, label, epoch):
    try:
        if samples.is_cuda:
            samples = samples.cpu()
            samples = samples.numpy()
    except:
        print()

    label_dir = os.path.join(output_dir, "images", label)
    samples_dir = os.path.join(str(label_dir), "samples")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(samples_dir)

    b, c, h, w = samples.shape
    fields = ['pressure', "vel_x", "vel_y"]
    for i in range(len(samples)):
        for j in range(c):
            field = np.copy(samples[i, j])
            field = np.flipud(field.transpose())

            min_value = np.min(field)
            max_value = np.max(field)
            field -= min_value
            max_value -= min_value
            field /= max_value

            # grayscale_field = ((field * 127) + 128).astype(np.uint8)  # Grayscale mapping
            # rgb_field = np.stack([grayscale_field] * 3, axis=-1)  # Convert to RGB (R=G=B for grayscale)
            # im = Image.fromarray(rgb_field)

            im = Image.fromarray(cm.magma(field, bytes=True))
            im = im.resize((h, w))
            file_path = os.path.join(str(samples_dir), "{}_{}.png".format(fields[j], i))
            im.save(file_path)






def plot_comparison(targets, predictions, output_dir, file_name):
    if targets.shape != predictions.shape or targets.shape[0] != 3:
        raise ValueError("Input arrays must have shape (3, H, W)")


    targets = np.rot90(targets, axes=(1,2))
    predictions = np.rot90(predictions, axes=(1, 2))
    delta = targets - predictions

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    column_labels = ['P', 'Ux', 'Uy']
    row_labels = ['Target', 'Prediction', 'Delta']

    for i in range(3):
        im = axes[0, i].imshow(targets[i] ,cmap=cm.magma)
        axes[0, i].set_title(column_labels[i])
    fig.colorbar(im, ax=axes[0, :], orientation='vertical')  # For vertical colorbar


    for i in range(3):
        im = axes[1, i].imshow(predictions[i], cmap=cm.magma)
    fig.colorbar(im, ax=axes[1, :], orientation='vertical')  # For vertical colorbar


    #for i in range(3):
    #    im = axes[2, i].imshow(delta[i], cmap=cm.magma)
    #fig.colorbar(im, ax=axes[2, :], orientation='vertical')  # For vertical colorbar



    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row_label, size='large')

    #plt.tight_layout()

    save_path = os.path.join(output_dir,  file_name+".png")
    plt.savefig(save_path)
    plt.close()


def plot_comparison_all(predictions, output_dir, file_name):


    predictions = np.rot90(predictions, axes=(2, 3))
    B, C, _, _ = predictions.shape

    fig, axes = plt.subplots(5, 6, figsize=(12, 8))

    column_labels = ['mean_P', 'mean_Ux', 'mean_Uy','std_P', 'std_Ux', 'std_Uy']
    row_labels = ['Target', 'Prediction', 'Delta']


    for i in range(B):
        for j in range(C):
            im = axes[i, j].imshow(predictions[i, j], cmap=cm.magma)
            fig.colorbar(im, ax=axes[i, :], orientation='vertical')  # For vertical colorbar



    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row_label, size='large')

    #plt.tight_layout()

    save_path = os.path.join(output_dir,  file_name+".png")
    plt.savefig(save_path)
    plt.close()


