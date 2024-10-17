import math

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

            im = Image.fromarray(cm.magma(field, bytes=True))
            im = im.resize((h, w))
            file_path = "{}/images/{}/{}/{}_{}.png".format(output_dir, mode,epoch, labels[j], i)
            im.save(file_path)


def save_samples(samples, output_dir):
    samples = np.rot90(samples, axes=(2, 3))
    b, c, h, w = samples.shape
    channel_labels = ['P', 'Ux', 'Uy']

    for i, sample in enumerate(samples):
        fig, axes = plt.subplots(1, c, figsize=(12, 8))
        for channel in range(c):
            im = axes[channel].imshow(sample[channel], cmap=cm.magma)
            axes[channel].set_title(channel_labels[channel])
            fig.colorbar(im, ax=axes[channel])

        file_path = os.path.join(output_dir, "sample_{}.png".format(i))
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()




def plot_comparison(targets, predictions, output_dir, file_name):
    if targets.shape != predictions.shape:
        raise ValueError("Input arrays must have same shape!")

    targets = np.rot90(targets, axes=(1,2))
    predictions = np.rot90(predictions, axes=(1, 2))
    delta = targets - predictions

    data = np.stack([targets, predictions], axis=0)
    rows, C, H, W = data.shape

    fig, axes = plt.subplots(rows, C, figsize=(12, 8))

    column_labels = ['mean_P', 'mean_Ux', 'mean_Uy', 'std_P', 'std_Ux', 'std_Uy']
    row_labels = ['Target', 'Prediction', 'Delta']

    for col in range(C):
        vmin, vmax = data[:, col, :, :].min(), data[:, col, :, :].max()
        axes[0, col].set_title(column_labels[col])
        for row in range(rows):
            im = axes[row, col].imshow(data[row, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
            #axes[row, col].axis('off')
        im = axes[0, col].imshow(data[0, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=axes[:, col], orientation='horizontal', pad=0.1)

    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row_label, size='medium')

    save_path = os.path.join(output_dir,  file_name+".png")
    plt.savefig(save_path)
    plt.close()





def plot_comparison_parameter_range(data, row_labels, table_label):
    rows, C, H, W = data.shape

    fig, axes = plt.subplots(rows, C, squeeze=False, figsize=(10,10))
    column_labels = ['mean_P', 'mean_Ux', 'mean_Uy', 'std_P', 'std_Ux', 'std_Uy']

    for col in range(C):
        vmin, vmax = data[:, col, :, :].min(), data[:, col, :, :].max()
        axes[0, col].set_title(column_labels[col])
        for row in range(rows):
            im = axes[row, col].imshow(data[row, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
            #axes[row, col].axis('off')
        im = axes[0, col].imshow(data[0, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=axes[:, col], orientation='horizontal', pad=0.1)


    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row_label, size='medium')

    plt.suptitle(table_label)
    return plt



def save_parameter_comparison(predictions, parameters, output_dir):
    num_REs, num_Angles, C, H, W = predictions.shape
    predictions = np.rot90(predictions, axes=(3, 4))

    reynolds_comparison_folder = os.path.join(output_dir, "Reynolds_comparison")
    angle_comparison_folder = os.path.join(output_dir, "Angle_comparison")

    os.makedirs(reynolds_comparison_folder, exist_ok=True)
    os.makedirs(angle_comparison_folder, exist_ok=True)

    for i in range(num_Angles):
        slice = predictions[:, i]
        re_nums = parameters[:, i, 0]
        angle = parameters[0, i, 1]
        angle = round(math.degrees(angle.item()),ndigits=2)

        re_nums_lables = ["Re:{}e-5".format(int(re.item())) for re in re_nums]
        angle_lable = "Angle of Attack:{} degrees".format(angle)

        plt = plot_comparison_parameter_range(slice, re_nums_lables, angle_lable)
        save_path = os.path.join(reynolds_comparison_folder,"Reynolds_comparison_at_{}.png".format(angle))
        plt.savefig(save_path)
        plt.close()

    for i in range(num_REs):
        slice = predictions[i, :]
        angles = parameters[i, :, 1]
        re = parameters[i,0,0]
        angles = [round(math.degrees(angle.item()),ndigits=2) for angle in angles]

        angle_lables = ["Angle:{}".format(angle) for angle in angles]
        re_lable = "Re:{}e-5 ".format(int(re.item()))

        plt = plot_comparison_parameter_range(slice, angle_lables, re_lable)
        save_path = os.path.join(angle_comparison_folder,"Angle_comparison_at_{}.png".format(re))
        plt.savefig(save_path)
        plt.close()




