import os
import json
import math

import torch
from PIL import Image
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")


def plot_losses(curves, x_label, y_label, title='Train/Val loss curves'):
    colors = ['blue', 'red', 'pink', 'black', 'orange', 'yellow']

    for idx, (label, values) in enumerate(curves.items()):
        x_values = np.arange(len(values))
        plt.plot(x_values, values, label=label, color=colors[idx])

    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    return plt


def plot_samples(samples, output_dir):
    samples = to_numpy(samples)
    samples = np.rot90(samples, axes=(2, 3))

    B, C, W, H, = samples.shape


    column_labels = ['P', 'Ux', 'Uy']
    for i in range(B):
        fig, axes = plt.subplots(1, C, figsize=(5, 5))
        for col in range(C):
            im = axes[col].imshow(samples[i, col], cmap=cm.magma)
            axes[col].set_title(column_labels[col])
            cbar = fig.colorbar(im, ax=axes[col], orientation='horizontal', pad=0.1)

        save_path = os.path.join(output_dir, "sample_{}.png".format(i))
        plt.savefig(save_path)
        plt.close()



def plot_moment_comparison(targets, predictions, file_name, output_dir, plot_delta=True):
    targets = to_numpy(targets)
    predictions = to_numpy(predictions)
    if targets.shape != predictions.shape:
        raise ValueError("Input arrays must have same shape!")

    targets = np.rot90(targets, axes=(1,2))
    predictions = np.rot90(predictions, axes=(1, 2))
    
    if plot_delta:
        delta = targets - predictions
        data = np.stack([targets, predictions, delta], axis=0)
    else:
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
    data = to_numpy(data)
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
    predictions = to_numpy(predictions)
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


def plot_std_curves(lines, x, output_dir):
    colors = ['blue', 'red', 'pink', 'black', 'orange', 'yellow']
    line_styles = ['-', '--', '-.', ':', 'solid']

    plt.figure(figsize=(10, 6))
    plt.xlim(min(x)-0.5, max(x)+0.5)

    for i, (label, line) in enumerate(lines.items()):
        means = []
        for re, (min_val, max_val, mean) in line.items():
            re = re/1000.0
            means.append(mean)
            plt.plot([re, re], [min_val, max_val], color=colors[i], linestyle='--')
            plt.plot([re - 0.01, re + 0.01], [min_val, min_val], color=colors[i], linestyle='solid')
            plt.plot([re - 0.01, re + 0.01], [max_val, max_val], color=colors[i], linestyle='solid')

        plt.plot(x, means, label=label, color=colors[i], linestyle=line_styles[i])


    plt.axvspan(xmin=min(x)-0.5, xmax=min(x)+0.5, color='gray', alpha=0.5)
    plt.axvspan(xmin=max(x)-0.5, xmax=max(x)+0.5, color='gray', alpha=0.5)

    plt.xlabel('Re_number 10\u2075')
    plt.ylabel('std')
    plt.title('Model sample/ground truth mean std comparison')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(output_dir, "average_std_comparison.png"))


def plot_multiple_mse_ratios(mse_list, plot_label, output_dir):
    if not mse_list:
        raise ValueError("The MSE list cannot be empty.")
    colors = ['red', 'red', 'green', 'green', 'orange', 'orange']
    line_styles = ['solid', '--', 'solid', '--', 'solid', '--']

    plt.figure(figsize=(10, 8))

    plt.xscale("log")

    for i, (label, mse_list) in enumerate(mse_list):
        min_curve = mse_list[0]
        mean_curve = mse_list[1]
        max_curve = mse_list[2]

        length = len(mean_curve)
        ratios = [idx / (length-1) for idx in range(length)]
        plt.plot(mean_curve, ratios, color=colors[i], linestyle=line_styles[i], label=label, scaley="log")
        plt.fill_betweenx(ratios, min_curve, max_curve, color=colors[i], alpha=0.1)

    plt.xlabel("Mean Squared Error (MSE)", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    plt.title(plot_label, fontsize=14)
    plt.grid(True, linestyle='solid') #, alpha=1.0)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{plot_label}.png"))




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)




