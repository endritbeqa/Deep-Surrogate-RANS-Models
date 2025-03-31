import os
import json
import math

import torch
from PIL import Image
from matplotlib import cm
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    targets = np.rot90(targets, axes=(1, 2))
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
            # axes[row, col].axis('off')
        im = axes[0, col].imshow(data[0, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=axes[:, col], orientation='horizontal', pad=0.1)

    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row_label, size='medium')

    save_path = os.path.join(output_dir, file_name + ".png")
    plt.savefig(save_path)
    plt.close()


def plot_comparison_parameter_range(data, row_labels, table_label):
    data = to_numpy(data)
    rows, C, H, W = data.shape

    fig, axes = plt.subplots(rows, C, squeeze=False, figsize=(10, 10))
    column_labels = ['mean_P', 'mean_Ux', 'mean_Uy', 'std_P', 'std_Ux', 'std_Uy']

    for col in range(C):
        vmin, vmax = data[:, col, :, :].min(), data[:, col, :, :].max()
        axes[0, col].set_title(column_labels[col])
        for row in range(rows):
            im = axes[row, col].imshow(data[row, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
            # axes[row, col].axis('off')
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
        angle = round(math.degrees(angle.item()), ndigits=2)

        re_nums_lables = ["Re:{}e-5".format(int(re.item())) for re in re_nums]
        angle_lable = "Angle of Attack:{} degrees".format(angle)

        plt = plot_comparison_parameter_range(slice, re_nums_lables, angle_lable)
        save_path = os.path.join(reynolds_comparison_folder, "Reynolds_comparison_at_{}.png".format(angle))
        plt.savefig(save_path)
        plt.close()

    for i in range(num_REs):
        slice = predictions[i, :]
        angles = parameters[i, :, 1]
        re = parameters[i, 0, 0]
        angles = [round(math.degrees(angle.item()), ndigits=2) for angle in angles]

        angle_lables = ["Angle:{}".format(angle) for angle in angles]
        re_lable = "Re:{}e-5 ".format(int(re.item()))

        plt = plot_comparison_parameter_range(slice, angle_lables, re_lable)
        save_path = os.path.join(angle_comparison_folder, "Angle_comparison_at_{}.png".format(re))
        plt.savefig(save_path)
        plt.close()


def plot_std_curves(lines, x, output_dir):
    colors = ['blue', 'red', 'pink', 'black', 'orange', 'yellow']
    line_styles = ['-', '--', '-.', ':', 'solid']

    plt.figure(figsize=(10, 6))
    plt.xlim(min(x) - 0.5, max(x) + 0.5)

    for i, (label, line) in enumerate(lines.items()):
        means = []
        for re, (min_val, max_val, mean) in line.items():
            re = re / 1000.0
            means.append(mean)
            plt.plot([re, re], [min_val, max_val], color=colors[i], linestyle='--')
            plt.plot([re - 0.01, re + 0.01], [min_val, min_val], color=colors[i], linestyle='solid')
            plt.plot([re - 0.01, re + 0.01], [max_val, max_val], color=colors[i], linestyle='solid')

        plt.plot(x, means, label=label, color=colors[i], linestyle=line_styles[i])

    plt.axvspan(xmin=min(x) - 0.5, xmax=min(x) + 0.5, color='gray', alpha=0.5)
    plt.axvspan(xmin=max(x) - 0.5, xmax=max(x) + 0.5, color='gray', alpha=0.5)

    plt.xlabel('Re_number 10\u2075')
    plt.ylabel('std')
    plt.title('Model sample/ground truth mean std comparison')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(output_dir, "average_std_comparison.png"))


def plot_mse_ratios(data, output_dir, save_plot=True):
    fig, axes = plt.subplots(2, 2, squeeze=False, figsize=(15, 12))

    for i, (mode_region, mse_lists) in enumerate(data.items()):

        row = i//2
        col = i%2

        if not mse_lists:
            raise ValueError("The MSE list cannot be empty.")
        colors = ['red', 'red', 'green', 'green', 'orange', 'orange']
        line_styles = ['solid', '--', 'solid', '--', 'solid', '--']

        axes[row,col].set_xscale("log")

        for i, (label, mse_list) in enumerate(mse_lists):
            min_curve = mse_list[0]
            mean_curve = mse_list[1]
            max_curve = mse_list[2]

            length = len(mean_curve)
            ratios = [idx / (length - 1) for idx in range(length)]
            axes[row,col].plot(mean_curve, ratios, color=colors[i], linestyle=line_styles[i], label=label, scaley="log")
            axes[row,col].fill_betweenx(ratios, min_curve, max_curve, color=colors[i], alpha=0.1)

        axes[row,col].set_xlabel("{} Uncertainty MSE".format(mode_region), fontsize=12)
        axes[row,col].set_ylabel("Ratio", fontsize=12)
        axes[row,col].grid(True, linestyle='solid')  # , alpha=1.0)
        axes[row,col].legend(fontsize=10)

    plt.suptitle("MSE Ratios", fontsize=20)
    plt.savefig(os.path.join(output_dir, "ratio_comparison.png"))


def plot_drag_coefficient_distribution(label, targets, predictions, output_dir, num_buckets):
    plt.figure(figsize=(10, 6))

    max_val = np.max(np.array([np.max(targets), np.max(predictions)]))
    min_val = np.min(np.array([np.min(targets), np.min(predictions)]))
    step = (max_val-min_val)/num_buckets

    x_vals = [min_val + x * step + 0.5 * step for x in range(num_buckets)]
    intervals = [(min_val + x * step, min_val + (x + 1) * step) for x in range(num_buckets)]
    target_counts = np.array([np.sum((targets >= low) & (targets < high)) for low, high in intervals])
    target_counts = target_counts/np.sum(target_counts)
    prediction_counts = [[np.sum((run > low) & (run <= high)) for low, high in intervals] for run in predictions]
    prediction_counts = np.array(prediction_counts)
    prediction_counts = prediction_counts/np.sum(prediction_counts, axis=1, keepdims=True)
    prediction_counts_mins = np.min(prediction_counts, axis=0)
    prediction_counts_mean = np.mean(prediction_counts, axis=0)
    prediction_counts_maxs = np.max(prediction_counts, axis=0)


    colors = ['red', 'red', 'green', 'green', 'orange', 'orange']
    line_styles = ['solid', '--', 'solid', '--', 'solid']

    plt.bar(x_vals, target_counts, color="white", edgecolor="black", width=step, label="Ground Truth")
    plt.plot(x_vals, prediction_counts_mean, label=label, color=colors[0], linestyle=line_styles[0])

    for i, x in enumerate(x_vals):
        plt.plot([x, x], [prediction_counts_mins[i], prediction_counts_maxs[i]], color=colors[0], linestyle='--', label="Prediction")
        plt.plot([x - step/100, x + step/100], [prediction_counts_mins[i], prediction_counts_mins[i]], color=colors[0], linestyle='solid')
        plt.plot([x - step/100, x + step/100], [prediction_counts_maxs[i], prediction_counts_maxs[i]], color=colors[0], linestyle='solid')

    plt.xlabel("Drag Coefficient", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    plt.title('Drag Coefficient distribution')

    plt.savefig(os.path.join(output_dir, "{}_drag_comparison.png".format(label)))



def plot_samples_different_models(samples, chart_label, row_labels,channel , output_dir):
        samples = to_numpy(samples)
        samples = np.rot90(samples, axes=(3, 4))

        Models, B, C, W, H, = samples.shape

        fig, axes = plt.subplots(Models, B, squeeze=False, figsize=(9, 6))

        channel_name = ["Pressure", "X-Velocity", "Y-Velocity"]
        column_labels = ["Sample {}".format(i) for i in range(1, B+1)]


        for i, ax in enumerate(axes[:,0]):
            ax.set_ylabel(row_labels[i], rotation=90, size="medium")

        for col in range(B):
            vmin, vmax = samples[:, col, channel, :, :].min(), samples[:, col, channel, :, :].max()
            axes[0, col].set_title(column_labels[col], fontsize=12, pad=8)
            for row in range(Models):
                ax = axes[row, col]
                im = ax.imshow(samples[row, col, channel], cmap=cm.magma, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
            cbar = fig.colorbar(im, ax=axes[:, col], orientation='horizontal', pad=0.03, shrink = 0.8)

        fig.suptitle("{} samples of models and ground truth".format(channel_name[channel]))
        plt.savefig(os.path.join(output_dir,channel_name[channel] ,"{}.png".format(chart_label)))



def plot_moment_comparison_models(moments, file_name, output_dir):
    moments = to_numpy(moments)
    moments = np.rot90(moments, axes=(2, 3))

    rows, C, H, W = moments.shape

    fig, axes = plt.subplots(rows, C, figsize=(12, 8))

    column_labels = ['µ P', 'µ Ux', 'µ Uy', 'σ P', 'σ Ux', 'σ Uy']
    row_labels = ['Ground Truth', 'FactFormer', 'Swin', 'DiT']

    for col in range(C):
        vmin, vmax = moments[:, col, :, :].min(), moments[:, col, :, :].max()
        axes[0, col].set_title(column_labels[col])
        for row in range(rows):
            im = axes[row, col].imshow(moments[row, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        im = axes[0, col].imshow(moments[0, col], cmap=cm.magma, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=axes[:, col], orientation='horizontal', pad=0.03, shrink = 0.8)

    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(row_label, rotation=90, size='medium')

    fig.suptitle("Moment Comparison")
    save_path = os.path.join(output_dir, file_name + ".png")
    plt.savefig(save_path)
    plt.close()



def plot_bar_chart(statistics, categories, bar_labels, chart_label, x_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    interpolation_means = statistics["interpolation"]["means"]
    interpolation_stds = statistics["interpolation"]["stds"]
    extrapolation_means = statistics["extrapolation"]["means"]
    extrapolation_stds = statistics["extrapolation"]["stds"]

    num_groups = len(bar_labels)
    x = np.arange(len(categories))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    colors = ['b', 'g', 'r', 'o', 'p']

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_useOffset(False)
    formatter.set_powerlimits((-4, 0))


    for i in range(num_groups):
        axes[0].bar(x + i * width - width, interpolation_means[:, i], width, yerr=interpolation_stds[:, i], capsize=5,
                    color=colors[i], label=bar_labels[i], alpha=0.7)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].yaxis.get_offset_text().set_visible(False)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel('σ MSE $10^{-4}$')
    axes[0].set_xlabel(x_label)
    axes[0].set_title('Interpolation region')
    axes[0].legend()

    for i in range(num_groups):
        axes[1].bar(x + i * width - width, extrapolation_means[:, i], width, yerr=extrapolation_stds[:, i], capsize=5,
                    color=colors[i], label=bar_labels[i], alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel('σ MSE $10^{-4}$')
    axes[1].set_xlabel(x_label)
    axes[1].set_title('Extrapolation region')
    axes[1].legend()

    plt.savefig(os.path.join(output_dir, "{}.png".format(chart_label)))


def plot_sampling_speed_bar_chart(statistics, categories, num_samples_compared, bar_labels, chart_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    num_samples_1_means = statistics[num_samples_compared[0]]["means"]
    num_samples_1_stds = statistics[num_samples_compared[0]]["stds"]
    num_samples_2_means = statistics[num_samples_compared[1]]["means"]
    num_samples_2_stds = statistics[num_samples_compared[1]]["stds"]

    num_groups = len(bar_labels)
    x = np.arange(len(categories))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    colors = ['b', 'g', 'r', 'o', 'p']

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_useOffset(False)
    formatter.set_powerlimits((-4, 0))


    for i in range(num_groups):
        axes[0].bar(x + i * width - width, num_samples_1_means[:, i], width, yerr=num_samples_1_stds[:, i], capsize=5,
                    color=colors[i], label=bar_labels[i], alpha=0.7)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].yaxis.get_offset_text().set_visible(False)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel('Time in seconds')
    axes[0].set_xlabel('Number of diffusion steps')
    axes[0].set_title(f'{num_samples_compared[0]} samples generated')
    axes[0].legend()

    for i in range(num_groups):
        axes[1].bar(x + i * width - width, num_samples_2_means[:, i], width, yerr=num_samples_2_stds[:, i], capsize=5,
                    color=colors[i], label=bar_labels[i], alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel('Time in seconds')
    axes[1].set_xlabel('Number of diffusion steps')
    axes[1].set_title(f'{num_samples_compared[1]} samples generated')
    axes[1].legend()

    plt.savefig(os.path.join(output_dir, "{}.png".format(chart_label)))


def plot_all_mse_ratios(data, output_dir):
    fig, axes = plt.subplots(3, 4, squeeze=False, figsize=(20, 15))
    colors = ['red', 'red', 'green', 'green', 'orange', 'orange']
    line_styles = ['solid', '--', 'solid', '--', 'solid', '--']

    for row, (model, ratio_statistics) in enumerate(data.items()):

        for col, (mode_region, mse_lists) in enumerate(ratio_statistics.items()):

            if not mse_lists:
                raise ValueError("The MSE list cannot be empty.")

            axes[row,col].set_xscale("log")

            for j, (label, mse_list) in enumerate(mse_lists):
                min_curve = mse_list[0]
                mean_curve = mse_list[1]
                max_curve = mse_list[2]

                length = len(mean_curve)
                ratios = [idx / (length - 1) for idx in range(length)]
                axes[row,col].plot(mean_curve, ratios, color=colors[j], linestyle=line_styles[j], label=label, scaley="log")
                axes[row,col].fill_betweenx(ratios, min_curve, max_curve, color=colors[j], alpha=0.1)

            if row == 0:
                axes[row,col].set_title("{} Uncertainty MSE".format(mode_region), fontsize=12)
            if col==0:
                axes[row, col].set_ylabel(model, fontsize=12)
            else:
                axes[row,col].set_ylabel("Ratio", fontsize=12)
            axes[row,col].grid(True, linestyle='solid')  # , alpha=1.0)
            axes[row,col].legend(fontsize=10)

    #plt.suptitle("MSE Ratios", fontsize=20)
    plt.savefig(os.path.join(output_dir, "ratio_comparison.png"))




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
