import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import dataset
from src import utils
from src.evaluation import evaluation_config


def replace_outliers(data, label, output_dir):
    flat_data = data.flatten(1)
    outlier_indexes = (torch.max(flat_data, 1)[0] > torch.tensor(1.0)) | (torch.min(flat_data, 1)[0] < torch.tensor(-1.0))
    mean = torch.mean(data[~outlier_indexes], dim=0)
    if any(outlier_indexes):
        outliers = data[outlier_indexes]
        outlier_dir = os.path.join(output_dir, "Outliers", label)
        os.makedirs(outlier_dir, exist_ok=True)
        utils.plot_samples(outliers, outlier_dir)
    data[outlier_indexes] = mean
    return data



class Inter_Extrapolation_Test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.device = config.device
        self.checkpoint = torch.load(config.checkpoint)
        self.num_samples = config.num_samples
        self.model = self.checkpoint['model']
        self.model.device = self.device
        self.model.load_state_dict(self.checkpoint['model_params'])
        self.model.device = self.device
        self.model.move_to_device(self.device)
        self.eta = config.eta
        self.interpolation_dataset = dataset.Test_Dataset(self.config, 'interpolation')
        self.extrapolation_dataset = dataset.Test_Dataset(self.config, 'extrapolation')
        self.interpolation_dataloader = DataLoader(self.interpolation_dataset, batch_size=None, shuffle=False)
        self.extrapolation_dataloader = DataLoader(self.extrapolation_dataset, batch_size=None, shuffle=False)
        self.output_dir = os.path.join(config.output_dir, "Inter_Extrapolation")
        self.interpolation_output_dir = os.path.join(self.output_dir, 'interpolation')
        self.extrapolation_output_dir = os.path.join(self.output_dir, 'extrapolation')
        self.interpolation_mse_log = os.path.join(self.interpolation_output_dir, "MSE.txt")
        self.interpolation_target_moment_log = os.path.join(self.interpolation_output_dir, "target_moments.txt")
        self.extrapolation_mse_log = os.path.join(self.extrapolation_output_dir, "MSE.txt")
        self.extrapolation_target_moment_log = os.path.join(self.extrapolation_output_dir, "target_moments.txt")

        for dir in [self.output_dir,
                    self.interpolation_output_dir,
                    self.extrapolation_output_dir,
                    os.path.join(self.interpolation_output_dir, "Samples"),
                    os.path.join(self.interpolation_output_dir, "Comparison"),
                    os.path.join(self.interpolation_output_dir, "Outliers"),
                    os.path.join(self.extrapolation_output_dir, "Samples"),
                    os.path.join(self.extrapolation_output_dir, "Comparison"),
                    os.path.join(self.extrapolation_output_dir, "Outliers"),
                    ]:
            os.makedirs(dir, exist_ok=True)

        for file_path in [self.interpolation_mse_log, self.extrapolation_mse_log]:
            with open(file_path, "w") as file:
                file.write("airfoil_name,mean_p,mean_Ux,mean_Uy,std_p,std_Ux,std_Uy,full_MSE\n")

        for file_path in [self.interpolation_target_moment_log, self.extrapolation_target_moment_log]:
            with open(file_path, "w") as file:
                file.write("airfoil_name,mean_p,mean_Ux,mean_Uy,std_p,std_Ux,std_Uy\n")

    def write_test_statistics(self, mode):
        if mode == 'interpolation':
            output_dir = self.interpolation_output_dir
            mse_log = self.interpolation_mse_log
            target_moment_log = self.interpolation_target_moment_log
        elif mode == 'extrapolation':
            output_dir = self.extrapolation_output_dir
            mse_log = self.extrapolation_mse_log
            target_moment_log = self.extrapolation_target_moment_log

        mse_data = ((np.genfromtxt(mse_log, delimiter=',', skip_header=1))[:, 1:7]).astype(np.float32)
        target_means_data = ((np.genfromtxt(target_moment_log, delimiter=',', skip_header=1))[:, 1:]).astype(np.float32)

        statistics = {}
        statistics_raw = {}

        target_means_low_mask = np.mean(target_means_data[:, 3:6], axis=1) < 5e-3
        target_means_low_mask = np.tile(target_means_low_mask[:, np.newaxis], (1, 6))

        statistics['std_MSE_all'] = np.mean(mse_data[:, 3:6])
        statistics['mean_MSE_all'] = np.mean(mse_data[:, 0:3])
        statistics['std_MSE_low'] = np.mean(mse_data[:, 3:6][target_means_low_mask[:, 3:6]])
        statistics['mean_MSE_low'] = np.mean(mse_data[:, 0:3][target_means_low_mask[:, 0:3]])
        statistics['std_MSE_high'] = np.mean(mse_data[:, 3:6][~target_means_low_mask[:, 3:6]])
        statistics['mean_MSE_high'] = np.mean(mse_data[:, 0:3][~target_means_low_mask[:, 0:3]])

        statistics_raw['std_MSE_all'] = np.mean(mse_data[:, 3:6], axis=1)
        statistics_raw['mean_MSE_all'] = np.mean(mse_data[:, 0:3], axis=1)

        #TODO fix this work around to deal with the boolen mask
        std_low_values = mse_data[:, 3:6][target_means_low_mask[:, 3:6]].reshape(-1, 3)
        mean_low_values = mse_data[:, 0:3][target_means_low_mask[:, 0:3]].reshape(-1, 3)
        std_high_values = mse_data[:, 3:6][~target_means_low_mask[:, 3:6]].reshape(-1, 3)
        mean_high_values = mse_data[:, 0:3][~target_means_low_mask[:, 0:3]].reshape(-1, 3)

        statistics_raw['std_MSE_low'] = np.mean(std_low_values, axis=1)
        statistics_raw['mean_MSE_low'] = np.mean(mean_low_values, axis=1)
        statistics_raw['std_MSE_high'] = np.mean(std_high_values, axis=1)
        statistics_raw['mean_MSE_high'] = np.mean(mean_high_values, axis=1)


        with open(os.path.join(output_dir, "test_statistics.json"), "w") as file:
            json.dump(statistics, file, indent=4, cls=utils.NumpyEncoder)

        with open(os.path.join(output_dir, "test_statistics_raw.json"), "w") as file:
            json.dump(statistics_raw, file, indent=4, cls=utils.NumpyEncoder)



    def calculate_moments(self, condition, targets, label, mode):
        if mode == 'interpolation':
            output_dir = self.interpolation_output_dir
            mse_log = self.interpolation_mse_log
            target_mean_log = self.interpolation_target_moment_log
        elif mode == 'extrapolation':
            output_dir = self.extrapolation_output_dir
            mse_log = self.extrapolation_mse_log
            target_mean_log = self.extrapolation_target_moment_log

        samples = self.model.sample(condition, self.num_samples, self.eta)
        samples = replace_outliers(samples, label, output_dir)

        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)
        target_mean = targets.mean(dim=0)
        target_std = targets.std(dim=0)

        prediction = torch.cat([sample_mean, sample_std], dim=0)
        target = torch.cat([target_mean, target_std], dim=0)

        SE = (prediction - target)**2
        MSE = torch.mean(SE, dim=(1,2))
        full_MSE = torch.mean(SE)
        means = torch.mean(target, dim=(1,2))

        if self.config.inter_extra.plot_samples:
            samples_dir = os.path.join(output_dir, "Samples", label)
            os.makedirs(samples_dir, exist_ok=True)
            utils.plot_samples(samples, samples_dir)

        if self.config.inter_extra.plot_moment_comparison:
            comparison_dir = os.path.join(output_dir, "Comparison")
            utils.plot_moment_comparison(target, prediction, label, comparison_dir, plot_delta=True)

        with open(mse_log, "a") as file:
            file.write("{},{},{},{},{},{},{},{}\n".format(label, *MSE, full_MSE))

        with open(target_mean_log, "a") as file:
            file.write("{},{},{},{},{},{},{}\n".format(label, *means))



    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (condition, targets, label) in tqdm(enumerate(self.interpolation_dataloader), total=len(self.interpolation_dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                self.calculate_moments(condition, targets, label, mode='interpolation')
            self.write_test_statistics(mode='interpolation')

            for idx, (condition, targets, label) in tqdm(enumerate(self.extrapolation_dataloader), total=len(self.extrapolation_dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                self.calculate_moments(condition, targets, label, mode='extrapolation')
            self.write_test_statistics(mode='extrapolation')

class Raf30_test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.device = config.device
        self.checkpoint = torch.load(config.checkpoint)
        self.num_samples = config.num_samples
        self.model = self.checkpoint['model']
        self.model.device = self.device
        self.model.load_state_dict(self.checkpoint['model_params'])
        self.model.move_to_device(self.device)
        self.eta = config.eta
        self.output_dir = os.path.join(config.output_dir,"1_parameter_test")
        self.dataset = dataset.Test_Dataset(self.config, '1_parameter')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)

        for dir in [self.output_dir,
                    os.path.join(self.output_dir, "Prediction"),
                    os.path.join(self.output_dir, "Target")
                    ]:
            os.makedirs(dir, exist_ok=True)

    def calculate_moments(self, condition, targets, label, save_samples=True):
        samples = self.model.sample(condition, self.num_samples, self.eta)
        samples = replace_outliers(samples)

        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)
        target_mean = targets.mean(dim=0)
        target_std = targets.std(dim=0)

        sample_moments = torch.cat([sample_mean, sample_std], dim=0)
        target_moments = torch.cat([target_mean, target_std], dim=0)

        if save_samples:
            samples_dir = os.path.join(self.output_dir, "Samples", label)
            os.makedirs(samples_dir, exist_ok=True)
            utils.plot_samples(samples,samples_dir)

        return sample_moments, target_moments

    def plot_std_prediction(self):
        sample_std_statistics = {}
        target_std_statistics = {}
        self.model.eval()
        with torch.no_grad():
            for i in range(self.config.single_parameter.num_runs):
                for idx, (conditions, targets, label) in enumerate(self.dataloader):
                    conditions = conditions[0].squeeze(dim=0)
                    conditions, targets = conditions.to(self.device), targets.to(self.device)
                    airfoil_name, RE, angle = label.split('_')
                    sample_moments, target_moments = self.calculate_moments(conditions, targets, label, False)
                    sample_std_statistics.setdefault(float(RE), []).append(torch.mean(sample_moments[3:6, :, :]).item())
                    target_std_statistics.setdefault(float(RE), []).append(torch.mean(target_moments[3:6, :, :]).item())

        sample_std_statistics = OrderedDict(sorted(sample_std_statistics.items()))
        target_std_statistics = OrderedDict(sorted(target_std_statistics.items()))

        with open(os.path.join(self.output_dir, "average_std_comparison.json"), 'w') as file:
            json.dump({"model": sample_std_statistics, 'ground_truth': target_std_statistics}, file, indent=4)

        x_values = [key/1000.0 for key, _ in target_std_statistics.items()]
        lines = {"model": sample_std_statistics, 'ground_truth': target_std_statistics}

        for label, line in lines.items():
            for re, stds in line.items():
                line[re] = [min(stds), max(stds), sum(stds)/len(stds)]

        utils.plot_std_curves(lines, x_values, self.output_dir)

    def evaluate(self):
        sample = {}
        target = {}

        self.plot_std_prediction()
        self.model.eval()
        with torch.no_grad():
            for idx, (conditions, targets, label) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                conditions = conditions[0].squeeze(dim=0)
                conditions, targets = conditions.to(self.device), targets.to(self.device)
                print("Case: {}, {}, {}".format(*label.split('_')))
                airfoil_name, RE, angle = label.split('_')
                RE = float(RE)/100
                angle = math.radians(float(angle)/100)
                sample_moments, target_moments = self.calculate_moments(conditions, targets, label)
                sample[RE] = sample_moments
                target[RE] = target_moments

                # TODO refactor this dirty code(this is to plot some samples)
                os.makedirs(os.path.join(self.output_dir, "Channel_samples"), exist_ok=True)
                samples = self.model.sample(conditions, self.num_samples, self.eta)
                p_samples = samples[0:5, 0, :, :]
                x_samples = samples[0:5, 1, :, :]
                y_samples = samples[0:5, 2, :, :]
                utils.plot_samples(p_samples.squeeze(), "Pressure_{}.png".format(RE), os.path.join(self.output_dir, "Channel_samples"))
                utils.plot_samples(x_samples.squeeze(), "U_x_{}.png".format(RE), os.path.join(self.output_dir, "Channel_samples"))
                utils.plot_samples(y_samples.squeeze(), "U_y_{}.png".format(RE), os.path.join(self.output_dir, "Channel_samples"))


        params = [[[key, angle]] for key, value in sorted(sample.items())]
        params = np.array(params)
        sample = [value for key, value in sorted(sample.items())]
        target = [value for key, value in sorted(target.items())]
        sample = torch.unsqueeze(torch.stack(sample), dim=1)
        target = torch.unsqueeze(torch.stack(target), dim=1)
        utils.save_parameter_comparison(sample, params, os.path.join(self.output_dir, "Prediction"))
        utils.save_parameter_comparison(target, params, os.path.join(self.output_dir, "Target"))



class Parameter_Comparison_Test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.device = config.device
        self.checkpoint = torch.load(config.checkpoint)
        self.num_samples = config.num_samples
        self.model = self.checkpoint['model']
        self.model.device = self.device
        self.model.load_state_dict(self.checkpoint['model_params'])
        self.model.move_to_device(self.device)
        self.eta = config.eta
        self.output_dir = os.path.join(config.output_dir, "parameter_comparison")
        self.dataset = dataset.Comparison_Dataset(self.config, mode='mask_only')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)

        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (airfoil_name, conditions, parameters) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                conditions = conditions.to(self.device)
                print("Airfoil {}".format(airfoil_name))
                case_dir = os.path.join(self.output_dir, airfoil_name)
                os.makedirs(case_dir, exist_ok=True)

                conditions = conditions.to(torch.float32)
                num_RE, num_angles, C, H, W = conditions.shape
                # data are array of shape (num_re, num_angles, num_samples, channels , height, width)
                samples_shape = (num_RE, num_angles, self.config.comparison.num_samples, C, H, W)
                samples = torch.tensor(samples_shape)

                for i in range(num_RE):
                    for j in range(num_angles):
                        condition = conditions[i, j]
                        condition = torch.tile(condition, (self.config.comparison.num_samples, 1, 1, 1))
                        samples = self.model.sample(condition, self.num_samples, self.eta)
                        samples[i, j] = samples

                sample_means = samples.mean(dim=2)
                sample_stds = samples.std(dim=2)
                sample_moments = np.concatenate([sample_means, sample_stds], axis=2)

                utils.save_parameter_comparison(sample_moments, parameters, case_dir)


class Sampling_Speed_Test(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.checkpoint = torch.load(config.checkpoint)
        self.num_samples = config.sampling_speed.num_samples
        self.model = self.checkpoint['model']
        self.model.device = self.device
        self.model.load_state_dict(self.checkpoint['model_params'])
        self.model.move_to_device(self.device)
        self.eta = config.eta
        self.output_dir = os.path.join(config.output_dir, "Sampling_speed_test")
        self.dataset = dataset.Test_Dataset(self.config, 'interpolation')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):

        self.model.eval()
        with torch.no_grad():
            sampling_times = {}
            sampling_times['device'] = self.device
            sampling_times_statistics = {}
            sampling_times_statistics['device'] = self.device
            for num_samples in self.num_samples:
                for idx, (conditions, targets, label) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                    condition = conditions[0].squeeze(dim=0)
                    condition = condition.to(self.device)
                    start_time = time.time()
                    samples = self.model.sample(condition, num_samples, self.eta)
                    end_time = time.time()
                    time_elapsed = end_time-start_time
                    sampling_times.setdefault(num_samples, []).append(time_elapsed)

            with open(os.path.join(self.output_dir, 'sampling_times_raw.json'), "w") as file:
                json.dump(sampling_times, file, indent=4)

            for key, item in sampling_times.items():
                if key == "device":
                    continue
                item = np.array(item)
                sampling_times_statistics[key] = {"mean": float(np.mean(item)),
                                                  "std": float(np.std(item))}

            with open(os.path.join(self.output_dir, 'sampling_times_statistics.json'), 'w') as file:
                json.dump(sampling_times_statistics, file, indent=4)


if __name__ == '__main__':
    config = evaluation_config.get_config()
    os.makedirs(config.output_dir, exist_ok=True)
    if config.inter_extrapolation_test:
        inter_extra_test = Inter_Extrapolation_Test(config)
        inter_extra_test.evaluate()
    if config.raf30_test:
        raf30_test = Raf30_test(config)
        raf30_test.evaluate()
    if config.sampling_speed_test:
        sampling_speed_test = Sampling_Speed_Test(config)
        sampling_speed_test.evaluate()
    if config.parameter_comparison_test:
        parameter_comparison_test = Parameter_Comparison_Test(config)
        parameter_comparison_test.evaluate()






