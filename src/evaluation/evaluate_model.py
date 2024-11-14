import json
import math
import os
import time

import numpy as np
import torch

from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import dataset
from src import utils
from src.evaluation import evaluation_config


class Inter_Extrapolation_Test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.device = config.device
        self.checkpoint = torch.load(config.checkpoint)
        self.num_samples = config.num_samples
        self.model = self.checkpoint['model']
        self.model.device = self.device
        self.model.load_state_dict(self.checkpoint['model_params'])
        self.model.move_to_device()
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
                    os.path.join(self.extrapolation_output_dir, "Samples"),
                    os.path.join(self.extrapolation_output_dir, "Comparison")
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

        target_means_low_mask = target_means_data[:, 3:6] < 5e-3
        target_means_low_mask = np.tile(target_means_low_mask, 6).reshape((-1, 6))

        statistics['std_MSE_all'] = np.mean(mse_data[:, 3:6])
        statistics['mean_MSE_all'] = np.mean(mse_data[:, 0:3])
        statistics['std_MSE_low'] = np.mean(mse_data[:, 3:6][target_means_low_mask[:, 3:6]])
        statistics['mean_MSE_low'] = np.mean(mse_data[:, 0:3][target_means_low_mask[:, 0:3]])
        statistics['std_MSE_high'] = np.mean(mse_data[:, 3:6][~target_means_low_mask[:, 3:6]])
        statistics['mean_MSE_high'] = np.mean(mse_data[:, 0:3][~target_means_low_mask[:, 0:3]])

        with open(os.path.join(output_dir, "test_statistics.json"), "w") as file:
            json.dump(statistics, file, indent=4, cls=utils.NumpyEncoder)



    def calculate_moments(self, condition, targets, label, mode):
        if mode == 'interpolation':
            output_dir = self.interpolation_output_dir
            mse_log = self.interpolation_mse_log
            target_mean_log = self.interpolation_target_moment_log
        elif mode == 'extrapolation':
            output_dir = self.extrapolation_output_dir
            mse_log = self.extrapolation_mse_log
            target_mean_log = self.extrapolation_target_moment_log

        samples = self.model.sample(condition, self.num_samples)

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


        with open(mse_log, "a") as file:
            file.write("{},{},{},{},{},{},{},{}\n".format(label, *MSE, full_MSE))

        with open(target_mean_log, "a") as file:
            file.write("{},{},{},{},{},{},{}\n".format(label, *means))

        samples_dir = os.path.join(output_dir, "Samples", label)
        os.makedirs(samples_dir, exist_ok=True)
        comparison_dir = os.path.join(output_dir, "Comparison")
        #utils.save_samples(samples, samples_dir)
        utils.plot_comparison(target, prediction, comparison_dir, label)

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
        self.model.move_to_device()
        self.output_dir = os.path.join(config.output_dir,"1_parameter_test")
        self.dataset = dataset.Test_Dataset(self.config, '1_parameter')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)
        self.model = self.model.to(self.device)

        for dir in [self.output_dir,
                    os.path.join(self.output_dir, "Prediction"),
                    os.path.join(self.output_dir, "Target")
                    ]:
            os.makedirs(dir, exist_ok=True)

    def calculate_moments(self, condition, targets, label):
        samples = self.model.sample(condition, self.num_samples)

        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)
        target_mean = targets.mean(dim=0)
        target_std = targets.std(dim=0)

        sample_moments = torch.cat([sample_mean, sample_std], dim=0)
        target_moments = torch.cat([target_mean, target_std], dim=0)

        samples_dir = os.path.join(self.output_dir, "Samples", label)
        os.makedirs(samples_dir, exist_ok=True)
        utils.save_samples(samples, samples_dir)

        return sample_moments, target_moments

    def plot_std_prediction(self, sample, target):
        sample_std = [torch.mean(value[3:6, :, :]).item() for key, value in sorted(sample.items())]
        target_std = [torch.mean(value[3:6, :, :]).item() for key, value in sorted(target.items())]
        x_values = [key/10 for key, value in sorted(target.items())]

        labels = ['Model', 'Ground Truth']
        lines = [sample_std, target_std]
        raf_30_curves = {"Model":sample_std, "Ground Truth":target_std}
        with open(os.path.join(self.output_dir, "average_std_comparison.json"), 'w') as file:
            json.dump(raf_30_curves, file, indent=4)
        utils.plot_std_curves(lines, x_values, labels, 1, 9, self.output_dir)


    def evaluate(self):
        sample = {}
        target = {}

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

        params = [[[key, angle]] for key, value in sorted(sample.items())]
        params = np.array(params)

        self.plot_std_prediction(sample, target)

        sample = [value for key, value in sorted(sample.items())]
        target = [value for key, value in sorted(target.items())]

        sample = torch.stack(sample)
        target = torch.stack(target)
        sample = torch.unsqueeze(sample, dim=1)
        target = torch.unsqueeze(target, dim=1)

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
        self.model.move_to_device()
        self.output_dir = os.path.join(config.output_dir, "parameter_comparison")
        self.dataset = dataset.Comparison_Dataset(self.config, mode='mask_only')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)
        self.device = self.model.to(self.device)

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
                        samples = self.model.sample(condition, self.num_samples)
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
        self.num_samples = config.num_samples
        self.model = self.checkpoint['model']
        self.model.device = self.device
        self.model.load_state_dict(self.checkpoint['model_params'])
        self.model.move_to_device()
        self.output_dir = os.path.join(config.output_dir, "Sampling_speed_test")
        self.dataset = dataset.Test_Dataset(self.config, 'interpolation')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)
        self.model = self.model.to(self.device)
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
                    samples = self.model.sample(condition, num_samples)
                    end_time = time.time()
                    time_elapsed = end_time-start_time
                    sampling_times.setdefault(num_samples, []).append(time_elapsed)

            with open(os.path.join(self.output_dir, 'sampling_times_raw.json'), "w") as file:
                json.dump(sampling_times, file, indent=4)

            for key, item in sampling_times.items():
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






