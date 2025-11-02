import json
import os

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.data import dataset
from src.evaluation.tests.base_test import Base_Test


class Inter_Extrapolation_Test(Base_Test):

    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.dataset = dataset.Test_Dataset(self.config, "interpolation")
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)
        self.mse_log = os.path.join(self.output_dir, "MSE.txt")
        self.target_moment_log = os.path.join(self.output_dir, "target_moments.txt")

        with open(self.mse_log, "w") as file:
            file.write("airfoil_name,mean_p,mean_Ux,mean_Uy,std_p,std_Ux,std_Uy,full_MSE\n")

        with open(self.target_moment_log, "w") as file:
            file.write("airfoil_name,mean_p,mean_Ux,mean_Uy,std_p,std_Ux,std_Uy\n")

    def write_test_statistics(self):
        mse_data = ((np.genfromtxt(self.mse_log, delimiter=",", skip_header=1))[:, 1:7]).astype(np.float32)
        target_means_data = ((np.genfromtxt(self.target_moment_log, delimiter=",", skip_header=1))[:, 1:]).astype(np.float32)

        statistics = {}
        statistics_raw = {}

        target_means_low_mask = np.mean(target_means_data[:, 3:6], axis=1) < 5e-3
        target_means_low_mask = np.tile(target_means_low_mask[:, np.newaxis], (1, 6))

        statistics["std_MSE_all"] = np.mean(mse_data[:, 3:6])
        statistics["mean_MSE_all"] = np.mean(mse_data[:, 0:3])
        statistics["std_MSE_low"] = np.mean(mse_data[:, 3:6][target_means_low_mask[:, 3:6]])
        statistics["mean_MSE_low"] = np.mean(mse_data[:, 0:3][target_means_low_mask[:, 0:3]])
        statistics["std_MSE_high"] = np.mean(mse_data[:, 3:6][~target_means_low_mask[:, 3:6]])
        statistics["mean_MSE_high"] = np.mean(mse_data[:, 0:3][~target_means_low_mask[:, 0:3]])

        statistics_raw["std_MSE_all"] = np.mean(mse_data[:, 3:6], axis=1)
        statistics_raw["mean_MSE_all"] = np.mean(mse_data[:, 0:3], axis=1)

        # TODO fix this work around to deal with the boolen mask
        std_low_values = mse_data[:, 3:6][target_means_low_mask[:, 3:6]].reshape(-1, 3)
        mean_low_values = mse_data[:, 0:3][target_means_low_mask[:, 0:3]].reshape(-1, 3)
        std_high_values = mse_data[:, 3:6][~target_means_low_mask[:, 3:6]].reshape(-1, 3)
        mean_high_values = mse_data[:, 0:3][~target_means_low_mask[:, 0:3]].reshape(-1, 3)

        statistics_raw["std_MSE_low"] = np.mean(std_low_values, axis=1)
        statistics_raw["mean_MSE_low"] = np.mean(mean_low_values, axis=1)
        statistics_raw["std_MSE_high"] = np.mean(std_high_values, axis=1)
        statistics_raw["mean_MSE_high"] = np.mean(mean_high_values, axis=1)

        with open(os.path.join(self.output_dir, "test_statistics.json"), "w") as file:
            json.dump(statistics, file, indent=4, cls=utils.NumpyEncoder)

        with open(os.path.join(self.output_dir, "test_statistics_raw.json"), "w") as file:
            json.dump(statistics_raw, file, indent=4, cls=utils.NumpyEncoder)

    def calculate_moments(self, condition, targets, label):
        samples = self.model.sample(condition, self.num_samples, self.eta)
        samples = self.replace_outliers(samples, label, self.output_dir)

        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)
        target_mean = targets.mean(dim=0)
        target_std = targets.std(dim=0)

        prediction = torch.cat([sample_mean, sample_std], dim=0)
        target = torch.cat([target_mean, target_std], dim=0)

        SE = (prediction - target) ** 2
        MSE = torch.mean(SE, dim=(1, 2))
        full_MSE = torch.mean(SE)
        means = torch.mean(target, dim=(1, 2))

        if self.config.inter_extra.plot_samples:
            samples_dir = os.path.join(self.output_dir, "Samples", label)
            os.makedirs(samples_dir, exist_ok=True)
            utils.plot_samples(samples, samples_dir)

        if self.config.inter_extra.plot_moment_comparison:
            comparison_dir = os.path.join(self.output_dir, "Comparison")
            utils.plot_moment_comparison(
                target, prediction, label, comparison_dir, plot_delta=True
            )

        with open(self.mse_log, "a") as file:
            file.write("{},{},{},{},{},{},{},{}\n".format(label, *MSE, full_MSE))

        with open(self.target_moment_log, "a") as file:
            file.write("{},{},{},{},{},{},{}\n".format(label, *means))

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (condition, targets, label) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                self.calculate_moments(condition, targets, label)
            self.write_test_statistics()


