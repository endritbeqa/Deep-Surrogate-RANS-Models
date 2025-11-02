import json
import math
import os
from collections import OrderedDict

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.data import dataset
from src.evaluation.tests.base_test import Base_Test


class Raf30_test(Base_Test):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.dataset = dataset.Test_Dataset(self.config, "1_parameter")
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)

        for dir in [
            self.output_dir,
            os.path.join(self.output_dir, "Prediction"),
            os.path.join(self.output_dir, "Target"),
        ]:
            os.makedirs(dir, exist_ok=True)

    def calculate_moments(self, condition, targets, label, save_samples=True):
        samples = self.model.sample(condition, self.num_samples, self.eta)
        samples = self.replace_outliers(samples, label, self.output_dir)

        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)
        target_mean = targets.mean(dim=0)
        target_std = targets.std(dim=0)

        sample_moments = torch.cat([sample_mean, sample_std], dim=0)
        target_moments = torch.cat([target_mean, target_std], dim=0)

        if save_samples:
            samples_dir = os.path.join(self.output_dir, "Samples", label)
            os.makedirs(samples_dir, exist_ok=True)
            utils.plot_samples(samples, samples_dir)

        return sample_moments, target_moments

    def plot_std_prediction(self):
        sample_std_statistics = {}
        target_std_statistics = {}
        self.model.eval()
        with torch.no_grad():
            for i in range(self.config.single_parameter.num_runs):
                for idx, (conditions, targets, label) in enumerate(self.dataloader):
                    conditions = conditions[0].squeeze(dim=0)
                    conditions, targets = conditions.to(self.device), targets.to(
                        self.device
                    )
                    airfoil_name, RE, angle = label.split("_")
                    sample_moments, target_moments = self.calculate_moments(
                        conditions, targets, label, False
                    )
                    sample_std_statistics.setdefault(float(RE), []).append(
                        torch.mean(sample_moments[3:6, :, :]).item()
                    )
                    target_std_statistics.setdefault(float(RE), []).append(
                        torch.mean(target_moments[3:6, :, :]).item()
                    )

        sample_std_statistics = OrderedDict(sorted(sample_std_statistics.items()))
        target_std_statistics = OrderedDict(sorted(target_std_statistics.items()))

        with open(
            os.path.join(self.output_dir, "average_std_comparison.json"), "w"
        ) as file:
            json.dump(
                {"model": sample_std_statistics, "ground_truth": target_std_statistics},
                file,
                indent=4,
            )

        x_values = [key / 1000.0 for key, _ in target_std_statistics.items()]
        lines = {"model": sample_std_statistics, "ground_truth": target_std_statistics}

        for label, line in lines.items():
            for re, stds in line.items():
                line[re] = [min(stds), max(stds), sum(stds) / len(stds)]

        utils.plot_std_curves(lines, x_values, self.output_dir)

    def evaluate(self):
        sample = {}
        target = {}

        self.plot_std_prediction()
        self.model.eval()
        with torch.no_grad():
            for idx, (conditions, targets, label) in tqdm(
                enumerate(self.dataloader), total=len(self.dataloader)
            ):
                conditions = conditions[0].squeeze(dim=0)
                conditions, targets = conditions.to(self.device), targets.to(
                    self.device
                )
                print("Case: {}, {}, {}".format(*label.split("_")))
                airfoil_name, RE, angle = label.split("_")
                RE = float(RE) / 100
                angle = math.radians(float(angle) / 100)
                sample_moments, target_moments = self.calculate_moments(
                    conditions, targets, label
                )
                sample[RE] = sample_moments
                target[RE] = target_moments

                # TODO refactor this dirty code(this is to plot some samples)
                os.makedirs(
                    os.path.join(self.output_dir, "Channel_samples"), exist_ok=True
                )
                samples = self.model.sample(conditions, self.num_samples, self.eta)
                p_samples = samples[0:5, 0, :, :]
                x_samples = samples[0:5, 1, :, :]
                y_samples = samples[0:5, 2, :, :]
                utils.plot_samples(
                    p_samples.squeeze(),
                    "Pressure_{}.png".format(RE),
                    os.path.join(self.output_dir, "Channel_samples"),
                )
                utils.plot_samples(
                    x_samples.squeeze(),
                    "U_x_{}.png".format(RE),
                    os.path.join(self.output_dir, "Channel_samples"),
                )
                utils.plot_samples(
                    y_samples.squeeze(),
                    "U_y_{}.png".format(RE),
                    os.path.join(self.output_dir, "Channel_samples"),
                )

        params = [[[key, angle]] for key, value in sorted(sample.items())]
        params = np.array(params)
        sample = [value for key, value in sorted(sample.items())]
        target = [value for key, value in sorted(target.items())]
        sample = torch.unsqueeze(torch.stack(sample), dim=1)
        target = torch.unsqueeze(torch.stack(target), dim=1)
        utils.save_parameter_comparison(
            sample, params, os.path.join(self.output_dir, "Prediction")
        )
        utils.save_parameter_comparison(
            target, params, os.path.join(self.output_dir, "Target")
        )
