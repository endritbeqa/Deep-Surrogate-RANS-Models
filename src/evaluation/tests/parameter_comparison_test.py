import os

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.data import dataset
from src.evaluation.tests.base_test import Base_Test


class Parameter_Comparison_Test(Base_Test):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
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

