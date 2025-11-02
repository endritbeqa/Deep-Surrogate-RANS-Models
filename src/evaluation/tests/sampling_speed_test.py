import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import dataset
from src.evaluation.tests.base_test import Base_Test


class Sampling_Speed_Test(Base_Test):

    def __init__(self, config):
        super().__init__(config)
        self.num_samples = config.sampling_speed.num_samples
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
