import os
from abc import ABC, abstractmethod
from enum import Enum

import torch
from ml_collections import ConfigDict

from src import utils


class TestType(Enum):
    SAMPLING_SPEED = 1
    INTERPOLATION = 2
    EXTRAPOLATION = 3
    DRAG_COEFFICIENT = 4
    PARAMETER_COMPARISON = 5


class Base_Test(ABC):
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
        self.num_samples = config.num_samples
        self.output_dir = config.output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def replace_outliers(data, label, output_dir):
        flat_data = data.flatten(1)
        outlier_indexes = (torch.max(flat_data, 1)[0] > torch.tensor(1.0)) | (
            torch.min(flat_data, 1)[0] < torch.tensor(-1.0)
        )
        mean = torch.mean(data[~outlier_indexes], dim=0)
        if any(outlier_indexes):
            outliers = data[outlier_indexes]
            outlier_dir = os.path.join(output_dir, "Outliers", label)
            os.makedirs(outlier_dir, exist_ok=True)
            utils.plot_samples(outliers, outlier_dir)
        data[outlier_indexes] = mean
        return data

    @abstractmethod
    def evaluate(self):
        pass
