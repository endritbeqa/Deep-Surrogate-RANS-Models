

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.data import dataset
from src.evaluation import drag_coefficient_utils
from src.evaluation.tests.base_test import Base_Test


class Drag_Coefficient_Test(Base_Test):
    def __init__(self, config):
        super().__init__(config)
        self.num_runs = config.drag_coefficient.num_runs
        self.num_buckets = config.drag_coefficient.num_buckets
        self.num_samples = config.num_samples
        self.dataset = dataset.Test_Dataset(self.config, 'interpolation')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)


    def calculate_drag(self, airfoil_shape, samples, AoA, velocity, cell_length):
        drag_coefficients = []
        for sample in samples:
            drag_coefficients.append(drag_coefficient_utils.get_lift_drag_coef(airfoil_shape, sample, AoA, velocity, cell_length=cell_length))
        drag_coefficients = np.array(drag_coefficients)
        return drag_coefficients


    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (conditions, targets, label) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                data = torch.concat((conditions, targets), dim=1)
                data = drag_coefficient_utils.reverse_preprocess_data(data, True, True, True)
                data = torch.tensor(data)
                conditions, targets = data[:, 0:3, :, :], data[:, 3:, :, :]

                condition = conditions[0].squeeze(dim=0)
                airfoil_shape = condition[2]
                airfoil_name, AoA, velocity = label.split("_")
                condition = condition.to(self.device)
                dragCoeff_targets = self.calculate_drag(airfoil_shape, targets, float(AoA), float(velocity), cell_length=2/64)
                dragCoeff_predictions = []
                for i in range(self.num_runs):
                    samples = self.model.sample(condition, self.num_samples, self.eta)
                    samples = samples.detach().cpu()
                    dragCoeff_predictions.append(self.calculate_drag(airfoil_shape, samples, float(AoA), float(velocity), cell_length=2/64))
                utils.plot_drag_coefficient_distribution(label, dragCoeff_targets, dragCoeff_predictions, self.output_dir, self.num_buckets)
