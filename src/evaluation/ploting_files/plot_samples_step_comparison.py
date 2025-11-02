import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from project_definitions import PROJECT_ROOT_DIR
from src import utils
from src.data import dataset
from src.evaluation.evaluation_config import get_config


class Step_Sample_Plotter():

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model_name = "FactFormer"

        steps = [20, 50, 100, 150, 200]

        self.checkpoints = { num_steps: torch.load(f"{PROJECT_ROOT_DIR}/results/res32/step_ablation_study/{self.model_name}/{num_steps}_steps/checkpoints/Final.pth")
                             for num_steps in steps }


        self.models = {num_steps: checkpoint["model"] for num_steps, checkpoint in self.checkpoints.items()}

        for num_steps, model in self.models.items():
            model.device = self.device
            model.move_to_device(self.device)
            model.load_state_dict(self.checkpoints[num_steps]["model_params"])


        self.num_samples = config.num_samples
        self.eta = config.eta
        self.interpolation_dataset = dataset.Test_Dataset(self.config, 'interpolation')
        self.extrapolation_dataset = dataset.Test_Dataset(self.config, 'extrapolation')
        self.interpolation_dataloader = DataLoader(self.interpolation_dataset, batch_size=None, shuffle=False)
        self.extrapolation_dataloader = DataLoader(self.extrapolation_dataset, batch_size=None, shuffle=False)
        self.output_dir = f"{PROJECT_ROOT_DIR}/results/Graphs/Step_Comparison_Samples/{self.model_name}"
        self.interpolation_output_dir = os.path.join(self.output_dir, 'interpolation')
        self.extrapolation_output_dir = os.path.join(self.output_dir, 'extrapolation')

        for dir in [self.output_dir,
                    self.interpolation_output_dir,
                    self.extrapolation_output_dir,
                    os.path.join(self.interpolation_output_dir, "Pressure"),
                    os.path.join(self.interpolation_output_dir, "X-Velocity"),
                    os.path.join(self.interpolation_output_dir, "Y-Velocity"),
                    os.path.join(self.extrapolation_output_dir, "Pressure"),
                    os.path.join(self.extrapolation_output_dir, "X-Velocity"),
                    os.path.join(self.extrapolation_output_dir, "Y-Velocity"),
                    ]:
            os.makedirs(dir, exist_ok=True)


    def plot(self):
        row_labels = ["Target", "20", "50", "100", "150", "200"]
        for num_steps, model in self.models.items():
            model.eval()

        with torch.no_grad():
            for idx, (condition, targets, label) in tqdm(enumerate(self.interpolation_dataloader),
                                                         total=len(self.interpolation_dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                targets = targets[:self.num_samples]
                samples = [targets]
                for num_steps, model in self.models.items():
                    samples.append(model.sample(condition, self.num_samples, self.eta))
                samples = torch.stack(samples, dim=0)
                utils.plot_samples_different_models(samples,label, row_labels, 0, self.interpolation_output_dir) # Dont touch order of stacking
                utils.plot_samples_different_models(samples, label,row_labels,  1, self.interpolation_output_dir)
                utils.plot_samples_different_models(samples, label,row_labels,  2, self.interpolation_output_dir)

            for idx, (condition, targets, label) in tqdm(enumerate(self.extrapolation_dataloader),
                                                         total=len(self.extrapolation_dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                targets = targets[:self.num_samples]
                samples = [targets]
                for num_steps, model in self.models.items():
                    samples.append(model.sample(condition, self.num_samples, self.eta))
                samples = torch.stack(samples, dim=0)
                utils.plot_samples_different_models(samples, label,row_labels,  0, self.extrapolation_output_dir)  # Dont touch order of stacking
                utils.plot_samples_different_models(samples, label,row_labels,  1, self.extrapolation_output_dir)
                utils.plot_samples_different_models(samples, label,row_labels,  2, self.extrapolation_output_dir)



if __name__ == '__main__':
    config = get_config()
    plotter = Step_Sample_Plotter(config)
    plotter.plot()