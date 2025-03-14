import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import dataset
from src import utils
from project_definitions import PROJECT_ROOT_DIR
from src.evaluation.evaluation_config import get_config


class Sample_Plotter():

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.checkpoints = {"Swin":f"{PROJECT_ROOT_DIR}/results/res32/Swin_test/run_1/checkpoints/Final.pth",
                            "DiT" :f"{PROJECT_ROOT_DIR}/results/res32/DiT_test/run_1/checkpoints/Final.pth",
                            "FactFormer":f"{PROJECT_ROOT_DIR}/results/res32/dataset_size_ablation_study/FactFormer/FactFormer_20_snapshots/checkpoints/Final.pth"}


        self.Swin_checkpoint = torch.load(self.checkpoints["Swin"])
        self.FactFormer_checkpoint = torch.load(self.checkpoints["FactFormer"])
        self.DiT_checkpoint = torch.load(self.checkpoints["DiT"])


        self.Swin = self.Swin_checkpoint['model']
        self.FactFormer = self.FactFormer_checkpoint['model']
        self.DiT = self.DiT_checkpoint['model']

        self.Swin.device = self.device
        self.Swin.move_to_device(self.device)
        self.FactFormer.device = self.device
        self.FactFormer.move_to_device(self.device)
        self.DiT.device = self.device
        self.DiT.move_to_device(self.device)

        self.Swin.load_state_dict(self.Swin_checkpoint['model_params'])
        self.FactFormer.load_state_dict(self.FactFormer_checkpoint['model_params'])
        self.DiT.load_state_dict(self.DiT_checkpoint['model_params'])


        self.num_samples = config.num_samples
        self.eta = config.eta
        self.interpolation_dataset = dataset.Test_Dataset(self.config, 'interpolation')
        self.extrapolation_dataset = dataset.Test_Dataset(self.config, 'extrapolation')
        self.interpolation_dataloader = DataLoader(self.interpolation_dataset, batch_size=None, shuffle=False)
        self.extrapolation_dataloader = DataLoader(self.extrapolation_dataset, batch_size=None, shuffle=False)
        self.output_dir = f"{PROJECT_ROOT_DIR}/results/Graphs/Samples"
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
        self.DiT.eval()
        self.Swin.eval()
        self.FactFormer.eval()
        with torch.no_grad():
            for idx, (condition, targets, label) in tqdm(enumerate(self.interpolation_dataloader),
                                                         total=len(self.interpolation_dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                targets = targets[:self.num_samples]
                samples_DiT = self.DiT.sample(condition, self.num_samples, self.eta)
                samples_Swin = self.Swin.sample(condition, self.num_samples, self.eta)
                samples_FactFormer = self.FactFormer.sample(condition, self.num_samples, self.eta)
                samples = torch.stack((targets, samples_FactFormer, samples_Swin, samples_DiT), dim=0)
                utils.plot_samples_different_models(samples,label, 0, self.interpolation_output_dir) # Dont touch order of stacking
                utils.plot_samples_different_models(samples, label, 1, self.interpolation_output_dir)
                utils.plot_samples_different_models(samples, label, 2, self.interpolation_output_dir)

            for idx, (condition, targets, label) in tqdm(enumerate(self.extrapolation_dataloader),
                                                         total=len(self.extrapolation_dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                targets = targets[:self.num_samples]
                samples_DiT = self.DiT.sample(condition, self.num_samples, self.eta)
                samples_Swin = self.Swin.sample(condition, self.num_samples, self.eta)
                samples_FactFormer = self.FactFormer.sample(condition, self.num_samples, self.eta)
                samples = torch.stack((targets, samples_FactFormer, samples_Swin, samples_DiT), dim=0)
                utils.plot_samples_different_models(samples, label, 0, self.extrapolation_output_dir)  # Dont touch order of stacking
                utils.plot_samples_different_models(samples, label, 1, self.extrapolation_output_dir)
                utils.plot_samples_different_models(samples, label, 2, self.extrapolation_output_dir)



if __name__ == '__main__':
    config = get_config()
    plotter = Sample_Plotter(config)
    plotter.plot()