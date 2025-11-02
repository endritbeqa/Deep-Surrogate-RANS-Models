import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from project_definitions import PROJECT_ROOT_DIR
from src import utils
from src.data import dataset
from src.evaluation.evaluation_config import get_config


def replace_outliers(data):
    flat_data = data.flatten(1)
    outlier_indexes = (torch.max(flat_data, 1)[0] > torch.tensor(1.0)) | (
            torch.min(flat_data, 1)[0] < torch.tensor(-1.0))
    mean = torch.mean(data[~outlier_indexes], dim=0)
    data[outlier_indexes] = mean
    return data

class Moment_Plotter():


    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.checkpoints = {"Swin":f"{PROJECT_ROOT_DIR}/results/res32/Swin/run_1/checkpoints/Final.pth",
                            "DiT" :f"{PROJECT_ROOT_DIR}/results/res32/DiT/run_1/checkpoints/Final.pth",
                            "FactFormer":f"{PROJECT_ROOT_DIR}/results/res32/dataSize_ablation_study/FactFormer/20_snapshots/checkpoints/Final.pth"}


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
        self.output_dir = f"{PROJECT_ROOT_DIR}/results/Graphs/Moments"
        self.interpolation_output_dir = os.path.join(self.output_dir, 'interpolation')
        self.extrapolation_output_dir = os.path.join(self.output_dir, 'extrapolation')

        for dir in [self.output_dir,
                    self.interpolation_output_dir,
                    self.extrapolation_output_dir,
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

                samples_DiT = replace_outliers(samples_DiT)
                samples_Swin = replace_outliers(samples_Swin)
                samples_FactFormer = replace_outliers(samples_FactFormer)

                target_mean = targets.mean(dim=0)
                target_std = targets.std(dim=0)
                target_moments = torch.cat([target_mean, target_std], dim=0)

                DiT_mean = samples_DiT.mean(dim=0)
                DiT_std = samples_DiT.std(dim=0)
                DiT_moments = torch.cat([DiT_mean, DiT_std], dim=0)

                Swin_mean = samples_Swin.mean(dim=0)
                Swin_std = samples_Swin.std(dim=0)
                Swin_moments = torch.cat([Swin_mean, Swin_std], dim=0)

                FactFormer_mean = samples_FactFormer.mean(dim=0)
                FactFormer_std = samples_FactFormer.std(dim=0)
                FactFormer_moments = torch.cat([FactFormer_mean, FactFormer_std], dim=0)


                moments = torch.stack((target_moments, FactFormer_moments, Swin_moments, DiT_moments), dim=0)
                utils.plot_moment_comparison_models(moments, label, self.interpolation_output_dir)


            for idx, (condition, targets, label) in tqdm(enumerate(self.extrapolation_dataloader),
                                                         total=len(self.extrapolation_dataloader)):
                condition = condition[0].squeeze(dim=0)
                condition, targets = condition.to(self.device), targets.to(self.device)
                targets = targets[:self.num_samples]
                samples_DiT = self.DiT.sample(condition, self.num_samples, self.eta)
                samples_Swin = self.Swin.sample(condition, self.num_samples, self.eta)
                samples_FactFormer = self.FactFormer.sample(condition, self.num_samples, self.eta)

                samples_DiT = replace_outliers(samples_DiT)
                samples_Swin = replace_outliers(samples_Swin)
                samples_FactFormer = replace_outliers(samples_FactFormer)

                target_mean = targets.mean(dim=0)
                target_std = targets.std(dim=0)
                target_moments = torch.cat([target_mean, target_std], dim=0)

                DiT_mean = samples_DiT.mean(dim=0)
                DiT_std = samples_DiT.std(dim=0)
                DiT_moments = torch.cat([DiT_mean, DiT_std], dim=0)

                Swin_mean = samples_Swin.mean(dim=0)
                Swin_std = samples_Swin.std(dim=0)
                Swin_moments = torch.cat([Swin_mean, Swin_std], dim=0)

                FactFormer_mean = samples_FactFormer.mean(dim=0)
                FactFormer_std = samples_FactFormer.std(dim=0)
                FactFormer_moments = torch.cat([FactFormer_mean, FactFormer_std], dim=0)

                moments = torch.stack((target_moments, FactFormer_moments, Swin_moments, DiT_moments), dim=0)
                utils.plot_moment_comparison_models(moments, label, self.extrapolation_output_dir)



if __name__ == '__main__':
    config = get_config()
    plotter = Moment_Plotter(config)
    plotter.plot()