import math
import os

import numpy as np
import torch

from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from src.models import model_select
from src.data import dataset
from src import loss
from src import utils
from src.evaluation import evaluation_config
from src import Noise_scheduler


class Model_Test(object):
    def __init__(self, config: ConfigDict):

        self.config = config
        self.checkpoint = torch.load(config.checkpoint)
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.output_dir = config.output_dir
        self.test_dataset = dataset.Test_Dataset(self.config)
        self.test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)
        self.noise_scheduler = Noise_scheduler.get_noise_scheduler(self.train_config)
        self.device = 'cpu'

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test_results', 'mean_comparisons'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test_results', 'std_comparisons'), exist_ok=True)

    def sinusoidal_embedding(self, timesteps, dim):
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embeddings)
        embeddings = embeddings.to(self.device)
        timesteps = timesteps.to(self.device)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def predict(self):
        self.model.eval()
        with torch.no_grad():

            for idx, (condition, targets, label) in enumerate(self.test_dataloader):
                print(idx)
                condition = torch.squeeze(condition)
                targets = torch.squeeze(targets)
                B, C, H, W = condition.shape

                x_t = torch.randn((B, C, H, W), device=self.device)

                for t in reversed(range(self.train_config.timesteps)):
                    t_tensor = torch.tensor([t] * B, device=self.device)
                    t_emb = self.sinusoidal_embedding(t_tensor, math.prod(self.model_config.swin_decoder.time_embedding))
                    t_emb = t_emb.view(B, *self.model_config.swin_decoder.time_embedding)

                    predicted_noise = self.model(condition, x_t, t_emb)



                    alpha_bar_t = self.noise_scheduler.get_alpha_bar(t_tensor).view(-1, 1, 1, 1)
                    alpha_bar_t_minus_1 = self.noise_scheduler.get_alpha_bar(torch.tensor([t - 1], device=self.device)).view(
                        -1, 1, 1, 1)

                    if t > 0:
                        noise = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
                        x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                        x_t = x_t + torch.sqrt(1 - alpha_bar_t_minus_1) * noise
                    else:
                        x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

                samples = x_t
                samples = np.array(samples)
                targets = np.array(targets)

                utils.save_images(samples, "{}/samples".format(self.output_dir), label, 0)

                prediction_mean = np.mean(samples, axis=0)
                prediction_std = np.std(samples, axis=0)
                target_mean = np.mean(targets, axis=0)
                target_std = np.std(targets, axis=0)

                # dirty fix with the label[0]
                utils.plot_comparison(target_mean, prediction_mean, os.path.join(self.output_dir, 'test_results','mean_comparisons'), label[0])
                utils.plot_comparison(target_std, prediction_std, os.path.join(self.output_dir, 'test_results', 'std_comparisons'), label[0])

class Parameter_comparison(object):
    def __init__(self, config: ConfigDict):

        self.config = config
        self.checkpoint = torch.load(config.checkpoint)
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.output_dir = os.path.join(config.output_dir,"parameter_comparison")
        self.test_dataset = dataset.Comparison_Dataset(self.config)
        self.test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)

        os.makedirs(self.output_dir, exist_ok=True)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (airfoil_name,conditions, labels) in enumerate(self.test_dataloader):
                print(idx)
                conditions = conditions.to(torch.float32)
                conditions = torch.squeeze(conditions, dim=0) #Remove the batch dimension
                labels = torch.squeeze(labels, dim=0)
                num_RE, num_angles, C, H, W = conditions.shape
                predictions_shape = (num_RE, num_angles, self.config.comparison.num_samples, C, H, W)
                predictions = np.ndarray(predictions_shape)

                for i in range(num_RE):
                    for j in range(num_angles):
                        condition = conditions[i, j]
                        condition = torch.tile(condition, (self.config.comparison.num_samples, 1, 1, 1))
                        samples = self.model.sample(condition)
                        predictions[i, j] = samples

                # data are array of shape (num_re, num_angles, B, channels , height, width)

                prediction_means = np.mean(predictions, axis=2)
                prediction_stds = np.std(predictions, axis=2)

                predictions_combined = np.concatenate((prediction_means, prediction_stds), axis=2)
                utils.save_parameter_comparison(predictions_combined, labels, os.path.join(self.output_dir, airfoil_name[0]))




if __name__ == '__main__':
    config = evaluation_config.get_config()
    model_test = Model_Test(config)
    parameter_comparison = Parameter_comparison(config)
    model_test.predict()
    #parameter_comparison.predict()