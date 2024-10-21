import math
import os

import numpy as np
import torch

from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from src.models import model_select
from src.data import dataset
from src import utils
from src.evaluation import evaluation_config


class Inter_Extrapolation_Test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.device = config.device
        self.checkpoint = torch.load(config.checkpoint)
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.model = self.model.to(self.device)
        self.output_dir = os.path.join(config.output_dir, "Inter_Extrapolation")
        self.interpolation_dataset = dataset.Test_Dataset(self.config, 'interpolation')
        self.extrapolation_dataset = dataset.Test_Dataset(self.config, 'extrapolation')
        self.interpolation_dataloader = DataLoader(self.interpolation_dataset, batch_size=None, shuffle=False)
        self.extrapolation_dataloader = DataLoader(self.extrapolation_dataset, batch_size=None, shuffle=False)
        self.interpolation_output_dir = os.path.join(self.output_dir, 'interpolation')
        self.extrapolation_output_dir = os.path.join(self.output_dir, 'extrapolation')
        self.mse_file = os.path.join(self.output_dir, "MSE.txt")
        self.target_mean_file = os.path.join(self.output_dir, "target_means.txt")

        for dir in [self.output_dir,
                    self.interpolation_output_dir,
                    self.extrapolation_output_dir,
                    os.path.join(self.interpolation_output_dir, "Samples"),
                    os.path.join(self.interpolation_output_dir, "Comparison"),
                    os.path.join(self.extrapolation_output_dir, "Samples"),
                    os.path.join(self.extrapolation_output_dir, "Comparison")
                    ]:
            os.makedirs(dir, exist_ok=True)

        with open(self.mse_file, "w") as file:
            file.write("airfoil_name,mean_p,mean_Ux,mean_Uy,std_p,std_Ux,std_Uy,full_MSE\n")

        with open(self.target_mean_file, "w") as file:
            file.write("airfoil_name,mean_p,mean_Ux,mean_Uy,std_p,std_Ux,std_Uy\n")




    def calculate_moments(self, condition, targets, label, output_dir):
        samples = self.model.sample(condition, self.device)
        samples = np.array(samples.to('cpu'))
        targets = np.array(targets)

        prediction_mean = np.mean(samples, axis=0)
        prediction_std = np.std(samples, axis=0)
        target_mean = np.mean(targets, axis=0)
        target_std = np.std(targets, axis=0)

        prediction = np.concatenate([prediction_mean, prediction_std], axis=0)
        target = np.concatenate([target_mean, target_std], axis=0)

        SE = (prediction - target)**2
        MSE = np.mean(SE, axis=(1,2))
        full_MSE = np.mean(SE)

        means = np.mean(target, axis=(1,2))


        with open(self.mse_file, "a") as file:
            file.write("{},{},{},{},{},{},{},{}\n".format(label, *MSE, full_MSE))

        with open(self.target_mean_file, "a") as file:
            file.write("{},{},{},{},{},{},{}\n".format(label, *means))

        samples_dir = os.path.join(output_dir, "Samples", label)
        os.makedirs(samples_dir, exist_ok=True)
        comparison_dir = os.path.join(output_dir, "Comparison")
        utils.save_samples(samples, samples_dir)
        utils.plot_comparison(target, prediction, comparison_dir, label)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (condition, targets, label) in enumerate(self.interpolation_dataloader):
                condition = condition.to(self.device)
                print("Interpolation case:{}/{}".format(idx, len(self.interpolation_dataloader)))
                self.calculate_moments(condition, targets, label, self.interpolation_output_dir)

            for idx, (condition, targets, label) in enumerate(self.extrapolation_dataloader):
                condition = condition.to(self.device)
                print("Extrapolation case:{}/{}".format(idx, len(self.extrapolation_dataloader)))
                self.calculate_moments(condition, targets, label, self.extrapolation_output_dir)


class Raf30_test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.checkpoint = torch.load(config.checkpoint)
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.output_dir = os.path.join(config.output_dir,"1_parameter_test")
        self.dataset = dataset.Test_Dataset(self.config, '1_parameter')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)

        for dir in [self.output_dir,
                    os.path.join(self.output_dir, "Prediction"),
                    os.path.join(self.output_dir, "Target")
                    ]:
            os.makedirs(dir, exist_ok=True)

    def calculate_moments(self, condition, targets):
        samples = self.model.sample(condition, 'cpu')
        samples = np.array(samples)
        targets = np.array(targets)

        prediction_mean = np.mean(samples, axis=0)
        prediction_std = np.std(samples, axis=0)
        target_mean = np.mean(targets, axis=0)
        target_std = np.std(targets, axis=0)

        predicted_moments = np.concatenate([prediction_mean, prediction_std], axis=0)
        target_moments = np.concatenate([target_mean, target_std], axis=0)

        return predicted_moments, target_moments

    def plot_std_prediction(self, prediction, target):
        prediction_std = [np.mean(value[5, :, :]) for key, value in sorted(prediction.items())]
        target_std = [np.mean(value[5, :, :]) for key, value in sorted(target.items())]
        x_values = [key/10 for key, value in sorted(target.items())]

        labels = ['Hierarchical VAE', 'Ground truth']
        lines = [prediction_std, target_std]
        utils.plot_std_curves(lines, x_values, labels, 1, 9, self.output_dir)


    def evaluate(self):
        prediction = {}
        target = {}

        self.model.eval()
        with torch.no_grad():
            for idx, (condition, targets, label) in enumerate(self.dataloader):
                print("Case: {}, {}, {}".format(*label.split('_')))
                airfoil_name, RE, angle = label.split('_')
                RE = float(RE)/100
                angle = math.radians(float(angle)/100)
                predicted_moments, target_moments = self.calculate_moments(condition, targets)
                prediction[RE] = predicted_moments
                target[RE] = target_moments

        params = [[[key, angle]] for key, value in sorted(prediction.items())]
        params = np.array(params)

        self.plot_std_prediction(prediction, target)

        prediction = [value for key,value in sorted(prediction.items())]
        target = [value for key, value in sorted(target.items())]

        prediction = np.array(prediction)
        target = np.array(target)
        prediction = np.expand_dims(prediction, axis=1)
        target = np.expand_dims(target, axis=1)

        #params = np.tile(params,2)
        #prediction = np.concatenate([prediction, target], axis=0)

        utils.save_parameter_comparison(prediction, params, os.path.join(self.output_dir, "Prediction"))
        utils.save_parameter_comparison(target, params, os.path.join(self.output_dir, "Target"))


class Parameter_Comparison_Test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.checkpoint = torch.load(config.checkpoint)
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.output_dir = os.path.join(config.output_dir, "parameter_comparison")
        self.dataset = dataset.Comparison_Dataset(self.config, mode='mask_only')
        self.dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False)

        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (airfoil_name, conditions, parameters) in enumerate(self.dataloader):
                print("Airfoil {}, Case:{}/{}".format(airfoil_name, idx, len(self.dataloader)))
                case_dir = os.path.join(self.output_dir, airfoil_name)
                os.makedirs(case_dir, exist_ok=True)

                conditions = conditions.to(torch.float32)
                num_RE, num_angles, C, H, W = conditions.shape
                # data are array of shape (num_re, num_angles, num_samples, channels , height, width)
                predictions_shape = (num_RE, num_angles, self.config.comparison.num_samples, C, H, W)
                predictions = np.ndarray(predictions_shape)

                for i in range(num_RE):
                    for j in range(num_angles):
                        condition = conditions[i, j]
                        condition = torch.tile(condition, (self.config.comparison.num_samples, 1, 1, 1))
                        samples = self.model.sample(condition)
                        predictions[i, j] = samples

                prediction_means = np.mean(predictions, axis=2)
                prediction_stds = np.std(predictions, axis=2)
                predictions_combined = np.concatenate([prediction_means, prediction_stds], axis=2)

                utils.save_parameter_comparison(predictions_combined, parameters, case_dir)


if __name__ == '__main__':
    config = evaluation_config.get_config()
    os.makedirs(config.output_dir, exist_ok=True)
    if config.inter_extrapolation_test:
        inter_extra_test = Inter_Extrapolation_Test(config)
        inter_extra_test.evaluate()
    if config.raf30_test:
        raf30_test = Raf30_test(config)
        raf30_test.evaluate()
    if config.parameter_comparison_test:
        parameter_comparison_test = Parameter_Comparison_Test(config)
        parameter_comparison_test.evaluate()






