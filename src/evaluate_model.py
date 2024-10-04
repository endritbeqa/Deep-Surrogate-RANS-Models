import json
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
from src import evaluation_config

class Model_Test(object):
    def __init__(self, config: ConfigDict):

        self.config = config
        self.checkpoint = torch.load(config.checkpoint)
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.output_dir = config.output_dir
        self.loss_func = loss.get_loss_function(config.loss)
        self.test_dataset = dataset.Test_Dataset(self.config)
        self.test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)

        os.makedirs(self.output_dir, exist_ok=True)


    def predict(self):

        target_means = []
        target_stds = []
        sample_means = []
        sample_stds = []

        self.model.eval()
        with torch.no_grad():

            for idx, (condition, targets, label) in enumerate(self.test_dataloader):
                print(idx)
                samples = []
                for i in range(25):
                    sample = self.model.sample(condition)

                    samples.append(sample)

                samples = np.array(samples)
                samples = np.squeeze(samples)

                targets = targets.numpy()
                targets = np.squeeze(targets)
                utils.save_images(samples, "{}/samples".format(self.output_dir), label, 0)

                sample_mean = np.mean(samples, axis=0)
                sample_std = np.std(samples, axis=0)
                target_mean = np.mean(targets, axis=0)
                target_std = np.std(targets, axis=0)

                target_means.append(np.squeeze(target_mean))
                target_stds.append(np.squeeze(target_std))
                sample_means.append(np.squeeze(sample_mean))
                sample_stds.append(np.squeeze(sample_std))

            target_means = np.array(target_means)
            target_stds = np.array(target_stds)
            sample_means = np.array(sample_means)
            sample_stds = np.array(sample_stds)

            utils.save_images(target_means, "{}/test_results".format(config.output_dir), "target_means", 0)
            utils.save_images(target_stds, "{}/test_results".format(config.output_dir), "target_stds", 0)
            utils.save_images(sample_means, "{}/test_results".format(config.output_dir),"predictions_means", 0)
            utils.save_images(sample_stds, "{}/test_results".format(config.output_dir),"predictions_stds", 0)


if __name__ == '__main__':
    config = evaluation_config.get_config()
    test = Model_Test(config)
    test.predict()