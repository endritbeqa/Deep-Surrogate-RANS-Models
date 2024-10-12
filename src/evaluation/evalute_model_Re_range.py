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


class Model_Test_RE_Range(object):
    def __init__(self, config: ConfigDict):

        self.config = config
        self.checkpoint = torch.load(config.checkpoint)
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.output_dir = config.output_dir
        self.loss_func = loss.get_loss_function(config.loss)
        self.test_dataset = dataset.Inference_Dataset(self.config)
        self.test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test_results','mean_comparisons'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test_results', 'std_comparisons'), exist_ok=True)


    def predict(self):

        self.model.eval()
        with torch.no_grad():
            for idx, (conditions, labels) in enumerate(self.test_dataloader):
                print(idx)
                num_RE, num_angles = labels.shape

                predictions = []
                for i in range(num_RE):
                    for j in range(num_angles):
                        condition = conditions[i,j]
                        condition = np.tile(condition, (25, 1, 1, 1))
                        samples = self.model.sample(condition)
                        predictions.append(samples)

                # predictions are array of shape (num_re, num_angles, num_samples, channels , height, width)
                predictions = np.array(predictions)
                predictions = predictions.reshape((num_RE, num_angles))

                prediction_means = np.mean(predictions, axis=2)
                prediction_stds = np.std(predictions, axis=2)

                predictions_combined = np.concatenate((prediction_means, prediction_stds), axis=2)


                #TODO  the function that paints this in utils
                # e.g utils.plot_comparison_all(predictions_combined[:, 0, :, :, :], labels[:, 0], self.output_dis, filename_smth_smth)








if __name__ == '__main__':
    config = evaluation_config.get_config()
    test = Model_Test_RE_Range(config)
    test.predict()