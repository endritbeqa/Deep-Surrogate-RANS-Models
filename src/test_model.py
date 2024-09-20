import json
import math
import torch

from ml_collections import config_dict
from torch.utils.data import DataLoader

from src.models import model_select
from src.data import dataset
from src import loss
from src import utils
from src import test_config

class Model_Test(object):
    def __init__(self, config: config_dict):
        self.config = config

        with open(config.train_config, 'r') as f:
            self.train_config = json.load(f)
        with open(config.model_config, 'r') as f:
            self.model_config = json.load(f)

        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, config.checkpoint)
        self.output_dir = config.output_dir
        self.loss_func = loss.get_loss_function(config.loss)
        self.test_dataset = dataset.Airfoil_Dataset(self.config, mode='test')
        self.test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)


    def predict(self):
        losses = []
        test_loss = 0.0
        self.model.eval()

        with torch.no_grad():

            for inputs, targets, label in self.test_dataloader:
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                if math.isinf(loss) | math.isnan(loss):
                    print("{}, {}".format(label, loss))
                test_loss += loss.item()
                losses.append(loss.item())
                utils.plot_comparison(targets, outputs, self.output_dir, label)

            print("Test loss: {}".format(sum(losses) / len(losses)))

if __name__ == '__main__':
    config = test_config.get_config()
    test = Model_Test(config)
    test.predict()


