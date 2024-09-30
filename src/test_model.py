import math
import os
import torch

from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from src.models import model_select
from src.data import dataset
from src import loss
from src import utils
from src import test_config

class Model_Test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.checkpoint = torch.load(config.checkpoint, map_location='cpu')
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = model_select.load_model(self.model_name, self.model_config, self.checkpoint)
        self.output_dir = config.output_dir
        self.loss_func = loss.get_loss_function(config.loss)
        self.test_dataset = dataset.Airfoil_Dataset(self.config, mode='test')
        self.test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)

        os.makedirs(self.output_dir, exist_ok=True)


    def predict(self):
        losses = []
        test_loss = 0.0
        self.model.eval()

        with torch.no_grad():

            for idx, (inputs, targets, label) in enumerate(self.test_dataloader):

                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                if math.isinf(loss) | math.isnan(loss):
                    print("{}, {}".format(label, loss))
                test_loss += loss.item()
                losses.append(loss.item())
                targets = targets.numpy().squeeze()
                outputs = outputs.numpy().squeeze()
                utils.plot_comparison(targets, outputs, os.path.join(self.output_dir, "images"), label[0][:-4])
                with open(os.path.join(self.output_dir, "log.txt"), 'a') as f:
                    f.write("Test foil, loss:{}, {}".format(label[0][:-4], loss.item()))
                print("Test foil, loss:{}, {}".format(label[0][:-4], loss.item()))

            with open(os.path.join(self.output_dir, "Final_Accuracy.txt"), 'a') as f:
                f.write("Average loss: {}".format(sum(losses) / len(losses)))
            print("Test loss: {}".format(sum(losses) / len(losses)))


if __name__ == '__main__':
    config = test_config.get_config()
    test = Model_Test(config)
    test.predict()


