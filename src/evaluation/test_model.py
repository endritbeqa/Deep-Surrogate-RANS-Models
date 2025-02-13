import math
import os
import torch
import torch.nn.functional as F

from ml_collections import ConfigDict
from torch import nn
from torch.utils.data import DataLoader

from src.data import dataset
from src import utils
from src.evaluation import test_config


def set_device(module, device):
    if hasattr(module, "device"):
        setattr(module, "device", device)

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue

        attr_value = getattr(module, attr_name)

        if isinstance(attr_value, nn.Module):
            set_device(attr_value, device)

def move_to_device(model, device):
        set_device(model, device)
        model.device = device
        model.to(device)



class Model_Test(object):
    def __init__(self, config: ConfigDict):
        self.config = config
        self.checkpoint = torch.load(config.checkpoint, map_location='cpu')
        self.train_config = self.checkpoint['train_config']
        self.model_config = self.checkpoint['model_config']
        self.model_name = self.train_config.model_name
        self.model = self.checkpoint['model']
        self.model.load_state_dict(self.checkpoint['model_params'])
        move_to_device(self.model, "cpu")
        self.output_dir = config.output_dir
        self.loss_func = self.loss_select(config.loss)
        self.test_dataset = dataset.Airfoil_Dataset(self.config, mode='test')
        self.test_dataloader = DataLoader(self.test_dataset, config.batch_size, shuffle=False)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

    def loss_select(self, loss: str):
        def mean_relative_loss_function(input, target):
            epsilon = 1e-10
            relative_difference = torch.sum(torch.abs(input - target)) / torch.sum(torch.abs(target))   #torch.max(torch.abs(target), torch.tensor(epsilon, dtype=target.dtype, device=target.device))
            return relative_difference

        if loss == 'mse':
            return F.mse_loss
        elif loss == 'l1':
            return F.l1_loss
        elif loss == 'huber_loss':
            return F.smooth_l1_loss
        elif loss == 'mrl':
            return mean_relative_loss_function
        else:
            raise ValueError(f"Unknown loss function: {loss}, available are mse, l1, mrl, huber.")


    def predict(self):
        losses = []
        test_loss = 0.0
        self.model.eval()

        with torch.no_grad():

            for idx, (inputs, targets, label) in enumerate(self.test_dataloader):
                inputs = inputs.to("cpu")
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                if math.isinf(loss) | math.isnan(loss):
                    print("{}, {}".format(label, loss))
                test_loss += loss.item()
                losses.append(loss.item())
                targets = targets.numpy().squeeze()
                outputs = outputs.numpy().squeeze()
                utils.plot_moment_comparison(targets, outputs,  label[0][:-4] ,os.path.join(self.output_dir, "images"))
                with open(os.path.join(self.output_dir, "log.txt"), 'a') as f:
                    f.write("Test foil: {}, Loss: {}".format(label[0][:-4], loss.item()))
                print("Test foil: {}, Loss: {}".format(label[0][:-4], loss.item()))

            with open(os.path.join(self.output_dir, "Average_Test_Loss.txt"), 'a') as f:
                f.write("Average test loss: {}".format(sum(losses) / len(losses)))
            print("Test loss: {}".format(sum(losses) / len(losses)))


if __name__ == '__main__':
    config = test_config.get_config()
    test = Model_Test(config)
    test.predict()


