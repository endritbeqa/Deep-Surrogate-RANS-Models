import os

import torch
from torch.utils.data import DataLoader

import numpy as np

from src import utils
from src.data import dataset
from src.models.U_net_SwinV2 import U_NET_Swin
from src.models import Config_UNet_Swin
from src import config


def all_arrays_equal(arrays):
    if len(arrays) < 2:
        return True  # List with 0 or 1 array is trivially equal

    first_array = arrays[0]
    return all(np.array_equal(first_array, arr) for arr in arrays[1:])



def sample_from_vae(model_config, checkpoint, condition):

    model = U_NET_Swin(model_config)
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    torch.manual_seed(42)

    condition = torch.unsqueeze(torch.from_numpy(condition), 0)
    condition = condition.to(device='cpu')

    predictions = []

    for i in range(5):
        with torch.no_grad():
            random_tensors = [torch.unsqueeze(torch.rand(128), dim=0) for _ in range(3)]
            prediction = model.inference(condition, random_tensors)
            predictions.append(prediction)
            utils.save_images(prediction, '/home/blin/PycharmProjects/Thesis/src/results', "predictions", i)

    print(all_arrays_equal(predictions))


if __name__ == '__main__':
    config = config.get_config()
    model_config = Config_UNet_Swin.get_config()
    test_dataset = dataset.Airfoil_Dataset(config, mode='test')

    checkpoint = "/home/blin/PycharmProjects/Thesis/src/Outputs/checkpoints/100.pth"

    for inputs, targets, label in test_dataset:
        for i in range(4):
            sample_from_vae(model_config, checkpoint, inputs)