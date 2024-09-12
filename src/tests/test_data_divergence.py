import os
import math

import numpy as np
from torch.utils.data import DataLoader
from src.data import dataset
from src import utils, loss
from src import config

def test_divergence():
    train_config = config.get_config()
    train_dataset = dataset.Airfoil_Dataset(train_config, mode='train')
    val_dataset = dataset.Airfoil_Dataset(train_config, mode='validation')
    train_dataloader = DataLoader(train_dataset, train_config.batch_size)
    val_dataloader = DataLoader(val_dataset, train_config.batch_size)
    train_dataset_divergence = []
    val_dataset_divergence = []

    for inputs, targets, label in train_dataloader:
        divergence = loss.con_of_mass(targets, inputs)
        train_dataset_divergence.append(divergence)

    for inputs, targets, label in val_dataloader:
        divergence = loss.con_of_mass(targets, inputs)
        val_dataset_divergence.append(divergence)

    print("Train dataset mean divergence:{}".format(np.mean(np.array(train_dataset_divergence))))
    print("Val dataset mean divergence:{}".format(np.mean(np.array(val_dataset_divergence))))




