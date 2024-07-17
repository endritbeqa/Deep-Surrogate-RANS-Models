import os
import numpy as np
from ml_collections import config_dict
from torch.utils.data import Dataset


class Airfoil_Dataset(Dataset):

    def __init__(self, config: config_dict, mode: str):
        self.data_dir = os.path.join(config.data_dir, mode)
        self.batch_size = config.batch_size
        self.normalize = config.normalize
        self.file_names = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]


    def normalize_target(self, data):
        return data

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)
        data = data['a']
        input = data[0:2,:,:]
        target = data[2:6,:,:]

        if self.normalize:
            target = self.normalize_target(target)

        return (input, target)








