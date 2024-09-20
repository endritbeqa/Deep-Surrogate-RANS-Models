import os
import numpy as np
from ml_collections import config_dict
from torch.utils.data import Dataset


class Airfoil_Dataset(Dataset):

    def __init__(self, config: config_dict, mode: str):
        self.data_dir = os.path.join(config.data_dir,config.data.type , mode)
        self.batch_size = config.batch_size
        self.fixedAirfoilNormalization = config.data_preprocessing.fixedAirfoilNormalization
        self.makeDimLess = config.data_preprocessing.makeDimLess
        self.removePOffset = config.data_preprocessing.removePOffset
        self.file_names = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.epsilon =1e-8 #constant for numerical stability
        data = np.load(os.path.join(self.data_dir, self.file_names[0]))['a']
        c, h, w = data.shape
        assert h == w, "Fields are not square"
        self.res = h

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir,self.file_names[idx]))
        data = self.preprocess_data(data['a'].astype(np.float32))
        input = data[0:3, :, :]
        target = data[3:, :, :]

        return (input, target, self.file_names[idx])

    def preprocess_data(self, data) -> list[np.ndarray]:

        if not any((self.removePOffset, self.makeDimLess, self.fixedAirfoilNormalization)):
            return data
        #TODO always double check this because datasets encode the mask differently
        boundary = ~data[2].flatten().astype(bool)
        num_field_elements = np.sum(boundary)
        c, h, w = data.shape

        data = data.reshape((c, h * w))
        fields = data[np.tile(boundary, (6, 1))]
        fields = fields.reshape((6, num_field_elements))
        p_mean = np.mean(fields[3])
        v_norm = (np.max(np.abs(fields[0])) ** 2 + np.max(np.abs(fields[1])) ** 2) ** 0.5

        if self.removePOffset:
            data[3][boundary] -= p_mean
            data[3][boundary][data[3][boundary] == 0] = self.epsilon

        if self.makeDimLess:
            data[3][boundary] /= (v_norm ** 2 + self.epsilon)
            data[4][boundary] /= (v_norm + self.epsilon)
            data[5][boundary] /= (v_norm + self.epsilon)

        if self.fixedAirfoilNormalization:
            # hard coded maxima , inputs dont change
            max_inputs_0 = 100.
            max_inputs_1 = 38.12
            max_inputs_2 = 1.0

            # targets depend on normalization
            if self.makeDimLess:
                max_targets_0 = 4.65
                max_targets_1 = 2.04
                max_targets_2 = 2.37

            else:  # full range
                max_targets_0 = 40000.
                max_targets_1 = 200.
                max_targets_2 = 216.

        else:
            max_inputs_0 = np.max(fields[0]) if np.max(fields[0]) !=0 else self.epsilon
            max_inputs_1 = np.max(fields[1]) if np.max(fields[1]) !=0 else self.epsilon

            max_targets_0 = np.max(fields[3]) if np.max(fields[3]) !=0 else self.epsilon
            max_targets_1 = np.max(fields[4]) if np.max(fields[4]) !=0 else self.epsilon
            max_targets_2 = np.max(fields[5]) if np.max(fields[5]) !=0 else self.epsilon

        data[0][boundary] *= (1.0 / max_inputs_0)
        data[1][boundary] *= (1.0 / max_inputs_1)

        data[3][boundary] *= (1.0 / max_targets_0)
        data[4][boundary] *= (1.0 / max_targets_1)
        data[5][boundary] *= (1.0 / max_targets_2)

        data = data.reshape((c, h, w))

        return data


