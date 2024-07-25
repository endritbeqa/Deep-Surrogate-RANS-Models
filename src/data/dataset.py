import os
import numpy as np
from ml_collections import config_dict
from torch.utils.data import Dataset


class Airfoil_Dataset(Dataset):

    def __init__(self, config: config_dict, mode: str):
        self.data_dir = os.path.join(config.data_dir, mode)
        self.batch_size = config.batch_size
        self.fixedAirfoilNormalization = config.data_preprocessing.fixedAirfoilNormalization
        self.makeDimLess = config.data_preprocessing.makeDimLess
        self.removePOffset = config.data_preprocessing.removePOffset
        self.file_names = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        data = np.load(os.path.join(self.data_dir, self.file_names[0]))['a']
        c, h, w = data.shape
        assert h == w, "Fields are not square"
        self.res = h

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)
        data = data['a'].astype(np.float32)
        data = self.data_preprocessing(data)

        input = data[0:3, :, :]
        target = data[3:, :, :]

        return (input, target, self.file_names[idx])

    def data_preprocessing(self, data: np.ndarray) -> np.ndarray:
        if any((self.removePOffset, self.makeDimLess, self.fixedAirfoilNormalization)):
            return data

        boundary = ~ data[2].flatten().astype(bool)
        num_field_elements = np.sum(boundary)
        c, h, w = data.shape

        data = data.reshape((c, h * w))
        fields = data[np.tile(boundary, (6, 1))]
        fields = fields.reshape((6, num_field_elements))
        p_mean = np.mean(fields[3])
        v_norm = (np.max(np.abs(fields[0])) ** 2 + np.max(np.abs(fields[1])) ** 2) ** 0.5

        if self.removePOffset:
            data[3][boundary] -= p_mean

        if self.makeDimLess:
            data[3][boundary] /= v_norm ** 2
            data[4][boundary] /= v_norm
            data[5][boundary] /= v_norm

        if self.fixedAirfoilNormalization:
            # hard coded maxima , inputs dont change
            data.max_inputs_0 = 100.
            data.max_inputs_1 = 38.12
            data.max_inputs_2 = 1.0

            # targets depend on normalization
            if self.makeDimLess:
                data.max_targets_0 = 4.65
                data.max_targets_1 = 2.04
                data.max_targets_2 = 2.37
                print("Using fixed maxima " + format([data.max_targets_0, data.max_targets_1, data.max_targets_2]))
            else:  # full range
                data.max_targets_0 = 40000.
                data.max_targets_1 = 200.
                data.max_targets_2 = 216.
                print("Using fixed maxima " + format([data.max_targets_0, data.max_targets_1, data.max_targets_2]))

        else:  # use current max values from loaded data
            data.max_inputs_0 = np.max(fields[0])
            data.max_inputs_1 = np.max(fields[1])
            print("Maxima inputs " + format([data.max_inputs_0, data.max_inputs_1, data.max_inputs_2]))

            data.max_targets_0 = np.max(fields[1])
            data.max_targets_1 = np.max(fields[1])
            data.max_targets_2 = np.max(fields[1])
            print("Maxima targets " + format([data.max_targets_0, data.max_targets_1, data.max_targets_2]))

        data[0][boundary] *= (1.0 / data.max_inputs_0)
        data[1][boundary] *= (1.0 / data.max_inputs_1)

        data[0][boundary] *= (1.0 / data.max_targets_0)
        data[1][boundary] *= (1.0 / data.max_targets_1)
        data[2][boundary] *= (1.0 / data.max_targets_2)

        return data.reshape((c, h, w))


