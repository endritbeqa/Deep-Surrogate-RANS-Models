import itertools
import math
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
        self.epsilon = 1e-8  # constant for numerical stability
        data = np.load(os.path.join(self.data_dir, self.file_names[0]))['a']
        c, h, w = data.shape
        assert h == w, "Fields are not square"
        self.res = h

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.file_names[idx]))
        data = self.preprocess_data(data['a'].astype(np.float32))
        input = data[0:3, :, :]
        target = data[3:, :, :]

        return (input, target, self.file_names[idx])

    def preprocess_data(self, data) -> np.ndarray:

        if not any((self.removePOffset, self.makeDimLess, self.fixedAirfoilNormalization)):
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
            data[3][boundary][data[3][boundary] == 0] = self.epsilon

        if self.makeDimLess:
            data[3][boundary] /= (v_norm ** 2 + self.epsilon)
            data[4][boundary] /= (v_norm + self.epsilon)
            data[5][boundary] /= (v_norm + self.epsilon)

        if self.fixedAirfoilNormalization:
            # hard coded maxima , inputs dont change
            max_inputs_0 = 100.
            max_inputs_1 = 38.5
            max_inputs_2 = 1.0

            # targets depend on normalization
            if self.makeDimLess:
                max_targets_0 = 4.3
                max_targets_1 = 2.15
                max_targets_2 = 2.35

            else:  # full range
                max_targets_0 = 40000.
                max_targets_1 = 200.
                max_targets_2 = 216.

        else:
            max_inputs_0 = np.max(fields[0]) if np.max(fields[0]) != 0 else self.epsilon
            max_inputs_1 = np.max(fields[1]) if np.max(fields[1]) != 0 else self.epsilon

            max_targets_0 = np.max(fields[3]) if np.max(fields[3]) != 0 else self.epsilon
            max_targets_1 = np.max(fields[4]) if np.max(fields[4]) != 0 else self.epsilon
            max_targets_2 = np.max(fields[5]) if np.max(fields[5]) != 0 else self.epsilon

        data[0][boundary] *= (1.0 / max_inputs_0)
        data[1][boundary] *= (1.0 / max_inputs_1)

        data[3][boundary] *= (1.0 / max_targets_0)
        data[4][boundary] *= (1.0 / max_targets_1)
        data[5][boundary] *= (1.0 / max_targets_2)

        data = data.reshape((c, h, w))

        return data


class Test_Dataset(Dataset):

    def __init__(self, config):
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.fixedAirfoilNormalization = config.data_preprocessing.fixedAirfoilNormalization
        self.makeDimLess = config.data_preprocessing.makeDimLess
        self.removePOffset = config.data_preprocessing.removePOffset
        self.epsilon = 1e-8  # constant for numerical stability
        self.simulation_folders = [f for f in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.simulation_folders)

    def __getitem__(self, idx):
        simulation_folder = os.path.join(self.data_dir, self.simulation_folders[idx])
        snapshots = os.listdir(simulation_folder)
        targets = []
        conditions = []
        for snapshot in snapshots:
            snapshot_path = os.path.join(simulation_folder, snapshot)
            snapshot = np.load(snapshot_path)
            snapshot = self.preprocess_data(snapshot['a'].astype(np.float32))
            conditions.append(snapshot[:3, :, :])
            targets.append(snapshot[3:, :, :])
        conditions = np.array(conditions)
        targets = np.array(targets)

        return (conditions, targets, self.simulation_folders[idx])

    def preprocess_data(self, data) -> np.ndarray:

        if not any((self.removePOffset, self.makeDimLess, self.fixedAirfoilNormalization)):
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
            data[3][boundary][data[3][boundary] == 0] = self.epsilon

        if self.makeDimLess:
            data[3][boundary] /= (v_norm ** 2 + self.epsilon)
            data[4][boundary] /= (v_norm + self.epsilon)
            data[5][boundary] /= (v_norm + self.epsilon)

        if self.fixedAirfoilNormalization:
            # hard coded maxima , inputs dont change
            max_inputs_0 = 100.
            max_inputs_1 = 38.5
            max_inputs_2 = 1.0

            # targets depend on normalization
            if self.makeDimLess:
                max_targets_0 = 4.3
                max_targets_1 = 2.15
                max_targets_2 = 2.35

            else:  # full range
                max_targets_0 = 40000.
                max_targets_1 = 200.
                max_targets_2 = 216.

        else:
            max_inputs_0 = np.max(fields[0]) if np.max(fields[0]) != 0 else self.epsilon
            max_inputs_1 = np.max(fields[1]) if np.max(fields[1]) != 0 else self.epsilon

            max_targets_0 = np.max(fields[3]) if np.max(fields[3]) != 0 else self.epsilon
            max_targets_1 = np.max(fields[4]) if np.max(fields[4]) != 0 else self.epsilon
            max_targets_2 = np.max(fields[5]) if np.max(fields[5]) != 0 else self.epsilon

        data[0][boundary] *= (1.0 / max_inputs_0)
        data[1][boundary] *= (1.0 / max_inputs_1)

        data[3][boundary] *= (1.0 / max_targets_0)
        data[4][boundary] *= (1.0 / max_targets_1)
        data[5][boundary] *= (1.0 / max_targets_2)

        data = data.reshape((c, h, w))

        return data


class Comparison_Dataset(Dataset):

    def __init__(self, config):
        self.data_dir = config.comparison.data_dir
        self.batch_size = config.batch_size
        self.fixedAirfoilNormalization = config.data_preprocessing.fixedAirfoilNormalization
        self.makeDimLess = config.data_preprocessing.makeDimLess
        self.removePOffset = config.data_preprocessing.removePOffset
        self.RE_numbers = config.comparison.freestream_velocities
        self.angles = config.comparison.angles
        self.labels = np.array([[[np.float32(re), np.float32(angle)] for angle in self.angles] for re in self.RE_numbers])
        self.epsilon = 1e-8  # constant for numerical stability
        self.airfoils = [f for f in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.airfoils)

    def __getitem__(self, idx):
        airfoil = self.airfoils[idx].split('.')[0]
        airfoil_path = os.path.join(self.data_dir, self.airfoils[idx])
        mask = np.load(airfoil_path)
        mask = mask['a'].astype(np.int32)
        field_mask = np.abs(mask - 1)
        H, W = mask.shape

        output_shape = (*self.labels.shape[0:2], 3, H, W)
        data = np.ndarray(shape=output_shape)
        for i, freestream in enumerate(self.RE_numbers):
            for j, angle in enumerate(self.angles):
                f_x = math.cos(angle) * freestream
                f_y = math.sin(angle) * freestream
                field_x = field_mask * f_x
                field_y = (field_mask * f_y)
                case_data = np.stack((field_x, field_y, mask), axis=0)
                case_data = self.preprocess_data(case_data)
                data[i, j] = case_data

        return airfoil, data, self.labels

    def preprocess_data(self, data) -> np.ndarray:
        max_inputs_0 = 100.
        max_inputs_1 = 38.5
        data[0] *= (1.0 / max_inputs_0)
        data[1] *= (1.0 / max_inputs_1)


        return data
