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
        c,h,w = data.shape
        assert h == w, "Fields are not square"
        self.res = h

    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)
        data = data['a'].astype(np.float32)
        data = self.data_preprocessing(data)

        input = data[0:3,:,:]
        target = data[3:,:,:]

        return (input, target)

    def data_preprocessing(self, data: np.ndarray) -> np.ndarray:
        # TODO this not is only for the 32 res dataset because the boundary is encoded as 1 there
        if not any((self.removePOffset, self.makeDimLess, self.fixedAirfoilNormalization)):
            return data

        boundary = ~ data[2].flatten().astype(bool)
        num_field_elements = np.sum(boundary)
        c, h, w = data.shape

        data = data.reshape((c, h*w))
        fields = data[np.tile(boundary, (6, 1))]
        fields = fields.reshape((6, num_field_elements))
        p_mean = np.mean(fields[3])
        v_norm = (np.max(np.abs(fields[0])) ** 2 + np.max(np.abs(fields[1])) ** 2) ** 0.5

        if self.removePOffset:
            data[3][boundary] -= p_mean

        if self.makeDimLess:
            data[3][boundary] /= v_norm**2
            data[4][boundary] /= v_norm
            data[5][boundary] /= v_norm

        return data.reshape((c, h, w))


    '''
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

        else:  # use current max values from loaded data_generation
            data.max_inputs_0 = self.find_absmax(data, 0, 0)
            data.max_inputs_1 = self.find_absmax(data, 0, 1)
            data.max_inputs_2 = self.find_absmax(data, 0, 2)  # mask, not really necessary
            print("Maxima inputs " + format([data.max_inputs_0, data.max_inputs_1, data.max_inputs_2]))

            data.max_targets_0 = self.find_absmax(data, 1, 0)
            data.max_targets_1 = self.find_absmax(data, 1, 1)
            data.max_targets_2 = self.find_absmax(data, 1, 2)
            print("Maxima targets " + format([data.max_targets_0, data.max_targets_1, data.max_targets_2]))

        data[0, :, :] *= (1.0 / data.max_inputs_0)
        data[1, :, :] *= (1.0 / data.max_inputs_1)

        data[3, :, :] *= (1.0 / data.max_targets_0)
        data[4, :, :] *= (1.0 / data.max_targets_1)
        data[5, :, :] *= (1.0 / data.max_targets_2)
     '''



    def find_absmax(self, data, use_targets, x):
        maxval = 0
        for i in range(data.totalLength):
            if use_targets == 0:
                temp_tensor = data.inputs[i]
            else:
                temp_tensor = data.targets[i]
            temp_max = np.max(np.abs(temp_tensor[x]))
            if temp_max > maxval:
                maxval = temp_max
        return maxval


