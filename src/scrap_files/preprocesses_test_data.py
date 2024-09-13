import os
import numpy as np
import torch
import torch.nn.functional as F



def preprocess_data(data) -> np.ndarray:
    removePOffset, makeDimLess, fixedAirfoilNormalization = True, True, True
    epsilon = 1e-8

    if not any((removePOffset, makeDimLess, fixedAirfoilNormalization)):
        return data

    boundary = ~ data[2].flatten().astype(bool)
    num_field_elements = np.sum(boundary)
    c, h, w = data.shape

    data = data.reshape((c, h * w))
    fields = data[np.tile(boundary, (6, 1))]
    fields = fields.reshape((6, num_field_elements))
    p_mean = np.mean(fields[3])
    v_norm = (np.max(np.abs(fields[0])) ** 2 + np.max(np.abs(fields[1])) ** 2) ** 0.5

    if removePOffset:
        data[3][boundary] -= p_mean
        data[3][boundary][data[3][boundary] == 0] = epsilon

    if makeDimLess:
        data[3][boundary] /= (v_norm ** 2 + epsilon)
        data[4][boundary] /= (v_norm + epsilon)
        data[5][boundary] /= (v_norm + epsilon)

    if fixedAirfoilNormalization:
        # hard coded maxima , inputs dont change
        max_inputs_0 = 100.
        max_inputs_1 = 38.5
        max_inputs_2 = 1.0

        # targets depend on normalization
        if makeDimLess:
            max_targets_0 = 4.3
            max_targets_1 = 2.15
            max_targets_2 = 2.35

        else:  # full range
            max_targets_0 = 40000.
            max_targets_1 = 200.
            max_targets_2 = 216.

    else:
        max_inputs_0 = np.max(fields[0]) if np.max(fields[0]) != 0 else epsilon
        max_inputs_1 = np.max(fields[1]) if np.max(fields[1]) != 0 else epsilon

        max_targets_0 = np.max(fields[3]) if np.max(fields[3]) != 0 else epsilon
        max_targets_1 = np.max(fields[4]) if np.max(fields[4]) != 0 else epsilon
        max_targets_2 = np.max(fields[5]) if np.max(fields[5]) != 0 else epsilon

    data[0][boundary] *= (1.0 / max_inputs_0)
    data[1][boundary] *= (1.0 / max_inputs_1)

    data[3][boundary] *= (1.0 / max_targets_0)
    data[4][boundary] *= (1.0 / max_targets_1)
    data[5][boundary] *= (1.0 / max_targets_2)

    data = data.reshape((c, h, w))

    return data






def preprocess_test_files():

    directory = "/home/blin/PycharmProjects/Thesis/src/Uncertainty_data_test/test/interpolation"
    cases = os.listdir(directory)

    output_dir = "/home/blin/PycharmProjects/Thesis/src/Uncertainty_data_test_preprocessed/interpolation_64"
    os.makedirs(output_dir)

    for case in cases:
        case_path = os.path.join(directory, case)
        os.makedirs(os.path.join(output_dir,case))

        for snapshot in os.listdir(case_path):
            snapshot_data = np.load(os.path.join(case_path, snapshot))
            snapshot_data = preprocess_data(snapshot_data['a'].astype(np.float32))

            arrays = torch.tensor(snapshot_data, dtype=torch.float32)
            arrays = torch.unsqueeze(arrays, dim=0)
            arrays = F.interpolate(arrays, size=(64, 64), mode='bilinear', align_corners=False)
            arrays = torch.squeeze(arrays)
            arrays = arrays.numpy()

            mask = arrays[2] != 0

            arrays[2][mask] = 1

            for i in [0, 1, 3, 4, 5]:
                arrays[i][mask] = 0

            arrays = preprocess_data(arrays)

            output_path = "{}/{}/{}".format(output_dir, case, snapshot)
            save_path = os.path.join(output_path)
            np.savez(save_path, a=arrays)



if __name__ == '__main__':
    preprocess_test_files()
