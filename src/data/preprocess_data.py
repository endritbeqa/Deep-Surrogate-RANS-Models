import os
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = "/home/blin/endrit/dataset/steady/original/train/combined"  # Directory where the dataset is downloaded
DEST_DIR = "/home/blin/endrit/dataset/steady/preprocessed_small/combined"  # Directory where the data should be moved when preprocessed

removePOffset = True
makeDimLess = True
fixedAirfoilNormalization = True
epsilon = 1e-8
res = 128
percentage = 0.05
train_val_split = 0.95

TRAIN_DIR = "{}/res_{}/full/train_val_split/train".format(DEST_DIR, res)
VALIDATION_DIR = "{}/res_{}/full/train_val_split/validation".format(DEST_DIR, res)
MASK_DIR = "{}/res_{}/masks".format(DEST_DIR, res)


def split_train_val():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    files = os.listdir(SRC_DIR)
    num_files = int(len(files)*percentage)
    files = files[:num_files]
    random.shuffle(files)
    train_count = int(len(files) * train_val_split)

    train_files = files[:train_count]
    validation_files = files[train_count:]

    print(f"Copying {len(train_files)} directories to the train set...")
    for file in train_files:
        snapshot_data = np.load(os.path.join(SRC_DIR, file))
        snapshot_data = snapshot_data['a'].astype(np.float32)
        snapshot_data = preprocess_data(snapshot_data)

        arrays = torch.tensor(snapshot_data, dtype=torch.float32)
        arrays = torch.unsqueeze(arrays, dim=0)
        arrays = F.interpolate(arrays, size=(res, res), mode='bilinear', align_corners=False)
        arrays = torch.squeeze(arrays)
        arrays = arrays.numpy()

        mask = arrays[2] != 0
        arrays[2][mask] = 1

        for i in [0, 1, 3, 4, 5]:
            arrays[i][mask] = 0

        output_path = "{}/{}".format(TRAIN_DIR, file)
        save_path = os.path.join(output_path)
        np.savez(save_path, a=arrays)


    print(f"Copying {len(validation_files)} directories to the validation set...")
    for file in validation_files:
        snapshot_data = np.load(os.path.join(SRC_DIR, file))
        snapshot_data = snapshot_data['a'].astype(np.float32)
        snapshot_data = preprocess_data(snapshot_data)

        arrays = torch.tensor(snapshot_data, dtype=torch.float32)
        arrays = torch.unsqueeze(arrays, dim=0)
        arrays = F.interpolate(arrays, size=(res, res), mode='bilinear', align_corners=False)
        arrays = torch.squeeze(arrays)
        arrays = arrays.numpy()

        mask = arrays[2] != 0
        arrays[2][mask] = 1

        for i in [0, 1, 3, 4, 5]:
            arrays[i][mask] = 0

        output_path = "{}/{}".format(VALIDATION_DIR, file)
        save_path = os.path.join(output_path)
        np.savez(save_path, a=arrays)


    print("Data split completed!")


def preprocess_data(data) -> np.ndarray:
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


def save_masks():
    all_cases = os.listdir(SRC_DIR)
    airfoils = {}

    for case in all_cases:
        airfoil_name = case.split('_')[0]
        if airfoil_name not in airfoils:
            snapshot = os.listdir(os.path.join(SRC_DIR, case))[0]
            airfoils[airfoil_name] = os.path.join(SRC_DIR, case, snapshot)

    os.makedirs(MASK_DIR, exist_ok=True)

    for airfoil_name, snapshot_path in airfoils.items():
        snapshot_data = np.load(snapshot_path)
        snapshot_data = snapshot_data['a'].astype(np.float32)

        arrays = torch.tensor(snapshot_data, dtype=torch.float32)
        mask = arrays[2] != 0
        arrays[2][mask] = 1

        arrays = torch.unsqueeze(arrays, dim=0)
        arrays = F.interpolate(arrays, size=(res, res), mode='bilinear', align_corners=False)
        arrays = torch.squeeze(arrays)
        arrays = arrays.numpy()
        arrays = arrays[2]

        output_path = "{}/{}".format(MASK_DIR, airfoil_name)
        save_path = os.path.join(output_path)
        np.savez(save_path, a=arrays)


if __name__ == '__main__':
    split_train_val()
    #save_masks()
