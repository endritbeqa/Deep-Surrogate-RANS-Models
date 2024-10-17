import os
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = "/home/blin/endrit/dataset/uncertainty/dataset_diffusion_based_flow_prediction/train"
PREPROCESS_DIR = "/home/blin/endrit/dataset/uncertainty/preprocessed/res_128/"
TRAIN_DIR = "/home/blin/endrit/dataset/uncertainty/preprocessed/res_128/train_val_split_tiny/train"
VALIDATION_DIR = "/home/blin/endrit/dataset/uncertainty/preprocessed/res_128/train_val_split_tiny/validation"


removePOffset = True
makeDimLess = True
fixedAirfoilNormalization = True
epsilon = 1e-8
res = 128
percentage = 1
train_val_split = 0.95


def split_train_val():

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    
    dirs = [os.path.join(PREPROCESS_DIR, d) for d in os.listdir(PREPROCESS_DIR) if os.path.isdir(os.path.join(PREPROCESS_DIR, d))]
    random.shuffle(dirs)
    train_count = int(len(dirs) * train_val_split)
    
    train_dirs = dirs[:train_count]
    validation_dirs = dirs[train_count:]
    
    print(f"Copying {len(train_dirs)} directories to the train set...")
    for dir_path in train_dirs:
        if len(os.listdir(dir_path))==0:
            print("Empty directory")
            continue
        for file in os.listdir(dir_path):
            shutil.copy(os.path.join(dir_path, file), os.path.join(TRAIN_DIR, file))
    
    print(f"Copying {len(validation_dirs)} directories to the validation set...")
    for dir_path in validation_dirs:
        if len(os.listdir(dir_path))==0:
            print("Empty directory")
            continue
        for file in os.listdir(dir_path):
            shutil.copy(os.path.join(dir_path, file), os.path.join(VALIDATION_DIR, file))
    
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



def preprocess_files():
    all_cases = os.listdir(SRC_DIR)
    random.shuffle(all_cases)
    num_cases = int(len(all_cases)*percentage)
    cases = all_cases[:num_cases]

    os.makedirs(PREPROCESS_DIR, exist_ok=True)

    for i, case in enumerate(cases):
        print("Case: {}/{}".format(i+1, num_cases))
        case_path = os.path.join(SRC_DIR, case)
        os.makedirs(os.path.join(PREPROCESS_DIR,case))

        for snapshot in os.listdir(case_path):
            snapshot_data = np.load(os.path.join(case_path, snapshot))
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

            output_path = "{}/{}/{}".format(PREPROCESS_DIR, case, snapshot)
            save_path = os.path.join(output_path)
            np.savez(save_path, a=arrays)


def save_mask_only():
    all_cases = os.listdir(SRC_DIR)
    airfoils = {}

    for case in all_cases:
        airfoil_name = case.split('_')[0]
        if airfoil_name not in airfoils:
            snapshot = os.listdir(os.path.join(SRC_DIR, case))[0]
            airfoils[airfoil_name] = os.path.join(SRC_DIR, case, snapshot)

    os.makedirs(PREPROCESS_DIR, exist_ok=True)

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

            output_path = "{}/{}".format(PREPROCESS_DIR, airfoil_name)
            save_path = os.path.join(output_path)
            np.savez(save_path, a=arrays)




if __name__ == '__main__':
    #save_mask_only()
    preprocess_files()
    #split_train_val()
