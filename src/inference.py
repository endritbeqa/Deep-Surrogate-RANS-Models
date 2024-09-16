import json
import os
import torch
import numpy as np
from ml_collections import ConfigDict

from src import utils
from src.models.swin_VAE.U_net_SwinV2_VAE import U_NET_Swin
from src.models.swin_VAE import Config_UNet_Swin
from src import config


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



def sample_from_vae(model_config, checkpoint, condition, num_samples):

    model = U_NET_Swin(model_config)
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    torch.manual_seed(42)
    condition = torch.unsqueeze(torch.from_numpy(condition), 0)
    condition = condition.to(device='cpu')
    predictions = []

    for i in range(num_samples):
        with torch.no_grad():
            prediction = model.inference(condition)
            predictions.append(prediction)

    return np.array(predictions)

def get_test_files(directory):
    cases = os.listdir(directory)

    data = {}

    for case in cases:

        case_path = os.path.join(directory, case)
        case_data =[]
        for snapshot in os.listdir(case_path):
            snapshot_data = np.load(os.path.join(case_path, snapshot))
            snapshot_data = snapshot_data['a'].astype(np.float32)
            case_data.append(snapshot_data)
        case_data = np.array(case_data)
        data[case] = case_data

    return data




if __name__ == '__main__':

    config = config.get_config()
    with open(config.sampling.model_config, 'r') as f:
        model_config_data = json.load(f)
    model_config = ConfigDict(model_config_data)
    interpolation_files = get_test_files(config.sampling.test_folder)
    checkpoint = config.sampling.checkpoint

    target_means = []
    target_stds = []
    predictions_means = []
    predictions_stds = []
    count = 0
    for case, data in interpolation_files.items():
        print(count)
        count+=1
        inputs = data[0][:3]
        predictions = sample_from_vae(model_config, checkpoint, inputs, 25)
        predictions = predictions.squeeze()
        utils.save_images(predictions, "{}/predictions".format(config.sampling.output_dir), case, 0)
        targets = data[:,3:,:,:]
        utils.save_images(targets, "{}/targets".format(config.sampling.output_dir), case, 0)
        target_mean = np.mean(targets, axis=0)
        target_std = np.std(targets, axis=0)
        predictions_mean = np.mean(predictions, axis=0)
        predictions_std = np.std(predictions, axis=0)

        target_means.append(target_mean)
        target_stds.append(target_std)
        predictions_means.append(predictions_mean)
        predictions_stds.append(predictions_std)


    target_means = np.array(target_means)
    target_stds = np.array(target_stds)
    predictions_means = np.array(predictions_means)
    predictions_stds = np.array(predictions_stds)

    utils.save_images(target_means, "{}/test_results".format(config.sampling.output_dir), "target_means", 0)
    utils.save_images(target_stds, "{}/test_results".format(config.sampling.output_dir), "target_stds", 0)
    utils.save_images(predictions_means, "{}/test_results".format(config.sampling.output_dir), "predictions_means", 0)
    utils.save_images(predictions_stds, "{}/test_results".format(config.sampling.output_dir), "predictions_stds", 0)

