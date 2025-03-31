import json
import os
import numpy as np

from src.utils import NumpyEncoder
from src import utils
from project_definitions import PROJECT_ROOT_DIR



def calculate_mse_ratios(files):
    runs_data = {}

    for i, f in enumerate(files):
        with open(f, 'r') as file:
            data = json.load(file)
            for key, values in data.items():
                sorted_values = sorted(values)
                runs_data.setdefault(key, []).append(sorted_values)

    for key, values in runs_data.items():
        data = np.array(values)
        min_values = np.min(data, axis=0)
        mean_values = np.mean(data, axis=0)
        max_values = np.max(data, axis=0)
        runs_data[key] = np.stack([min_values, mean_values, max_values], axis=0)

    return runs_data


def read_ratios(resolutions, model):
    data = {}

    for mode in ['interpolation', 'extrapolation']:
        for res in resolutions:
            paths = []
            for run in ["run_1", "run_2", "run_3"]:
                file_path = os.path.join(RESULT_DIR, "res{}".format(res), model, run,
                                         "evaluation/Inter_Extrapolation", mode, "test_statistics_raw.json")
                paths.append(file_path)

            case_data = calculate_mse_ratios(paths)
            for key, item in case_data.items():
                if key in ['std_MSE_all', 'mean_MSE_all']:
                    continue
                moment, _, region = key.split("_")
                if moment == "std":
                    moment = "σ"
                elif moment == "mean":
                    moment =  "µ"
                line_label = "{} {}x{}".format(moment, res, res)
                data.setdefault("{} {}".format(mode.capitalize(), region.capitalize()), []).append((line_label, item))


    return data





if __name__ == '__main__':

    OUTPUT_DIR = f"{PROJECT_ROOT_DIR}/results/Graphs/all_MSE_ratio"
    RESULT_DIR = f"{PROJECT_ROOT_DIR}/results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resolutions = [32, 64]
    models = ["FactFormer", "Swin", "DiT"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = {}

    for model in models:
        data[model] = read_ratios(resolutions, model)

    utils.plot_all_mse_ratios(data, OUTPUT_DIR)
