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


def plot_ratios(resolutions, model, output_dir):
    data = {}

    for mode in ['interpolation', 'extrapolation']:
        for res in resolutions:
            paths = []
            for run in ["run_1", "run_2", "run_3"]:
                file_path = os.path.join(RESULT_DIR, "res{}".format(res), "{}_test".format(model), run,
                                         "evaluation/Inter_Extrapolation", mode, "test_statistics_raw.json")
                paths.append(file_path)

            case_data = calculate_mse_ratios(paths)
            for key, item in case_data.items():
                if key in ['std_MSE_all', 'mean_MSE_all']:
                    continue
                moment, _, region = key.split("_")
                line_label = "{} {}".format(moment, res)
                data.setdefault("{} {}".format(mode.capitalize(), region.capitalize()), []).append((line_label, item))

    with open(os.path.join(output_dir, "{}_{}_RatioData.json".format(model, mode)), "w+") as fp:
        json.dump(data, fp, indent=4, cls=NumpyEncoder)

    plot = utils.plot_mse_ratios(data, output_dir)
    return plot





if __name__ == '__main__':

    OUTPUT_DIR = f"{PROJECT_ROOT_DIR}/results/Graphs/MSE_ratio"
    RESULT_DIR = f"{PROJECT_ROOT_DIR}/results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resolutions = [32, 64]
    models = ["DiT", "FactFormer", "Swin"]

    for model in models:
        output_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(output_dir, exist_ok=True)
        plot_ratios(resolutions, model, output_dir)

