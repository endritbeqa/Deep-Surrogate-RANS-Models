import json
import numpy as np


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
        min_values = np.min(data, axis=1)
        mean_values = np.mean(data, axis=1)
        max_values = np.max(data, axis=1)
        runs_data[key] = np.stack([min_values, mean_values, max_values], axis=0)

    return runs_data



if __name__ == '__main__':
    Fact_FormerFiles = ["", ""]
    Swin_files = ["",""]

    Fact_Formerdata = calculate_mse_ratios(Fact_FormerFiles, "FactFormer")
    Swin_data = calculate_mse_ratios(Swin_files, "SwinV2")



