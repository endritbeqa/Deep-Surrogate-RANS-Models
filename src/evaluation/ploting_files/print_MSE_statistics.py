import json
import os

import numpy as np

from project_definitions import PROJECT_ROOT_DIR


def print_smth():
    models = ["FactFormer", "Swin", "DiT"]
    regions = ["mean_MSE_low",
               "mean_MSE_high",
               "mean_MSE_all",
               "std_MSE_low",
               "std_MSE_high",
               "std_MSE_all"]



    for mode in ['interpolation', 'extrapolation']:
        print()
        print()
        print(mode)
        data = {}

        for model in models:
            data[model] = {}
            for region in regions:
                data[model][region] = []
        for model in models:
            for run in ["run_1", "run_2", "run_3"]:
                file_path = os.path.join(f"{PROJECT_ROOT_DIR}/results", "res32", model, run, "evaluation/Inter_Extrapolation", mode, "test_statistics.json")
                with open(file_path, 'r') as file:
                    file_data = json.load(file)
                    for region in regions:
                        data[model][region].append(file_data[region]*1000)

            for region in regions:
                vals = data[model][region]
                data[model][region] = (np.mean(vals), np.std(vals))



        for region in regions:
            print()
            print(region+":  ", end=" ")
            for model in models:
                print( "& "+f"{data[model][region][0]:.3f}"+"±"+f"{data[model][region][1]:.3f}", end = " ")

if __name__ == '__main__':
    print_smth()