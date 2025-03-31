import os
import json

from project_definitions import PROJECT_ROOT_DIR


def print_sample_statistics():
    src_dir = f"{PROJECT_ROOT_DIR}/results"

    num_samples = [1, 5, 10, 25, 50]
    resolutions = [32, 64]
    models = ["FactFormer", "Swin", "DiT"]
    statistics = {}

    for model in models:
        statistics[model] = {}
        for res in resolutions:
            statistics[model][res] = {}
            test_statistics_file_path = os.path.join(src_dir, f"res{res}", model, "run_1", "evaluation" ,"Sampling_speed_test", "sampling_times_statistics.json")
            with open(test_statistics_file_path, 'r') as file:
                data = json.load(file)
            for num in num_samples:
                statistics[model][res][num] = [data[str(num)]["mean"], data[str(num)]["std"]]

    for num in num_samples:
        print()
        for model in models:
            for res in resolutions:
                print(f"& {statistics[model][res][num][0]:.2f}±{statistics[model][res][num][1]:.2f}" , end=" ")



if __name__ == '__main__':
    print_sample_statistics()





