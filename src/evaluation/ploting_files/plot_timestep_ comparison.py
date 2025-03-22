import os
import json

import numpy as np
from src.utils import plot_sampling_speed_bar_chart
from project_definitions import PROJECT_ROOT_DIR


def plot_sampling_time_statistics():

    timesteps = [20, 50, 100, 150, 200]
    num_samples = [1, 5, 10, 25, 50]
    num_samples_compared = [10, 50]
    statistics = {}

    for num in num_samples:
        statistics[num] = {}
        statistics[num]["means"] = []
        statistics[num]["stds"] = []

    src_dir = f"{PROJECT_ROOT_DIR}/results/res32/step_ablation_study"
    models = ["FactFormer", "Swin", "DiT"]

    for steps in timesteps:
        for model in models:

            test_statistics_file_path = os.path.join(src_dir, model, f"{steps}_steps", "evaluation1", "Sampling_speed_test", "sampling_times_statistics.json")
            with open(test_statistics_file_path, 'r') as file:
                data = json.load(file)
                for num in num_samples:
                    statistics[num]["means"].append(data[str(num)]["mean"])
                    statistics[num]["stds"].append(data[str(num)]["std"])

    for num in num_samples:
        statistics[num]["means"] = np.reshape(np.array(statistics[num]["means"]), (len(timesteps),len(models)))
        statistics[num]["stds"] = np.reshape(np.array(statistics[num]["stds"]), (len(timesteps),len(models)))


    plot_sampling_speed_bar_chart(statistics, timesteps, num_samples_compared
                   ,models, f"timestep_sampling_speed_comparison_{num_samples_compared[0]}_vs_{num_samples_compared[1]}"
                   ,"{}/results/Graphs/timestep_sampling_speed_ablation_study".format(PROJECT_ROOT_DIR))


if __name__ == '__main__':
    plot_sampling_time_statistics()