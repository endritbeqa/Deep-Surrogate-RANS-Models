import os
import json

import numpy as np
from src.utils import plot_bar_chart
from project_definitions import PROJECT_ROOT_DIR


def plot_bar_statistics():

    ablation_study = "dataSize" # step or dataSize

    if ablation_study == "step":
        categories = [20, 50, 100, 150, 200]
    elif ablation_study == "dataSize":
        categories = [5, 10] #, 15, 20, 25]
    else:
        Exception("Unknown ablation Study")


    statistics = {}
    statistics["interpolation"] = {}
    statistics["extrapolation"] = {}

    src_dir = "{}/results/res32/{}_ablation_study".format(PROJECT_ROOT_DIR, ablation_study)
    models = ["FactFormer", "Swin", "Swin"]
    modes = ["interpolation", "extrapolation"]


    for mode in modes:
        means = []
        stds = []
        for category in categories:
            for model in models:
                #for calculating std of multiple evaluation runs
                #run_means = []
                #for evaluation_run in range(1,4):
                #if ablation_study == "step":
                #    test_statistics_file_path = os.path.join(src_dir, model, "{}_steps".format(category), "evaluation{}".format(evaluation_run), "Inter_Extrapolation", mode, "test_statistics.json")
                #elif ablation_study == "dataSize":
                #    test_statistics_file_path = os.path.join(src_dir, model, "{}_snapshots".format(category), "evaluation{}".format(evaluation_run), "Inter_Extrapolation", mode, "test_statistics.json")
                #    with open(test_statistics_file_path, 'r') as file:
                #        data = json.load(file)
                #        run_means.append(data["std_MSE_all"])
                #    means.append(np.mean(run_means))
                #    stds.append(np.std(means))

                if ablation_study == "step":
                    test_statistics_file_path = os.path.join(src_dir, model, "{}_steps".format(category), "evaluation", "Inter_Extrapolation", mode, "test_statistics.json")
                elif ablation_study == "dataSize":
                    test_statistics_file_path = os.path.join(src_dir, model, "{}_snapshots".format(category), "evaluation", "Inter_Extrapolation", mode, "test_statistics.json")

                with open(test_statistics_file_path, 'r') as file:
                    data = json.load(file)
                    means.append(data["std_MSE_all"])

        means = np.array(means)
        means = np.reshape(means, (len(categories),len(models)))
        stds = np.random.uniform(0.00000001, 0.000000005, size=(len(categories),len(models)))

        statistics[mode]["means"] = means
        statistics[mode]["stds"] = stds


    plot_bar_chart(statistics, categories, models, "timestep_comparison",  "{}/results/Graphs/{}_ablation_study".format(PROJECT_ROOT_DIR,ablation_study))


if __name__ == '__main__':
    plot_bar_statistics()