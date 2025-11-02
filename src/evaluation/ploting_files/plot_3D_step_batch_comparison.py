import json
import os

import matplotlib.pyplot as plt
import numpy as np

from project_definitions import PROJECT_ROOT_DIR


def plot_sampling_time_statistics(model):

    timesteps = [20, 50, 100, 150, 200]
    num_samples = [1, 5, 10, 25, 50]
    statistics = {}

    for num in num_samples:
        statistics[num] = {}
        for timestep in timesteps:
            statistics[num][timestep] = {}

    src_dir = f"{PROJECT_ROOT_DIR}/results/res32/step_ablation_study"

    for steps in timesteps:
        test_statistics_file_path = os.path.join(src_dir, model, f"{steps}_steps", "evaluation1", "Sampling_speed_test", "sampling_times_statistics.json")
        with open(test_statistics_file_path, 'r') as file:
            data = json.load(file)
            for num in num_samples:
                statistics[num][steps]["mean"] = data[str(num)]["mean"]
                statistics[num][steps]["std"] =  data[str(num)]["std"]

    X, Y = np.meshgrid(num_samples, timesteps)  # Create a grid
    Z = np.array([[statistics[x][y]["mean"] for x in num_samples] for y in timesteps])
    Z_std = np.array([[statistics[x][y]["std"] for x in num_samples] for y in timesteps])

    print()
    print(model)
    for num in num_samples:
        print()
        for timestep in timesteps:
            mean = statistics[num][timestep]["mean"]
            std = statistics[num][timestep]["std"]
            print("& "+f"{mean:.2f}"+"±"+f"{std:.2f}", end = " ")
        print("\\\\", end = " ")
    # Compute upper and lower bounds
    Z_upper = Z + Z_std
    Z_lower = Z - Z_std

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surfaces
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=1, label="Mean Surface")
    #ax.plot_surface(X, Y, Z_upper, cmap='Blues', alpha=0.3, label="+1 Std Dev")
    #ax.plot_surface(X, Y, Z_lower, cmap='Reds', alpha=0.3, label="-1 Std Dev")

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Number of Diffusion steps')
    ax.set_zlabel('Time in seconds')
    ax.set_title(model)

    #plt.show()
    plt.savefig(f"{PROJECT_ROOT_DIR}/results/Graphs/timestep_sampling_speed_ablation_study/{model}.png")
    plt.close()


if __name__ == '__main__':

    for model in ["FactFormer", "Swin", "DiT"]:
        plot_sampling_time_statistics(model)



