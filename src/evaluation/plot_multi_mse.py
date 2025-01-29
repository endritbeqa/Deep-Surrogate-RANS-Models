import json
import os
import numpy as np

from src.utils import NumpyEncoder
from src import utils

'''
   First read all files and create a dict of list  with the files 
   { "32":
   [results/res32/FactFormer_test/run_1/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json,
   results/res32/FactFormer_test/run_2/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json
   results/res32/FactFormer_test/run_3/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json]

   "64":
   [results/res64/FactFormer_test/run_1/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json,
   results/res64/FactFormer_test/run_2/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json
   results/res64/FactFormer_test/run_3/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json]

   "128":
   [results/res128/FactFormer_test/run_1/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json,
   results/res128/FactFormer_test/run_2/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json
   results/res128/FactFormer_test/run_3/evaluation/Inter_Extrapolation/interpolation/test_statistics_raw.json]
   }

   then extract the desired mode (e.g low uncetainty interpolation) from the json files and sort and calculate 
   min, mean, max values to create the following structure
   


{
    'mean_low':{
        '32':[[--min_values--],[--mean_values--],[--max_values--]]
        '64':[[--min_values--],[--mean_values--],[--max_values--]]
        '128':[[--min_values--],[--mean_values--],[--max_values--]]
    }

    'std_low':{
        '32':[[--min_values--],[--mean_values--],[--max_values--]]
        '64':[[--min_values--],[--mean_values--],[--max_values--]]
        '128':[[--min_values--],[--mean_values--],[--max_values--]]
    }


}

'''


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


def plot_ratios(resolutions, model, mode, uncertainty_region, output_dir):
    data = []

    for res in resolutions:
        res_paths = []
        for run in ["run_1", "run_2", "run_3"]:
            file_path = os.path.join(RESULT_DIR, "res{}".format(res), "{}_test".format(model), run,
                                     "evaluation/Inter_Extrapolation", mode, "test_statistics_raw.json")
            res_paths.append(file_path)

        case_data = calculate_mse_ratios(res_paths)
        for key, item in case_data.items():
            if uncertainty_region not in key:
                continue
            moment = key.split("_")[0]
            line_label = "{} {}".format(moment, res)
            data.append((line_label, item))

    with open(os.path.join(output_dir, "{}_{}_RatioData.json".format(model, mode)), "w+") as fp:
        json.dump(data, fp, indent=4, cls=NumpyEncoder)

    plot = utils.plot_mse_ratios(data, "{} {} {} uncertainty region".format(model, mode, uncertainty_region), output_dir)
    return plot





if __name__ == '__main__':

    OUTPUT_DIR = "/local/disk1/ebeqa/Thesis/results/Graphs"
    RESULT_DIR = "/local/disk1/ebeqa/Thesis/results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resolutions = [32, 64]
    models = ["DiT", "FactFormer"]
    modes = ["interpolation", "extrapolation"]
    uncertainty_regions = ["high", "low"]



    for model in models:
        output_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(output_dir, exist_ok=True)
        plots = []
        for mode in modes:
                for region in uncertainty_regions:
                    plots.append(plot_ratios(resolutions, model, mode, region, output_dir))

