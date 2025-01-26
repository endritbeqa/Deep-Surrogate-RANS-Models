import os
import concurrent.futures
from src.evaluation import evaluate_model
from src.evaluation import evaluation_config


def run_evaluations(config):
    if config.inter_extrapolation_test:
        print("Started Inter/Extrapolation Test")
        inter_extra_test = evaluate_model.Inter_Extrapolation_Test(config)
        inter_extra_test.evaluate()
        print("Ended Inter/Extrapolation Test")
    if config.raf30_test:
        print("Started Single Parameter Test")
        raf30_test = evaluate_model.Raf30_test(config)
        raf30_test.evaluate()
        print("Ended Single Parameter Test")
    if config.sampling_speed_test:
        sampling_speed_test = evaluate_model.Sampling_Speed_Test(config)
        sampling_speed_test.evaluate()
    if config.parameter_comparison_test:
        parameter_comparison_test = evaluate_model.Parameter_Comparison_Test(config)
        parameter_comparison_test.evaluate()


if __name__ == '__main__':

    MAX_WORKERS = 3
    experiments = [
        "FactFormer_test/run_1",
        "FactFormer_test/run_2",
        "FactFormer_test/run_3",
        "DiT_test/run_1",
        "DiT_test/run_2",
        "DiT_test/run_3"
    ]
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:0", "cuda:1", "cuda:2", ]
    jobs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, ex in enumerate(experiments):
            config = evaluation_config.get_config_parametrized(experiment=ex, device=devices[i])
            os.makedirs(config.output_dir, exist_ok=True)
            executor.submit(run_evaluations, config)
