import os
from src.evaluation import evaluate_model
from src.evaluation import evaluation_config


def run_evaluations():
    experiments = [
        #"FactFormer_test/run_1",
        "FactFormer_test/run_2",
        "FactFormer_test/run_3",
        "DiT_test/run_1",
        "DiT_test/run_2",
        "DiT_test/run_3",
        "Swin_UNet_test/run_1",
        "Swin_UNet_test/run_2",
        "Swin_UNet_test/run_3", ]

    for ex in experiments:
        config = evaluation_config.get_config_parametrized(experiment=ex)
        os.makedirs(config.output_dir, exist_ok=True)
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
    run_evaluations()
