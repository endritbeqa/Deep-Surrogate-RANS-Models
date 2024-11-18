import evaluate_model
import evaluation_config
import os


def run_evaluations():
    experiments = [
          "diffusion_ViT_UNet_32_1parameter_test_cosine_l1"
        , "diffusion_ViT_UNet_32_1parameter_test_cosine_mse"
        , "diffusion_ViT_UNet_32_1parameter_test_linear_l1"
        , "diffusion_ViT_UNet_32_1parameter_test_linear_mse"]

    for ex in experiments:
        config = evaluation_config.get_config(experiment=ex)
        os.makedirs(config.output_dir, exist_ok=True)
        if config.inter_extrapolation_test:
            inter_extra_test = evaluate_model.Inter_Extrapolation_Test(config)
            inter_extra_test.evaluate()
        if config.raf30_test:
            raf30_test = evaluate_model.Raf30_test(config)
            raf30_test.evaluate()
        if config.sampling_speed_test:
            sampling_speed_test = evaluate_model.Sampling_Speed_Test(config)
            sampling_speed_test.evaluate()
        if config.parameter_comparison_test:
            parameter_comparison_test = evaluate_model.Parameter_Comparison_Test(config)
            parameter_comparison_test.evaluate()


if __name__ == '__main__':
    run_evaluations()
