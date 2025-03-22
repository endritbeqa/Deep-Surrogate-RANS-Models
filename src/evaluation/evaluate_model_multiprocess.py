import os
import concurrent.futures
from src.evaluation import evaluate_model
from src.evaluation import evaluation_config

MAX_WORKERS = 2

def run_evaluation(config):
    if config.inter_extrapolation_test:
        print("Started Inter/Extrapolation Test")
        inter_extra_test = evaluate_model.Inter_Extrapolation_Test(config)
        inter_extra_test.evaluate()
    if config.raf30_test:
        print("Started Single Parameter Test")
        raf30_test = evaluate_model.Raf30_test(config)
        raf30_test.evaluate()
    if config.drag_coefficient_test:
        print("Started Drag Coefficient Test")
        drag_coefficient_test = evaluate_model.Drag_Coefficient_Test(config)
        drag_coefficient_test.evaluate()
    if config.sampling_speed_test:
        print("Started Sampling Speed Test")
        sampling_speed_test = evaluate_model.Sampling_Speed_Test(config)
        sampling_speed_test.evaluate()
    if config.parameter_comparison_test:
        print("Started Parameter Comparison Test")
        parameter_comparison_test = evaluate_model.Parameter_Comparison_Test(config)
        parameter_comparison_test.evaluate()


def evaluate_all():


    experiments = [
        #"DiT/20_steps",
        #"DiT/50_steps",
        #"DiT/100_steps",
        #"DiT/150_steps",
        #"DiT/200_steps",
        "DiT/20_snapshots"
        #"DiT/10_snapshots",
        #"Swin/100_steps",
        #"Swin/150_steps",
        #"Swin/200_steps",
        #"FactFormer/20_steps",
        #"FactFormer/50_steps",
        #"FactFormer/100_steps",
        #"FactFormer/150_steps",
        #"FactFormer/200_steps",
    ]
    checkpoints = ["Final.pth"]# , "Final.pth", "Final.pth", "Final.pth", "Final.pth", "Final.pth",
                   #"Final.pth", "Final.pth", "Final.pth", "Final.pth", "Final.pth",
                   #"Final.pth", "Final.pth", "Final.pth", "Final.pth", "Final.pth"]

    devices = ["cuda:2"]#, "cuda:1" ,"cuda:2", "cuda:2","cuda:2", "cuda:2",
               #"cuda:2","cuda:2", "cuda:2","cuda:2", "cuda:2",
               #"cuda:2","cuda:2", "cuda:2","cuda:2", "cuda:2"]
    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, ex in enumerate(experiments):
            config = evaluation_config.get_config_parametrized(experiment=ex, device=devices[i], checkpoint=checkpoints[i])
            os.makedirs(config.output_dir, exist_ok=True)
            futures.append(executor.submit(run_evaluation, config))

        for future in futures:
            print(future.result())


if __name__ == '__main__':
    evaluate_all()
