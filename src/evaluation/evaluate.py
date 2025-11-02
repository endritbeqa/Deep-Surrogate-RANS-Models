import concurrent.futures

from src.evaluation import  evaluation_config
from src.evaluation.test_factory import TestType, Test_Factory

MAX_WORKERS = 2

def evaluate_all():
    # configs is a touple of (experiment_name, checkpoint_name, device, dataset). (None, None, None) to use default values in config file
    configs = [(None, "195.pth", None)]

    tests = [TestType.SAMPLING_SPEED]

    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, (experiment, checkpoint, device) in enumerate(configs):
            config = evaluation_config.get_config(experiment=experiment, device=device, checkpoint=checkpoint)
            for test_type in tests:
                test = Test_Factory.get_evaluation_test(test_type=test_type, config=config)
                futures.append(executor.submit(test.evaluate(), config))

        for future in futures:
            print(future.result())


if __name__ == '__main__':
    evaluate_all()