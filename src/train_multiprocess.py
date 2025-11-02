import concurrent.futures

from project_definitions import PROJECT_ROOT_DIR
from src import trainers
from src.train_configs.train_uncertainty_config import get_config_parametrized, get_config_restart

MAX_WORKERS = 2
MODE = 'restart'  # train or restart


def train_run(study, model_name, cuda, seed):
    train_config = get_config_parametrized(study, model_name, cuda, seed)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()


def train():
    study_name = ["Swin/run_1", "Swin/run_2", "Swin/run_3"]
    model_name = ["Swin", "Swin", "Swin"]
    seeds = [12847847, 22763727, 89197372]
    devices = ["cuda:0", "cuda:1", "cuda:1"]

    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, study in enumerate(study_name):
            futures.append(executor.submit(train_run, study, model_name[i], devices[i], seeds[i]))

        for future in futures:
            print(future.result())


def restart_train_run(checkpoint_path, device="", data_dir="", output_dir=""):
    train_config = get_config_restart(checkpoint_path, device, data_dir, output_dir)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()


def restart_training():
    checkpoints = []

    data_dir = f'{PROJECT_ROOT_DIR}/data/preprocessed/res_128/full/train_val_split'
    output_dir = f'{PROJECT_ROOT_DIR}/results/res32'
    devices = ["cuda:2", "cuda:3"]

    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, checkpoint in enumerate(checkpoints):
            futures.append(executor.submit(restart_train_run, checkpoint, devices[i], data_dir, output_dir))

        for future in futures:
            print(future.result())


if __name__ == '__main__':
    if MODE == "train":
        train()
    elif MODE == "restart":
        restart_training()
    else:
        raise Exception("Mode {} not implemented.".format(MODE))
