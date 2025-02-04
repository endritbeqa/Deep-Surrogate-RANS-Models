import concurrent.futures
import multiprocessing
import os
import time

from src import trainers
from src.train_config import get_config_parametrized, get_config_restart

MAX_WORKERS = 3
MODE = 'restart'  # train or restart


def train_run(study, model_name, cuda, seed):
    train_config = get_config_parametrized(study, model_name, cuda, seed)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()


def train():


    study_name = ["Swin_test/run_1", "Swin_test/run_2", "Swin_test/run_3"]
    model_name = ["Swin", "Swin", "Swin"]
    seed = [12847847, 22763727, 89197372]
    cuda = ["cuda:0", "cuda:1", "cuda:2"]

    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, study in enumerate(study_name):
            futures.append(executor.submit(train_run, (study, model_name[i], cuda[i], seed[i])))

        for future in futures:
            print(future.result())


def restart_train_run(checkpoint_path, device="", data_dir="", output_dir=""):
    train_config = get_config_restart(checkpoint_path, device, data_dir, output_dir)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()



def restart_training():

    checkpoints = ["/home/blin/PycharmProjects/Thesis/results/res64/Swin_test/run_1/checkpoints/80.pth",
                   "/home/blin/PycharmProjects/Thesis/results/res64/Swin_test/run_2/checkpoints/80.pth",
                   "/home/blin/PycharmProjects/Thesis/results/res64/Swin_test/run_3/checkpoints/75.pth"]

    data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_64/full/train_val_split'
    output_dir = '/home/blin/PycharmProjects/Thesis/results/res64'

    devices = ["cuda:0", "cuda:0", "cuda:0"]

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
