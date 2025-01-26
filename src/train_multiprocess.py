import concurrent.futures
from src import trainers
from src.train_config import get_config_parametrized

MAX_WORKERS = 3
MODE = 'train'  # train or restart

def train():
    def train_run(study, model_name, cuda, seed):
        train_config = get_config_parametrized(study, model_name, cuda, seed)
        trainer = trainers.DiffusionTrainer(train_config)
        trainer.train_model()

    study_name = ["Swin_test/run_1", "Swin_test/run_2", "Swin_test/run_3"]
    model_name = ["Swin", "Swin", "Swin"]
    seed = [12847847, 22763727, 89197372]
    cuda = ["cuda:0", "cuda:1", "cuda:2"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, study in enumerate(study_name):
            executor.submit(train_run, (study, model_name[i], cuda[i], seed[i]))


def restart_training():
    def restart_train_run(checkpoint):
        train_config = get_config_parametrized(checkpoint=checkpoint, load_training=True)
        trainer = trainers.DiffusionTrainer(train_config)
        trainer.train_model()


    checkpoints = ["/local/disk1/ebeqa/Thesis/results/res64/FactFormer_test/run_1/checkpoints/140.pth",
                   "/local/disk1/ebeqa/Thesis/results/res64/DiT_test/run_2/checkpoints/55.pth",
                   "/local/disk1/ebeqa/Thesis/results/res64/FactFormer_test/run_3/checkpoints/145.pth"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for checkpoint in enumerate(checkpoints):
            executor.submit(restart_train_run, checkpoint)




if __name__ == '__main__':
    if MODE == "train":
        train()
    elif MODE == "restart":
        restart_training()
    else:
        raise Exception("Mode {} not implemented.".format(MODE))
