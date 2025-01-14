from multiprocessing import Process
from src import trainers
from src.train_config import get_config_parametrized



def train_run(study, model_name, cuda, seed):
    train_config = get_config_parametrized(study, model_name, cuda, seed)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()

def run():
    study_name = ["DiT_test/run_1", "DiT_test/run_2", "DiT_test/run_3"]
    model_name = ["DiT", "DiT", "DiT"]
    seed = [95375464, 4627575, 6587468]
    cuda = ["cuda:1", "cuda:2", "cuda:3"]

    for i, study in enumerate(study_name):
        process = Process(target=train_run, args=(study, model_name[i], cuda[i], seed[i]))
        process.start()


def restart_train_run(checkpoint:str, load_training):
    train_config = get_config_parametrized(checkpoint=checkpoint, load_training=load_training)
    print(train_config)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()

def restart_run():
    checkpoints = ["/local/disk1/ebeqa/Thesis/results/res64/DiT_test/run_1/checkpoints/55.pth",
                   "/local/disk1/ebeqa/Thesis/results/res64/DiT_test/run_2/checkpoints/55.pth",
                   "/local/disk1/ebeqa/Thesis/results/res64/DiT_test/run_3/checkpoints/55.pth"]

    for checkpoint in checkpoints:
        process = Process(target=restart_train_run, args=(checkpoint, True))
        process.start()




if __name__ == '__main__':
    #run()
    restart_run()
