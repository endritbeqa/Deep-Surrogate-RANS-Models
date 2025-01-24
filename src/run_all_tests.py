from multiprocessing import Process
from src import trainers
from src.train_config import get_config_parametrized



def train_run(study, model_name, cuda, seed):
    train_config = get_config_parametrized(study, model_name, cuda, seed)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()

def run():


    #study_name = ["FactFormer_test/run_1", "FactFormer_test/run_2", "FactFormer_test/run_3"]
    #model_name = ["FactFormer", "FactFormer", "FactFormer"]
    #seed = [74363476, 9274257, 2123782]
    #cuda = ["cuda:0", "cuda:1", "cuda:2"]

    study_name = ["Swin_test/run_1", "Swin_test/run_2", "Swin_test/run_3"]
    model_name = ["Swin", "Swin", "Swin"]
    seed = [ 12847847, 22763727, 89197372]
    cuda = [ "cuda:0", "cuda:1", "cuda:2"]
    jobs = []

    for i, study in enumerate(study_name):
        process = Process(target=train_run, args=(study, model_name[i], cuda[i], seed[i]))
        jobs.append(process)
        process.start()

    for job in jobs:
        job.join()


def restart_train_run(checkpoint:str, load_training):
    train_config = get_config_parametrized(checkpoint=checkpoint, load_training=load_training)
    trainer = trainers.DiffusionTrainer(train_config)
    trainer.train_model()

def restart_run():
    checkpoints = ["/local/disk1/ebeqa/Thesis/results/res64/FactFormer_test/run_1/checkpoints/140.pth",
                   #"/local/disk1/ebeqa/Thesis/results/res64/DiT_test/run_2/checkpoints/55.pth",
                   "/local/disk1/ebeqa/Thesis/results/res64/FactFormer_test/run_3/checkpoints/145.pth"]
    jobs = []

    for checkpoint in checkpoints:
        process = Process(target=restart_train_run, args=(checkpoint, True))
        jobs.append(process)
        process.start()

    for job in jobs:
        job.join()



if __name__ == '__main__':
    run()
    #restart_run()
