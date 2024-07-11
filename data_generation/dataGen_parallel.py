import multiprocessing
import shutil
from sim_Funcs import *
from dataGen import generator
from config import get_config
from ml_collections import config_dict

def work(config: config_dict ,samples: list, directory: str ):
    generator(config, samples, directory)

if __name__ == '__main__':

    config = get_config()

    samples = utils.generate_uniform_random_parameters(config.num_samples)
    num_workers = config.num_workers
    jobs = []
    k, m = divmod(len(samples), num_workers)
    parts = [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_workers)]

    work_dir = os.getcwd()

    for idx in range(num_workers):
        os.mkdir("working_case_{}".format(idx))
        shutil.copytree(airfoil_database, "working_case_{}/airfoil_database".format(idx))
        shutil.copytree("./OpenFOAM", "working_case_{}/OpenFOAM".format(idx))
        p = multiprocessing.Process(target=work, args=(config,parts[idx],os.path.join(work_dir, "working_case_{}".format(idx))))
        jobs.append(p)
        p.start()

for job in jobs:
    job.join()

print("Done")



