import os
import multiprocessing
import shutil
from ml_collections import config_dict
from data_generation import simFunctions, utils
from config import get_config

def work(config: config_dict ,samples: list, directory: str ):
    simFunctions.generator(config, samples, directory)



config = get_config()
work_dir = os.getcwd()

for res, res_params in (config.res_params):

    os.chdir(work_dir)
    jobs = []

    config.res = res
    config.num_samples, config.simulation_timeout = res_params
    res_dir = "data_res_{}".format(config.res)
    os.mkdir(res_dir)

    samples = utils.generate_uniform_random_parameters(config.num_samples)
    k, m = divmod(len(samples), config.num_workers)
    parts = [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(config.num_workers)]

    utils.write_point_coordinates('./OpenFOAM/system/internalCloud_template', config.res)
    utils.write_control_dict('./OpenFOAM/system/controlDict_template', config)

    for idx in range(config.num_workers):
        os.mkdir("{}/worker_{}".format(res_dir, idx))
        shutil.copytree(config.airfoil_database, "{}/worker_{}/airfoil_database".format(res_dir, idx))
        shutil.copytree("./OpenFOAM", "{}/worker_{}/OpenFOAM".format(res_dir, idx))
        p = multiprocessing.Process(target=work, args=(config,parts[idx],"{}/{}/worker_{}".format(work_dir,res_dir, idx)))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
        jobs=[]

    if config.clean_res_dir:
        utils.clean_res_dir(config, res_dir)


print("Done")



