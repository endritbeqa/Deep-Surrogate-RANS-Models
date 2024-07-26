import os
import multiprocessing
import shutil
from ml_collections import config_dict
import simFunctions, utils
from config import get_config

config = get_config()
work_dir = os.getcwd()
save_timesteps = set(config.save_timestep)
disk_timesteps = set([config.end_time - i * config.write_interval for i in range(config.purge_write)])

assert save_timesteps.issubset(disk_timesteps), "Timesteps to intepolate are not written to disk"


def work(config: config_dict, samples: list, directory: str):
    simFunctions.generator(config, samples, directory)


samples = utils.generate_uniform_random_parameters(config.num_samples)
k, m = divmod(len(samples), config.num_workers)
parts = [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(config.num_workers)]

for (res, timeout) in config.res_timeout:
    os.chdir(work_dir)
    jobs = []

    config.res = res
    config.simulation_timeout = timeout
    res_dir = "data_res_{}".format(config.res)
    os.mkdir(res_dir)

    samples = utils.generate_uniform_random_parameters(config.num_samples)
    k, m = divmod(len(samples), config.num_workers)
    parts = [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(config.num_workers)]

    utils.write_point_coordinates('./OpenFOAM/system/internalCloud_template', config.res)
    utils.write_control_dict('./OpenFOAM/system/controlDict_template', config)

    for idx in range(config.num_workers):
        os.mkdir("{}/worker_{}".format(res_dir, idx))
        shutil.copytree(config.airfoil_database, "{}/worker_{}/airfoil_database_{}".format(res_dir, idx, config.mode))
        shutil.copytree("./OpenFOAM", "{}/worker_{}/OpenFOAM".format(res_dir, idx))
        p = multiprocessing.Process(target=work,
                                    args=(config, parts[idx], "{}/{}/worker_{}".format(work_dir, res_dir, idx)))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
        jobs = []

    if config.clean_res_dir:
        utils.clean_res_dir(config, res_dir)

print("Done")
