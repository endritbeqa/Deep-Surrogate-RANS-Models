import os
import multiprocessing
import shutil
from ml_collections import config_dict

from utils import generate_uniform_random_parameters, write_point_coordinates, clean_res_dir
from simFunctions import generator
from config import get_config

def work(config: config_dict ,samples: list, directory: str ):
    generator(config, samples, directory)


if __name__ == '__main__':
    ###TODO find a way to run this since you would have to manually
    ###do it in a console and then run python dataGeneration.py there
    # exec(open("/opt/openfoam9/etc/bashrc").read())
    config = get_config()
    work_dir = os.getcwd()

    for res, res_params in (config.res_params):

        os.chdir(work_dir)
        jobs = []

        config.res = res
        config.num_samples, config.simulation_timeout = res_params
        res_dir = "data_res_{}".format(config.res)
        os.mkdir(res_dir)

        samples = generate_uniform_random_parameters(config.num_samples)
        k, m = divmod(len(samples), config.num_workers)
        parts = [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(config.num_workers)]

        write_point_coordinates('./OpenFOAM/system/internalCloud_template', config.res)

        for idx in range(config.num_workers):
            os.mkdir("{}/working_case_{}".format(res_dir, idx))
            shutil.copytree(config.airfoil_database, "{}/working_case_{}/airfoil_database".format(res_dir, idx))
            shutil.copytree("./OpenFOAM", "{}/working_case_{}/OpenFOAM".format(res_dir, idx))
            p = multiprocessing.Process(target=work, args=(config,parts[idx],"{}/{}/working_case_{}".format(work_dir,res_dir, idx)))
            jobs.append(p)
            p.start()

        for job in jobs:
            job.join()
            jobs=[]

        if config.clean_res_dir:
            clean_res_dir(config, res_dir)


    print("Done")



