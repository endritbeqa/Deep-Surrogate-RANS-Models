import os
import multiprocessing
import shutil
from ml_collections import config_dict

from utils import generate_uniform_random_parameters, write_point_coordinates
from simFunctions import generator
from config import get_config

def work(config: config_dict ,samples: list, directory: str ):
    generator(config, samples, directory)


if __name__ == '__main__':


    ###TODO find a way to run this since you would have to manually
    ###do it in a console and then run python dataGeneration.py there
    # exec(open("/opt/openfoam9/etc/bashrc").read())

    config = get_config()

    for res, res_params in (config.res_params).items():


        config.res = int(res)
        config.num_samples, config.simulation_timeout  = res_params
        res_folder = "data_res_{}".format(config.res)

        os.mkdir(res_folder)
        samples = generate_uniform_random_parameters(config.num_samples)
        num_workers = config.num_workers
        jobs = []

        k, m = divmod(len(samples), num_workers)
        parts = [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_workers)]

        work_dir = os.getcwd()

        write_point_coordinates('./OpenFOAM/system/internalCloud_template', config.res)

        for idx in range(num_workers):
            os.mkdir("{}/working_case_{}".format(res_folder,idx))
            shutil.copytree(config.airfoil_database, "{}/working_case_{}/airfoil_database".format(res_folder,idx))
            shutil.copytree("./OpenFOAM", "{}/working_case_{}/OpenFOAM".format(res_folder,idx))
            p = multiprocessing.Process(target=work, args=(config,parts[idx],os.path.join(work_dir, "{}/working_case_{}".format(res_folder,idx))))
            jobs.append(p)
            p.start()

        for job in jobs:
            job.join()
            jobs=[]

    print("Done")



