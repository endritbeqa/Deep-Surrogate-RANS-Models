################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import math, random
import multiprocessing
import os
import shutil
import threading
from asyncio import threads

import concurrent.futures
from sim_Funcs import *
from dataGen import generator

airfoil_database  = "./airfoil_database/"




def work(samples: list, directory: str ):
    generator(samples, directory)

if __name__ == '__main__':

    samples = utils.generate_uniform_random_parameters(20)
    num_workers = 4
    jobs = []
    k, m = divmod(len(samples), num_workers)
    parts = [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_workers)]

    work_dir = os.getcwd()

    for idx in range(num_workers):
        os.mkdir("working_case_{}".format(idx))
        shutil.copytree(airfoil_database, "working_case_{}/airfoil_database".format(idx))
        shutil.copytree("./OpenFOAM", "working_case_{}/OpenFOAM".format(idx))
        p = multiprocessing.Process(target=work, args=(parts[idx],os.path.join(work_dir, "working_case_{}".format(idx))))
        jobs.append(p)
        p.start()

for job in jobs:
    job.join()

print("Done")



