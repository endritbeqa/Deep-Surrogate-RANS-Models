import math

from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.num_samples = 4 ## make sure this is divisible by the number of workers
    config.num_workers = 2
    config.res = 64
    config.freestream_angle = math.pi / 8.  # -angle ... angle
    config.freestream_length = 10.  # len * (1. ... factor)
    config.freestream_length_factor = 10.  # length factor
    config.airfoil_database = "./airfoil_database/"
    config.output_dir = "./train/"
    config.save_images = True
    config.gmsh_timeout = 20
    config.gmshToFoam_timeout = 100
    config.simulation_timeout = 500



    return config