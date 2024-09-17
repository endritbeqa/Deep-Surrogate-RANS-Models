import math
import random
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.end_time = 3500  # max number of simulation time steps
    config.purge_write = 1  # number of simulation timepoints to save
    config.write_interval = 1  # interval to write simulation to disk
    config.num_snapshots = 25
    config.save_timestep = [random.randint(2500, 3500) for _ in range(config.num_snapshots)]  # simulation timesteps to interpolate into arrays
    config.res_params = [(32, [2000, 300])]# res: [num_samples, simulation_timeout]
    config.validation_split = 0.2
    config.num_workers = 4
    config.num_samples = 0  ## this will be set by the program from the num_samples_per_res dict
    config.res = 0  # this will be set by the program from the num_samples_per_res dict
    config.freestream_angle = math.pi / 8.  # -angle ... angle
    config.freestream_length = 10.  # len * (1. ... factor)
    config.freestream_length_factor = 10.  # length factor
    config.airfoil_database = "./airfoil_database/"
    config.output_dir = "./train/"
    config.clean_res_dir = True
    config.save_images = False
    config.gmsh_timeout = 20
    config.gmshToFoam_timeout = 100
    config.simulation_timeout = 500

    return config
