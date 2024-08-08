import math
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.mode = "test"
    config.end_time = 30  # max number of simulation time steps
    config.purge_write = 1  # number of simulation timepoints to save
    config.write_interval = 30  # interval to write simulation to disk
    config.save_timestep = [30]  # simulation timesteps to interpolate into arrays
    config.res_params = [(32, [2, 100])]# res: [num_samples, simulation_timeout]
    config.validation_split = 0.2
    config.num_workers = 4
    config.num_samples = 0  ## this will be set by the program from the num_samples_per_res dict
    config.res = 0  # this will be set by the program from the num_samples_per_res dict
    config.min_AoA = -math.pi / 8.  # -angle ... angle
    config.max_AoA = math.pi / 8.  # -angle ... angle
    config.min_velocity = 10
    config.max_velocity = 100
    config.freestream_length = 10.  # len * (1. ... factor)
    config.freestream_length_factor = 10.  # length factor
    config.airfoil_database = "./airfoil_database_{}/".format(config.mode)
    config.output_dir = "./train/"
    config.clean_res_dir = True
    config.save_images = True
    config.gmsh_timeout = 20
    config.gmshToFoam_timeout = 100
    config.simulation_timeout = 500

    return config
