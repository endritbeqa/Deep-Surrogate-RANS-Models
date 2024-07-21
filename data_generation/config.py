import math

from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.end_time = 500 # max number of simulation time steps
    config.purge_write = 0 # number of simulation timepoints to save
    config.write_interval = 500 # interval to write simulation to disk
    # TODO write assertion to make sure the time steps below are available after the simulation
    config.save_timestep = [1500,1800,1900] # simulation timesteps to interpolate into arrays
    config.res_params = [(32, [10,100])]# res: [num_samples, simulation_timeout]
    config.num_workers = 4
    config.num_samples = 0 ## this will be set by the program from the num_samples_per_res dict
    config.res = 0 #this will be set by the program from the num_samples_per_res dict
    config.freestream_angle = math.pi / 8.  # -angle ... angle
    config.freestream_length = 10.  # len * (1. ... factor)
    config.freestream_length_factor = 10.  # length factor
    config.airfoil_database = "./airfoil_database/"
    config.output_dir = "./train/"
    config.clean_res_dir = True
    config.save_images = True
    config.gmsh_timeout = 20
    config.gmshToFoam_timeout = 100
    config.simulation_timeout = 500

    return config