import math

from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.res = (32,32)
    config.num_samples = 1000
    config.freestream_angle = math.pi / 8.  # -angle ... angle
    config.freestream_length = 10.  # len * (1. ... factor)
    config.freestream_length_factor = 10.  # length factor
    config.airfoil_database = "./airfoil_database/"
    config.output_dir = "./train/"






    return config