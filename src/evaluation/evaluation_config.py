import math
import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = 'Swin_UNet_test'
    config.model_folder = os.path.join("/home/blin/endrit/tests/uncertainty", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "39900.pth")
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/test'
    config.output_dir = os.path.join(config.model_folder, "evaluation4")
    config.batch_size = 1 #TODO this is only needed for the dataset __init__. Maybe can get rid of it??
    config.num_samples = 5
    config.device = 'cuda:0'
    config.eta = 1.1

    config.inter_extrapolation_test = False
    config.raf30_test = True
    config.sampling_speed_test = False
    config.parameter_comparison_test = False


    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.single_parameter = config_dict.ConfigDict()
    config.single_parameter.num_runs = 5

    config.sampling_speed = config_dict.ConfigDict()
    config.sampling_speed.num_samples = [1, 10, 25, 50, 100]

    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]


    return config




def get_config_parametrized(experiment:str):

    config = config_dict.ConfigDict()
    config.test_name = experiment
    config.model_folder = os.path.join("/home/blin/endrit/tests/uncertainty/run_all", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "39")
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/test'
    config.output_dir = os.path.join(config.model_folder, "evaluation")
    config.batch_size = 1 #TODO this is only needed for the dataset __init__. Maybe can get rid of it??
    config.num_samples = 100
    config.device = 'cuda:0'

    config.inter_extrapolation_test = False
    config.raf30_test = True
    config.sampling_speed_test = False
    config.parameter_comparison_test = False


    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.single_parameter = config_dict.ConfigDict()
    config.single_parameter.num_runs = 5

    config.sampling_speed = config_dict.ConfigDict()
    config.sampling_speed.num_samples = [1, 10, 25, 50, 100]

    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]


    return config
