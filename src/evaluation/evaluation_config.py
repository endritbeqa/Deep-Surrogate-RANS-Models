import math
import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = 'FactFormer_test/run_1'
    config.model_folder = os.path.join("/local/disk1/ebeqa/Thesis/results/res64", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "105.pth")
    config.data_dir = '/local/disk1/ebeqa/Thesis/data/preprocessed/res_64/test'
    config.output_dir = os.path.join(config.model_folder, "evaluation2")
    config.batch_size = 1 #TODO this is only needed for the dataset __init__. Maybe can get rid of it??
    config.num_samples = 100
    config.device = 'cuda:0'
    config.eta = 1.0

    config.inter_extrapolation_test = True
    config.raf30_test = False
    config.sampling_speed_test = False
    config.parameter_comparison_test = False


    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.inter_extra = config_dict.ConfigDict()
    config.inter_extra.plot_samples = False
    config.inter_extra.plot_moment_comparison = True

    config.single_parameter = config_dict.ConfigDict()
    config.single_parameter.num_runs = 5

    config.sampling_speed = config_dict.ConfigDict()
    config.sampling_speed.num_samples = [1, 5, 10, 25, 50]

    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]


    return config




def get_config_parametrized(experiment:str):

    config = config_dict.ConfigDict()
    config.test_name = experiment
    config.model_folder = os.path.join("/local/disk1/ebeqa/Thesis/results/res32", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "Final.pth")
    config.data_dir = '/local/disk1/ebeqa/Thesis/data/preprocessed/res_32/test'
    config.output_dir = os.path.join(config.model_folder, "evaluation")
    config.batch_size = 1  # TODO this is only needed for the dataset __init__. Maybe can get rid of it??
    config.num_samples = 100
    config.device = 'cuda:3'
    config.eta = 1

    config.inter_extrapolation_test = True
    config.raf30_test = False
    config.sampling_speed_test = False
    config.parameter_comparison_test = False

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.inter_extra = config_dict.ConfigDict()
    config.inter_extra.plot_samples = False
    config.inter_extra.plot_moment_comparison = True

    config.single_parameter = config_dict.ConfigDict()
    config.single_parameter.num_runs = 5

    config.sampling_speed = config_dict.ConfigDict()
    config.sampling_speed.num_samples = [1, 5, 10, 25, 50]

    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]

    return config
