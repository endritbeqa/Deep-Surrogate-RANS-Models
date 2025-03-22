import math
import os
from ml_collections import config_dict
from project_definitions import PROJECT_ROOT_DIR



def get_config():

    config = config_dict.ConfigDict()
    config.test_name = 'FactFormer_test/run_1'
    config.model_folder = os.path.join("{}/results/res32".format(PROJECT_ROOT_DIR), config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "Final.pth")
    config.data_dir = '{}/data/preprocessed/res_32/test'.format(PROJECT_ROOT_DIR)
    config.output_dir = os.path.join(config.model_folder, "evaluation")
    config.batch_size = 1 #TODO this is only needed for the dataset __init__. Maybe can get rid of it??
    config.num_samples = 5
    config.device = 'cuda:0'
    config.eta = 1.0

    config.inter_extrapolation_test = False
    config.raf30_test = False
    config.drag_coefficient_test = True
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

    config.drag_coefficient = config_dict.ConfigDict()
    config.drag_coefficient.num_runs = 3
    config.drag_coefficient.num_buckets = 10

    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]


    return config




def get_config_parametrized(experiment:str, device = "cuda:0", checkpoint = "Final.pth"):

    config = config_dict.ConfigDict()
    config.test_name = experiment
    config.model_folder = os.path.join("{}/results/res32/dataSize_ablation_study".format(PROJECT_ROOT_DIR), config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", checkpoint)
    config.data_dir = '{}/data/preprocessed/res_32/test'.format(PROJECT_ROOT_DIR)
    config.output_dir = os.path.join(config.model_folder, "evaluation3")
    config.batch_size = 1  # TODO this is only needed for the dataset __init__. Maybe can get rid of it??
    config.num_samples = 100
    config.device = device
    config.eta = 1

    config.inter_extrapolation_test = True
    config.raf30_test = False
    config.drag_coefficient_test = False
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

    config.drag_coefficient = config_dict.ConfigDict()
    config.drag_coefficient.num_runs = 3
    config.drag_coefficient.num_buckets = 5


    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]

    return config
