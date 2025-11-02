import math

from ml_collections import config_dict

from project_definitions import PROJECT_ROOT_DIR


def get_config(experiment:str = None, device:str = None, checkpoint:str = None, dataset:str = None):

    config = config_dict.ConfigDict()
    config.test_name = 'res128/Swin_big' if experiment is None else experiment
    config.model_dir = f"{PROJECT_ROOT_DIR}/results/{config.test_name}"
    config.checkpoint = f"{config.model_dir}/checkpoints/{'Final.pth' if checkpoint is None else checkpoint}"#TODO ugly
    config.data_dir = f"{PROJECT_ROOT_DIR}/data/test"
    config.output_dir = f"{config.model_dir}/evaluation"
    config.batch_size = 1 #TODO this is only needed for the dataset __init__.
    config.num_samples = 10
    config.device = 'cuda:0' if device is None else device
    config.eta = 1.0

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
    config.sampling_speed.num_samples = [1, 5, 10]

    config.drag_coefficient = config_dict.ConfigDict()
    config.drag_coefficient.num_runs = 3
    config.drag_coefficient.num_buckets = 10

    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]

    return config
