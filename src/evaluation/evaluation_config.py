import math
import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    #config.test_name = 'diffusion_swin_UNet_32_1parameter_test6'
    #config.test_name = 'diffusion_ViT_UNet_1parameter_test'
    config.test_name = 'swin_NVAE_1parameter_test'
    config.model_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/uncertainty", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "8000.pth")
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/test'
    config.output_dir = os.path.join(config.model_folder, "evaluation")
    config.batch_size = 1 #TODO this is only needed for the dataset __init__. Maybe can get rid of it??
    config.num_samples = 100
    config.device = 'cuda:1'

    config.inter_extrapolation_test = False
    config.raf30_test = True
    config.sampling_speed_test = False
    config.parameter_comparison_test = False


    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.sampling_speed = config_dict.ConfigDict()
    config.sampling_speed.num_samples = [1, 10, 25, 50, 100]

    config.comparison = config_dict.ConfigDict()
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]


    return config