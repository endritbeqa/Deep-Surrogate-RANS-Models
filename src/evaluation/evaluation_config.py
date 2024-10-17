import math
import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "test_swin_NVAE_128_full"
    config.model_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/uncertainty", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "30.pth")
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_128/test'
    config.output_dir = os.path.join(config.model_folder, "evaluation")
    config.batch_size = 1

    config.inter_extrapolation_test = True
    config.raf30_test = True
    config.parameter_comparison_test = True

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.comparison = config_dict.ConfigDict()
    config.comparison.num_samples = 100
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]


    return config