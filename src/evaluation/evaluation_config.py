import math
import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "test_swin_NVAE_128_full"
    config.model_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/uncertainty", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "90.pth")
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/data_small_pre'#'/home/blin/endrit/dataset/uncertainty/data_small' #
    config.output_dir = os.path.join(config.model_folder, "predictions")
    config.batch_size = 1

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = True
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True

    config.comparison = config_dict.ConfigDict()
    config.comparison.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_128/mask_only'
    config.comparison.num_samples = 100
    config.comparison.freestream_velocities = [10, 40, 80, 100]
    config.comparison.angles = [math.radians(-10), math.radians(5), math.radians(10), math.radians(20)]


    return config