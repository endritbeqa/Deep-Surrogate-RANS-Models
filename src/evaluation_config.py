import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "test_NVAE"
    config.model_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/uncertainty", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "40.pth")
    config.data_dir = '/media/blin/VOL REC Blin/endrit/datasets/uncertainty/test/Uncertainty_data_test_preprocessed/interpolation'
    config.output_dir = os.path.join(config.model_folder, "predictions")
    config.batch_size = 1
    config.loss = ['mrl']

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = True
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True

    config.data = config_dict.ConfigDict()
    config.data.type = ""


    return config