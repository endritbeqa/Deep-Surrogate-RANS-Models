import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "test_NVAE"
    config.model_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/uncertainty", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "40.pth")
    config.data_dir = '/media/blin/VOL REC Blin/endrit/datasets/uncertainty/preprocessed/test/small/res_32'
    config.output_dir = os.path.join(config.model_folder, "predictions")
    config.batch_size = 1
    config.loss = ['mrl']

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config