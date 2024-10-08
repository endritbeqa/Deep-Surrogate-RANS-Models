import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "test_ViT_VAE_32_full"
    config.model_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/uncertainty", config.test_name)
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "15.pth")
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/test_small'
    config.output_dir = os.path.join(config.model_folder, "predictions")
    config.batch_size = 1
    config.loss = ['mrl']

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config