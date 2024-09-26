import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "test_11"
    config.model_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/uncertainty/", config.test_name)
    config.model_config = os.path.join(config.model_folder, "configs", "model_config.json")
    config.train_config = os.path.join(config.model_folder, "configs", "config.json")
    config.checkpoint = os.path.join(config.model_folder, "checkpoints", "20.pth")
    config.data_dir = '/media/blin/VOL REC Blin/endrit/datasets/uncertainty/test/Uncertainty_data_test_preprocessed'
    config.output_dir = os.path.join(config.model_folder, "predictions")
    config.batch_size = 1
    config.loss = ['mrl']

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.data = config_dict.ConfigDict()
    config.data.type = "interpolation"


    return config