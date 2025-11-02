import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "DiT"
    config.train_folder = os.path.join("/home/blin/PycharmProjects/Thesis/results/Steady/res128", config.test_name)
    config.checkpoint = os.path.join(config.train_folder, "checkpoints", "Final.pth")
    config.data_dir = '/home/blin/endrit/dataset/steady/preprocessed/test'
    config.output_dir = os.path.join(config.train_folder, "evaluation")
    config.batch_size = 1
    config.loss = 'mrl'

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    config.data = config_dict.ConfigDict()
    config.data.type = "test"


    return config