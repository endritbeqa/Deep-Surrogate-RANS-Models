import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test_name = "tiny"
    config.train_folder = os.path.join("/media/blin/VOL REC Blin/endrit/tests/Steady/best_results/", config.test_name)
    config.checkpoint = os.path.join(config.train_folder, "checkpoints", "Final.pth")
    config.data_dir = '/media/blin/VOL REC Blin/endrit/datasets/steady-state/UIUC_dataset'
    config.output_dir = os.path.join(config.train_folder, "predictions")
    config.batch_size = 1
    config.loss = ['mrl']

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = True
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True

    config.data = config_dict.ConfigDict()
    config.data.type = "test"


    return config