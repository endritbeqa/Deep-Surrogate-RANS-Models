from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.model = "swin" # swin or swin_cnn
    config.data_dir = '/home/endrit/PycharmProjects/data/Sequence/128_tra'
    config.output_dir = 'Outputs'
    config.context_size = 32
    config.num_epochs = 100
    config.batch_size = 600
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-4
    config.weight_decay = 0.01
    config.loss_function = ['mae']
    config.checkpoint_every = 10

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.preprocess_once = False
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True



    return config