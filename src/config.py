from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.model = "swin" # swin or swin_cnn
    config.data_dir = '/media/blin/VOL REC Blin/endrit/naca_dataset/medium_dataset'
    config.output_dir = 'Outputs'
    config.num_epochs = 30
    config.batch_size = 30
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-4
    config.weight_decay = 0.01
    config.loss_function = ['mae']
    config.checkpoint_every = 1

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True



    return config