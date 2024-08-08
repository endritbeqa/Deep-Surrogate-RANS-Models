from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.model = "swin" # swin or swin_cnn
    config.data_dir = './data_res_32_uncertainty/data'
    config.output_dir = 'Outputs'
    config.num_epochs = 51
    config.batch_size = 6000
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-4
    config.weight_decay = 0.01
    config.loss_function = ['mae']
    config.checkpoint_every = 10

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True



    return config