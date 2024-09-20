from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test = "small_model_2"
    config.model_name = "swin" # swin or swin_cnn
    config.data_dir = '/media/blin/VOL REC Blin/endrit/datasets/steady-state/UIUC_dataset/train'
    config.output_dir = '/media/blin/VOL REC Blin/endrit/results/Steady/{}'.format(config.test)
    config.num_epochs = 50
    config.batch_size = 30
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-5
    config.weight_decay = 0.01
    config.loss_function = ['mae']
    config.checkpoint_every = 1

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = True
    config.data_preprocessing.makeDimLess = True
    config.data_preprocessing.removePOffset = True


    config.data = config_dict.ConfigDict()
    config.data.type = "combined"


    return config