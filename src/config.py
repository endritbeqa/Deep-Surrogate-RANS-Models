from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test = ""
    config.model_name = "swin"
    config.data_dir = '/home/blin/endrit/dataset/preprocesed'
    config.output_dir = '/media/blin/VOL REC Blin/endrit/tests/Steady/{}'.format(config.test)
    config.num_epochs = 101
    config.batch_size = 1
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-5
    config.loss_function = ['mae']
    config.checkpoint_every = 5

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False


    config.data = config_dict.ConfigDict()
    config.data.type = "combined"


    return config