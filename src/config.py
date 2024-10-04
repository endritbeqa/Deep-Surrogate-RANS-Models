from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.test = "swin_Big"
    config.model_name = "swin"
    config.data_dir = '/local/disk1/ebeqa/'
    config.output_dir = "/local/disk1/ebeqa/results/{}".format(config.test)   #'/media/blin/VOL REC Blin/endrit/tests/Steady/{}'.format(config.test)
    config.num_epochs = 51
    config.batch_size = 50
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 2e-5
    config.loss_function = ['mae']
    config.checkpoint_every = 5
    config.device = "cuda:3"

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False


    config.data = config_dict.ConfigDict()
    config.data.type = "combined"


    return config
