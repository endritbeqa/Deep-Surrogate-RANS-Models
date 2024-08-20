from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.study_name = 'Run_2_res_32'
    config.model = "swin" # swin or swin_cnn
    config.data_dir = '/home/blin/PycharmProjects/Thesis/src/Uncertainty_preprocessed/res_32'
    config.output_dir = '/media/blin/VOL REC Blin/endrit/{}'.format(config.study_name)
    config.device = 'cuda:1'
    config.num_epochs = 51
    config.batch_size = 20
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-4
    config.weight_decay = 0.01
    config.loss_function = []
    config.checkpoint_every = 5

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False



    return config