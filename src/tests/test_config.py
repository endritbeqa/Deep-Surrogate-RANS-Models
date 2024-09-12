from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.model_name = "swin"
    config.data_dir = '/media/blin/VOL REC Blin/endrit/datasets/uncertainty/Uncertainty_preprocessed/res_{}'.format(config.study_name.split('_')[-1])
    config.output_dir = '/media/blin/VOL REC Blin/endrit/{}'.format(config.study_name)
    config.device = 'cuda:0'
    config.batch_size = 240
    config.normalize = True


    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config