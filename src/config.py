from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.study_name = 'test_5'
    config.model_name = "swin_VAE"#"swin_hierarchical_VAE"
    config.data_dir = '/media/blin/VOL REC Blin/endrit/datasets/uncertainty/Uncertainty_preprocessed/res_32'#.format(config.study_name.split('_')[-1])
    config.output_dir = '/media/blin/VOL REC Blin/endrit/tests/uncertainty/{}'.format(config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 100
    config.batch_size = 30
    config.normalize = True
    config.optimizer = 'adam'
    config.lr = 1e-4
    config.weight_decay = 0.01
    config.loss_function = ['mae', 'beta_KLD']# available losses: mse, mae, hubber_loss, mrl, con_of_mass, beta_KLD
    config.KLD_beta = 0.01
    config.checkpoint_every = 5

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False


    return config