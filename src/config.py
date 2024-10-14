from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.study_name = 'test_diffusion_swin_UNet_res_128_small_1'
    config.model_name = "diffusion_swin_UNet"
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_128/train_val_split_small'
    config.output_dir = '/media/blin/VOL REC Blin/endrit/tests/uncertainty/{}'.format(config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 101
    config.batch_size = 30
    config.optimizer = 'adamW' # TODO doesnt do anything right now (AdamW is used)
    config.lr = 5e-4
    config.weight_decay = 1e-2
    config.scheduler_restart_epochs = int(config.num_epochs/4)#  TODO   currently does nothing
    config.loss_function = ['mae', 'beta_KLD'] # available losses: mse, mae, hubber_loss, mrl, con_of_mass, beta_KLD
    config.KLD_beta = 0.01
    config.checkpoint_every = 5

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config
