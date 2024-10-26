from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.load_training = False
    config.checkpoint_path = "/media/blin/VOL REC Blin/endrit/tests/uncertainty/test_diffusion_swin_UNet_128_small/checkpoints/50.pth"

    #config.study_name = 'test'#'test_swin_NVAE_GMM_128_small'
    #config.model_name = "swin_NVAE"
    config.study_name = 'diffusion_fully_swin_UNet'
    config.model_name = "diffusion_fully_swin_UNet"
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/small/train_val_split'
    config.output_dir = '/media/blin/VOL REC Blin/endrit/tests/uncertainty/{}'.format(config.study_name)
    config.device = 'cuda:1'
    config.num_epochs = 701
    config.batch_size = 30
    config.optimizer = 'adamW' # TODO doesnt do anything right now (AdamW is used)
    config.lr = 1e-4
    config.weight_decay = 1e-4
    config.scheduler_restart_epochs = int(config.num_epochs/4)#  TODO   currently does nothing
    config.loss_function = 'mae' # TODO currently does nothing # available losses: mse, mae, hubber_loss, mrl, con_of_mass, beta_KLD
    config.KLD_beta = 0.01
    config.checkpoint_every = 5

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config
