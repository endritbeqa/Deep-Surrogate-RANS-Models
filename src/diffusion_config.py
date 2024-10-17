from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.study_name = 'test_diffusion_swin_UNet_128_small'
    config.model_name = "diffusion_swin_UNet"
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_128/train_val_split_small'
    config.output_dir = '/media/blin/VOL REC Blin/endrit/tests/uncertainty/{}'.format(config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 501
    config.batch_size = 30
    config.optimizer = 'adamW' # TODO doesnt do anything right now (AdamW is used)
    config.lr = 2e-4
    config.weight_decay = 1e-4
    config.scheduler_restart_epochs = int(config.num_epochs/4)#  TODO   currently does nothing
    config.checkpoint_every = 5

    config.noise_scheduler = 'cosine'
    config.timesteps = 300
    config.start_beta = 1e-4
    config.end_beta = 0.02

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config
