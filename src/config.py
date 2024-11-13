import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.load_training = False
    config.checkpoint_path = ""

    config.trainer = "VAE"  # diffusion or VAE
    #config.study_name = 'diffusion_swin_UNet_32_1parameter_test'
    #config.model_name = "diffusion_swin_UNet"
    #config.study_name = 'diffusion_ViT_UNet_32_1parameter_test'
    #config.model_name = "diffusion_ViT_UNet"
    config.study_name = 'swin_NVAE_1parameter_test'
    config.model_name = "swin_NVAE"
    #config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/full/train_val_split'
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/1_parameter/train_val_split'
    config.output_dir = os.path.join('/media/blin/VOL REC Blin/endrit/tests/uncertainty/{}', config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 25000
    config.batch_size = 15
    config.optimizer = 'adamW'  # TODO doesnt do anything right now (AdamW is used)
    config.scheduler = 'cosine'  # cosine or lambda
    config.cosine_anneling_TMax = 1000
    config.lr = 1e-4
    config.final_lr = 0
    config.weight_decay = 1e-4
    config.gradient_clip_norm = None  # None to turn off
    config.loss_function = 'mse' # TODO currently does nothing # available losses: mse, mae, hubber_loss, mrl, con_of_mass, beta_KLD
    config.checkpoint_every = 20

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config
