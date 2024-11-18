import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.load_training = False
    config.checkpoint_path = ""

    config.trainer = "diffusion"  # diffusion or VAE
    #train_config.study_name = 'diffusion_swin_UNet_32_1parameter_cosine_l1'
    #train_config.model_name = "diffusion_swin_UNet"
    config.study_name = 'diffusion_ViT_UNet_32_1parameter_test_cosine_l1'
    config.model_name = "diffusion_ViT_UNet"
    #train_config.study_name = 'swin_NVAE_1parameter'
    #train_config.model_name = "swin_NVAE"
    #train_config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/full/train_val_split'
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/1_parameter/train_val_split'
    config.output_dir = os.path.join('/media/blin/VOL REC Blin/endrit/tests/uncertainty', config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 40000
    config.batch_size = 15
    config.optimizer = 'AdamW'  # available are AdamW, Adam
    config.scheduler = 'lambda'  # cosine or lambda
    config.cosine_anneling_TMax = 3000
    config.lr = 1e-3
    config.final_lr = 0
    config.weight_decay = 1e-4
    config.gradient_clip_norm = None  # None to turn off
    config.loss_function = 'l1'  # available losses: mse, l1, hubber_loss, mrl
    config.checkpoint_every = 100

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config



def get_config(output_dir, study_name, model_name, data_dir, loss):

    config = config_dict.ConfigDict()
    config.load_training = False
    config.checkpoint_path = ""

    config.trainer = "diffusion"  # diffusion or VAE
    config.study_name = study_name
    config.model_name = model_name
    config.data_dir = data_dir
    config.output_dir = os.path.join(output_dir, config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 40
    config.batch_size = 15
    config.optimizer = 'AdamW'  # available are AdamW, Adam
    config.scheduler = 'lambda'  # cosine or lambda
    config.cosine_anneling_TMax = 3000
    config.lr = 1e-3
    config.final_lr = 0
    config.weight_decay = 1e-4
    config.gradient_clip_norm = None  # None to turn off
    config.loss_function = loss
    config.checkpoint_every = 1

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config
