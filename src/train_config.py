import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.load_training = False
    config.checkpoint_path = ""

    config.trainer = "diffusion"  # diffusion or VAE
    config.study_name = 'Swin_UNet_test'
    config.model_name = "Swin_UNet"
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/full/train_val_split'
    #config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_32/1_parameter/train_val_split'
    config.output_dir = os.path.join('/home/blin/endrit/tests/uncertainty/res32', config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 150
    config.batch_size = 20
    config.optimizer = 'AdamW'  # available are AdamW, Adam
    config.scheduler = 'lambda'  # cosine or lambda
    config.cosine_anneling_TMax = 1000
    config.lr = 1e-4
    config.final_lr = 0
    config.weight_decay = 1e-4
    config.gradient_clip_norm = None  # None to turn off
    config.loss_function = 'mse'  # available losses: mse, l1, hubber_loss, mrl
    config.checkpoint_every = 5

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config

def get_config_parametrized(output_dir, study_name, model_name, data_dir, loss):

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
    config.scheduler = 'cosine'  # cosine or lambda
    config.cosine_anneling_TMax = 1000
    config.lr = 1e-4
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
