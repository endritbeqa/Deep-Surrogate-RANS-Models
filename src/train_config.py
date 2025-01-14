import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.load_training = False
    config.checkpoint_path = ""

    config.seed = 66826386
    config.trainer = "diffusion"  # diffusion or VAE
    config.study_name = 'DiT_test/run_1'
    config.model_name = "DiT"
    config.data_dir = '/local/disk1/ebeqa/Thesis/data/preprocessed/res_64/full/train_val_split'
    config.output_dir = os.path.join('/local/disk1/ebeqa/Thesis/results/res64', config.study_name)
    config.device = 'cuda:1'
    config.num_epochs = 151
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


#TODO fix cuda assigment when loading from checkpoint
def get_config_parametrized(study_name="", model_name="", cuda="", seed=42, checkpoint="", load_training=False):
    config = config_dict.ConfigDict()
    config.load_training = load_training
    config.checkpoint_path = checkpoint

    config.seed = seed
    config.trainer = "diffusion"  # diffusion or VAE
    config.study_name = study_name
    config.model_name = model_name
    config.data_dir = '/local/disk1/ebeqa/Thesis/data/preprocessed/res_64/full/train_val_split'
    config.output_dir = os.path.join('/local/disk1/ebeqa/Thesis/results/res64', config.study_name)
    config.device = cuda
    config.num_epochs = 151
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
