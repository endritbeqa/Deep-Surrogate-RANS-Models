import os
from ml_collections import config_dict


def get_config():

    config = config_dict.ConfigDict()
    config.load_training = False
    config.checkpoint_path = ""

    config.seed = 66826386
    config.resolution = 128
    config.study_name = "FactFormer_128"
    config.model_name = "FactFormer" # DiT, Swin, FactFormer, Swin_UNet or Swin_NVAE
    config.data_dir = '/home/blin/endrit/dataset/steady/preprocessed_small/combined/res_{}/full/train_val_split'.format(config.resolution)
    config.output_dir = os.path.join('/home/blin/PycharmProjects/Thesis/results/Steady/res{}'.format(config.resolution), config.study_name)
    config.device = 'cuda:0'
    config.num_epochs = 151
    config.batch_size = 20
    config.optimizer = 'AdamW'  # available are AdamW, Adam
    config.scheduler = 'lambda'  # cosine or lambda
    config.cosine_anneling_TMax = 0
    config.lr = 1e-4
    config.final_lr = 0
    config.weight_decay = 1e-4
    config.gradient_clip_norm = None  # None to turn off
    config.loss_function = 'mse'  # available losses: mse, l1, hubber_loss, mrl
    config.checkpoint_every = 5
    config.num_checkpoints_keep = 10

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config


# config used to run several studies at oncewith a ProcessPoolExecuter in train_multiprocess file
def get_config_parametrized(study_name="", model_name="", device="", seed=42, checkpoint="", load_training=False):
    config = config_dict.ConfigDict()
    config.load_training = load_training
    config.checkpoint_path = checkpoint

    config.seed = seed
    config.trainer = "diffusion"  # diffusion or VAE
    config.resolution = 64
    config.study_name = study_name
    config.model_name = model_name
    config.data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/res_64/full/train_val_split'.format(config.resolution)
    config.output_dir = os.path.join('/home/blin/PycharmProjects/Thesis/results/res{}'.format(config.resolution),config.study_name)
    config.device = device
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
    config.num_checkpoints_keep = 20

    config.data_preprocessing = config_dict.ConfigDict()
    config.data_preprocessing.fixedAirfoilNormalization = False
    config.data_preprocessing.makeDimLess = False
    config.data_preprocessing.removePOffset = False

    return config


#config to restart the training and rewriting the checkpoint config
def get_config_restart(checkpoint_path, device, data_dir, output_dir):

    config = config_dict.ConfigDict()
    config.load_training = True
    config.checkpoint_path = checkpoint_path
    config.data_dir = data_dir
    config.output_dir = output_dir
    config.device = device

    return config

