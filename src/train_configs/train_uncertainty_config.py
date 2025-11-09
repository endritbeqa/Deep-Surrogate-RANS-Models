import os

from ml_collections import config_dict

from project_definitions import PROJECT_ROOT_DIR
from src.trainers.unceartainty.trainer_factory import TrainerType


def get_config(study_name:str = None, model_name:str = None, device:str = None, seed:int = None, checkpoint:str = None):

    config = config_dict.ConfigDict()
    config.checkpoint_path = checkpoint

    config.trainer_type = TrainerType.DIFFUSION
    config.seed = 42 if seed is None else seed
    config.resolution = 128
    config.study_name = "Swin_big" if study_name is None else study_name
    config.model_name = "Swin" if model_name is None else model_name
    config.data_dir = f'{PROJECT_ROOT_DIR}/data/preprocessed/res128/data'
    config.output_dir = f'{PROJECT_ROOT_DIR}/results/res128/{config.study_name}'
    config.device = 'cuda:0' if device is None else device
    config.num_epochs = 251
    config.batch_size = 15
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
