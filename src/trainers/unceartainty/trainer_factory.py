from enum import Enum

from bottle import ConfigDict

from src.trainers.unceartainty.diffusion_trainer import DiffusionTrainer
from src.trainers.unceartainty.VAE_trainer import VAE_Trainer

class TrainerType(Enum):
    DIFFUSION = 1
    VAE = 2

class trainer_factory:
    def __init__(self):
        pass

    @staticmethod
    def create_trainer(trainer_type:TrainerType, train_config:ConfigDict):
        if trainer_type == TrainerType.DIFFUSION:
            return DiffusionTrainer(train_config=train_config)
        elif trainer_type == TrainerType.VAE:
            return VAE_Trainer(train_config=train_config)