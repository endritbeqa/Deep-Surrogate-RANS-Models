from src import trainers
from src.train_configs.train_uncertainty_config import get_config

if __name__ == "__main__":
    train_config = get_config()
    if train_config.trainer == "diffusion":
        trainer = trainers.DiffusionTrainer(train_config)
        trainer.train_model()
    elif train_config.trainer == "VAE":
        trainer = trainers.VAE_Trainer(train_config)
        trainer.train_model()
