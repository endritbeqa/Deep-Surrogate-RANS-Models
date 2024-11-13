from src import config
from src import trainers

if __name__ == '__main__':
    train_config = config.get_config()
    if train_config.trainer == "diffusion":
        trainer = trainers.DiffusionTrainer(train_config)
        trainer.train_model()
    elif train_config.trainer == "VAE":
        trainer = trainers.VAE_Trainer(train_config)
        trainer.train_model()
