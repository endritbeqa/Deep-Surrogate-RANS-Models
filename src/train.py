from src.train_config import get_config
from src import trainers

if __name__ == '__main__':
    train_config = get_config()
    trainer = trainers.Trainer(train_config)
    trainer.train_model()