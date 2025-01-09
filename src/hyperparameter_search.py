import os
import optuna
from src import trainers
from src import train_config
from sqlalchemy import create_engine



#TODO update this code
def objective(trial):
    train_config = config.get_config()
    trial_config = train_config.copy_and_resolve_references()
    batch_size = trial.suggest_int('num_samples', 50, 300, step=1)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    directory_name = "{}/batch_{}_learningRate{}".format(trial_config.output_dir,batch_size, learning_rate)
    trial_config.output_dir = directory_name
    trial_config.batch_size = batch_size
    trial_config.learning_rate = learning_rate

    trainer = trainers.Base_Trainer(trial_config)
    final_val_loss = trainer.train_model()

    return final_val_loss


if __name__ == '__main__':

    os.makedirs(config.output_dir)
    DATABASE_URL = 'sqlite:///{}.sqlite'.format(Config.study_name)
    engine = create_engine(DATABASE_URL, echo=True)

    storage = optuna.storages.RDBStorage(
        url=DATABASE_URL,
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )

    study = optuna.create_study(study_name=Config.study_name, storage=storage)
    study.optimize(objective, n_trials=10, n_jobs=2)
