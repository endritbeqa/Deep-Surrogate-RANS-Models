import os
import optuna
from train import Trainer
from config import get_config



#TODO fix this trash
def objective(trial):

    config = get_config()
    trial_config = config.copy_and_resolve_references()

    batch_size = trial.suggest_int('batch_size', 4, 20, step=4)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2)
    directory_name = "batch_{}_learningRate{}".format(batch_size, learning_rate)
    trial_config.output_dir = os.path.join(trial_config.output_dir, 'trials', directory_name)
    trial_config.batch_size = batch_size
    trial_config.learning_rate = learning_rate

    trainer = Trainer(trial_config)

    return trainer.train_model(config)

storage = optuna.storages.RDBStorage(
    url="thesis.db",
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)

study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=10)


