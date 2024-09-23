import optuna
from sqlalchemy import create_engine

from src import train
from src import config

def objective(trial):
    trial_config = config.get_config().copy_and_resolve_references()
    batch_size = trial.suggest_int('batch_size', 10, 50, step=1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, step=1e-6)
    directory_name = "/media/blin/VOL REC Blin/endrit/tests/Steady/hyperparameter_search/tiny/batch_{}_learningRate{}".format(batch_size, learning_rate)
    trial_config.output_dir = directory_name
    trial_config.batch_size = batch_size
    trial_config.learning_rate = learning_rate

    trainer = train.Trainer(trial_config)
    final_val_loss = trainer.train_model()

    return final_val_loss


if __name__ == '__main__':
    DATABASE_URL = 'sqlite:///Steady-state-tiny.sqlite'
    engine = create_engine(DATABASE_URL, echo=True)

    storage = optuna.storages.RDBStorage(
        url=DATABASE_URL,
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )

    study = optuna.create_study(study_name="test-tiny", storage=storage)
    study.optimize(objective, n_trials=20)
