import optuna
import train
import config
def objective(trial):
    config = get_config()
    trial_config = config.copy_and_resolve_references()

    batch_size = trial.suggest_int('batch_size', 100, 400, step=4)
    learning_rate = trial.suggest_float('learning_rate', 1e-9, 1e-3)
    # scheduler = trial.suggest_categorical('scheduler', ['constant','sgdr'])

    directory_name = "batch_size{}_learning_rate{}".format(batch_size, learning_rate)

    trial_config.output_dir = os.path.join(trial_config.output_dir, 'trials', directory_name)

    trial_config.batch_size = batch_size
    # trial_config.learning_rate_scheduler = scheduler
    trial_config.learning_rate_end_value = learning_rate

    loss = train_and_evaluate(trial_config)

    return loss

storage = optuna.storages.RDBStorage(
    url="sqlite:///:memory:",
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)

study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=150)