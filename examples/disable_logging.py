import random
import time

import optuna

import optuna_distributed


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    time.sleep(random.uniform(0.0, 2.0))
    return x**2 + y


if __name__ == "__main__":
    optuna.logging.disable_default_handler()
    optuna_distributed.config.disable_logging()

    study = optuna_distributed.from_study(optuna.create_study())
    print("Running 10 trials without logging...")
    study.optimize(objective, n_trials=10)
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    optuna_distributed.config.enable_logging()
    print("Running 10 more trials with logging...")
    study.optimize(objective, n_trials=10)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
