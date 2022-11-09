"""
This example adds Optuna-distributed semantics on top of
https://github.com/optuna/optuna-examples/blob/main/quadratic_simple.py

Optuna example that optimizes a simple quadratic function.
In this example, we optimize a simple quadratic function. We also demonstrate how to continue an
optimization and to use timeouts.
"""

import random
import socket
import time

import optuna
from optuna.samplers import NSGAIISampler
from optuna.storages import RDBStorage

import optuna_distributed


# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    # Let's simulate long running job and identify worker doing the job.
    time.sleep(random.uniform(0.0, 2.0))
    trial.set_user_attr("worker", socket.gethostname())
    return x**2 + y


if __name__ == "__main__":
    # Using Dask client, we can easily scale up to multiple machines.
    # from dask.distributed import Client
    # client = Client(<your.cluster.scheduler.address>)
    client = None

    # All standard Optuna storage, sampler and pruner options are supported.
    storage = RDBStorage("sqlite:///:memory:")
    sampler = NSGAIISampler()

    # Optuna-distributed just wraps standard Optuna study. The resulting object behaves
    # just like regular study, but optimization process is asynchronous.
    study = optuna_distributed.from_study(
        optuna.create_study(storage=storage, sampler=sampler), client=client
    )

    # And let's continue with original Optuna example from here.
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    study.optimize(objective, n_trials=10)
    worker = study.best_trial.user_attrs["worker"]
    print(f"Best value: {study.best_value} (params: {study.best_params}) calculated by {worker}\n")

    # We can continue the optimization as follows.
    print("Running 20 additional trials...")
    study.optimize(objective, n_trials=20)
    worker = study.best_trial.user_attrs["worker"]
    print(f"Best value: {study.best_value} (params: {study.best_params}) calculated by {worker}\n")

    # We can specify the timeout.
    print("Running additional trials in 2 seconds...")
    study.optimize(objective, n_trials=100, timeout=2.0)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
