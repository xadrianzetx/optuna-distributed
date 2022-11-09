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
    # By default, we are relying on process based parallelism to run
    # all trials on a single machine. However, with Dask client, we can easily scale up
    # to Dask cluster spanning multiple physical workers. To learn how to setup and use
    # Dask cluster, please refer to https://docs.dask.org/en/stable/deploying.html.
    # from dask.distributed import Client
    # client = Client(<your.cluster.scheduler.address>)
    client = None

    # Optuna-distributed just wraps standard Optuna study. The resulting object behaves
    # just like regular study, but optimization process is asynchronous.
    study = optuna_distributed.from_study(optuna.create_study(), client=client)

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
