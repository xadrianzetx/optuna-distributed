# optuna-distributed

An extension to [Optuna](https://github.com/optuna/optuna) which makes distributed hyperparameter optimization easy, and keeps all of the original Optuna semantics. Optuna-distributed can run locally, by default utilising all CPU cores, or can easily scale to many machines in [Dask cluster](https://docs.dask.org/en/stable/deploying.html).

> **Note**
> 
> Optuna-distributed is still in the early stages of development. While core Optuna functionality is supported, few missing APIs (especially around Optuna integrations) might prevent this extension from being entirely plug-and-play for some users. Bug reports, feature requests and PRs are more than welcome.

## Features

* Asynchronous optimization by default. Scales from single machine to many machines in cluster.
* Distributed study walks and quacks just like regular Optuna study, making it plug-and-play.
* Compatible with all standard Optuna storages, samplers and pruners.
* No need to modify existing objective functions.

## Installation

```sh
pip install optuna-distributed
```
Optuna-distributed requires Python 3.8 or newer.

## Basic example
Optuna-distributed wraps standard Optuna study. The resulting object behaves just like regular study, but optimization process is asynchronous. Depending on setup of [Dask client](https://docs.dask.org/en/stable/10-minutes-to-dask.html#scheduling), each trial is scheduled to run on available CPU core on local machine, or physical worker in cluster.

> **Note**
>
> Running distributed optimization requires a Dask cluster with environment closely matching one on the client machine. For more information on cluster setup and configuration, please refer to https://docs.dask.org/en/stable/deploying.html.

```python
import random
import time

import optuna
import optuna_distributed
from dask.distributed import Client


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    # Some expensive model fit happens here...
    time.sleep(random.uniform(1.0, 2.0))
    return x**2 + y


if __name__ == "__main__":
    # client = Client("<your.cluster.scheduler.address>")  # Enables distributed optimization.
    client = None  # Enables local asynchronous optimization.
    study = optuna_distributed.from_study(optuna.create_study(), client=client)
    study.optimize(objective, n_trials=10)
    print(study.best_value)
```

But there's more! All of the core Optuna APIs, including [storages, samplers](https://github.com/xadrianzetx/optuna-distributed/blob/main/examples/simple_storages.py) and [pruners](https://github.com/xadrianzetx/optuna-distributed/blob/main/examples/simple_pruning.py) are supported! If you'd like to know how Optuna-distributed works, then check out [this article on Optuna blog](https://medium.com/optuna/running-distributed-hyperparameter-optimization-with-optuna-distributed-17bb2f7d422d).

## What's missing?
* Support for callbacks and Optuna integration modules.
* Study APIs such as [`study.stop`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.stop) can't be called from trial at the moment.
* Local asynchronous optimization on Windows machines. Distributed mode is still available.
* Support for [`optuna.terminator`](https://optuna.readthedocs.io/en/stable/reference/terminator.html).