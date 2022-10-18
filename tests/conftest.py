from dask.distributed import Client
from dask.distributed import LocalCluster
import optuna
import pytest


_test_cluster = LocalCluster(n_workers=1, threads_per_worker=1)
_test_client = Client(_test_cluster.scheduler_address)


@pytest.fixture
def client() -> Client:
    return _test_client


@pytest.fixture
def study() -> optuna.Study:
    study = optuna.create_study()
    study.ask()
    return study
