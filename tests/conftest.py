from dask.distributed import Client
import optuna
import pytest


_test_client = Client(n_workers=1, threads_per_worker=1)


@pytest.fixture
def client() -> Client:
    return _test_client


@pytest.fixture
def study() -> optuna.Study:
    study = optuna.create_study()
    study.ask()
    return study
