import optuna
import pytest


@pytest.fixture
def study() -> optuna.Study:
    study = optuna.create_study()
    study.ask()
    return study
