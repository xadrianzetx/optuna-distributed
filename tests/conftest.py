from typing import Any
from unittest.mock import MagicMock

import optuna
import pytest


@pytest.fixture
def study() -> optuna.Study:
    study = optuna.create_study()
    study.ask()
    return study


@pytest.fixture
def optimization_manager() -> Any:
    manager = MagicMock()
    manager.get_connection = MagicMock(return_value=MagicMock())
    return manager
