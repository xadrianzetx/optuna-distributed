import time

import optuna
import pytest

from optuna_distributed.eventloop import EventLoop
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.terminal import Terminal
from optuna_distributed.trial import DistributedTrial


def test_raises_on_trial_exception() -> None:
    def _objective(trial: DistributedTrial) -> float:
        raise ValueError()

    n_trials = 5
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective)
    with pytest.raises(ValueError):
        event_loop.run(terminal=Terminal(show_progress_bar=False, n_trials=n_trials))


def test_catches_on_trial_exception() -> None:
    def _objective(trial: DistributedTrial) -> float:
        raise ValueError()

    n_trials = 5
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective)
    event_loop.run(
        terminal=Terminal(show_progress_bar=False, n_trials=n_trials), catch=(ValueError,)
    )


def test_stops_optimization() -> None:
    uninterrupted_execution_time = 60.0

    def _objective(trial: DistributedTrial) -> float:
        time.sleep(uninterrupted_execution_time)
        return 1.0

    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective)
    started_at = time.time()
    event_loop.run(terminal=Terminal(show_progress_bar=False, n_trials=n_trials), timeout=1.0)
    interrupted_execution_time = time.time() - started_at
    assert interrupted_execution_time < uninterrupted_execution_time
