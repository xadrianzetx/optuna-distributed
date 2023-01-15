import sys
import time

import optuna
import pytest

from optuna_distributed.eventloop import EventLoop
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.terminal import Terminal
from optuna_distributed.trial import DistributedTrial


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Local optimization not supported on Windows."
)


def _objective_raises(trial: DistributedTrial) -> float:
    raise ValueError()


def test_raises_on_trial_exception() -> None:
    n_trials = 5
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective_raises, interrupt_patience=10.0)
    with pytest.raises(ValueError):
        event_loop.run(terminal=Terminal(show_progress_bar=False, n_trials=n_trials))


def test_catches_on_trial_exception() -> None:
    n_trials = 5
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective_raises, interrupt_patience=10.0)
    event_loop.run(
        terminal=Terminal(show_progress_bar=False, n_trials=n_trials), catch=(ValueError,)
    )


def _objective_sleeps(trial: DistributedTrial) -> float:
    uninterrupted_execution_time = 60.0
    time.sleep(uninterrupted_execution_time)
    return 1.0


def test_stops_optimization_after_timeout() -> None:
    uninterrupted_execution_time = 60.0
    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective_sleeps, interrupt_patience=10.0)
    started_at = time.time()
    event_loop.run(terminal=Terminal(show_progress_bar=False, n_trials=n_trials), timeout=1.0)
    interrupted_execution_time = time.time() - started_at
    assert interrupted_execution_time < uninterrupted_execution_time
