import optuna
import pytest

from optuna_distributed.eventloop import EventLoop
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.trial import DistributedTrial


def test_raises_on_trial_exception() -> None:
    def _objective(trial: DistributedTrial) -> float:
        raise ValueError()

    n_trials = 5
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective)
    with pytest.raises(ValueError):
        event_loop.run(n_trials)


def test_catches_on_trial_exception() -> None:
    def _objective(trial: DistributedTrial) -> float:
        raise ValueError()

    n_trials = 5
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    event_loop = EventLoop(study, manager, objective=_objective)
    event_loop.run(n_trials, catch=(ValueError,))
