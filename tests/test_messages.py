from unittest.mock import MagicMock

from optuna.study import Study
from optuna.trial import TrialState
import pytest

from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import ShouldPruneMessage


def test_completed_with_correct_value(study: Study, optimization_manager: MagicMock) -> None:
    msg = CompletedMessage(0, 0.0)
    assert msg.closing
    msg.process(study, optimization_manager)
    assert optimization_manager.register_trial_exit.called_once_with(0)
    trial = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    assert len(trial) == 1
    assert trial[0].value == 0.0


def test_completed_with_incorrect_values(study: Study, optimization_manager: MagicMock) -> None:
    msg = CompletedMessage(0, "foo")  # type: ignore
    assert msg.closing
    with pytest.warns():
        msg.process(study, optimization_manager)
    assert optimization_manager.register_trial_exit.called_once_with(0)


def test_failed(study: Study, optimization_manager: MagicMock) -> None:
    exc = ValueError("foo")
    msg = FailedMessage(0, exc, exc_info=MagicMock())
    assert msg.closing
    with pytest.raises(ValueError):
        msg.process(study, optimization_manager)

    assert optimization_manager.register_trial_exit.called_once_with(0)
    trial = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
    assert len(trial) == 1


def test_should_prune(study: Study, optimization_manager: MagicMock) -> None:
    msg = ShouldPruneMessage(0)
    assert not msg.closing
    msg.process(study, optimization_manager)

    assert optimization_manager.get_connection.called_once_with(0)
    assert optimization_manager.get_connection.return_value.called_once_with(False)
    trial = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    assert len(trial) == 1
    assert trial[0]._trial_id == 0
