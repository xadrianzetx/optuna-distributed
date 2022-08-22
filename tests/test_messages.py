from unittest.mock import MagicMock

from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import TrialState
import pytest

from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.messages import PrunedMessage
from optuna_distributed.messages import RepeatedTrialMessage
from optuna_distributed.messages import ReportMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.messages import ShouldPruneMessage
from optuna_distributed.messages import TrialProperty
from optuna_distributed.messages import TrialPropertyMessage


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


def test_pruned(study: Study, optimization_manager: MagicMock) -> None:
    msg = PrunedMessage(0, TrialPruned())
    assert msg.closing
    msg.process(study, optimization_manager)
    assert optimization_manager.register_trial_exit.called_once_with(0)
    trial = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    assert len(trial) == 1


def test_failed(study: Study, optimization_manager: MagicMock) -> None:
    exc = ValueError("foo")
    msg = FailedMessage(0, exc, exc_info=MagicMock())
    assert msg.closing
    with pytest.raises(ValueError):
        msg.process(study, optimization_manager)

    assert optimization_manager.register_trial_exit.called_once_with(0)
    trial = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
    assert len(trial) == 1


def test_heartbeat() -> None:
    msg = HeartbeatMessage()
    assert not msg.closing


def test_response() -> None:
    msg = ResponseMessage(0, data="foo")
    assert not msg.closing
    assert msg.data == "foo"


@pytest.mark.parametrize(
    "name,property",
    [
        ("params", TrialProperty.PARAMS),
        ("distributions", TrialProperty.DISTRIBUTIONS),
        ("user_attrs", TrialProperty.USER_ATTRS),
        ("system_attrs", TrialProperty.SYSTEM_ATTRS),
        ("datetime_start", TrialProperty.DATETIME_START),
        ("number", TrialProperty.NUMBER),
    ],
)
def test_trial_property(
    study: Study, optimization_manager: MagicMock, name: str, property: TrialProperty
) -> None:
    msg = TrialPropertyMessage(0, property)
    assert not msg.closing
    msg.process(study, optimization_manager)
    expected = getattr(study.get_trials(deepcopy=False)[0], name)
    assert optimization_manager.get_connection.return_value_called_once_with(expected)


def test_should_prune(study: Study, optimization_manager: MagicMock) -> None:
    msg = ShouldPruneMessage(0)
    assert not msg.closing
    msg.process(study, optimization_manager)

    assert optimization_manager.get_connection.called_once_with(0)
    assert optimization_manager.get_connection.return_value.called_once_with(False)
    trial = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    assert len(trial) == 1
    assert trial[0]._trial_id == 0


def test_repeated_trial(study: Study, optimization_manager: MagicMock) -> None:
    msg = RepeatedTrialMessage(0)
    assert not msg.closing

    study.tell(0, state=TrialState.PRUNED)
    msg.process(study, optimization_manager)
    assert optimization_manager.get_connection.return_value.called_once_with(True)


def test_report_intermediate(study: Study, optimization_manager: MagicMock) -> None:
    msg = ReportMessage(0, value=0.0, step=1)
    assert not msg.closing

    msg.process(study, optimization_manager)
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.intermediate_values[1] == 0.0
