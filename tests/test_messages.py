from typing import Any
from unittest.mock import MagicMock

from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import TrialState
import pytest

from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.messages import Message
from optuna_distributed.messages import PrunedMessage
from optuna_distributed.messages import RepeatedTrialMessage
from optuna_distributed.messages import ReportMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.messages import ShouldPruneMessage
from optuna_distributed.messages import TrialProperty
from optuna_distributed.messages import TrialPropertyMessage


class MockConnection:
    def __init__(self, manager: "MockOptimizationManager") -> None:
        self._manager = manager

    def put(self, message: Message) -> None:
        assert isinstance(message, ResponseMessage)
        self._manager.message_response = message.data


class MockOptimizationManager:
    def __init__(self) -> None:
        self.trial_exit_called = False
        self.message_response = None

    def register_trial_exit(self, trial_id: int) -> None:
        self.trial_exit_called = True

    def get_connection(self, trial_id: int) -> MockConnection:
        return MockConnection(self)


@pytest.fixture
def manager() -> Any:
    # FIXME: Type annotations are too relaxed here.
    return MockOptimizationManager()


def _message_responds_with(value: Any, manager: Any) -> bool:
    return manager.message_response == value


def test_completed_with_correct_value(study: Study, manager: Any) -> None:
    msg = CompletedMessage(0, 0.0)
    assert msg.closing
    msg.process(study, manager)
    assert manager.trial_exit_called
    trial = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    assert len(trial) == 1
    assert trial[0].value == 0.0


def test_completed_with_incorrect_values(study: Study, manager: Any) -> None:
    msg = CompletedMessage(0, "foo")  # type: ignore
    assert msg.closing
    with pytest.warns():
        msg.process(study, manager)
    assert manager.trial_exit_called


def test_pruned(study: Study, manager: Any) -> None:
    msg = PrunedMessage(0, TrialPruned())
    assert msg.closing
    msg.process(study, manager)
    assert manager.trial_exit_called
    trial = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    assert len(trial) == 1


def test_failed(study: Study, manager: Any) -> None:
    exc = ValueError("foo")
    msg = FailedMessage(0, exc, exc_info=MagicMock())
    assert msg.closing
    with pytest.raises(ValueError):
        msg.process(study, manager)

    assert manager.trial_exit_called
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
def test_trial_property(study: Study, manager: Any, name: str, property: TrialProperty) -> None:
    msg = TrialPropertyMessage(0, property)
    assert not msg.closing
    msg.process(study, manager)
    expected = getattr(study.get_trials(deepcopy=False)[0], name)
    assert _message_responds_with(expected, manager=manager)


def test_should_prune(study: Study, manager: Any) -> None:
    msg = ShouldPruneMessage(0)
    assert not msg.closing
    msg.process(study, manager)

    assert _message_responds_with(False, manager=manager)
    trial = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    assert len(trial) == 1
    assert trial[0]._trial_id == 0


def test_repeated_trial(study: Study, manager: Any) -> None:
    msg = RepeatedTrialMessage(0)
    assert not msg.closing

    study.tell(0, state=TrialState.PRUNED)
    msg.process(study, manager)
    assert _message_responds_with(True, manager=manager)


def test_report_intermediate(study: Study, manager: Any) -> None:
    msg = ReportMessage(0, value=0.0, step=1)
    assert not msg.closing

    msg.process(study, manager)
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.intermediate_values[1] == 0.0
