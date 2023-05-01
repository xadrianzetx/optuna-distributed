from contextlib import contextmanager
import logging
from typing import Any
from typing import Generator
from unittest.mock import MagicMock

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import TrialState
import pytest

import optuna_distributed
from optuna_distributed.eventloop import EventLoop
from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.managers import ObjectiveFuncType
from optuna_distributed.managers import OptimizationManager
from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.messages import Message
from optuna_distributed.messages import PrunedMessage
from optuna_distributed.messages import ReportMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.messages import SetAttributeMessage
from optuna_distributed.messages import ShouldPruneMessage
from optuna_distributed.messages import SuggestMessage
from optuna_distributed.messages import TrialProperty
from optuna_distributed.messages import TrialPropertyMessage


class MockConnection(IPCPrimitive):
    def __init__(self, manager: "MockOptimizationManager") -> None:
        self._manager = manager

    def get(self) -> "Message":
        return HeartbeatMessage()

    def put(self, message: Message) -> None:
        assert isinstance(message, ResponseMessage)
        self._manager.message_response = message.data

    def close(self) -> None:
        ...


class MockOptimizationManager(OptimizationManager):
    def __init__(self) -> None:
        self.trial_exit_called = False
        self.message_response = None

    def create_futures(self, study: "Study", objective: ObjectiveFuncType) -> None:
        ...

    def before_message(self, event_loop: "EventLoop") -> None:
        ...

    def get_message(self) -> Generator["Message", None, None]:
        yield HeartbeatMessage()

    def after_message(self, event_loop: "EventLoop") -> None:
        ...

    def get_connection(self, trial_id: int) -> "IPCPrimitive":
        return MockConnection(self)

    def stop_optimization(self, patience: float) -> None:
        ...

    def should_end_optimization(self) -> bool:
        return True

    def register_trial_exit(self, trial_id: int) -> None:
        self.trial_exit_called = True


@pytest.fixture
def manager() -> MockOptimizationManager:
    return MockOptimizationManager()


@contextmanager
def _forced_log_propagation(logger_name: str) -> Generator[None, None, None]:
    try:
        # Local fix for https://github.com/pytest-dev/pytest/issues/3697
        logging.getLogger(logger_name).propagate = True
        yield
    finally:
        logging.getLogger(logger_name).propagate = False


def _message_responds_with(value: Any, manager: MockOptimizationManager) -> bool:
    return manager.message_response == value


def test_completed_with_correct_value(
    study: Study, manager: MockOptimizationManager, caplog: pytest.LogCaptureFixture
) -> None:
    msg = CompletedMessage(0, 0.0)
    assert msg.closing
    with _forced_log_propagation(logger_name=optuna_distributed.__name__):
        msg.process(study, manager)
    assert manager.trial_exit_called
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    trial = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    assert len(trial) == 1
    assert trial[0].value == 0.0


def test_completed_with_incorrect_values(study: Study, manager: MockOptimizationManager) -> None:
    msg = CompletedMessage(0, "foo")  # type: ignore
    assert msg.closing
    with pytest.warns():
        msg.process(study, manager)
    assert manager.trial_exit_called


def test_pruned(
    study: Study, manager: MockOptimizationManager, caplog: pytest.LogCaptureFixture
) -> None:
    msg = PrunedMessage(0, TrialPruned())
    assert msg.closing
    with _forced_log_propagation(logger_name=optuna_distributed.__name__):
        msg.process(study, manager)
    assert manager.trial_exit_called
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    trial = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    assert len(trial) == 1


def test_failed(
    study: Study, manager: MockOptimizationManager, caplog: pytest.LogCaptureFixture
) -> None:
    exc = ValueError("foo")
    msg = FailedMessage(0, exc, exc_info=MagicMock())
    assert msg.closing
    logger_name = optuna_distributed.__name__
    with pytest.raises(ValueError), _forced_log_propagation(logger_name):
        msg.process(study, manager)
    assert manager.trial_exit_called
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
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
    "property",
    [
        "params",
        "distributions",
        "user_attrs",
        "system_attrs",
        "datetime_start",
        "number",
    ],
)
def test_trial_property(
    study: Study, manager: MockOptimizationManager, property: TrialProperty
) -> None:
    msg = TrialPropertyMessage(0, property)
    assert not msg.closing
    msg.process(study, manager)
    expected = getattr(study.get_trials(deepcopy=False)[0], property)
    assert _message_responds_with(expected, manager=manager)


def test_should_prune(study: Study, manager: MockOptimizationManager) -> None:
    msg = ShouldPruneMessage(0)
    assert not msg.closing
    msg.process(study, manager)

    assert _message_responds_with(False, manager=manager)
    trial = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    assert len(trial) == 1
    assert trial[0]._trial_id == 0


def test_report_intermediate(study: Study, manager: MockOptimizationManager) -> None:
    msg = ReportMessage(0, value=0.0, step=1)
    assert not msg.closing

    msg.process(study, manager)
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.intermediate_values[1] == 0.0


def test_set_user_attributes(study: Study, manager: MockOptimizationManager) -> None:
    msg = SetAttributeMessage(0, key="foo", value=0, kind="user")
    assert not msg.closing

    msg.process(study, manager)
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.user_attrs["foo"] == 0


def test_set_system_attributes(study: Study, manager: MockOptimizationManager) -> None:
    msg = SetAttributeMessage(0, value=0, key="foo", kind="system")
    assert not msg.closing

    msg.process(study, manager)
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.system_attrs["foo"] == 0


@pytest.mark.parametrize(
    "distribution",
    [
        FloatDistribution(low=0.0, high=1.0),
        IntDistribution(low=0, high=1),
        CategoricalDistribution(choices=["foo", "bar"]),
    ],
)
def test_suggest(
    study: Study, manager: MockOptimizationManager, distribution: BaseDistribution
) -> None:
    msg = SuggestMessage(0, name="x", distribution=distribution)
    assert not msg.closing

    msg.process(study, manager)
    trial = study.get_trials(deepcopy=False)[0]
    assert "x" in trial.distributions
    assert trial.distributions["x"] == distribution
    assert "x" in trial.params
    assert _message_responds_with(trial.params["x"], manager=manager)
