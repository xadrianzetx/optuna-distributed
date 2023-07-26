from __future__ import annotations

from collections import deque

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
import pytest

from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.messages import Message
from optuna_distributed.messages import ReportMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.messages import SetAttributeMessage
from optuna_distributed.messages import ShouldPruneMessage
from optuna_distributed.messages import SuggestMessage
from optuna_distributed.messages import TrialPropertyMessage
from optuna_distributed.trial import DistributedTrial


class MockIPC(IPCPrimitive):
    def __init__(self) -> None:
        self.captured: list[Message] = []
        self.responses: deque[ResponseMessage] = deque()

    def get(self) -> "Message":
        return self.responses.popleft()

    def put(self, message: "Message") -> None:
        self.captured.append(message)

    def close(self) -> None:
        ...

    def enqueue_response(self, response: ResponseMessage) -> None:
        self.responses.append(response)


@pytest.fixture
def connection() -> MockIPC:
    return MockIPC()


def test_suggest_float(connection: MockIPC) -> None:
    connection.enqueue_response(ResponseMessage(0, data=0.0))
    trial = DistributedTrial(0, connection)
    x = trial.suggest_float("x", low=0.0, high=1.0)
    assert x == 0.0
    captured = connection.captured[0]
    assert isinstance(captured, SuggestMessage)
    assert captured._trial_id == 0
    assert captured._name == "x"

    distribution = captured._distribution
    assert isinstance(distribution, FloatDistribution)
    assert distribution.low == 0.0
    assert distribution.high == 1.0
    assert not distribution.log
    assert distribution.step is None


def test_suggest_int(connection: MockIPC) -> None:
    connection.enqueue_response(ResponseMessage(0, data=0))
    trial = DistributedTrial(0, connection)
    x = trial.suggest_int("x", low=0, high=1)
    assert x == 0
    captured = connection.captured[0]
    assert isinstance(captured, SuggestMessage)
    assert captured._trial_id == 0
    assert captured._name == "x"

    distribution = captured._distribution
    assert isinstance(distribution, IntDistribution)
    assert distribution.low == 0
    assert distribution.high == 1
    assert distribution.step == 1
    assert not distribution.log


def test_suggest_categorical(connection: MockIPC) -> None:
    connection.enqueue_response(ResponseMessage(0, data="foo"))
    trial = DistributedTrial(0, connection)
    x = trial.suggest_categorical("x", choices=["foo", "bar", "baz"])
    assert x == "foo"
    captured = connection.captured[0]
    assert isinstance(captured, SuggestMessage)
    assert captured._trial_id == 0
    assert captured._name == "x"

    distribution = captured._distribution
    assert isinstance(distribution, CategoricalDistribution)
    assert distribution.choices == ("foo", "bar", "baz")


def test_report(connection: MockIPC) -> None:
    trial = DistributedTrial(0, connection)
    trial.report(value=0.0, step=1)
    captured = connection.captured[0]
    assert isinstance(captured, ReportMessage)
    assert captured._trial_id == 0
    assert captured._step == 1
    assert captured._value == 0.0


def test_should_prune(connection: MockIPC) -> None:
    connection.enqueue_response(ResponseMessage(0, data=False))
    trial = DistributedTrial(0, connection)
    assert not trial.should_prune()
    captured = connection.captured[0]
    assert isinstance(captured, ShouldPruneMessage)
    assert captured._trial_id == 0


def test_set_user_attr(connection: MockIPC) -> None:
    trial = DistributedTrial(0, connection)
    trial.set_user_attr(key="foo", value="bar")
    captured = connection.captured[0]
    assert isinstance(captured, SetAttributeMessage)
    assert captured._trial_id == 0
    assert captured._kind == "user"
    assert captured._key == "foo"
    assert captured._value == "bar"


def test_set_system_attr(connection: MockIPC) -> None:
    trial = DistributedTrial(0, connection)
    trial.set_system_attr(key="foo", value="bar")
    captured = connection.captured[0]
    assert isinstance(captured, SetAttributeMessage)
    assert captured._trial_id == 0
    assert captured._kind == "system"
    assert captured._key == "foo"
    assert captured._value == "bar"


@pytest.mark.parametrize(
    "property",
    ["params", "distributions", "user_attrs", "system_attrs", "datetime_start", "number"],
)
def test_get_properties(connection: MockIPC, property: str) -> None:
    connection.enqueue_response(ResponseMessage(0, "foo"))
    trial = DistributedTrial(0, connection)
    assert getattr(trial, property) == "foo"
    captured = connection.captured[0]
    assert isinstance(captured, TrialPropertyMessage)
    assert captured._property == property
