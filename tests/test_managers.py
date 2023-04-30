from dataclasses import dataclass
import multiprocessing
import sys
import time
from unittest.mock import Mock
import uuid

from dask.distributed import Client
from dask.distributed import Variable
from dask.distributed import wait
import optuna
import pytest

from optuna_distributed.managers import DistributedOptimizationManager
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.managers import ObjectiveFuncType
from optuna_distributed.managers.distributed import _StateSynchronizer
from optuna_distributed.managers.distributed import _TaskContext
from optuna_distributed.managers.distributed import _TaskState
from optuna_distributed.managers.distributed import _distributable
from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.trial import DistributedTrial


def test_distributed_get_message(client: Client) -> None:
    n_trials = 5
    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials)
    manager.create_futures(study, lambda trial: 0.0)
    completed = 0
    for message in manager.get_message():
        assert isinstance(message, CompletedMessage)
        completed += 1
        if completed == n_trials:
            break


def test_distributed_heartbeat_on_timeout(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> float:
        time.sleep(2.0)
        return 0.0

    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials=1, heartbeat_interval=1)
    manager.create_futures(study, _objective)
    start = time.time()
    for message in manager.get_message():
        assert isinstance(message, HeartbeatMessage)
        assert 0.8 < time.time() - start < 1.2
        break

    wait(manager._futures)


def test_distributed_should_end_optimization(client: Client) -> None:
    n_trials = 5
    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials)
    manager.create_futures(study, lambda trial: 0.0)
    closing_messages_recieved = 0
    for message in manager.get_message():
        assert not isinstance(message, HeartbeatMessage)
        if message.closing:
            closing_messages_recieved += 1
            manager.register_trial_exit(message._trial_id)  # type: ignore

        if manager.should_end_optimization():
            break

    assert closing_messages_recieved == n_trials


def test_distributed_stops_optimization(client: Client) -> None:
    uninterrupted_execution_time = 100

    def _objective(trial: DistributedTrial) -> float:
        # Sleep needs to be fragemnted to read error indicator.
        for _ in range(uninterrupted_execution_time):
            time.sleep(1.0)
        return 0.0

    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials=5)
    manager.create_futures(study, _objective)
    stopped_at = time.time()
    manager.stop_optimization(patience=10.0)
    interrupted_execution_time = time.time() - stopped_at
    assert interrupted_execution_time < uninterrupted_execution_time
    for future in manager._futures:
        assert future.cancelled()


def test_distributed_connection_management(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> float:
        requested = trial.connection.get()
        assert isinstance(requested, ResponseMessage)
        data = {"requested": requested.data, "actual": trial.trial_id}
        trial.connection.put(ResponseMessage(trial.trial_id, data))
        return 0.0

    n_trials = 5
    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials)
    manager.create_futures(study, _objective)
    for trial in study.get_trials(deepcopy=False):
        connection = manager.get_connection(trial._trial_id)
        connection.put(ResponseMessage(0, data=trial._trial_id))

    for message in manager.get_message():
        if message.closing:
            manager.register_trial_exit(message._trial_id)  # type: ignore
        if isinstance(message, ResponseMessage):
            assert message.data["requested"] == message.data["actual"]
        if manager.should_end_optimization():
            break


def test_distributed_task_deduped(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> float:
        run_count = Variable("run_count")
        run_count.set(run_count.get() + 1)
        return 0.0

    run_count = Variable("run_count")
    run_count.set(0)
    state_id = uuid.uuid4().hex
    Variable(state_id).set(_TaskState.WAITING)

    # Simulate scenario where task run was repeated.
    # https://stackoverflow.com/a/41965766
    func = _distributable(_objective)
    context = _TaskContext(DistributedTrial(0, Mock()), stop_flag="foo", state_id=state_id)
    for _ in range(5):
        client.submit(func, context).result()

    assert run_count.get() == 1


def test_synchronizer_optimization_enabled() -> None:
    synchronizer = _StateSynchronizer()
    optimization_enabled = Variable(synchronizer.stop_flag)
    assert optimization_enabled.get()


def test_synchronizer_emits_stop() -> None:
    synchronizer = _StateSynchronizer()
    synchronizer.emit_stop_and_wait(1)
    optimization_enabled = Variable(synchronizer.stop_flag)
    assert not optimization_enabled.get()


def test_synchronizer_states_created() -> None:
    synchronizer = _StateSynchronizer()
    states = [Variable(synchronizer.set_initial_state()) for _ in range(10)]
    assert all(_TaskState(state.get()) is _TaskState.WAITING for state in states)


def test_synchronizer_timeout() -> None:
    synchronizer = _StateSynchronizer()
    task_state = Variable(synchronizer.set_initial_state())
    task_state.set(_TaskState.RUNNING)
    with pytest.raises(TimeoutError):
        synchronizer.emit_stop_and_wait(0)


def _objective_local_get_message(trial: DistributedTrial) -> float:
    trial.connection.put(ResponseMessage(0, data=None))
    return 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="Local optimization not supported on Windows.")
def test_local_get_message() -> None:
    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    manager.create_futures(study, _objective_local_get_message)
    completed = 0
    for message in manager.get_message():
        assert isinstance(message, ResponseMessage)
        completed += 1
        if completed == n_trials:
            break


def _objective_local_should_end_optimization(trial: DistributedTrial) -> float:
    return 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="Local optimization not supported on Windows.")
def test_local_should_end_optimization() -> None:
    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    manager.create_futures(study, _objective_local_should_end_optimization)
    closing_messages_recieved = 0
    for message in manager.get_message():
        if message.closing:
            closing_messages_recieved += 1
            manager.register_trial_exit(message._trial_id)  # type: ignore

        if manager.should_end_optimization():
            break

    assert closing_messages_recieved == n_trials


def _objective_local_stops_optimziation(trial: DistributedTrial) -> float:
    uninterrupted_execution_time = 5.0
    time.sleep(uninterrupted_execution_time)
    return 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="Local optimization not supported on Windows.")
def test_local_stops_optimziation() -> None:
    uninterrupted_execution_time = 5.0
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials=10, n_jobs=1)
    manager.create_futures(study, _objective_local_stops_optimziation)
    stopped_at = time.time()
    manager.stop_optimization(patience=10.0)
    interrupted_execution_time = time.time() - stopped_at
    assert interrupted_execution_time < uninterrupted_execution_time
    for process in manager._processes:
        assert not process.is_alive()


def _objective_local_connection_management(trial: DistributedTrial) -> float:
    requested = trial.connection.get()
    assert isinstance(requested, ResponseMessage)
    data = {"requested": requested.data, "actual": trial.trial_id}
    trial.connection.put(ResponseMessage(trial.trial_id, data))
    return 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="Local optimization not supported on Windows.")
def test_local_connection_management() -> None:
    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    manager.create_futures(study, _objective_local_connection_management)
    for trial in study.get_trials(deepcopy=False):
        connection = manager.get_connection(trial._trial_id)
        connection.put(ResponseMessage(0, data=trial._trial_id))

    recieved = 0
    for message in manager.get_message():
        assert isinstance(message, ResponseMessage)
        assert message.data["requested"] == message.data["actual"]
        recieved += 1
        if recieved == n_trials:
            break


def _objective_local_worker_pool_management(trial: DistributedTrial) -> float:
    return 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="Local optimization not supported on Windows.")
def test_local_worker_pool_management() -> None:
    @dataclass
    class _MockEventLoop:
        study: optuna.Study
        objective: ObjectiveFuncType

    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials=10, n_jobs=-1)
    eventloop = _MockEventLoop(study, _objective_local_worker_pool_management)

    manager.create_futures(study, _objective_local_worker_pool_management)
    for message in manager.get_message():
        message.process(study, manager)
        manager.after_message(eventloop)  # type: ignore
        if not manager.should_end_optimization():
            assert 0 < len(manager._pool) <= multiprocessing.cpu_count()
        else:
            break
