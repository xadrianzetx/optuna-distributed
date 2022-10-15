from dataclasses import dataclass
import multiprocessing
import time

from dask.distributed import Client
from dask.distributed import wait
import optuna

from optuna_distributed.managers import DistributedOptimizationManager
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.managers import ObjectiveFuncType
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


def test_distributed_stops_optimziation(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> float:
        time.sleep(2.0)
        return 0.0

    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials=5)
    manager.create_futures(study, _objective)
    manager.stop_optimization()
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


def test_local_get_message() -> None:
    def _objective(trial: DistributedTrial) -> float:
        trial.connection.put(ResponseMessage(0, data=None))
        return 0.0

    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    manager.create_futures(study, _objective)
    completed = 0
    for message in manager.get_message():
        assert isinstance(message, ResponseMessage)
        completed += 1
        if completed == n_trials:
            break


def test_local_should_end_optimization() -> None:
    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    manager.create_futures(study, lambda trial: 0.0)
    closing_messages_recieved = 0
    for message in manager.get_message():
        if message.closing:
            closing_messages_recieved += 1
            manager.register_trial_exit(message._trial_id)  # type: ignore

        if manager.should_end_optimization():
            break

    assert closing_messages_recieved == n_trials


def test_local_connection_management() -> None:
    def _objective(trial: DistributedTrial) -> float:
        requested = trial.connection.get()
        assert isinstance(requested, ResponseMessage)
        data = {"requested": requested.data, "actual": trial.trial_id}
        trial.connection.put(ResponseMessage(trial.trial_id, data))
        return 0.0

    n_trials = 1
    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials, n_jobs=1)
    manager.create_futures(study, _objective)
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


def test_local_worker_pool_management() -> None:
    def _objective(trial: DistributedTrial) -> float:
        return 0.0

    @dataclass
    class _MockEventLoop:
        study: optuna.Study
        objective: ObjectiveFuncType

    study = optuna.create_study()
    manager = LocalOptimizationManager(n_trials=10, n_jobs=-1)
    eventloop = _MockEventLoop(study, _objective)

    manager.create_futures(study, _objective)
    for message in manager.get_message():
        message.process(study, manager)
        manager.after_message(eventloop)  # type: ignore
        if not manager.should_end_optimization():
            assert 0 < len(manager._pool) <= multiprocessing.cpu_count()
        else:
            break
