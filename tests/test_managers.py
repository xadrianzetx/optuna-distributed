import time

from dask.distributed import Client
import optuna

from optuna_distributed.managers import DistributedOptimizationManager
from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.trial import DistributedTrial


def test_distributed_get_message(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> None:
        trial.connection.put(ResponseMessage(0, data=None))

    n_trials = 5
    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials)
    manager.create_futures(study, _objective)
    completed = 0
    for message in manager.get_message():
        assert isinstance(message, ResponseMessage)
        completed += 1
        if completed == n_trials:
            break


def test_distributed_get_message_timeout(client: Client) -> None:
    # TODO(xadrianzetx) Manager has to be able to pump event loop with heartbeat messages first.
    pass


def test_distributed_should_end_optimization(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> None:
        # Non-closing message first.
        trial.connection.put(ResponseMessage(trial.trial_id, data=None))
        # Closing message follows.
        trial.connection.put(CompletedMessage(trial.trial_id, 0.0))

    n_trials = 5
    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials)
    manager.create_futures(study, _objective)
    messages_recieved = 0
    for message in manager.get_message():
        messages_recieved += 1
        assert not isinstance(message, HeartbeatMessage)
        if isinstance(message, CompletedMessage):
            manager.register_trial_exit(message._trial_id)

        if manager.should_end_optimization():
            break

    assert messages_recieved == n_trials * 2


def test_distributed_registers_future_failure(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> None:
        # Simulate ungraceful faliure.
        assert False

    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials=5)
    manager.create_futures(study, _objective)
    messages = []
    for message in manager.get_message():
        messages.append(message)
        if manager.should_end_optimization():
            break

    assert all(isinstance(msg, HeartbeatMessage) for msg in messages)


def test_distributed_stops_optimziation(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> None:
        time.sleep(2.0)

    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials=5)
    manager.create_futures(study, _objective)
    manager.stop_optimization()
    for future in manager._futures:
        assert future.cancelled()


def test_distributed_connection_management(client: Client) -> None:
    def _objective(trial: DistributedTrial) -> None:
        requested = trial.connection.get()
        assert isinstance(requested, ResponseMessage)
        data = {"requested": requested.data, "actual": trial.trial_id}
        trial.connection.put(ResponseMessage(trial.trial_id, data))

    n_trials = 5
    study = optuna.create_study()
    manager = DistributedOptimizationManager(client, n_trials)
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
