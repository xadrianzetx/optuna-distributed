import asyncio
import ctypes
from dataclasses import dataclass
from enum import IntEnum
import logging
import sys
import threading
from threading import Thread
import time
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import TYPE_CHECKING
import uuid

from dask.distributed import Client
from dask.distributed import Future
from dask.distributed import LocalCluster
from dask.distributed import Variable
from optuna.exceptions import TrialPruned

from optuna_distributed.ipc import Queue
from optuna_distributed.managers import ObjectiveFuncType
from optuna_distributed.managers import OptimizationManager
from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.messages import PrunedMessage
from optuna_distributed.trial import DistributedTrial


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.eventloop import EventLoop
    from optuna_distributed.ipc import IPCPrimitive
    from optuna_distributed.messages import Message


DistributableWithContext = Callable[["_TaskContext"], None]
_logger = logging.getLogger(__name__)


class WorkerInterrupted(Exception):
    ...


class _TaskState(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


@dataclass
class _TaskContext:
    trial: DistributedTrial
    stop_flag: str
    task_state_var: str


class _StateSynchronizer:
    def __init__(self) -> None:
        self._optimization_enabled = Variable()
        self._optimization_enabled.set(True)
        self._task_states: List[Variable] = []

    @property
    def stop_flag(self) -> str:
        return self._optimization_enabled.name

    def set_initial_state(self) -> str:
        task_state = Variable()
        task_state.set(_TaskState.WAITING)
        self._task_states.append(task_state)
        return task_state.name

    def emit_stop_and_wait(self, patience: int) -> None:
        self._optimization_enabled.set(False)
        disabled_at = time.time()
        _logger.info("Interrupting running tasks...")
        while any(state.get() == _TaskState.RUNNING for state in self._task_states):
            if time.time() - disabled_at > patience:
                raise TimeoutError("Timed out while trying to interrupt running tasks.")
            time.sleep(0.1)
        _logger.info("All tasks have been stopped.")


class DistributedOptimizationManager(OptimizationManager):
    """Controls optimization process spanning multiple physical machines.

    This implementation uses dask as parallel computing backend.

    Args:
        client:
            An instance of dask client.
        n_trials:
            Number of trials to run.
        heartbeat_interval:
            Delay (in seconds) before
            :func:`optuna_distributed.managers.DistributedOptimizationManager.get_message`
            produces a heartbeat message if no other message is sent by the worker.
    """

    def __init__(self, client: Client, n_trials: int, heartbeat_interval: int = 60) -> None:
        self._client = client
        self._n_trials = n_trials
        self._completed_trials = 0
        self._public_channel = str(uuid.uuid4())

        # Manager has write access to its own message queue as a sort of health check.
        # Basically that means we can pump event loop from callbacks running in
        # main process with e.g. HeartbeatMessage.
        self._message_queue = Queue(
            publishing=self._public_channel,
            recieving=self._public_channel,
            timeout=heartbeat_interval,
        )
        self._private_channels: Dict[int, str] = {}
        self._futures: List[Future] = []

    def _ensure_safe_exit(self, future: Future) -> None:
        if future.status in ["error", "cancelled"]:
            # FIXME: I'm not sure if there is a way to get
            # id of a trial that failed this way.
            self.register_trial_exit(-1)
            self._message_queue.put(HeartbeatMessage())

    def _assign_private_channel(self, trial_id: int) -> "Queue":
        private_channel = str(uuid.uuid4())
        self._private_channels[trial_id] = private_channel
        return Queue(self._public_channel, private_channel)

    def create_futures(self, study: "Study", objective: ObjectiveFuncType) -> None:
        # HACK: It's kinda naughty to access _trial_id, but this is gonna make
        # our lifes much easier in messaging system.
        distributable = _distributable(objective)
        trial_ids = [study.ask()._trial_id for _ in range(self._n_trials)]
        trials = [DistributedTrial(id, self._assign_private_channel(id)) for id in trial_ids]
        self._futures = self._client.map(distributable, trials, pure=False)
        for future in self._futures:
            future.add_done_callback(self._ensure_safe_exit)

    def before_message(self, event_loop: "EventLoop") -> None:
        ...

    def get_message(self) -> Generator["Message", None, None]:
        while True:
            try:
                # TODO(xadrianzetx) At some point we might need a mechanism
                # that allows workers to repeat messages to master.
                # A deduplication algorithm would go here then.
                yield self._message_queue.get()
            except asyncio.TimeoutError:
                # Pumping event loop with heartbeat messages on timeout
                # allows us to handle potential problems gracefully
                # e.g. in `after_message`.
                yield HeartbeatMessage()

    def after_message(self, event_loop: "EventLoop") -> None:
        ...

    def get_connection(self, trial_id: int) -> "IPCPrimitive":
        return Queue(self._private_channels[trial_id])

    def stop_optimization(self) -> None:
        # This will only cancel scheduled tasks.
        # There's not much we can do about ones already running.
        # https://stackoverflow.com/a/49203129
        # https://github.com/dask/distributed/issues/4694

        # FIXME: Could use variable as global stopping condition for all workers,
        # but we can't expect users to check for it in objective function.
        # I guess objective wrapper must check for it in another thread and be able to interrupt.
        # https://docs.dask.org/en/stable/futures.html#distributed.Variable
        self._client.cancel(self._futures)

    def should_end_optimization(self) -> bool:
        return self._completed_trials == self._n_trials

    def register_trial_exit(self, trial_id: int) -> None:
        self._completed_trials += 1


def _distributable(func: ObjectiveFuncType) -> DistributableFuncType:
    def _wrapper(trial: DistributedTrial) -> None:
        # FIXME: Re-introduce task deduplication.
        message: Message
        try:
            value_or_values = func(trial)
            message = CompletedMessage(trial.trial_id, value_or_values)
            trial.connection.put(message)

        except TrialPruned as e:
            message = PrunedMessage(trial.trial_id, e)
            trial.connection.put(message)

        except Exception as e:
            exc_info = sys.exc_info()
            message = FailedMessage(trial.trial_id, e, exc_info)
            trial.connection.put(message)

        finally:
            trial.connection.close()

    return _wrapper
