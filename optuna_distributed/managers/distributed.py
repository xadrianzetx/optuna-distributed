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
        self._synchronizer = _StateSynchronizer()
        self._is_distributed = not isinstance(client.cluster, LocalCluster)

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
        return Queue(self._public_channel, private_channel, timeout=5)

    def _create_trials_with_context(self, trial_ids: List[int]) -> List[_TaskContext]:
        context: List[_TaskContext] = []
        for trial_id in trial_ids:
            trial = DistributedTrial(trial_id, self._assign_private_channel(trial_id))
            context.append(
                _TaskContext(
                    trial,
                    stop_flag=self._synchronizer.stop_flag,
                    task_state_var=self._synchronizer.set_initial_state(),
                )
            )

        return context

    def create_futures(self, study: "Study", objective: ObjectiveFuncType) -> None:
        # HACK: It's kinda naughty to access _trial_id, but this is gonna make
        # our lifes much easier in messaging system.
        trial_ids = [study.ask()._trial_id for _ in range(self._n_trials)]
        distributable = _distributable(objective, with_supervisor=self._is_distributed)
        trials = self._create_trials_with_context(trial_ids)
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
        # Only want to cleanup cluster that does not belong to us.
        # TODO(xadrianzetx) Notebooks might be a special case (cleanup even with LocalCluster).
        self._client.cancel(self._futures)
        if self._is_distributed:
            # Twice the timeout of task connection.
            # This way even tasks waiting for message will have chance to exit.
            self._synchronizer.emit_stop_and_wait(patience=10)

    def should_end_optimization(self) -> bool:
        return self._completed_trials == self._n_trials

    def register_trial_exit(self, trial_id: int) -> None:
        self._completed_trials += 1


def _distributable(func: ObjectiveFuncType, with_supervisor: bool) -> DistributableWithContext:
    def _wrapper(context: _TaskContext) -> None:
        # FIXME: Re-introduce task deduplication.
        task_state = Variable(context.task_state_var)
        task_state.set(_TaskState.RUNNING)
        message: Message

        try:
            if with_supervisor:
                args = (threading.get_ident(), context)
                Thread(target=_task_supervisor, args=args, daemon=True).start()

            value_or_values = func(context.trial)
            message = CompletedMessage(context.trial.trial_id, value_or_values)
            context.trial.connection.put(message)

        except TrialPruned as e:
            message = PrunedMessage(context.trial.trial_id, e)
            context.trial.connection.put(message)

        except WorkerInterrupted:
            ...

        except Exception as e:
            exc_info = sys.exc_info()
            message = FailedMessage(context.trial.trial_id, e, exc_info)
            context.trial.connection.put(message)

        finally:
            context.trial.connection.close()
            task_state.set(_TaskState.FINISHED)

    return _wrapper


def _task_supervisor(thread_id: int, context: _TaskContext) -> None:
    optimization_enabled = Variable(context.stop_flag)
    task_state = Variable(context.task_state_var)
    while True:
        time.sleep(0.1)
        if task_state.get() == _TaskState.FINISHED:
            break

        if not optimization_enabled.get():
            # https://gist.github.com/liuw/2407154
            # https://distributed.dask.org/en/stable/worker-state.html#task-cancellation
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), ctypes.py_object(WorkerInterrupted)
            )
            break
