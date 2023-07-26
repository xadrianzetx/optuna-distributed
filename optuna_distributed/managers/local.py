from __future__ import annotations

from collections.abc import Generator
import multiprocessing
from multiprocessing import Pipe as MultiprocessingPipe
from multiprocessing import Process
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
import sys
from typing import TYPE_CHECKING

from optuna import Study
from optuna.exceptions import TrialPruned

from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.ipc import Pipe
from optuna_distributed.managers import ObjectiveFuncType
from optuna_distributed.managers import OptimizationManager
from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.messages import Message
from optuna_distributed.messages import PrunedMessage
from optuna_distributed.trial import DistributedTrial


if TYPE_CHECKING:
    from optuna_distributed.eventloop import EventLoop


class LocalOptimizationManager(OptimizationManager):
    """Controls optimization process on local machine.

    In contrast to Optuna, this implementation uses process based parallelism.

    Args:
        n_trials:
            Number of trials to run.
        n_jobs:
            Maximum number of processes allowed to run trials at the same time.
            If less or equal to 0, then this argument is overridden with CPU count.
    """

    def __init__(self, n_trials: int, n_jobs: int) -> None:
        if n_jobs <= 0 or n_jobs > multiprocessing.cpu_count():
            self._n_jobs = multiprocessing.cpu_count()
        else:
            self._n_jobs = n_jobs

        self._workers_to_spawn = min(self._n_jobs, n_trials)
        self._trials_remaining = n_trials - self._workers_to_spawn
        self._pool: dict[int, Connection] = {}
        self._processes: list[Process] = []

    def create_futures(self, study: Study, objective: ObjectiveFuncType) -> None:
        trial_ids = [study.ask()._trial_id for _ in range(self._workers_to_spawn)]
        for trial_id in trial_ids:
            master, worker = MultiprocessingPipe()
            trial = DistributedTrial(trial_id, Pipe(worker))
            p = Process(target=_trial_runtime, args=(objective, trial), daemon=True)
            p.start()
            self._processes.append(p)
            self._pool[trial_id] = master
            worker.close()

    def get_message(self) -> Generator[Message, None, None]:
        while True:
            messages: list[Message] = []
            for incoming in wait(self._pool.values(), timeout=10):
                # FIXME: This assertion is true only for Unix systems.
                # Some refactoring is needed to support Windows as well.
                # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.connection.wait
                assert isinstance(incoming, Connection)
                try:
                    message = incoming.recv()
                    messages.append(message)

                except EOFError:
                    for trial_id, connection in self._pool.items():
                        if incoming == connection:
                            break
                    self._pool.pop(trial_id)

            self._workers_to_spawn = min(self._n_jobs - len(self._pool), self._trials_remaining)
            if messages:
                yield from messages
            else:
                yield HeartbeatMessage()

    def after_message(self, event_loop: "EventLoop") -> None:
        if self._workers_to_spawn > 0:
            self.create_futures(event_loop.study, event_loop.objective)
            self._trials_remaining -= self._workers_to_spawn
            self._workers_to_spawn = 0

    def get_connection(self, trial_id: int) -> IPCPrimitive:
        return Pipe(self._pool[trial_id])

    def stop_optimization(self, patience: float) -> None:
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=patience)

    def should_end_optimization(self) -> bool:
        return len(self._pool) == 0 and self._trials_remaining == 0

    def register_trial_exit(self, trial_id: int) -> None:
        # Noop, as worker informs us about exit by closing connection.
        ...


def _trial_runtime(func: ObjectiveFuncType, trial: DistributedTrial) -> None:
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
