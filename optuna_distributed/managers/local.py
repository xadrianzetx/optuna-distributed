import multiprocessing
from multiprocessing import Pipe as MultiprocessingPipe
from multiprocessing import Process
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import TYPE_CHECKING

from optuna_distributed.ipc import Pipe
from optuna_distributed.managers import OptimizationManager
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.trial import DistributedTrial


if TYPE_CHECKING:
    from optuna import Study

    from optuna_distributed.eventloop import EventLoop
    from optuna_distributed.ipc import IPCPrimitive
    from optuna_distributed.messages import Message


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
        self._pool: Dict[int, Connection] = {}

    def create_futures(
        self, study: "Study", objective: Callable[[DistributedTrial], None]
    ) -> None:
        trial_ids = [study.ask()._trial_id for _ in range(self._workers_to_spawn)]
        for trial_id in trial_ids:
            master, worker = MultiprocessingPipe()
            trial = DistributedTrial(trial_id, Pipe(worker))
            Process(target=objective, args=(trial,), daemon=True).start()
            self._pool[trial_id] = master
            worker.close()

    def before_message(self, event_loop: "EventLoop") -> None:
        ...

    def get_message(self) -> Generator["Message", None, None]:
        while True:
            messages: List["Message"] = []
            for incoming in wait(self._pool.values(), timeout=10):
                try:
                    message = incoming.recv()
                    messages.append(message)

                except EOFError:
                    for trial_id, connection in self._pool.items():
                        if incoming == connection:
                            break
                    self._pool.pop(trial_id)

            self._workers_to_spawn = min(self._n_jobs - len(self._pool), self._trials_remaining)
            if len(messages) > 0:
                yield from messages
            else:
                yield HeartbeatMessage()

    def after_message(self, event_loop: "EventLoop") -> None:
        if self._workers_to_spawn > 0:
            self.create_futures(event_loop.study, event_loop.objective)
            self._trials_remaining -= self._workers_to_spawn

    def get_connection(self, trial_id: int) -> "IPCPrimitive":
        return Pipe(self._pool[trial_id])

    def stop_optimization(self) -> None:
        # Noop here, deamonic processes are terminated when parent exits.
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.daemon
        ...

    def should_end_optimization(self) -> bool:
        return len(self._pool) == 0 and self._trials_remaining == 0

    def register_trial_exit(self, trial_id: int) -> None:
        # Noop, as worker informs us about exit by closing connection.
        ...