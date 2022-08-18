from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import TYPE_CHECKING
import uuid

from dask.distributed import Client
from dask.distributed import Future

from optuna_distributed.ipc import Queue
from optuna_distributed.managers import OptimizationManager
from optuna_distributed.messages import HeartbeatMessage
from optuna_distributed.trial import DistributedTrial


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.eventloop import EventLoop
    from optuna_distributed.ipc import IPCPrimitive
    from optuna_distributed.messages import Message


class DistributedOptimizationManager(OptimizationManager):
    """Controls optimization process spanning multiple physical machines.

    This implementation uses dask as parallel computing backend.
    """

    def __init__(self, client: Client, n_trials: int) -> None:
        self._client = client
        self._n_trials = n_trials
        self._completed_trials = 0
        self._public_channel = str(uuid.uuid4())

        # Manager has write access to its own message queue as a sort of health check.
        # Basically that means we can pump event loop from callbacks running in
        # main process with e.g. HeartbeatMessage.
        self._message_queue = Queue(self._public_channel, self._public_channel)
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

    def create_futures(
        self, study: "Study", objective: Callable[[DistributedTrial], None]
    ) -> None:
        # HACK: It's kinda naughty to access _trial_id, but this is gonna make
        # our lifes much easier in messaging system.
        trial_ids = [study.ask()._trial_id for _ in range(self._n_trials)]
        trials = [DistributedTrial(id, self._assign_private_channel(id)) for id in trial_ids]
        self._futures = self._client.map(objective, trials, pure=False)
        for future in self._futures:
            future.add_done_callback(self._ensure_safe_exit)

    def before_message(self, event_loop: "EventLoop") -> None:
        ...

    def get_message(self) -> Generator["Message", None, None]:
        while True:
            # TODO(xadrianzetx) At some point we might need a mechanism
            # that allows workers to repeat messages to master.
            # A deduplication algorithm would go here then.
            yield self._message_queue.get()

    def after_message(self, event_loop: "EventLoop") -> None:
        ...

    def get_connection(self, trial_id: int) -> "IPCPrimitive":
        return Queue(self._private_channels[trial_id])

    def stop_optimization(self) -> None:
        for future in self._futures:
            if not future.done():
                future.release()

    def should_end_optimization(self) -> bool:
        return self._completed_trials == self._n_trials

    def register_trial_exit(self, trial_id: int) -> None:
        self._completed_trials += 1
