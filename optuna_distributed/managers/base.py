import abc
from abc import ABC
from typing import Callable
from typing import Generator
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.eventloop import EventLoop
    from optuna_distributed.ipc import IPCPrimitive
    from optuna_distributed.messages import Message
    from optuna_distributed.trial import DistributedTrial


class OptimizationManager(ABC):
    """Controls and provides context in event loop.

    Managers serve as a layer of abstraction between main process event loop
    and distributed workers. They can provide workers with context necessary
    to do the job, and pump event loop with messages to process.
    """

    @abc.abstractmethod
    def create_futures(
        self, study: "Study", objective: Callable[["DistributedTrial"], None]
    ) -> None:
        """Spawns a set of workers running objective function.

        Args:
            study:
                An instance of Optuna study.
            objective:
                Objective function to run on worker.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def before_message(self, event_loop: "EventLoop") -> None:
        """A hook allowing to run additional operations before recieved
        message is processed.

        Args:
            event_loop:
                An instance of :class:`~optuna_distributed.eventloop.EventLoop`
                providing context to study and manager.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_message(self) -> Generator["Message", None, None]:
        """Fetches incoming messages from workers."""
        raise NotImplementedError

    @abc.abstractmethod
    def after_message(self, event_loop: "EventLoop") -> None:
        """A hook allowing to run additional operations after recieved
        message is processed.

        Args:
            event_loop:
                An instance of :class:`~optuna_distributed.eventloop.EventLoop`
                providing context to study and manager.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_connection(self, trial_id: int) -> "IPCPrimitive":
        """Fetches private connection to worker.

        Args:
            trial_id:
                A connection to worker running trial with specified
                id will be fetched.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop_optimization(self) -> None:
        """Stops all running trials and sets thier statuses to failed."""
        raise NotImplementedError

    @abc.abstractmethod
    def should_end_optimization(self) -> bool:
        """Indicates whether optimization process can be finished.

        Returns :obj:`True` when all workers have send one of closing
        messages, indicating completed, pruned or failed trials.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_run_repeated(self, study: "Study", trial_id: int) -> bool:
        """Indicates whether trial with specified id has already been completed.

        Some backends might allow tasks to be re-run in which case is usefull to
        check for that case ahead of time.

        Args:
            study:
                An instance of Optuna study.
            trial_id:
                Trial id to check a status of.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def register_trial_exit(self, trial_id: int) -> None:
        """Informs manager about worker finishing a trial.

        This should be called in one of closing messages to indicate
        worker finishing with expected state.

        Args:
            trial_id:
                Id of a trial that was being run on exiting worker.
        """
        raise NotImplementedError