import abc
from abc import ABC
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers.base import OptimizationManager


class Message(ABC):
    """Base class for for IPC messages.

    These messages are used to pass data and code between client and workers.
    """

    @abc.abstractmethod
    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        """Process a message data with context available in main process.

        Concrete implementations of this method should contain operations that
        worker wants to execute using resources available only to the main process.
        This means stuff like hyperparameter suggestions, prune commands and general
        data passing.

        Args:
            study:
                An instance of Optuna study.
            manager:
                :class:`~optuna_distributed.managers.Manager` providing additional
                execution context.
        """
        raise NotImplementedError
