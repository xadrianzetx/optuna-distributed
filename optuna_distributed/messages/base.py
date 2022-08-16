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
        raise NotImplementedError
