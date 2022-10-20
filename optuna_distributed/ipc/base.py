import abc
from abc import ABC

from optuna_distributed.messages.base import Message


class IPCPrimitive(ABC):
    """An inter process communication primitive.

    This interface defines a common way to pass messages
    between processes hosted on the same machine or in cluster setups.
    """

    @abc.abstractmethod
    def get(self) -> Message:
        """Retrieves a single message."""
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, message: Message) -> None:
        """Publishes a single message.

        Args:
            message:
                An instance of :class:'~optuna_distributed.messages.Message'.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Closes communication channel."""
        raise NotImplementedError
