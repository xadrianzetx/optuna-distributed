from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar

from optuna.study import Study

from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager


# FIXME: Would be nice to bound it to pickable interface but there is no such thing at the moment.
# https://stackoverflow.com/questions/50328386/python-typing-pickle-and-serialisation
T = TypeVar("T")


class ResponseMessage(Message, Generic[T]):
    """A generic message.

    Response messages are used by client to pass data back to workers.
    These do not carry any code to execute, and should be used as a wrapper
    around data served as response to workers request.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        data:
            Serializable data that should be carried as a part of response.
    """

    closing = False

    def __init__(self, trial_id: int, data: T) -> None:
        self.trial_id = trial_id
        self.data = data

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        ...
