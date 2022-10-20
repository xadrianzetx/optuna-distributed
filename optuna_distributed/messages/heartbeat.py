from typing import TYPE_CHECKING

from optuna.study import Study

from optuna_distributed.messages.base import Message


if TYPE_CHECKING:
    from optuna_distributed.managers.base import OptimizationManager


class HeartbeatMessage(Message):
    """A heartbeat message.

    Heartbeat messages do not carry any code or data. Their purpose
    is to generate some traffic on communication channels to confirm
    some things are still alive and well.
    """

    closing = False

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        ...
