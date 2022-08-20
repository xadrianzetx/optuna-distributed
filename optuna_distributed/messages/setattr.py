from enum import Enum
from typing import Any
from typing import TYPE_CHECKING

from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers import OptimizationManager


class AttributeType(Enum):
    USER = "user"
    SYSTEM = "system"


class SetAttributeMessage(Message):
    """Sets either user or system value on a trial.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        kind:
            An option from :class:`~optuna_distributed.messages.AttributeType` enum.
        key:
            A key string of the attribute.
        value:
            A value of the attribute. The value should be able to serialize with pickle.
    """

    closing = False

    def __init__(self, trial_id: int, kind: AttributeType, key: str, value: Any) -> None:
        self._trial_id = trial_id
        self._kind = kind
        self._key = key
        self._value = value

    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        if self._kind == AttributeType.USER:
            study._storage.set_trial_user_attr(self._trial_id, self._key, self._value)
        elif self._kind == AttributeType.SYSTEM:
            study._storage.set_trial_system_attr(self._trial_id, self._key, self._value)
        else:
            assert False, "Should not reach."
