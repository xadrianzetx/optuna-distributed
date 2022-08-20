from enum import auto
from enum import Enum
from typing import TYPE_CHECKING

from optuna.trial import Trial

from optuna_distributed.messages import Message
from optuna_distributed.messages import ResponseMessage


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers import OptimizationManager


class TrialProperty(Enum):
    PARAMS = auto()
    DISTRIBUTIONS = auto()
    USER_ATTRS = auto()
    SYSTEM_ATTRS = auto()
    DATETIME_START = auto()
    NUMBER = auto()


class TrialPropertyMessage(Message):
    """Requests one of trial properties.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        trial_property:
            An option from :class:`~optuna_distributed.messages.TrialProperty` enum.
    """

    closing = False

    def __init__(self, trial_id: int, trial_property: TrialProperty) -> None:
        self._trial_id = trial_id
        self._trial_property = trial_property

    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        data = {
            TrialProperty.PARAMS: trial.params,
            TrialProperty.DISTRIBUTIONS: trial.distributions,
            TrialProperty.USER_ATTRS: trial.user_attrs,
            TrialProperty.SYSTEM_ATTRS: trial.system_attrs,
            TrialProperty.DATETIME_START: trial.datetime_start,
            TrialProperty.NUMBER: trial.number,
        }[self._trial_property]

        conn = manager.get_connection(self._trial_id)
        conn.put(ResponseMessage(self._trial_id, data))
