from typing import Literal
from typing import TYPE_CHECKING

from optuna.study import Study
from optuna.trial import Trial

from optuna_distributed.messages import Message
from optuna_distributed.messages.response import ResponseMessage


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager


TrialProperty = Literal[
    "params", "distributions", "user_attrs", "system_attrs", "datetime_start", "number"
]


class TrialPropertyMessage(Message):
    """Requests one of trial properties.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        property:
            An option from :class:`~optuna_distributed.messages.TrialProperty` enum.
    """

    closing = False

    def __init__(self, trial_id: int, property: TrialProperty) -> None:
        self._trial_id = trial_id
        self._property = property

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        conn = manager.get_connection(self._trial_id)
        conn.put(ResponseMessage(self._trial_id, getattr(trial, self._property)))
