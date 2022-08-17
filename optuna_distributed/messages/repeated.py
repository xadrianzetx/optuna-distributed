from typing import TYPE_CHECKING

from optuna_distributed.messages import GenericMessage
from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers import OptimizationManager


class RepeatedTrialMessage(Message):
    """A repeated trial message.

    This message is sent by worker to confirm that it's not about
    to re-run a completed trial. This is a safeguard against situation
    described in https://stackoverflow.com/a/41965766.
    """

    def __init__(self, trial_id: int) -> None:
        self._trial_id = trial_id

    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        is_repeated = manager.is_run_repeated(study, self._trial_id)
        conn = manager.get_connection(self._trial_id)
        conn.put(GenericMessage(self._trial_id, data=is_repeated))
