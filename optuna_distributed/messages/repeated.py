from typing import TYPE_CHECKING

from optuna_distributed.messages import Message
from optuna_distributed.messages.response import ResponseMessage


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers import OptimizationManager


class RepeatedTrialMessage(Message):
    """A repeated trial message.

    This message is sent by worker to confirm that it's not about
    to re-run a completed trial. This is a safeguard against situation
    described in https://stackoverflow.com/a/41965766.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
    """

    closing = False

    def __init__(self, trial_id: int) -> None:
        self._trial_id = trial_id

    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        trial = study._storage.get_trial(self._trial_id)
        conn = manager.get_connection(self._trial_id)
        conn.put(ResponseMessage(self._trial_id, data=trial.state.is_finished()))
