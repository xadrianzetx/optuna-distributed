from typing import TYPE_CHECKING

from optuna.study import Study
from optuna.trial import Trial

from optuna_distributed.messages import Message
from optuna_distributed.messages.response import ResponseMessage


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager


class ShouldPruneMessage(Message):
    """A should prune trial message.

    This message is sent by :class:`~optuna_distributed.trial.DistributedTrial` to
    main process asking for whether trial should be pruned or not.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
    """

    closing = False

    def __init__(self, trial_id: int) -> None:
        self._trial_id = trial_id

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        conn = manager.get_connection(self._trial_id)
        conn.put(ResponseMessage(self._trial_id, trial.should_prune()))
