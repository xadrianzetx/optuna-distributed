import logging
from typing import TYPE_CHECKING

from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import Trial
from optuna.trial import TrialState

from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager


_logger = logging.getLogger(__name__)


class PrunedMessage(Message):
    """A pruned trial message.

    This message is sent after :obj:`TrialPruned` exception has been rised
    in objective function and tells study to set associated trial to pruned state.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        exception:
            Instance of :obj:`TrialPruned` exception.
    """

    closing = True

    def __init__(self, trial_id: int, exception: TrialPruned) -> None:
        self._trial_id = trial_id
        self._exception = exception

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        frozen_trial = study.tell(trial, state=TrialState.PRUNED)
        manager.register_trial_exit(self._trial_id)
        _logger.info(f"Trial {frozen_trial.number} pruned. {repr(self._exception)}")
