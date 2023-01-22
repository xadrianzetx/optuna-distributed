import logging
from typing import Any
from typing import TYPE_CHECKING

from optuna.study import Study
from optuna.trial import Trial
from optuna.trial import TrialState

from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager


_logger = logging.getLogger(__name__)


class FailedMessage(Message):
    """A failed trial message.

    This message is sent after objective function has failed while being evaluated
    and tells study to fail associated trial. Also, if exception that caused objective
    function to fail is not explicitly ignored by user, it will be re-raised in main
    process, failing it entirely.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        exception:
            Instance of exception that was raised in objective function.
        exc_info:
            Information about exception that was raised in objective function.
    """

    closing = True

    def __init__(self, trial_id: int, exception: Exception, exc_info: Any) -> None:
        self._trial_id = trial_id
        self._exception = exception
        self._exc_info = exc_info

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        frozen_trial = study.tell(trial, state=TrialState.FAIL)
        manager.register_trial_exit(self._trial_id)
        _logger.warning(
            f"Trial {frozen_trial.number} failed with parameters: {frozen_trial.params} "
            f"because of the following error: {repr(self._exception)}.",
            exc_info=self._exc_info,
        )
        raise self._exception
