from typing import Any
from typing import TYPE_CHECKING

from optuna import logging
from optuna.trial import Trial
from optuna.trial import TrialState

from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers import OptimizationManager


_logger = logging.get_logger(__name__)


class FailedMessage(Message):
    """A failed trial message.

    This message is sent to inform a client about failed execution
    of the objective function. Client can then tell study about it.
    """

    def __init__(self, trial_id: int, exception: Exception, exc_info: Any) -> None:
        self._trial_id = trial_id
        self._exception = exception
        self._exc_info = exc_info

    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        frozen_trial = study.tell(trial, state=TrialState.FAIL)
        manager.register_trial_exit(self._trial_id)
        _logger.warning(
            f"Trial {frozen_trial.number} failed because "
            f"of the following error: {repr(self._exception)}",
            exc_info=self._exc_info,
        )
        # TODO(xadrianzetx) Implement exception catching.
        # https://github.com/optuna/optuna/blob/5d19e5e1f5dd9b3f9a11c74d215bd2a9c7ff43d2/optuna/study/_optimize.py#L229-L234
        raise self._exception
