from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

from optuna import logging
from optuna.trial import Trial
from optuna.trial import TrialState

from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers import OptimizationManager


_logger = logging.get_logger(__name__)


class CompletedMessage(Message):
    """A completed trial message.

    This message is sent after objective function has been successfully evaluated
    and tells study about objective value (in case of single objective optimization)
    or sequence of objective values (in case of multi-objective optimization).

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        value_or_values:
            Objective value or sequence of objective values.
    """

    closing = True

    def __init__(self, trial_id: int, value_or_values: Union[Sequence[float], float]) -> None:
        self._trial_id = trial_id
        self._value_or_values = value_or_values

    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        try:
            frozen_trial = study.tell(trial, self._value_or_values, skip_if_finished=True)

        except Exception:
            frozen_trial = study._storage.get_trial(self._trial_id)
            raise

        finally:
            manager.register_trial_exit(self._trial_id)
            if frozen_trial.state == TrialState.COMPLETE:
                study._log_completed_trial(frozen_trial)
            else:
                # Tell failed to postprocess trial, so state has changed.
                _logger.warning(
                    f"Trial {frozen_trial.number} failed because "
                    "of the following error: STUDY_TELL_WARNING"
                )
