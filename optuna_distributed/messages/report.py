from typing import TYPE_CHECKING

from optuna.study import Study
from optuna.trial import Trial

from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager


class ReportMessage(Message):
    """Reports trial intermediate values.

    This message is sent by :class:`~optuna_distributed.trial.DistributedTrial` to
    main process reporting on intermediate value in trial.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        value:
            An itermediate value returned from the objective function.
        step:
            Step of the trial (e.g., Epoch of neural network training).
    """

    closing = False

    def __init__(self, trial_id: int, value: float, step: int) -> None:
        self._trial_id = trial_id
        self._value = value
        self._step = step

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        trial.report(self._value, self._step)
