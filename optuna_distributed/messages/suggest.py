from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import Trial

from optuna_distributed.messages import GenericMessage
from optuna_distributed.messages import Message


if TYPE_CHECKING:
    from optuna.study import Study

    from optuna_distributed.managers import OptimizationManager


class SuggestMessage(Message):
    """ A request for value suggestions.

    This message is sent by :class:`~optuna_distributed.trial.DistributedTrial` to
    main process asking for value suggestions. Main process provides them by
    using regular Optuna suggest APIs and responding via connection provided by worker.
    """

    def __init__(self, trial_id: int, name: str, distribution: BaseDistribution) -> None:
        self._trial_id = trial_id
        self._name = name
        self._distribution = distribution

    def process(self, study: "Study", manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        if manager.is_run_repeated(self._trial_id):
            return

        if isinstance(self._distribution, FloatDistribution):
            value = trial.suggest_float(
                name=self._name,
                low=self._distribution.low,
                high=self._distribution.high,
                step=self._distribution.step,
                log=self._distribution.log,
            )
        elif isinstance(self._distribution, IntDistribution):
            value = trial.suggest_int(
                name=self._name,
                low=self._distribution.low,
                high=self._distribution.high,
                step=self._distribution.step,
                log=self._distribution.log,
            )
        elif isinstance(self._distribution, CategoricalDistribution):
            value = trial.suggest_categorical(name=self._name, choices=self._distribution.choices)
        else:
            raise ValueError()

        conn = manager.get_connection(self._trial_id)
        conn.put(GenericMessage(self._trial_id, value))
