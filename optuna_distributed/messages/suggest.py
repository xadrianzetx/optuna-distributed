from __future__ import annotations

from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.trial import Trial

from optuna_distributed.messages import Message
from optuna_distributed.messages.response import ResponseMessage


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager


class SuggestMessage(Message):
    """A request for value suggestions.

    This message is sent by :class:`~optuna_distributed.trial.DistributedTrial` to
    main process asking for value suggestions. Main process provides them by
    using regular Optuna suggest APIs and responding via connection provided by worker.

    Args:
        trial_id:
            Id of a trial to which the message is referring.
        name:
            A parameter name.
        distribution:
            A parameter distribution.
    """

    closing = False

    def __init__(self, trial_id: int, name: str, distribution: BaseDistribution) -> None:
        self._trial_id = trial_id
        self._name = name
        self._distribution = distribution

    def process(self, study: Study, manager: "OptimizationManager") -> None:
        trial = Trial(study, self._trial_id)
        value: float | int | CategoricalChoiceType
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
            assert False, "Should not reach."

        conn = manager.get_connection(self._trial_id)
        conn.put(ResponseMessage(self._trial_id, value))
