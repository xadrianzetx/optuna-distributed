import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution

from optuna_distributed.messages import GenericMessage
from optuna_distributed.messages import SuggestMessage


if TYPE_CHECKING:
    from optuna_distributed.ipc import IPCPrimitive


class DistributedTrial:
    """Version of Optuna Trial designed to run on process or machine separate to the client.

    Args:
        trial_id:
            A trial ID that is automatically generated.
        connection:
            An instance of :class:'~optuna_distributed.connections.Connection'.
    """

    def __init__(self, trial_id: int, connection: "IPCPrimitive") -> None:
        self.trial_id = trial_id
        self.connection = connection

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        message = SuggestMessage(self.trial_id, name, distribution)
        self.connection.put(message)
        response = self.connection.get()
        assert isinstance(response, GenericMessage)
        return response.data

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:
        distribution = FloatDistribution(low, high, step=step, log=log)
        return self._suggest(name, distribution)

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        return self.suggest_float(name, low, high)

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        return self.suggest_float(name, low, high, log=True)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        return self.suggest_float(name, low, high, step=q)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        distribution = IntDistribution(low, high, log=log, step=step)
        return self._suggest(name, distribution)

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        distribution = CategoricalDistribution(choices)
        return self._suggest(name, distribution)

    def report(self, value: float, step: int) -> None:
        raise NotImplementedError

    def should_prune(self) -> bool:
        raise NotImplementedError

    def set_user_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def set_system_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    @property
    def params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        raise NotImplementedError

    @property
    def user_attrs(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def system_attrs(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:
        raise NotImplementedError

    @property
    def number(self) -> int:
        raise NotImplementedError
