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

from optuna_distributed.messages import ReportMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.messages import SuggestMessage
from optuna_distributed.messages.shouldprune import ShouldPruneMessage


if TYPE_CHECKING:
    from optuna_distributed.ipc import IPCPrimitive


class DistributedTrial:
    """Version of Optuna Trial designed to run on process or machine separate to the client.

    Args:
        trial_id:
            A trial ID that is automatically generated.
        connection:
            An instance of :class:`~optuna_distributed.connections.Connection`.
    """

    def __init__(self, trial_id: int, connection: "IPCPrimitive") -> None:
        self.trial_id = trial_id
        self.connection = connection

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        message = SuggestMessage(self.trial_id, name, distribution)
        self.connection.put(message)
        response = self.connection.get()
        assert isinstance(response, ResponseMessage)
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
        """Suggest a value for the floating point parameter.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
                ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,
                ``low`` must be larger than 0.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
                ``high`` must be greater than or equal to ``low``.
            step:
                A step of discretization.
            log:
                A flag to sample the value from the log domain or not.
                If ``log`` is true, the value is sampled from the range in the log domain.
                Otherwise, the value is sampled from the range in the linear domain.
        """
        distribution = FloatDistribution(low, high, step=step, log=log)
        return self._suggest(name, distribution)

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value for the continuous parameter.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_uniform

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
        """
        return self.suggest_float(name, low, high)

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value for the continuous parameter.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_loguniform

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
        """
        return self.suggest_float(name, low, high, log=True)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        """Suggest a value for the discrete parameter.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_discrete_uniform

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
            q:
                A step of discretization.
        """
        return self.suggest_float(name, low, high, step=q)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        """Suggest a value for the integer parameter.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int

        Args:
            name:
                A parameter name.
            low:
                Lower endpoint of the range of suggested values. ``low`` is included in the range.
                ``low`` must be less than or equal to ``high``. If ``log`` is :obj:`True`,
                ``low`` must be larger than 0.
            high:
                Upper endpoint of the range of suggested values. ``high`` is included in the range.
                ``high`` must be greater than or equal to ``low``.
            step:
                A step of discretization.
            log:
                A flag to sample the value from the log domain or not.
        """
        distribution = IntDistribution(low, high, log=log, step=step)
        return self._suggest(name, distribution)

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        """Suggest a value for the categorical parameter.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical

        Args:
            name:
                A parameter name.
            choices:
                Parameter value candidates.
        """
        distribution = CategoricalDistribution(choices)
        return self._suggest(name, distribution)

    def report(self, value: float, step: int) -> None:
        """Report an objective function value for a given step.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report

        Args:
            value:
                An intermediate value returned from the objective function.
            step:
                Step of the trial (e.g., Epoch of neural network training).
        """
        message = ReportMessage(self.trial_id, value, step)
        self.connection.put(message)

    def should_prune(self) -> bool:
        """Suggest whether the trial should be pruned or not.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.should_prune
        """
        message = ShouldPruneMessage(self.trial_id)
        self.connection.put(message)
        response = self.connection.get()
        assert isinstance(response, ResponseMessage)
        return response.data

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
