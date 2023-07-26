from __future__ import annotations

from collections.abc import Sequence
import datetime
from typing import Any
from typing import TypeVar

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution

from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.messages import ReportMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.messages import SetAttributeMessage
from optuna_distributed.messages import ShouldPruneMessage
from optuna_distributed.messages import SuggestMessage
from optuna_distributed.messages import TrialProperty
from optuna_distributed.messages import TrialPropertyMessage
from optuna_distributed.messages.base import Message


T = TypeVar("T", bound=CategoricalChoiceType)


class DistributedTrial:
    """A trial is a process of evaluating an objective function.

    This is a version of Optuna trial designed to run in process or machine separate
    to the study and its resources. Communication with study is held via messaging
    system, allowing remote workers to use standard Optuna trial APIs.

    For complete documentation, please refer to:
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna-trial-trial

    Args:
        trial_id:
            A trial ID that is automatically generated.
        connection:
            An instance of :class:`~optuna_distributed.ipc.IPCPrimitive`.
    """

    def __init__(self, trial_id: int, connection: IPCPrimitive) -> None:
        self.trial_id = trial_id
        self.connection = connection

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        message = SuggestMessage(self.trial_id, name, distribution)
        return self._send_message_and_wait_response(message)

    def _get_property(self, property: TrialProperty) -> Any:
        message = TrialPropertyMessage(self.trial_id, property)
        return self._send_message_and_wait_response(message)

    def _send_message_and_wait_response(self, message: Message) -> Any:
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
        step: float | None = None,
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

    def suggest_categorical(self, name: str, choices: Sequence[T]) -> T:
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
        return self._send_message_and_wait_response(message)

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set user attributes to the trial.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.set_user_attr

        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be able to serialize with pickle.
        """
        message = SetAttributeMessage(self.trial_id, key, value, kind="user")
        self.connection.put(message)

    def set_system_attr(self, key: str, value: Any) -> None:
        """set system attributes to the trial.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.set_system_attr

        Args:
            key:
                A key string of the attribute.
            value:
                A value of the attribute. The value should be able to serialize with pickle.
        """
        message = SetAttributeMessage(self.trial_id, key, value, kind="system")
        self.connection.put(message)

    @property
    def params(self) -> dict[str, Any]:
        """Return parameters to be optimized."""
        return self._get_property("params")

    @property
    def distributions(self) -> dict[str, BaseDistribution]:
        """Return distributions of parameters to be optimized."""
        return self._get_property("distributions")

    @property
    def user_attrs(self) -> dict[str, Any]:
        """Return user attributes."""
        return self._get_property("user_attrs")

    @property
    def system_attrs(self) -> dict[str, Any]:
        """Return system attributes."""
        return self._get_property("system_attrs")

    @property
    def datetime_start(self) -> datetime.datetime | None:
        """Return start datetime."""
        return self._get_property("datetime_start")

    @property
    def number(self) -> int:
        """Return trial's number which is consecutive and unique in a study."""
        return self._get_property("number")
