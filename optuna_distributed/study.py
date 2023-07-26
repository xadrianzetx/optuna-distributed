from __future__ import annotations

from collections.abc import Callable
from collections.abc import Container
from collections.abc import Iterable
from collections.abc import Sequence
import sys
from typing import Any
from typing import TYPE_CHECKING

from dask.distributed import Client
from dask.distributed import LocalCluster
from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState

from optuna_distributed.eventloop import EventLoop
from optuna_distributed.managers import DistributedOptimizationManager
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.managers import ObjectiveFuncType
from optuna_distributed.terminal import Terminal


if TYPE_CHECKING:
    import pandas as pd


class DistributedStudy:
    """Extends regular Optuna study by distributing trials across multiple workers.

    This object behaves like regular Optuna study, except trials will be evaluated in parallel
    after :func:`optuna_distributed.DistributedStudy.optimize` is called. When :obj:`client`
    is :obj:`None`, work is distributed among available CPU cores by using multiprocessing.
    If Dask client is specified, `optuna_distributed` can use it to distribute trials across
    many physical workers in the cluster.

    .. note::
        Using `optuna_distributed` in distributed mode requires a Dask cluster with matching
        environment. To read more about the deployment and usage of Dask clusters, please refer
        to https://docs.dask.org/en/stable/deploying.html.

    .. note::
        Any APIs besides :func:`optuna_distributed.DistributedStudy.optimize` are just
        passthrough to regular Optuna study and can be used in standard ways.

    .. note::
        There are no known compatibility issues at the moment. All Optuna storages, samplers
        and pruners can be used.

    Args:
        study:
            An isntance of Optuna study.
        client:
            A Dask client. When specified, all trials will be passed to
            Dask scheduler to distribute across available workers.
            If :obj:`None`, multiprocessing backend is used for
            process based parallelism.
    """

    def __init__(self, study: Study, client: Client | None = None) -> None:
        self._study = study
        self._client = client

    @property
    def best_params(self) -> dict[str, Any]:
        """Return parameters of the best trial in the study."""
        return self._study.best_params

    @property
    def best_value(self) -> float:
        """Return the best objective value in the study."""
        return self._study.best_value

    @property
    def best_trial(self) -> FrozenTrial:
        """Return the best trial in the study."""
        return self._study.best_trial

    @property
    def best_trials(self) -> list[FrozenTrial]:
        """Return trials located at the Pareto front in the study."""
        return self._study.best_trials

    @property
    def direction(self) -> StudyDirection:
        """Return the direction of the study."""
        return self._study.direction

    @property
    def directions(self) -> list[StudyDirection]:
        """Return the directions of the study."""
        return self._study.directions

    @property
    def trials(self) -> list[FrozenTrial]:
        """Return all trials in the study."""
        return self._study.trials

    @property
    def user_attrs(self) -> dict[str, Any]:
        """Return user attributes."""
        return self._study.user_attrs

    @property
    def system_attrs(self) -> dict[str, Any]:
        """Return system attributes."""
        return self._study.system_attrs

    def into_study(self) -> Study:
        """Returns regular Optuna study."""
        return self._study

    def get_trials(
        self, deepcopy: bool = True, states: Container[TrialState] | None = None
    ) -> list[FrozenTrial]:
        """Return all trials in the study.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.get_trials

        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.
            states:
                Trial states to filter on. If :obj:`None`, include all states.
        """
        return self._study.get_trials(deepcopy, states)

    def optimize(
        self,
        func: ObjectiveFuncType,
        n_trials: int | None = None,
        timeout: float | None = None,
        n_jobs: int = -1,
        catch: Iterable[type[Exception]] | type[Exception] = (),
        callbacks: list[Callable[["Study", FrozenTrial], None]] | None = None,
        show_progress_bar: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Optimize an objective function.

        Optimization is done by choosing a suitable set of hyperparameter values from a given
        range. If Dask client has been specified, evaluations of objective function (trials)
        will be distributed among available workers, otherwise parallelism is process based.

        For additional notes on some args, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize

        Args:
            func:
                A callable that implements objective function.
            n_trials:
                The number of trials to run in total.
            timeout:
                Stop study after the given number of second(s).
            n_jobs:
                The number of parallel jobs when using multiprocessing backend. Values less than
                one or greater than :obj:`multiprocessing.cpu_count()` will default to number of
                logical CPU cores available.
            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Currently
                not supported.
            show_progress_bar:
                Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
        """
        if n_trials is None:
            raise ValueError("Only finite number of trials supported at the moment.")

        terminal = Terminal(show_progress_bar, n_trials, timeout)
        catch = tuple(catch) if isinstance(catch, Iterable) else (catch,)
        manager = (
            DistributedOptimizationManager(self._client, n_trials)
            if self._client is not None and not isinstance(self._client.cluster, LocalCluster)
            else LocalOptimizationManager(n_trials, n_jobs)
        )

        if isinstance(manager, LocalOptimizationManager) and sys.platform == "win32":
            raise ValueError(
                "Local asynchronous optimization is currently not supported on Windows. "
                "Please specify Dask client to continue in distributed mode."
            )

        try:
            event_loop = EventLoop(self._study, manager, objective=func, interrupt_patience=10.0)
            event_loop.run(terminal, timeout, catch)

        except KeyboardInterrupt:
            with terminal.spin_while_trials_interrupted():
                manager.stop_optimization(patience=10.0)

            states = (TrialState.RUNNING, TrialState.WAITING)
            trials = self._study.get_trials(deepcopy=False, states=states)
            for trial in trials:
                self._study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
            raise

        finally:
            self._study._storage.remove_session()

    def ask(self, fixed_distributions: dict[str, BaseDistribution] | None = None) -> Trial:
        """Create a new trial from which hyperparameters can be suggested.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.ask

        Args:
            fixed_distributions:
                A dictionary containing the parameter names and parameter's distributions.
        """
        return self._study.ask(fixed_distributions)

    def tell(
        self,
        trial: Trial | int,
        values: float | Sequence[float] | None = None,
        state: TrialState | None = None,
        skip_if_finished: bool = False,
    ) -> FrozenTrial:
        """Finish a trial created with :func:`~optuna_distributed.study.DistributedStudy.ask`.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.tell

        Args:
            trial:
                A :obj:`optuna.trial.Trial` object or a trial number.
            values:
                Optional objective value or a sequence of such values in case the study is used
                for multi-objective optimization.
            state:
                State to be reported.
            skip_if_finished:
                Flag to control whether exception should be raised when values for already
                finished trial are told.
        """
        return self._study.tell(trial, values, state, skip_if_finished)

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a user attribute to the study.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.set_user_attr

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.
        """
        self._study.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        """Set a system attribute to the study.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.set_system_attr

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable
        """
        self._study.set_system_attr(key, value)

    def trials_dataframe(
        self,
        attrs: tuple[str, ...] = (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        ),
        multi_index: bool = False,
    ) -> "pd.DataFrame":
        """Export trials as a pandas DataFrame.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.trials_dataframe

        Args:
            attrs:
                Specifies field names of :obj:`optuna.trial.FrozenTrial` to include them to a
                DataFrame of trials.
            multi_index:
                Specifies whether the returned DataFrame employs MultiIndex or not.
        """
        return self._study.trials_dataframe(attrs, multi_index)

    def stop(self) -> None:
        """Exit from the current optimization loop after the running trials finish.

        This method is effectively a noop, sice there is no way to reach study from the
        objective function at the moment. TODO(xadrianzetx) Implement this.
        """
        self._study.stop()

    def enqueue_trial(
        self,
        params: dict[str, Any],
        user_attrs: dict[str, Any] | None = None,
        skip_if_exists: bool = False,
    ) -> None:
        """Enqueue a trial with given parameter values.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.enqueue_trial

        Args:
            params:
                Parameter values to pass your objective function.
            user_attrs:
                A dictionary of user-specific attributes other than :obj:`params`.
            skip_if_exists:
                When :obj:`True`, prevents duplicate trials from being enqueued again.
        """
        self._study.enqueue_trial(params, user_attrs, skip_if_exists)

    def add_trial(self, trial: FrozenTrial) -> None:
        """Add trial to study.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.add_trial

        Args:
            trial: Trial to add.
        """
        self._study.add_trial(trial)

    def add_trials(self, trials: Iterable[FrozenTrial]) -> None:
        """Add trials to study.

        For complete documentation, please refer to:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.add_trials

        Args:
            trial: Trials to add.
        """
        self._study.add_trials(trials)


def from_study(study: Study, client: Client | None = None) -> DistributedStudy:
    """Takes regular Optuna study and extends it to :class:`~optuna_distributed.DistributedStudy`.

    This creates an object which behaves like regular Optuna study, except trials
    will be evaluated in parallel after :func:`optuna_distributed.DistributedStudy.optimize`
    is called. When :obj:`client` is :obj:`None`, work is distributed among available CPU cores
    by using multiprocessing. If Dask client is specified, `optuna_distributed` can use it to
    distribute trials across many physical workers in the cluster.

    .. note::
        Using `optuna_distributed` in distributed mode requires a Dask cluster with matching
        environment. To read more about the deployment and usage of Dask clusters, please refer
        to https://docs.dask.org/en/stable/deploying.html.

    .. note::
        Any APIs besides :func:`optuna_distributed.DistributedStudy.optimize` are just
        passthrough to regular Optuna study and can be used in standard ways.

    .. note::
        There are no known compatibility issues at the moment. All Optuna storages, samplers
        and pruners can be used.

    Args:
        study:
            A regular Optuna study isntance.
        client:
            Dask client, as described in https://distributed.dask.org/en/stable/client.html#client
    """
    return DistributedStudy(study, client)
