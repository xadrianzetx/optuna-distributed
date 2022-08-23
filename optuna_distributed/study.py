import sys
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Type
from typing import Union

from dask.distributed import Client
from optuna.distributions import BaseDistribution
from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState

from optuna_distributed.eventloop import EventLoop
from optuna_distributed.managers import DistributedOptimizationManager
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import Message
from optuna_distributed.messages import PrunedMessage
from optuna_distributed.messages import RepeatedTrialMessage
from optuna_distributed.messages import ResponseMessage
from optuna_distributed.trial import DistributedTrial


if TYPE_CHECKING:
    import pandas as pd


ObjectiveFuncType = Callable[[DistributedTrial], Union[float, Sequence[float]]]
DistributableFuncType = Callable[[DistributedTrial], None]


class DistributedStudy:
    """An extenstion of Optuna study, able to distribute trials across
    multiple processes or machines.

    Args:
        study:
            An Optuna study.
        client:
            A dask client.
    """

    def __init__(self, study: Study, client: Optional[Client] = None) -> None:
        self._study = study
        self._client = client

    @property
    def best_params(self) -> Dict[str, Any]:
        pass

    @property
    def best_value(self) -> float:
        pass

    @property
    def best_trial(self) -> FrozenTrial:
        return self._study.best_trial

    @property
    def best_trials(self) -> List[FrozenTrial]:
        pass

    @property
    def direction(self) -> StudyDirection:
        pass

    @property
    def directions(self) -> List[StudyDirection]:
        pass

    @property
    def trials(self) -> List[FrozenTrial]:
        pass

    @property
    def user_attrs(self) -> Dict[str, Any]:
        pass

    @property
    def system_attrs(self) -> Dict[str, Any]:
        pass

    def get_trials(
        self, deepcopy: bool = True, states: Optional[Container[TrialState]] = None
    ) -> List[FrozenTrial]:
        pass

    def optimize(
        self,
        func: ObjectiveFuncType,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = -1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[["Study", FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        """Optimize an objective function."""
        if n_trials is None:
            raise ValueError("Only finite number of trials supported at the moment.")

        distributable = _wrap_objective(func)
        manager = (
            DistributedOptimizationManager(self._client, n_trials)
            if self._client is not None
            else LocalOptimizationManager(n_trials, n_jobs)
        )
        try:
            event_loop = EventLoop(self._study, manager, distributable)
            event_loop.run(n_trials, timeout, catch, callbacks, show_progress_bar)

        except KeyboardInterrupt:
            manager.stop_optimization()
            states = (TrialState.RUNNING, TrialState.WAITING)
            trials = self._study.get_trials(deepcopy=False, states=states)
            for trial in trials:
                self._study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
            raise

        finally:
            self._study._storage.remove_session()

    def ask(self, fixed_distributions: Optional[Dict[str, BaseDistribution]] = None) -> Trial:
        pass

    def tell(
        self,
        trial: Union[Trial, int],
        values: Optional[Union[float, Sequence[float]]] = None,
        state: Optional[TrialState] = None,
        skip_if_finished: bool = False,
    ) -> FrozenTrial:
        pass

    def set_user_attr(self, key: str, value: Any) -> None:
        pass

    def set_system_attr(self, key: str, value: Any) -> None:
        pass

    def trials_dataframe(
        self,
        attrs: Tuple[str, ...] = (
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
        pass

    def stop(self) -> None:
        pass

    def enqueue_trial(
        self,
        params: Dict[str, Any],
        user_attrs: Optional[Dict[str, Any]] = None,
        skip_if_exists: bool = False,
    ) -> None:
        pass

    def add_trial(self, trial: FrozenTrial) -> None:
        pass

    def add_trials(self, trials: Iterable[FrozenTrial]) -> None:
        pass


def _wrap_objective(func: ObjectiveFuncType) -> DistributableFuncType:
    def _objective_wrapper(trial: DistributedTrial) -> None:
        trial.connection.put(RepeatedTrialMessage(trial.trial_id))
        is_repeated = trial.connection.get()
        assert isinstance(is_repeated, ResponseMessage)
        message: Message
        if is_repeated.data:
            return

        try:
            value_or_values = func(trial)
            message = CompletedMessage(trial.trial_id, value_or_values)
            trial.connection.put(message)

        except TrialPruned as e:
            message = PrunedMessage(trial.trial_id, e)
            trial.connection.put(message)

        except Exception as e:
            exc_info = sys.exc_info()
            message = FailedMessage(trial.trial_id, e, exc_info)
            trial.connection.put(message)

        finally:
            trial.connection.close()

    return _objective_wrapper


def from_optuna_study(study: Study) -> DistributedStudy:
    pass
