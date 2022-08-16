import sys
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

from dask.distributed import Client
from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial

from optuna_distributed.eventloop import EventLoop
from optuna_distributed.managers import DistributedOptimizationManager
from optuna_distributed.managers import LocalOptimizationManager
from optuna_distributed.messages import CompletedMessage
from optuna_distributed.messages import FailedMessage
from optuna_distributed.messages import PrunedMessage
from optuna_distributed.messages import RepeatedTrialMessage
from optuna_distributed.trial import DistributedTrial


ObjectiveFuncType = Callable[[Trial], Union[float, Sequence[float]]]


class DistributedStudy:
    """An extenstion of Optuna study, able to distribute trials across
    multiple processes or machines.

    Args:
        study:
            An Optuna study.
        client:
            A dask client.
    """

    def __init__(self, study: Study, client: Optional[Client]) -> None:
        self._study = study
        self._client = client

    @property
    def best_trial(self) -> FrozenTrial:
        return self._study.best_trial

    @classmethod
    def from_optuna_study(cls, study: Study, client: Optional[Client]) -> "DistributedStudy":
        pass

    def optimize(
        self,
        func: ObjectiveFuncType,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[["Study", FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        """Optimize an objective function using multiple workers.

        Args: TODO
        """

        def _objective_wrapper(trial: DistributedTrial) -> None:
            trial.connection.put(RepeatedTrialMessage(trial.trial_id))
            if trial.connection.get():
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

        manager = (
            DistributedOptimizationManager(self._client, n_trials)
            if self._client is not None
            else LocalOptimizationManager(n_trials, n_jobs)
        )
        try:
            event_loop = EventLoop(self._study, manager, _objective_wrapper)
            event_loop.run(n_trials, timeout, catch, callbacks, show_progress_bar)

        except KeyboardInterrupt:
            manager.stop_optimization()
            raise

        finally:
            self._study._storage.remove_session()
