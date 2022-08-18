from datetime import datetime
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING

# FIXME: We should probably implement our own progress bar.
from optuna.progress_bar import _ProgressBar
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna_distributed.managers import OptimizationManager
    from optuna_distributed.trial import DistributedTrial


class EventLoop:
    """Collects and acts upon all that is happening in optimization process.

    After trials are dispatched to run across many workers, all communication with
    them is held via central point in the event loop. From here we can wait for requests
    made by workers (e.g. to suggest a hyperparameter value) and act upon them using local
    resources. This ensures sequential access to storages, samplers etc.

    Args:
        study:
            An instance of Optuna study.
        manager:
            An instance of :class:`~optuna_distributed.managers.Manager`.
        objective:
            An objective function to optimize.
    """

    def __init__(
        self,
        study: Study,
        manager: "OptimizationManager",
        objective: Callable[["DistributedTrial"], None],
    ) -> None:
        self.study = study
        self.manager = manager
        self.objective = objective

    def run(
        self,
        n_trials: Optional[int],
        timeout: Optional[float],
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
        show_progress_bar: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Starts the event loop.

        Args:
            n_trials:
                The number of trials for each process.
            timeout:
                Stops study after the given number of second(s).
            catch:
                A tuple of exceptions to ignore if any is raised while optimizing a function.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Not supported
                at the moment.
            show_progress_bar:
                A flag to include tqdm-style progress bar.
        """
        time_start = datetime.now()
        progress_bar = _ProgressBar(show_progress_bar, n_trials, timeout)

        self.manager.create_futures(self.study, self.objective)
        for message in self.manager.get_message():
            try:
                self.manager.before_message(self)
                message.process(self.study, self.manager)
                self.manager.after_message(self)

            except Exception:
                self.manager.stop_optimization()
                # TODO(xadrianzetx) Is there a better way to do this in Optuna?
                states = (TrialState.RUNNING, TrialState.WAITING)
                trials = self.study.get_trials(deepcopy=False, states=states)
                for trial in trials:
                    self.study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
                raise

            progress_bar.update((datetime.now() - time_start).total_seconds())
            # TODO(xadrianzetx): Stop optimization on timeout here.
            # TODO(xadrianzetx): Call callbacks here.
            if self.manager.should_end_optimization():
                break
