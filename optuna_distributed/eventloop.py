from datetime import datetime
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type

from optuna.study import Study
from optuna.trial import TrialState

from optuna_distributed.managers import ObjectiveFuncType
from optuna_distributed.managers import OptimizationManager
from optuna_distributed.terminal import Terminal


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
        manager: OptimizationManager,
        objective: ObjectiveFuncType,
    ) -> None:
        self.study = study
        self.manager = manager
        self.objective = objective

    def run(
        self,
        terminal: Terminal,
        timeout: Optional[float] = None,
        catch: Tuple[Type[Exception], ...] = (),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Starts the event loop.

        Args:
            terminal:
                An instance of :obj:`optuna_distributed.terminal.Terminal`.
            timeout:
                Stops study after the given number of second(s).
            catch:
                A tuple of exceptions to ignore if any is raised while optimizing a function.
        """
        time_start = datetime.now()
        self.manager.create_futures(self.study, self.objective)
        for message in self.manager.get_message():
            try:
                self.manager.before_message(self)
                message.process(self.study, self.manager)
                self.manager.after_message(self)

            except Exception as e:
                if not isinstance(e, catch):
                    with terminal.spin_while_trials_interrupted():
                        self.manager.stop_optimization()
                        self._fail_unfinished_trials()
                    raise

            elapsed = (datetime.now() - time_start).total_seconds()
            if timeout is not None and elapsed > timeout:
                with terminal.spin_while_trials_interrupted():
                    self.manager.stop_optimization()
                break

            if message.closing:
                terminal.update_progress_bar()

            # TODO(xadrianzetx): Call callbacks here.
            if self.manager.should_end_optimization():
                terminal.close_progress_bar()
                break

    def _fail_unfinished_trials(self) -> None:
        # TODO(xadrianzetx) Is there a better way to do this in Optuna?
        states = (TrialState.RUNNING, TrialState.WAITING)
        trials = self.study.get_trials(deepcopy=False, states=states)
        for trial in trials:
            self.study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
