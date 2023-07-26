from __future__ import annotations

from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.status import Status
from rich.style import Style


class Terminal:
    """Provides styled terminal output.

    Args:
        show_progress_bar:
            Enables progress bar.
        n_trials:
            The number of trials to run in total.
        timeout:
            Stops study after the given number of second(s).
    """

    def __init__(
        self, show_progress_bar: bool, n_trials: int, timeout: float | None = None
    ) -> None:
        self._timeout = timeout
        self._progbar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style=Style(color="light_coral")),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        )

        self._task = self._progbar.add_task("[blue]Running trials...[/blue]", total=n_trials)
        if show_progress_bar:
            self._progbar.start()

    def update_progress_bar(self) -> None:
        """Advance progress bar by one trial."""
        self._progbar.advance(self._task)

    def close_progress_bar(self) -> None:
        """Closes progress bar."""
        self._progbar.stop()

    def spin_while_trials_interrupted(self) -> Status:
        """Renders spinner animation while trials are being interrupted."""
        self._progbar.stop()
        return self._progbar.console.status(
            "[blue]Interrupting running trials...[/blue]", spinner_style=Style(color="blue")  # type: ignore # noqa: E501
        )
