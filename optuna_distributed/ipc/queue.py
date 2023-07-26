from __future__ import annotations

import asyncio
import pickle

from dask.distributed import Queue as DaskQueue

from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.messages import Message


class Queue(IPCPrimitive):
    """IPC primitive based on dask distributed queue.

    All messages are pickled before sending and unpickled
    after recieving to ensure data is msgpack-encodable.

    Args:
        publishing:
            A name of the queue used to publish messages to.
        recieving:
            A name of the queue used to recieve messages from.
        timeout:
            Time (in seconds) to wait for message to be fetched
            before raising an exception. Should not be set if
            `max_retries` is used.
        max_retries:
            Specifies maximum number of attempts to fetch a message.
            After each attempt, timeout is extended exponentially.
    """

    def __init__(
        self,
        publishing: str,
        recieving: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._publishing = publishing
        self._recieving = recieving

        if max_retries is not None and timeout is not None:
            raise ValueError("Exponentially growing timeout is used when `max_retries` is set.")

        self._timeout = timeout
        self._max_retries = max_retries
        self._publisher: DaskQueue | None = None
        self._subscriber: DaskQueue | None = None
        self._initialized = False

    def _initialize(self) -> None:
        if not self._initialized:
            # Lazy initialization, since we have to make sure
            # channels are opened on target machine.
            self._publisher = DaskQueue(self._publishing)
            if self._recieving is not None:
                self._subscriber = DaskQueue(self._recieving)
            self._initialized = True

    def get(self) -> Message:
        self._initialize()
        if self._subscriber is None:
            raise RuntimeError("Trying to get message with publish-only connection.")

        attempt = 0
        while True:
            try:
                timeout = self._timeout if self._max_retries is None else 2**attempt
                return pickle.loads(self._subscriber.get(timeout))

            except asyncio.TimeoutError:
                attempt += 1
                if self._max_retries is None or attempt == self._max_retries:
                    raise

    def put(self, message: Message) -> None:
        self._initialize()
        assert self._publisher is not None
        self._publisher.put(pickle.dumps(message))

    def close(self) -> None:
        # Cleanup is handled by dask.
        # For us it's enough to just drop references to queue objects.
        ...
