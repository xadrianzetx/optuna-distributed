import pickle
from typing import Optional
from typing import TYPE_CHECKING

from dask.distributed import Queue as DaskQueue

from optuna_distributed.ipc import IPCPrimitive


if TYPE_CHECKING:
    from optuna_distributed.messages import Message


class Queue(IPCPrimitive):
    """IPC primitive based on dask distributed queue.

    All messages are pickled before sending and unpickled
    after recieving to ensure data is msgpack-encodable.

    Args:
        public_channel:
            A name of the queue used to publish messages to.
        private_channel:
            A name of the queue used to recieve messages from.
        timeout:
            Time (in seconds) to wait for message to be fetched
            before raising an exception.
    """

    def __init__(
        self,
        public_channel: str,
        private_channel: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self._public_channel = public_channel
        self._private_channel = private_channel
        self._timeout = timeout
        self._publisher: Optional[DaskQueue] = None
        self._subscriber: Optional[DaskQueue] = None
        self._initialized = False

    def _initialize(self) -> None:
        if not self._initialized:
            # Lazy initialization, since we have to make sure
            # channels are opened on target machine.
            self._publisher = DaskQueue(self._public_channel)
            if self._private_channel is not None:
                self._subscriber = DaskQueue(self._private_channel)
            self._initialized = True

    def get(self) -> "Message":
        self._initialize()
        if self._subscriber is None:
            raise RuntimeError("Trying to get message with publish-only connection.")
        return pickle.loads(self._subscriber.get(timeout=self._timeout))

    def put(self, message: "Message") -> None:
        self._initialize()
        assert self._publisher is not None
        self._publisher.put(pickle.dumps(message))

    def close(self) -> None:
        # Cleanup is handled by dask.
        # For us it's enough to just drop references to queue objects.
        ...
