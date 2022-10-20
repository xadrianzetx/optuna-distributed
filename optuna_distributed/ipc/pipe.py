from multiprocessing.connection import Connection

from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.messages import Message


class Pipe(IPCPrimitive):
    """IPC primitive based on multiprocessing Pipe.

    Forms a thin layer of abstraction over one end of multiprocessing
    pipe connection, with get/put semantics.

    Args:
        connection:
            One of the ends of multiprocessing pipe.
            https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Pipe
    """

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def get(self) -> Message:
        return self._connection.recv()

    def put(self, message: Message) -> None:
        return self._connection.send(message)

    def close(self) -> None:
        return self._connection.close()
