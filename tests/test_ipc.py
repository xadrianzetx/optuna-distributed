from multiprocessing import Process
from multiprocessing.connection import Pipe as MultiprocessingPipe

from dask.distributed import Client
from dask.distributed import wait
import pytest

from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.ipc import Pipe
from optuna_distributed.ipc import Queue
from optuna_distributed.messages import ResponseMessage


def _ping_pong(conn: IPCPrimitive) -> None:
    msg = conn.get()
    assert isinstance(msg, ResponseMessage)
    assert msg.data == "ping"
    conn.put(ResponseMessage(0, "pong"))


def test_pipe_ping_pong() -> None:
    a, b = MultiprocessingPipe()
    p = Process(target=_ping_pong, args=(Pipe(b),))
    p.start()

    master = Pipe(a)
    master.put(ResponseMessage(0, "ping"))
    response = master.get()
    assert isinstance(response, ResponseMessage)
    assert response.data == "pong"
    p.join()
    assert p.exitcode == 0


def test_queue_ping_pong(client: Client) -> None:
    public = "public"
    private = "private"
    future = client.submit(_ping_pong, Queue(public, private))
    master = Queue(private, public)
    master.put(ResponseMessage(0, "ping"))
    response = master.get()
    assert isinstance(response, ResponseMessage)
    assert response.data == "pong"
    wait(future)
    assert future.done()
    assert future.status == "finished"


def test_queue_publishing_only(client: Client) -> None:
    q = Queue("foo")
    with pytest.raises(RuntimeError):
        q.get()
