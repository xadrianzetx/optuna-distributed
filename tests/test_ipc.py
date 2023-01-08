import asyncio
from multiprocessing import Process
from multiprocessing.connection import Pipe as MultiprocessingPipe
import time

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


def test_queue_raises_on_timeout_and_backoff(client: Client) -> None:
    with pytest.raises(ValueError):
        Queue("foo", timeout=1, max_retries=1)


def test_queue_raises_after_timeout(client: Client) -> None:
    q = Queue("foo", "bar", timeout=1)
    with pytest.raises(asyncio.TimeoutError):
        q.get()


def test_queue_raises_after_retries(client: Client) -> None:
    q = Queue("foo", "bar", max_retries=1)
    with pytest.raises(asyncio.TimeoutError):
        q.get()


def test_queue_get_delayed_message(client: Client) -> None:
    public = "public"
    private = "private"
    future = client.submit(_ping_pong, Queue(public, private, max_retries=5))
    master = Queue(private, public)

    # With exponential timeout, attempts are made after 1, 3, 7, 15... seconds.
    # To ensure at least one retry, message should be delayed between 1 and 3 seconds.
    time.sleep(2.0)
    master.put(ResponseMessage(0, "ping"))
    response = master.get()
    assert isinstance(response, ResponseMessage)
    assert response.data == "pong"
    wait(future)
    assert future.done()
    assert future.status == "finished"
