from collections import deque
from typing import Deque
from typing import List

import pytest

from optuna_distributed.ipc import IPCPrimitive
from optuna_distributed.messages import Message
from optuna_distributed.messages import ResponseMessage


class MockIPC(IPCPrimitive):
    def __init__(self) -> None:
        self.captured: List[Message] = []
        self.responses: Deque[ResponseMessage] = deque()

    def get(self) -> "Message":
        return self.responses.popleft()

    def put(self, message: "Message") -> None:
        self.captured.append(message)

    def close(self) -> None:
        ...

    def enqueue_response(self, response: ResponseMessage) -> None:
        self.responses.append(response)


@pytest.fixture
def connection() -> MockIPC:
    return MockIPC()
