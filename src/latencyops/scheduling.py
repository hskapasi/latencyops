"""Local queue and scheduling abstractions for inference requests."""

from dataclasses import dataclass
from queue import Empty, Queue
from time import monotonic
from typing import Generic, Mapping, TypeVar

from .models import SystemSignals

T = TypeVar("T")


@dataclass(frozen=True)
class QueuedRequest(Generic[T]):
    """A request with a priority and enqueue timestamp."""

    payload: T
    priority: int = 0
    enqueued_at: float = 0.0


class InMemoryRequestQueue(Generic[T]):
    """Bounded FIFO queue suitable for local development and tests."""

    def __init__(self, maxsize: int = 100):
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._queue: Queue[QueuedRequest[T]] = Queue(maxsize=maxsize)

    def put(self, payload: T, priority: int = 0) -> None:
        self._queue.put_nowait(QueuedRequest(payload, priority, monotonic()))

    def get(self) -> QueuedRequest[T]:
        try:
            return self._queue.get_nowait()
        except Empty as exc:
            raise LookupError("request queue is empty") from exc

    @property
    def depth(self) -> int:
        return self._queue.qsize()


@dataclass(frozen=True)
class CachePressureSignal:
    """Normalized cache capacity signal used by request profiling."""

    used: int
    capacity: int

    def __post_init__(self) -> None:
        if self.used < 0 or self.capacity <= 0 or self.used > self.capacity:
            raise ValueError("cache usage must be between zero and capacity")

    @property
    def pressure(self) -> float:
        return self.used / self.capacity


class PriorityScheduler(Generic[T]):
    """Select higher-priority requests first, preserving FIFO within a priority."""

    def __init__(self):
        self._items: list[QueuedRequest[T]] = []

    def submit(self, payload: T, priority: int = 0) -> None:
        self._items.append(QueuedRequest(payload, priority, monotonic()))

    def next(self) -> QueuedRequest[T]:
        if not self._items:
            raise LookupError("scheduler is empty")
        index = max(range(len(self._items)), key=lambda i: self._items[i].priority)
        return self._items.pop(index)

    @property
    def depth(self) -> int:
        return len(self._items)
