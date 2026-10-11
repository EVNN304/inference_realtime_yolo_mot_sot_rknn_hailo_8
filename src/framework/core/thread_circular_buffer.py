import time
import threading
import weakref
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Literal, Iterator, Iterable

import numpy as np

T = TypeVar("T")


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass(frozen=True)
class SampleView(Generic[T]):
    slot_id: int
    timestamp_ns: int
    obj: T
    seq: int


class _Slot(Generic[T]):
    __slots__ = ("seq", "timestamp_ns", "obj")

    def __init__(self) -> None:
        self.seq: int = 0
        self.timestamp_ns: int = 0
        self.obj: Optional[T] = None


class ThreadCircularBuffer(Generic[T], Iterable[T]):
    class Listener(Generic[T]):
        def __init__(self, ring: "ThreadCircularBuffer[T]") -> None:
            self._ring = ring
            self._lock = threading.Lock()
            self._cond = threading.Condition(self._lock)
            self._gen: int = 0
            self._last_seen: int = 0
            self._closed: bool = False

        def _notify(self) -> None:
            with self._cond:
                self._gen += 1
                self._cond.notify_all()

        def wait_latest(
            self,
            timeout: Optional[float] = None,
            retries: int = 8,
        ) -> Optional[SampleView[T]]:
            with self._cond:
                if self._closed:
                    return None

                ok = self._cond.wait_for(
                    lambda: self._closed or self._gen != self._last_seen,
                    timeout=timeout,
                )
                if not ok or self._closed:
                    return None

                self._last_seen = self._gen

            return self._ring.latest(retries=retries)

        def try_latest(self, retries: int = 8) -> Optional[SampleView[T]]:
            with self._cond:
                if self._closed or self._gen == self._last_seen:
                    return None
                self._last_seen = self._gen

            return self._ring.latest(retries=retries)

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self._capacity = int(capacity)
        self._slots: List[_Slot[T]] = [_Slot() for _ in range(self._capacity)]

        self._write_pos: int = 0
        self._last_slot: int = -1

        self._listeners_lock = threading.Lock()
        self._listeners: List[weakref.ref[ThreadCircularBuffer.Listener[T]]] = []

    def subscribe(self) -> "ThreadCircularBuffer.Listener[T]":
        lst: ThreadCircularBuffer.Listener[T] = ThreadCircularBuffer.Listener(self)
        with self._listeners_lock:
            self._listeners.append(weakref.ref(lst))
        return lst

    def _notify_all(self) -> None:
        with self._listeners_lock:
            refs = list(self._listeners)

        any_dead = False
        for wr in refs:
            lst = wr()
            if lst is None:
                any_dead = True
            else:
                lst._notify()

        if any_dead:
            with self._listeners_lock:
                self._listeners = [wr for wr in self._listeners if wr() is not None]

    def enqueue(self, obj: T, timestamp_ns: Optional[int] = None) -> int:
        ts = now_ns() if (timestamp_ns is None) else int(timestamp_ns)

        slot_id = self._write_pos
        slot = self._slots[slot_id]

        slot.seq += 1
        slot.timestamp_ns = ts
        slot.obj = obj
        slot.seq += 1

        self._last_slot = slot_id
        self._write_pos = (slot_id + 1) % self._capacity

        self._notify_all()
        return slot_id

    def try_read_slot(self, slot_id: int) -> Optional[SampleView[T]]:
        if slot_id < 0 or slot_id >= self._capacity:
            raise IndexError("slot_id out of range")

        slot = self._slots[slot_id]

        s1 = slot.seq
        if s1 & 1:
            return None

        ts = slot.timestamp_ns
        obj = slot.obj

        s2 = slot.seq
        if s1 != s2 or (s2 & 1) or obj is None:
            return None

        return SampleView(slot_id=slot_id, timestamp_ns=ts, obj=obj, seq=s2)

    def get_by_index(self, slot_id: int) -> Optional[SampleView[T]]:
        return self.try_read_slot(slot_id)

    def get_by_timestamp(
        self,
        timestamp_ns: int,
        mode: str = "le",
    ) -> Optional[SampleView[T]]:
        """
        mode:
            "le"      -> newest sample <= timestamp
            "nearest" -> closest sample by absolute delta
        """

        q = int(timestamp_ns)

        best: Optional[SampleView[T]] = None
        best_metric: Optional[int] = None

        for i in range(self._capacity):
            v = self.try_read_slot(i)
            if v is None:
                continue

            ts = v.timestamp_ns

            if mode == "le":
                if ts > q:
                    continue
                metric = q - ts
            else:  # nearest
                metric = abs(ts - q)

            if best is None or metric < best_metric:
                best = v
                best_metric = metric

        return best

    def latest(self, retries: int = 8) -> Optional[SampleView[T]]:
        last = self._last_slot
        if last < 0:
            return None

        for k in range(min(retries, self._capacity)):
            idx = (last - k) % self._capacity
            v = self.try_read_slot(idx)
            if v is not None:
                return v
        return None

    def get_next_by_timestamp(
        self,
        timestamp_ns: int,
        count: int = 1,
    ) -> List[SampleView[T]]:
        from bisect import bisect_left

        if count <= 0:
            return []

        items = list(self)
        if not items:
            return []

        q = int(timestamp_ns)
        ts_list = [v.timestamp_ns for v in items]
        pos = bisect_left(ts_list, q)

        return items[pos : pos + count]

    def clear(self) -> None:
        # Reset all slots
        for slot in self._slots:
            slot.seq = 0
            slot.timestamp_ns = 0
            slot.obj = None

        # Reset indices (same as freshly initialized state)
        self._write_pos = 0
        self._last_slot = -1

    def empty(self) -> bool:
        # Fast logical check
        return self._last_slot < 0

    def __iter__(self) -> Iterator[SampleView[T]]:
        start = self._write_pos

        for k in range(self._capacity):
            idx = (start + k) % self._capacity
            v = self.try_read_slot(idx)
            if v is not None:
                yield v

    def __len__(self) -> int:
        return self._capacity


def get_average_time_seconds(circular_buffer: ThreadCircularBuffer) -> float:
    timestamps = [(1e-9 * sample.timestamp_ns) for sample in circular_buffer]
    if len(timestamps) < 2:
        return 0.0

    return np.average(np.diff(timestamps), axis=0)
