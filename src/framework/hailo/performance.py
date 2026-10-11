"""Completed-frame rates and rolling stage timings, independent of source timestamps."""
from collections import defaultdict, deque
import threading
import time


class PipelineStats:
    def __init__(self, window=3.0, clock=time.monotonic):
        self.window, self.clock = window, clock
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.started = self.clock()
            self.events = defaultdict(deque)
            self.values = defaultdict(deque)

    def tick(self, stage, count=1):
        with self.lock:
            now = self.clock()
            self.events[stage].append((now, count))
            self._trim(self.events[stage], now)

    def observe(self, name, value):
        with self.lock:
            now = self.clock()
            self.values[name].append((now, value))
            self._trim(self.values[name], now)

    def _trim(self, entries, now):
        while entries and entries[0][0] <= now-self.window:
            entries.popleft()

    def rate(self, stage):
        with self.lock:
            now = self.clock()
            entries = self.events[stage]
            self._trim(entries, now)
            elapsed = min(self.window, now-self.started)
            return sum(v for _, v in entries)/elapsed if elapsed > 0 else 0.0

    def mean(self, name):
        with self.lock:
            entries = self.values[name]
            self._trim(entries, self.clock())
            return sum(v for _, v in entries)/len(entries) if entries else 0.0


def get_stats(state):
    # Called by detection construction before the source and MOT threads start.
    stats = getattr(state, 'hailo_performance', None)
    if stats is None:
        stats = PipelineStats()
        state.hailo_performance = stats
    return stats
