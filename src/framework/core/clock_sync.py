import time
from typing import List


class ClockSynchronizer:
    def __init__(self):
        pass
        # self._timestamp_monotonic =

    def make_pair_ns(self) -> List[int]:
        return [time.time_ns(), time.perf_counter_ns()]

