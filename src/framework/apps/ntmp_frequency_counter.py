import threading
import time
from collections import defaultdict
from typing import Optional, Set, Dict

from src.framework.loop.loop import SimpleSeparateThreadLoopRunner, IOLoop
from src.framework.protocols.ntmp.base import NTMPHeader
from src.framework.protocols.ntmp.client import NTMPClientUDP, NTMPClientUPDSettings


class FrequencyCounter:
    def __init__(
        self,
        interesting_ids: Optional[Set[int]] = None,
        names: Optional[Dict[int, str]] = None,
    ):
        self._counter: Dict[int, int] = defaultdict(lambda: 0)
        self._interesting_ids = interesting_ids
        self._names = names
        self._previous_report_time = time.time()
        self._mutex = threading.Lock()

    def __call__(self, header: NTMPHeader, payload: bytes) -> None:
        msg_id = header.msg_id
        count_this_message = False
        if self._interesting_ids is not None:
            if msg_id in self._interesting_ids:
                count_this_message = True
        else:
            count_this_message = True
        if count_this_message:
            with self._mutex:
                self._counter[header.msg_id] += 1

    def write_report(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        delta = now - self._previous_report_time
        with self._mutex:
            for id, count in self._counter.items():
                report_name = hex(id).upper().replace("X", "x")
                if self._names is not None:
                    if report_name not in self._names:
                        report_name = self._names.get(id, report_name)
                print(f"{report_name}:\t {count / delta:.1f} Hz")

            self._counter.clear()

        print("*****************************")
        self._previous_report_time = now


def main():
    # filter = [0x9FD491D1]
    filter = None
    names = {0x9FD491D1: "INF_TRACKER"}

    freq_counter = FrequencyCounter(filter, names)
    ntmp_udp_client = NTMPClientUDP(
        freq_counter,
        settings=NTMPClientUPDSettings(iface="eth0"),
    )
    io_loop = SimpleSeparateThreadLoopRunner(loop=IOLoop(ntmp_udp_client))
    io_loop.start()

    while True:
        time.sleep(1)
        freq_counter.write_report()


if __name__ == "__main__":
    main()
