import time
from threading import Thread

from src.framework.core.thread_circular_buffer import ThreadCircularBuffer

ring = ThreadCircularBuffer[str](5)


def consumer():
    listener = ring.subscribe()
    while True:
        v = listener.wait_latest(timeout=1.0)
        if v is None:
            continue
        recv_time = time.perf_counter_ns()
        print(f"received({recv_time}) => {v.obj}, delta time {recv_time - v.timestamp_ns} ns")



Thread(target=consumer, daemon=True).start()

ring.enqueue(f"hello")
time.sleep(1e-3)
ring.enqueue(f"world")
time.sleep(1e-3)
ring.enqueue(f"I")
time.sleep(1e-3)
ring.enqueue(f"am")
time.sleep(1e-3)
ring.enqueue(f"buffer")
time.sleep(1e-3)
ring.enqueue(f"ring")
time.sleep(1e-3)
ring.enqueue(f"!!!")
time.sleep(1e-2)
