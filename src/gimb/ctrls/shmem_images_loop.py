import time

from src.gimb.domain.cameras import TrOsCamId
from src.gimb.state import TrOsState
from src.framework.frame.shm.reader import SharedMemoryCapReader
from src.framework.loop.loop import ILoop


class ShmemImagesLoop(ILoop):
    def __init__(self, state: TrOsState) -> None:
        super().__init__()
        self._state = state
        self._reader: SharedMemoryCapReader = None

    def on_start(self) -> None:
        self._reader = SharedMemoryCapReader(path="/dev/shm/tv_n")
        self._reader.start()

    def update(self) -> None:
        ret, frame = self._reader.read_no_wait()
        if ret and frame is not None:
            self._state.cameras[TrOsCamId.TV_N].frames.enqueue(
                frame, timestamp_ns=1000 * self._reader.last_update_time
            )
            if self._state.cameras[TrOsCamId.TV_N].resolution is None:
                self._state.cameras[TrOsCamId.TV_N].resolution = frame.shape[:2]
        time.sleep(5e-3)

    def on_stop(self) -> None:
        pass
