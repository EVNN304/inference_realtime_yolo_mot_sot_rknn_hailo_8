import time
from pathlib import Path

from src.gimb.domain.cameras import TrOsCamHolder
from src.gimb.state import TrOsState
from src.framework.frame.folder import FolderVideoCapture
from src.framework.loop.loop import ILoop

import config.config as config


class TrOsDatasetImagesLoop(ILoop):
    def __init__(self, state: TrOsState):
        super().__init__()
        self._state = state
        self._im_readers: TrOsCamHolder[FolderVideoCapture | None] = None
        self._framerate = 30

    def on_start(self) -> None:
        self._im_readers = TrOsCamHolder(
            ir_n=self._try_create_dataset(folder=config.DatasetFolderPath.ir_n),
            tv_n=self._try_create_dataset(folder=config.DatasetFolderPath.tv_n),
            ir_w=self._try_create_dataset(folder=config.DatasetFolderPath.ir_w),
            tv_w=self._try_create_dataset(folder=config.DatasetFolderPath.tv_w),
        )

    def _try_create_dataset(self, folder: str | Path) -> FolderVideoCapture | None:
        try:
            return FolderVideoCapture(folder=folder, ping_pong=True, loop=True)
        except Exception as e:
            print(e)
            return None

    def update(self) -> None:
        now_ns = time.monotonic_ns()
        for cam_id, reader in self._im_readers.items():
            if reader is None:
                continue

            ret, frame = reader.read()
            if not ret:
                continue

            self._state.cameras[cam_id].frames.enqueue(frame, now_ns)
            self._state.cameras[cam_id].resolution = frame.shape[:2]

        time.sleep(1 / self._framerate)

    def on_stop(self) -> None:
        pass
