import time

from src.gimb.domain.cameras import TrOsCamHolder
from src.gimb.state import TrOsState
from src.framework.loop.loop import ILoop

import config.config as config


class RtspImagesLoop(ILoop):
    def __init__(self, state: TrOsState) -> None:
        super().__init__()
        self._state = state
        self._readers: TrOsCamHolder["GstRtspReader"] = None
        self._latest_timestamp = TrOsCamHolder(-1, -1, -1, -1)

    def on_start(self) -> None:
        from src.framework.frame.gst.rtsp_reader import GstRtspReader
        self._readers = TrOsCamHolder(
            ir_n=GstRtspReader(rtsp_url=config.RtspPath.ir_n),
            tv_n=GstRtspReader(rtsp_url=config.RtspPath.tv_n),
            ir_w=GstRtspReader(rtsp_url=config.RtspPath.ir_w),
            tv_w=GstRtspReader(rtsp_url=config.RtspPath.tv_w),
        )

    def update(self) -> None:
        now = time.monotonic_ns()
        for cam_id, reader in self._readers.items():
            sample = reader.get_latest_sample()
            if sample is None:
                continue
            if sample.timestamp_ns == self._latest_timestamp[cam_id]:
                continue

            frame = sample.arr()
            self._state.cameras[cam_id].frames.enqueue(frame, sample.timestamp_ns)
            self._latest_timestamp[cam_id] = sample.timestamp_ns
            if self._state.cameras[cam_id].resolution is None:
                self._state.cameras[cam_id].resolution = frame.shape[:2]
        time.sleep(5e-3)

    def on_stop(self) -> None:
        pass
