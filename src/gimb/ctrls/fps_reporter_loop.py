import logging

from src.gimb.domain.cameras import TrOsCamId
from src.gimb.state import TrOsState
from src.framework.loop.loop import ILoop


class TrOsFpsReporterLoop(ILoop):
    def __init__(self, state: TrOsState) -> None:
        super().__init__()
        self._state = state
        self._logger = logging.getLogger("fps")

    def on_start(self) -> None:
        pass

    def update(self) -> None:
        for cam_id in TrOsCamId:

            if self._state.det_durations[cam_id] is None:
                continue
            
            det_fps = self._state.get_det_fps(cam_id)
            mot_fps = self._state.get_mot_fps(cam_id)
            sot_fps = self._state.get_sot_fps(cam_id)
            self._logger.info(
                f"Cam '{cam_id}' stat: det_fps: {det_fps:.1f}; mot_fps: {mot_fps:.1f}; sot_fps: {sot_fps:.1f}"
            )

    def on_stop(self) -> None:
        pass
