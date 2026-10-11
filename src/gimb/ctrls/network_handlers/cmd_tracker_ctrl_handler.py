import math
import time
from typing import Tuple

from src.gimb.ctrls.tracking_controller import TrOsTrackingController
from src.gimb.domain.cameras import TrOsChannel
from src.gimb.domain.tracking import TrOsTrackingDesiredState, TrOsTrackInitMode
from src.gimb.state import TrOsState
from src.framework.protocols.tracktor.network.handlers import Handler, TMessage
from src.framework.protocols.tracktor.protocols.services_commands_protocol.protocol import (
    TrackerCtrlCmdPacket,
    TrackerCtrlCmdModeSettings,
)


class TrOsCmdTrackerCtrlHandler(Handler[TrackerCtrlCmdPacket]):
    def __init__(self, tracking_controller: TrOsTrackingController) -> None:
        super().__init__(TrackerCtrlCmdPacket)
        self._tracking_controller = tracking_controller

    def handle(self, msg: TrackerCtrlCmdPacket) -> None:
        self._tracking_controller.on_new_cmd(self._msg_to_desired_state(msg))

    def _msg_to_desired_state(
        self, msg: TrackerCtrlCmdPacket
    ) -> TrOsTrackingDesiredState:
        recv_time = time.monotonic_ns()
        init_mode = TrOsTrackInitMode(msg.payload.commands.flags.init_mode)
        return TrOsTrackingDesiredState(
            enabled=msg.payload.commands.flags.enable,
            channel=TrOsChannel(msg.payload.commands.flags.channel),
            init_mode=init_mode,
            init_data_rad=self._extract_init_data(init_mode, msg.payload.mode_settings),
            recv_time_ns=recv_time,
        )

    @staticmethod
    def _extract_init_data(
        mode: TrOsTrackInitMode, mode_settings: TrackerCtrlCmdModeSettings
    ) -> Tuple[float, float]:
        if TrOsTrackInitMode.DETECTION == mode:
            near_mode = mode_settings.detector_near_mode
            values_milli_deg = near_mode.reserved01, near_mode.reserved02
        elif TrOsTrackInitMode.CENTER == mode:
            center_mode = mode_settings.center_mode
            if center_mode.hor_size == 0:
                value_milli_deg_hor = 2860 / 3
            else:
                value_milli_deg_hor = center_mode.hor_size
            if center_mode.ver_size == 0:
                value_milli_deg_ver = 2860 / 3
            else:
                value_milli_deg_ver = center_mode.ver_size
            values_milli_deg = value_milli_deg_hor, value_milli_deg_ver
        else:
            segm_mode = mode_settings.segmentation_mode
            values_milli_deg = segm_mode.hor_pos, segm_mode.ver_pos

        return math.radians(1e-3 * values_milli_deg[0]), math.radians(
            1e-3 * values_milli_deg[1]
        )
