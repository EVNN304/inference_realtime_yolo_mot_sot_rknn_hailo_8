import logging
import time
from typing import Optional

from src.gimb.ctrls.senders.base_sender import (
    TracktorDataBaseSender,
    TracktorPacketWithTimestamp,
    rad_to_mdeg,
)
from src.gimb.domain.tracking import TrOsTrackingStatus
from src.gimb.state import TrOsState
from src.framework.protocols.tracktor.protocols.tracktor_data_protocol.protocol import (
    TrackerExtPacket,
    DeviceIDs,
    TrackerExtPayload,
)


class TrOsTrackerSender(TracktorDataBaseSender):
    def __init__(self, state: TrOsState) -> None:
        super().__init__()
        self._empty_period_ns = 20_000_000
        self._latest_empty_ns = -1
        self._state = state
        self._logger = logging.getLogger("state")

    def _prepare_msg_internal(self) -> Optional[TracktorPacketWithTimestamp]:
        now = self._state.time_ns()
        payload = TrackerExtPayload()
        tracked_target = self._state.tracked_target
        if tracked_target.status == TrOsTrackingStatus.DISABLED:
            if now - self._latest_empty_ns >= self._empty_period_ns:
                self._latest_empty_ns = now
                payload.status.flags.status = 0
                payload.target.timestamp_us = now // 1000
                packet = self._pack_packet(payload, now)
            else:
                return None
        else:
            target_time_ns = self._state.tracked_target.timestamp_ns
            target_data_box = self._state.tracked_target.angular_box
            payload.status.flags.status = 2
            payload.target.delta_azimuth = rad_to_mdeg(target_data_box.azimuth)
            payload.target.delta_elevation = rad_to_mdeg(target_data_box.elevation)
            payload.target.angular_width = rad_to_mdeg(target_data_box.width)
            payload.target.angular_height = rad_to_mdeg(target_data_box.height)
            payload.target.timestamp_us = target_time_ns // 1000
            packet = self._pack_packet(payload, target_time_ns)

        self._logger.debug(
            "prepared (status=%d) %s", packet.packet.payload.status.flags.status, packet
        )
        return packet

    @staticmethod
    def _pack_packet(payload, timestamp_ns: int) -> TracktorPacketWithTimestamp:
        return TracktorPacketWithTimestamp(
            packet=TrackerExtPacket.create(src_id=DeviceIDs.TRACKER, payload=payload),
            timestamp_ns=timestamp_ns,
        )
