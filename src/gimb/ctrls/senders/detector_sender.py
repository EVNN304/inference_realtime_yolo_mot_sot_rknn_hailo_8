import time

from src.gimb.ctrls.senders.base_sender import (
    TracktorDataBaseSender,
    TracktorPacketWithTimestamp,
    rad_to_mdeg,
)
from src.gimb.domain.cameras import TrOsCamId
from src.gimb.state import TrOsState
from src.framework.mot.track import Track
import config.config as config
from src.framework.protocols.tracktor.protocols.tracktor_data_protocol.protocol import (
    DetectionData,
    DetectorPacket,
    DetectionStatus,
    MAX_DETECTIONS,
    DeviceIDs,
)

class TrOsDetectorSender(TracktorDataBaseSender):
    def __init__(self, state: TrOsState):
        super().__init__()
        self._state = state
        self._det_timeout_ns = config.Tracking.detections_timeout_ns
        self._empty_period_ns = 20_000_000
        self._latest_empty_sent = -1

    def _prepare_msg_internal(self) -> TracktorPacketWithTimestamp:
        now = self._state.time_ns()
        packet = DetectorPacket.create(src_id=DeviceIDs.DETECTOR)
        det_user_state = self._state.detector_desired_state
        if not det_user_state.enabled:
            if now - self._latest_empty_sent >= self._empty_period_ns:
                self._latest_empty_sent = now
                return TracktorPacketWithTimestamp(packet=packet, timestamp_ns=now)

        detections = packet.payload.detections

        # pack tracks from all channels
        pack_index = 0
        newest_timestamp = -1
        # TODO: remove duplicates (one on another) if exist
        for cam_id in TrOsCamId:
            if det_user_state.channel != cam_id.channel:
                continue

            tracks_sample = self._state.mot_tracks[cam_id].latest()
            if tracks_sample is None:
                continue

            timestamp_ns = tracks_sample.timestamp_ns
            if now > timestamp_ns + self._det_timeout_ns:
                continue

            if timestamp_ns > newest_timestamp:
                newest_timestamp = timestamp_ns

            tracks = tracks_sample.obj

            for track in tracks:
                if pack_index > MAX_DETECTIONS:
                    break
                detections[pack_index] = self._pack_detection(cam_id, track)
                pack_index += 1

        packet.payload.count = pack_index
        packet.payload.status.flags.is_enabled = 1
        packet.payload.status.flags.channel = int(det_user_state.channel)

        return TracktorPacketWithTimestamp(packet=packet, timestamp_ns=newest_timestamp)

    def _pack_detection(self, cam_id: TrOsCamId, track: Track) -> DetectionData:
        ang_box = self._state.cameras[cam_id].alignment.px2ang_box(track.box)
        status = DetectionStatus()
        status.flags.status = 1
        return DetectionData(
            status=status,
            id=track.track_id,
            det_class=int(track.cls),
            det_type=0,
            class_confidence=int(100 * track.conf),
            type_confidence=0,
            age=0,
            delta_azimuth=rad_to_mdeg(ang_box.azimuth),
            delta_elevation=rad_to_mdeg(ang_box.elevation),
            angular_width=rad_to_mdeg(ang_box.width),
            angular_height=rad_to_mdeg(ang_box.height),
            distance=1000,
        )
