import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.framework.protocols.tracktor.protocols.tracktor_data_protocol.base import TracktorPacket


@dataclass(slots=True, frozen=True)
class TracktorPacketWithTimestamp:
    packet: TracktorPacket
    timestamp_ns: int


class TracktorDataBaseSender(ABC):
    def __init__(self) -> None:
        self._latest_send_ts = -1

    def try_prepare_pckt(self) -> Optional[TracktorPacket]:
        maybe_packet = self._prepare_msg_internal()
        if maybe_packet is None:
            return None
        if maybe_packet.timestamp_ns == self._latest_send_ts:
            return None

        self._latest_send_ts = maybe_packet.timestamp_ns
        return maybe_packet.packet

    @abstractmethod
    def _prepare_msg_internal(self) -> Optional[TracktorPacketWithTimestamp]:
        ...


def rad_to_mdeg(val: float) -> int:
    return round(1e3 * val * 180 / math.pi)
