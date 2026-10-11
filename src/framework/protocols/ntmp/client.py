import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass

from src.framework.loop.io_tickable import IOTickable
from src.framework.protocols.ntmp.base import NTMPMessage, NTMPHeader, NTMP_HEADER_SIZE


def _noop(header, payload):
    pass


@dataclass
class NTMPClientSettings:
    buffer_size: int = 1500


class NTMPClientBase(ABC):
    """Sends packets to NTMP receivers"""

    __slots__ = (
        "_settings",
        "_logger",
        "_packet_handler",
    )

    def __init__(
        self,
        packet_handler: Optional[Callable[[NTMPHeader, bytes], None]] = None,
        settings: Optional[NTMPClientSettings] = None,
    ) -> None:
        self._logger = logging.getLogger("ntmp")
        self._settings = settings if settings is not None else NTMPClientSettings()
        self._packet_handler = packet_handler or _noop

    def send_msg(self, msg: NTMPMessage) -> None:
        self._logger.debug("Sending %s", msg)
        self._send_raw_data(bytes(msg))

    def _on_raw_data_received(self, data: bytes, received: int) -> None:
        if received >= NTMP_HEADER_SIZE:
            header = NTMPHeader.from_buffer(data, 0)
            self._logger.debug("Received %s", header)
            self._packet_handler(header, data)
        else:
            self._logger.warning("Received data less than NTMP header size")

    @abstractmethod
    def _send_raw_data(self, data: bytes) -> None:
        ...
