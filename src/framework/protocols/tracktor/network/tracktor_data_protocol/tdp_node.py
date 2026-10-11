import logging

from ..udp_node import MulticastUDPNode
from ..dispatcher import Dispatcher

from src.framework.protocols.tracktor.protocols.tracktor_data_protocol.base import (
    TRACKTOR_DATA_MAX_PACKET_SIZE,
    TracktorPacket,
)
from src.framework.protocols.tracktor.protocols.tracktor_data_protocol.protocol import (
    parse_tracktor_packet,
)


class TracktorDataProtocolNode(MulticastUDPNode[TracktorPacket]):
    def __init__(self, dispatcher: Dispatcher, port: int = 15000):
        super().__init__(
            logging.getLogger("tdp"),
            port,
            recv_buffer_size=TRACKTOR_DATA_MAX_PACKET_SIZE,
        )
        self._dispatcher = dispatcher

    def _process(self, buffer: bytes, received: int):
        try:
            message = parse_tracktor_packet(buffer[:received])
            if message is None:
                return

            self._dispatcher.dispatch(message)
        except Exception as e:
            self._logger.error(e)

    def _pack(self, msg):
        return bytearray(msg)
