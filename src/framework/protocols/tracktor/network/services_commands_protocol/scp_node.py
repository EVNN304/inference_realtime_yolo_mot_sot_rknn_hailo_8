import logging

from ..udp_node import MulticastUDPNode
from ..dispatcher import Dispatcher

from src.framework.protocols.tracktor.protocols.services_commands_protocol.base import (
    SERVICES_COMMANDS_MAX_PACKET_SIZE,
    ServicesCommandsPacket,
)
from src.framework.protocols.tracktor.protocols.services_commands_protocol.protocol import (
    parse_services_command_packet,
)


class ServicesCommandsProtocolNode(MulticastUDPNode[ServicesCommandsPacket]):
    def __init__(self, dispatcher: Dispatcher, port: int = 16000):
        super().__init__(
            logging.getLogger("scp"),
            port,
            recv_buffer_size=SERVICES_COMMANDS_MAX_PACKET_SIZE,
        )
        self._dispatcher = dispatcher

    def _process(self, buffer: bytes, received: int):
        message = parse_services_command_packet(buffer[:received])
        if message is None:
            return

        self._dispatcher.dispatch(message)

    def _pack(self, msg):
        return bytearray(msg)
