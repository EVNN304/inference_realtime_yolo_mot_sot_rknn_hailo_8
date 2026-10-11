from typing import final
from ctypes import c_uint8, c_uint16

from ..base import BaseStructure


SERVICES_COMMANDS_ID_IDX = 0
SERVICES_COMMANDS_ID_SIZE = 2
SERVICES_COMMANDS_MAX_PACKET_SIZE = 19
SERVICES_COMMANDS_MAX_PAYLOAD_SIZE = 16
NON_SPECIFIED_ID = object()


class ServicesCommandsHeader(BaseStructure):
    """Services Commands Protocol Header"""

    _pack_ = 1
    _fields_ = [
        ("id", c_uint16),  # ID of the command
        ("src_id", c_uint8),  # ID of the block command came from
    ]


class ServicesCommandsPacket(BaseStructure):
    """Base class for all Services Commands Protocol packets"""

    _pack_ = 1
    _fields_ = [
        ("header", ServicesCommandsHeader),
    ]
    CMD_ID = NON_SPECIFIED_ID
    PROTOCOL = "scp"

    @final
    @classmethod
    def msg_id(self):
        return self.CMD_ID

    @classmethod
    def create(cls, src_id=0, payload=None):
        """Create a packet with header and payload"""
        if payload is None:
            # Create default payload if none provided
            payload_class = getattr(cls, "_payload_class", None)
            if payload_class:
                payload = payload_class()

        header = ServicesCommandsHeader(
            id=cls.CMD_ID,
            src_id=src_id,
        )

        # Create instance with header
        instance = cls()
        instance.header = header

        # Set payload if it exists
        if hasattr(instance, "payload") and payload is not None:
            instance.payload = payload

        return instance
    
    def __str__(self):
        return f"{self.__class__.__name__}(header={self.header}, payload={self.payload})"
    
    def __repr__(self):
        return self.__str__()
