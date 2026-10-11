import ctypes
from ctypes import c_uint8, c_uint16
from typing import final

from ..base import BaseStructure


TRACKTOR_DATA_MAX_PACKET_SIZE = 1432
TRACKTOR_MAX_PAYLOAD_SIZE = 1426
NON_SPECIFIED_ID = object()


class TracktorHeader(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("src_id", c_uint8),
        ("dst_id", c_uint8),
        ("msg_id", c_uint16),
        ("payload_len", c_uint16),
    ]


class TracktorPacket(BaseStructure):
    _pack_ = 1
    _fields_ = [
        ("header", TracktorHeader),
    ]

    MSG_ID = NON_SPECIFIED_ID
    PROTOCOL = "tdp"

    @final
    @classmethod
    def msg_id(self):
        return self.MSG_ID

    @classmethod
    def create(cls, src_id=0, dst_id=0, payload=None):
        """Create a packet with header and payload"""
        payload_len = 0
        payload_class = getattr(cls, "_payload_class", None)
        if payload_class is not None:
            if payload is None:
                # Create default payload if none provided
                payload = payload_class()
            payload_len = ctypes.sizeof(payload_class)

        header = TracktorHeader(
            msg_id=cls.MSG_ID, src_id=src_id, dst_id=dst_id, payload_len=payload_len
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
