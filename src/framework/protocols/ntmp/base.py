from copy import copy
from ctypes import LittleEndianStructure
from ctypes import c_uint16
from ctypes import c_uint32
from ctypes import sizeof


class NTMPHeader(LittleEndianStructure):
    _fields_ = (
        ("src_id", c_uint16),
        ("dst_id", c_uint16),
        ("msg_timestamp", c_uint32),
        ("msg_id", c_uint32),
        ("data_len", c_uint32),
    )

    _pack_ = 1

    def __repr__(self):
        return (
            "NTMPHeader(src_id=0x{0:04X}, dst_id=0x{1:04X}, "
            "msg_timestamp={2}, msg_id=0x{3:08X}, data_len={4})"
        ).format(
            self.src_id,
            self.dst_id,
            self.msg_timestamp,
            self.msg_id,
            self.data_len,
        )


class NTMPStatus(LittleEndianStructure):
    _fields_ = ()

    def __repr__(self):
        def field_to_str(field):
            return "{}={}".format(field[0], getattr(self, field[0]))

        return "".join(
            (
                "{}(".format(self.__class__.__name__),
                ", ".join([field_to_str(field) for field in self._fields_]),
                ")",
            )
        )


NTMPButtons = NTMPStatus


class NTMPMessage(LittleEndianStructure):
    _header_ = NTMPHeader()

    _fields_ = [("header", NTMPHeader)]

    _pack_ = 1

    def __new__(cls, *args, **kwargs):
        message = super(NTMPMessage, cls).__new__(cls, *args, **kwargs)
        message.header = copy(cls._header_)

        message.header.data_len = sizeof(message) - sizeof(message.header)

        if "msg_timestamp" in kwargs:
            message.header.msg_timestamp = kwargs["msg_timestamp"]

        return message

    def __repr__(self):
        def field_to_str(field):
            return "{}={}".format(field[0], getattr(self, field[0]))

        return "".join(
            (
                "{}(header={}, ".format(self.__class__.__name__, self.header),
                ", ".join([field_to_str(field) for field in self._fields_]),
                ")",
            )
        )

    @classmethod
    def get_msg_id(cls) -> int:
        return cls._header_.msg_id


NTMP_HEADER_SIZE = sizeof(NTMPHeader)
