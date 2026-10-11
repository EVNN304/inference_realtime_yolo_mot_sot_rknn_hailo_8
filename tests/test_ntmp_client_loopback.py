import ctypes
import time

from src.framework.protocols.ntmp.base import NTMPMessage, NTMPHeader
from src.framework.protocols.ntmp.client import NTMPClientUDP, NTMPClientUPDSettings


class TestNTMPMessage(NTMPMessage):
    _header_ = NTMPHeader(msg_id=0x300)

    _fields_ = (
        ("reserved", ctypes.c_uint16),
    )


def test_ntmp_client_loopback() -> None:
    data = {}
    def handler(header, body):  
        data["received"] = True
        data["msg_id"] = header.msg_id
        data["body"] = body
        data["msg"] = TestNTMPMessage.from_buffer(body)

    test_msg = TestNTMPMessage()
    test_msg.reserved = 1
    sut = NTMPClientUDP(handler, settings=NTMPClientUPDSettings(loopback=True))
    sut.open()
    sut.send_msg(test_msg)
    sut.tick_write()
    # Data is sent. Waiting to receive through loopback...
    time.sleep(1)
    sut.tick_read()
    print("Test complete.")

    assert "received" in data
    assert data["received"] is True
    assert data["msg_id"] == 0x300
    assert len(data["body"]) == ctypes.sizeof(test_msg)
