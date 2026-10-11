import logging
import select
import socket
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List

from src.framework.loop.io_tickable import IOTickable
from src.framework.protocols.ip import _is_multicast, _get_ip_address
from src.framework.protocols.ntmp.base import NTMPHeader, NTMPMessage, NTMP_HEADER_SIZE
from src.framework.protocols.ntmp.client import NTMPClientSettings, NTMPClientBase

@dataclass
class NTMPClientUPDSettings(NTMPClientSettings):
    dst_end_point: Tuple[str, int] = ("224.224.224.1", 7878)
    loopback: bool = True
    iface: str = "eth0"
    tick_time: float = 1e-2


def _init_mcast_socket(sock: socket.socket, settings: NTMPClientUPDSettings) -> None:
    addr, _ = settings.dst_end_point
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    src_ip = _get_ip_address(settings.iface)
    sock.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_MULTICAST_IF,
        socket.inet_aton(src_ip),
    )
    dst_inet_aton = socket.inet_aton(addr)
    mreq = dst_inet_aton + socket.inet_aton("0.0.0.0")
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_MULTICAST_LOOP,
        int(settings.loopback),
    )
    sock.bind(settings.dst_end_point)


class NTMPClientUDP(NTMPClientBase, IOTickable):
    __slots__ = (
        "_udp_sock",
        "_send_queue",
        "_recv_buffer",
    )

    def __init__(
        self,
        packet_handler: Optional[Callable[[NTMPHeader, bytes], None]] = None,
        settings: Optional[NTMPClientUPDSettings] = None,
    ) -> None:
        super().__init__(packet_handler, settings)
        self._settings = settings if settings is not None else NTMPClientUPDSettings()
        self._udp_sock: socket.socket = None
        self._send_queue = deque()
        self._recv_buffer = bytearray(self._settings.buffer_size)

    def open(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self._settings.tick_time)
        assert isinstance(self._settings, NTMPClientUPDSettings)
        addr, port = self._settings.dst_end_point
        if _is_multicast(addr):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            src_ip = _get_ip_address(self._settings.iface)
            sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(src_ip)
            )
            dst_inet_aton = socket.inet_aton(self._settings.dst_end_point[0])
            mreq = dst_inet_aton + socket.inet_aton("0.0.0.0")
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_LOOP,
                int(self._settings.loopback),
            )
            sock.bind(self._settings.dst_end_point)

        self._udp_sock = sock

    def _send_raw_data(self, data: bytes) -> None:
        self._send_queue.append(data)

    def tick_read(self) -> None:
        wait_time = self._settings.tick_time
        read_socks, _, _ = select.select([self._udp_sock], [], [], wait_time)
        if read_socks:
            received = self._udp_sock.recv_into(
                self._recv_buffer, self._settings.buffer_size
            )
            self._on_raw_data_received(self._recv_buffer, received)

    def tick_write(self) -> None:
        if self._send_queue:
            data = self._send_queue.popleft()
            self._udp_sock.sendto(data, self._settings.dst_end_point)


class NTMPClientUDPSync:
    """
    Standalone synchronous UDP NTMP client.

    - send_msg(): immediate send
    - receive(timeout_s): blocks up to timeout_s using select()
      and returns all packets received during that period as
      (header, raw_bytes) tuples.
    """

    __slots__ = (
        "_settings",
        "_logger",
        "_udp_sock",
        "_recv_buffer",
    )

    def __init__(
        self,
        settings: Optional[NTMPClientUPDSettings] = None,
    ) -> None:
        self._logger = logging.getLogger("ntmp")
        self._settings = settings if settings is not None else NTMPClientUPDSettings()
        self._recv_buffer = bytearray(self._settings.buffer_size)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(True)

        if _is_multicast(self._settings.dst_end_point[0]):
            _init_mcast_socket(sock, self._settings)

        self._udp_sock = sock

    def close(self) -> None:
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            finally:
                self._udp_sock = None

    def send_msg(self, msg: NTMPMessage) -> None:
        if self._udp_sock is None:
            raise RuntimeError("Socket is not open()")

        data = bytes(msg)
        self._logger.debug("Sending %s", msg)
        self._udp_sock.sendto(data, self._settings.dst_end_point)

    def receive(self, timeout_s: Optional[float]) -> List[Tuple[NTMPHeader, bytes]]:
        """
        Wait up to timeout_s for incoming packets and return all packets received
        during that interval as (header, raw_bytes).

        timeout_s:
            None  -> block indefinitely
            0.0   -> poll (non-blocking)
            >0.0  -> block up to that many seconds
        """
        if self._udp_sock is None:
            raise RuntimeError("Socket is not open()")

        out: List[Tuple[NTMPHeader, bytes]] = []

        # First wait (blocking up to timeout_s)
        r, _, _ = select.select([self._udp_sock], [], [], timeout_s)
        if not r:
            return out

        # At least one packet is ready
        while True:
            pkt = self._recv_one()
            if pkt is not None:
                out.append(pkt)

            # Drain additional packets without blocking
            r, _, _ = select.select([self._udp_sock], [], [], 0.0)
            if not r:
                break

        return out

    def _recv_one(self) -> Optional[Tuple[NTMPHeader, bytes]]:
        assert self._udp_sock is not None

        received = self._udp_sock.recv_into(
            self._recv_buffer,
            self._settings.buffer_size,
        )

        if received < NTMP_HEADER_SIZE:
            self._logger.warning(
                "Received data less than NTMP header size (%d < %d)",
                received,
                NTMP_HEADER_SIZE,
            )
            return None

        data = bytearray(memoryview(self._recv_buffer)[:received])

        header = NTMPHeader.from_buffer(data, 0)
        self._logger.debug("Received %s", header)

        return header, data
