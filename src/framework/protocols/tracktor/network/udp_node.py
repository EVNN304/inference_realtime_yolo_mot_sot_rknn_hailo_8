import select
import time
import socket
import struct
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Generic, Optional, TypeVar, final

TMessage = TypeVar("TMessage")


class MulticastUDPNode(ABC, Generic[TMessage]):
    MULTICAST_GROUP = "224.224.224.1"
    INTERFACE_IP = "192.168.65.116"

    def __init__(
        self,
        logger: logging.Logger,
        port: Optional[int] = None,
        recv_buffer_size: int = 1432,
    ) -> None:
        self._logger = logger
        if port is None:
            self._bind_address = None
            self._recv_buffer = bytearray(0)
        else:
            self._bind_address = ("", port)
            self._recv_buffer = bytearray(recv_buffer_size)

        if port is None:
            self._dest_address = None
        else:
            self._dest_address = (self.MULTICAST_GROUP, port)

        self._send_queue: Deque[TMessage] = deque()
        self._sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)

    @final
    def open(self) -> None:
        RETRIES = 8
        self._sock.bind(self._bind_address)
        if self._bind_address is not None:
            for i in range(RETRIES):
                try:
                    #self._sock.bind(self._bind_address)
                    mreq = struct.pack(
                        "4s4s", socket.inet_aton(self.MULTICAST_GROUP), socket.inet_aton(self.INTERFACE_IP)
                    )
                    self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                except OSError as e:
                    if e.errno == 19:
                        self._logger.error("Device not ready, waiting...")
                        time.sleep(2)
                    else:
                        raise e
                else:
                    break


    @final
    def send(self, msg: TMessage) -> None:
        self._logger.debug("Sent %s", msg)
        self._send_queue.append(msg)

    @final
    def tick(self) -> None:
        readable, writable, exceptional = select.select(
            [self._sock],
            [self._sock],
            [self._sock],
            0,
        )
        if self._sock in exceptional:
            self._logger.error("Socket error")
        if self._sock in readable:
            received = self._sock.recv_into(self._recv_buffer)
            self._process(self._recv_buffer, received)
        if self._sock in writable:
            if len(self._send_queue) != 0:
                msg = self._send_queue.popleft()
                packed_msg = self._pack(msg)
                self._sock.sendto(packed_msg, self._dest_address)

    @abstractmethod
    def _process(self, buffer: bytes, received: int) -> None: ...

    @abstractmethod
    def _pack(self, msg: TMessage) -> bytes: ...