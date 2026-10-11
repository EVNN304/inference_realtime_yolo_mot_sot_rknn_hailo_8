import fcntl
import ipaddress
import socket
import struct


def _is_multicast(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_multicast
    except ValueError:
        return False


def _get_ip_address(ifname: str):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", ifname[:15].encode("utf-8")),
        )[20:24]
    )
