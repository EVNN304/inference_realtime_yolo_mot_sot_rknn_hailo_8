import socket
import time

from protocols.tracktor_data_protocol.protocol import (
    BCCorPacket,
    BCPacket,
    DetectionData,
    DetectorPacket,
    DistPacket,
    NSPacket,
    OSPacket,
    RFPacket,
    T2Packet,
    T55Packet,
    TCBPacket,
    TrackerPacket,
    TTIPacket,
    parse_tracktor_packet,
)

if __name__ == "__main__":
    MULTICAST_GROUP = "224.224.224.1"
    INTERFACE_IP = "192.168.65.121"
    PORT = 15000

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.bind((MULTICAST_GROUP, PORT))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    mreq = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton(INTERFACE_IP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    detector_packet = DetectorPacket.create(0, 0)
    detector_packet.payload.add_detections([DetectionData()] * 5)

    dist_packet = DistPacket.create(0, 0)
    dist_packet.payload.info.flags.source = 2

    packets = [
        OSPacket.create(0, 0),
        T55Packet.create(0, 0),
        T2Packet.create(0, 0),
        TCBPacket.create(0, 0),
        NSPacket.create(0, 0),
        TTIPacket.create(0, 0),
        TrackerPacket.create(0, 0),
        detector_packet,
        BCPacket.create(0, 0),
        RFPacket.create(0, 0),
        BCCorPacket.create(0, 0),
        DistPacket.create(0, 0),
        dist_packet,
    ]

    while True:
        for packet in packets:
            print(packet)
            print(packet.header, packet.payload)
            # tracktor_packet = packet.encode()
            packed = bytearray(packet)
            print(packed)
            # packed = tracktor_packet.pack(0, 0)
            unpacked = parse_tracktor_packet(packed)
            print(unpacked)

            sock.sendto(bytearray(packet), (MULTICAST_GROUP, PORT))
            print("Sent")
            print(bytearray(packet))
            time.sleep(0.25)
