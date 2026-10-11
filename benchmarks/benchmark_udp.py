import queue
import random
import socket
import struct
import timeit
from collections import deque

# Multicast group and port
MCAST_GRP = "224.224.224.1"  # Multicast IP (must be in 224.0.0.0 to 239.255.255.255)
MCAST_PORT = 7878  # Port to bind to

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

# Allow multiple sockets to use the same PORT number
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind to the server address
sock.bind(("", MCAST_PORT))  # '' allows receiving data sent to the multicast group

# Tell the kernel that we want to add ourselves to a multicast group
# The address for the multicast group is converted to a 32-bit packed binary format
group = socket.inet_aton(MCAST_GRP)
mreq = struct.pack("4sL", group, socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

data = bytearray(100)
for i in range(len(data)):
    data[i] = random.randint(0, 255)

iters = 1_000
t = timeit.timeit(
    "sock.sendto(data, ('224.224.224.1', 7878))", number=iters, globals=globals()
)

print(f"{t / iters * 1e6:.3f} us per iter")

q = deque()

iters = 1_000
t = timeit.timeit(
    "q.append(data)", number=iters, globals=globals()
)

print(f"{t / iters * 1e6:.3f} us per iter")


q2 = queue.Queue(1000)

iters = 1_000
t = timeit.timeit(
    "q2.put(data)", number=iters, globals=globals()
)

print(f"{t / iters * 1e6:.3f} us per iter")