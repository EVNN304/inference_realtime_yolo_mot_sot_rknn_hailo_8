import time
from threading import Thread

from src.framework.protocols.tracktor.network.services_commands_protocol.scp_node import (
    ServicesCommandsProtocolNode,
)
from src.framework.protocols.tracktor.network.tracktor_data_protocol.tdp_node import (
    TracktorDataProtocolNode,
)

from src.framework.protocols.tracktor.protocols.services_commands_protocol.protocol import (
    LrfCtrlCmdPacket,
)
from src.framework.protocols.tracktor.protocols.tracktor_data_protocol.protocol import (
    TCBPacket,
)


def get_period_tickable(tick, period):
    def loop():
        while True:
            start = time.perf_counter()
            tick()
            end = time.perf_counter()
            time.sleep(max(0, period - (end - start)))

    return loop


if __name__ == "__main__":
    # node = TracktorDataProtocolNode()
    # node.open()
    # thread = Thread(target=get_period_tickable(node.tick, 0.5), daemon=True)
    # thread.start()

    # while True:
    #    packet = TCBPacket.create()
    #    node.send(packet)
    #    time.sleep(1)

    node = ServicesCommandsProtocolNode()
    node.open()
    thread = Thread(target=get_period_tickable(node.tick, 0.5), daemon=True)
    thread.start()

    while True:
        packet = LrfCtrlCmdPacket.create()
        node.send(packet)
        time.sleep(1)
