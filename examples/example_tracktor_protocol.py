import time
from threading import Thread

from src.framework.protocols.tracktor.network.dispatcher import Dispatcher
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
    data_node = TracktorDataProtocolNode(Dispatcher())
    data_node.open()
    data_thread = Thread(target=get_period_tickable(data_node.tick, 0.5), daemon=True)
    data_thread.start()

    cmd_node = ServicesCommandsProtocolNode(Dispatcher())
    cmd_node.open()
    cmd_thread = Thread(target=get_period_tickable(cmd_node.tick, 0.5), daemon=True)
    cmd_thread.start()

    while True:
        data_packet = TCBPacket.create()
        data_node.send(data_packet)

        cmd_packet = LrfCtrlCmdPacket.create()
        cmd_node.send(cmd_packet)

        time.sleep(1)
