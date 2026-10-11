from src.framework.protocols.ntmp.base import NTMPMessage
from src.framework.protocols.ntmp.messages import InfDetector, InfTarget


def test_ntmp_message_inf_detection() -> None:
    sut = InfDetector.from_targets(
        status=InfDetector.STATUS(is_selftest_ok=True),
        targets=[InfTarget(id=1), InfTarget(id=2), InfTarget(id=3)],
    )

    assert sut.header.msg_id == 0x9FD491D1
    assert isinstance(sut, NTMPMessage)
    assert sut.count == 3
    assert sut.targets[0].id == 1
