from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Tuple

import numpy as np

from src.framework.frame.gst.appsink_reader import GstAppSinkReader, GstSampleData


class RtspCodec(str, Enum):
    H264 = "h264"


class RtspProtocols(str, Enum):
    TCP = "tcp"
    UDP = "udp"


@dataclass(frozen=True)
class RtspDecoder:
    """
    For H264 now.
    Example:
      RtspDecoder(element="avdec_h264")
      RtspDecoder(element="vaapih264dec")
      RtspDecoder(element="nvv4l2decoder", properties="enable-max-performance=1")
    """

    element: str = "avdec_h264"
    properties: str = ""


@dataclass(frozen=True)
class RtspVideoConvert:
    """
    videoconvert + desired raw format caps for appsink.
    For OpenCV you'd typically use BGR.
    """

    element: str = "videoconvert"
    format: str = "RGB"  # default requested


@dataclass(frozen=True)
class VideoConvertConfig:
    """
    videoconvert + caps output.
    Typical:
      format="RGB" or "BGR"
    """

    element: str = "videoconvert"
    format: str = "RGB"  # default requested


class GstRtspReader:
    def __init__(
        self,
        rtsp_url: str,
        *,
        codec: RtspCodec = RtspCodec.H264,
        decoder: RtspDecoder = RtspDecoder(element="avdec_h264"),
        videoconvert: RtspVideoConvert = RtspVideoConvert(format="RGB"),
        latency: int = 0,
        protocols: RtspProtocols = RtspProtocols.TCP,
        drop_on_latency: bool = True,
        appsink_name: str = "sink",
        appsink_drop: bool = True,
        appsink_max_buffers: int = 1,
        appsink_sync: bool = False,
        keep_latest: bool = True,
        on_sample: Optional[Callable[["GstSampleData"], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_state: Optional[Callable] = None,
    ) -> None:
        self._rtsp_url = rtsp_url
        self._codec = codec
        self._decoder = decoder
        self._videoconvert = videoconvert
        self._latency = int(latency)
        self._protocols = protocols
        self._drop_on_latency = bool(drop_on_latency)

        self._appsink_name = appsink_name
        self._appsink_drop = bool(appsink_drop)
        self._appsink_max_buffers = int(appsink_max_buffers)
        self._appsink_sync = bool(appsink_sync)

        self._pipeline_str = self._build_pipeline_str()

        self._grabber = GstAppSinkReader(
            self._pipeline_str,
            appsink_name=self._appsink_name,
            keep_latest=keep_latest,
            on_sample=on_sample,
            on_error=on_error,
            on_state=on_state,
        )

    @property
    def pipeline_str(self) -> str:
        return self._pipeline_str

    def stop(self) -> None:
        self._grabber.stop()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._grabber.read()

    def get_latest_sample(self):
        return self._grabber.get_latest_sample()

    def get_frame_seq(self) -> int:
        return self._grabber.get_frame_seq()

    def wait_for_sample(
        self,
        *,
        after_seq: Optional[int] = None,
        timeout_s: Optional[float] = None,
        raise_on_error: bool = True,
    ) -> Optional[GstSampleData]:
        sample = self._grabber.wait_for_sample(
            after_seq=after_seq,
            timeout_s=timeout_s,
            raise_on_error=raise_on_error,
        )
        # Guard against None (can happen if keep_latest=False or timing edge)
        if sample is None:
            if raise_on_error:
                raise RuntimeError("New sample signaled but latest sample is None")
            return None

        return sample

    def _build_pipeline_str(self) -> str:
        src = (
            f'rtspsrc location="{self._rtsp_url}" '
            f"protocols={self._protocols.value} "
            f"latency={self._latency} "
            f'drop-on-latency={"true" if self._drop_on_latency else "false"}'
        )

        if self._codec != RtspCodec.H264:
            raise ValueError(f"Unsupported codec: {self._codec} (H264 only for now)")

        depay = "rtph264depay"
        parse = "h264parse"

        dec = self._decoder.element.strip()
        if not dec:
            raise ValueError("decoder.element must be non-empty")

        if self._decoder.properties.strip():
            dec = f"{dec} {self._decoder.properties.strip()}"

        vc = self._videoconvert.element.strip()
        if not vc:
            raise ValueError("videoconvert.element must be non-empty")

        out_fmt = self._videoconvert.format.strip().upper()
        if not out_fmt:
            raise ValueError("videoconvert.format must be non-empty")

        appsink = (
            f"appsink name={self._appsink_name} "
            f"emit-signals=true "
            f'drop={"true" if self._appsink_drop else "false"} '
            f"max-buffers={self._appsink_max_buffers} "
            f'sync={"true" if self._appsink_sync else "false"}'
        )

        # No queues by default (lower latency). Add later only if needed.
        return (
            f"{src} ! "
            f"{depay} ! {parse} ! {dec} ! "
            f"{vc} ! video/x-raw,format={out_fmt} ! "
            f"{appsink}"
        )
