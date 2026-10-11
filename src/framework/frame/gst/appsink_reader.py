import threading
from dataclasses import dataclass
from typing import Callable, Optional, Any, Tuple

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib


@dataclass(frozen=True)
class GstVideoInfo:
    width: int
    height: int
    format: str  # e.g. "RGB", "BGR", "GRAY8"


@dataclass(frozen=True)
class GstSampleData:
    """
    Minimal sample info. Payload is raw bytes (already copied) to avoid lifetime issues.
    If you want NumPy, convert in your callback using payload + width/height/format.
    """

    info: GstVideoInfo
    pts_ns: Optional[int]
    dts_ns: Optional[int]
    duration_ns: Optional[int]
    data: bytes
    timestamp_ns: int

    def arr(self, copy: bool = False) -> np.ndarray:
        """
        Convert raw data into numpy array.

        Returns:
            HxWxC ndarray (uint8)

        Supported formats:
            RGB, BGR, RGBA, BGRA, GRAY8

        copy:
            False -> zero-copy view over data (fastest)
            True  -> return independent copy
        """
        h = self.info.height
        w = self.info.width
        fmt = self.info.format.upper()

        if fmt in ("RGB", "BGR"):
            channels = 3
        elif fmt in ("RGBA", "BGRA"):
            channels = 4
        elif fmt in ("GRAY8", "GRAY"):
            channels = 1
        else:
            raise RuntimeError(f"Unsupported pixel format: {fmt}")

        expected = h * w * channels
        if len(self.data) != expected:
            raise RuntimeError(
                f"Buffer size mismatch. Expected {expected} bytes, got {len(self.payload)} "
                f"(format={fmt}, {w}x{h})"
            )

        arr = np.frombuffer(self.data, dtype=np.uint8)

        if channels == 1:
            arr = arr.reshape((h, w))
        else:
            arr = arr.reshape((h, w, channels))

        return arr.copy() if copy else arr

    # @property
    # def timestamp_ns(self) -> Optional[int]:
    #     if self.pts_ns is None:
    #         return None
    #     return self.base_time + self.pts_ns


OnSampleFn = Callable[[GstSampleData], None]
OnErrorFn = Callable[[Exception], None]
OnStateFn = Callable[[Gst.State, Gst.State, Gst.State], None]  # old, new, pending


class GstAppSinkReader:
    def __init__(
        self,
        pipeline_str: str,
        *,
        appsink_name: str = "sink",
        keep_latest: bool = True,
        on_sample: Optional[OnSampleFn] = None,
        on_error: Optional[OnErrorFn] = None,
        on_state: Optional[OnStateFn] = None,
    ) -> None:
        Gst.init(None)

        self._pipeline_str = pipeline_str
        self._appsink_name = appsink_name
        self._keep_latest = keep_latest

        self._on_sample = on_sample
        self._on_error = on_error
        self._on_state = on_state

        self._pipeline = Gst.parse_launch(self._pipeline_str)

        self._appsink = self._pipeline.get_by_name(self._appsink_name)
        if self._appsink is None:
            raise RuntimeError(
                f"appsink named '{self._appsink_name}' not found in pipeline"
            )

        self._appsink.set_property("emit-signals", True)
        self._appsink.connect("new-sample", self._on_new_sample)

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", self._on_bus_message)

        # self._loop: Optional[GLib.MainLoop] = None
        # self._thread: Optional[threading.Thread] = None
        self._running = False

        # Latest sample storage + frame notification
        self._latest_lock = threading.Lock()
        self._latest: Optional[GstSampleData] = None

        self._cv = threading.Condition()
        self._frame_seq: int = 0  # increments when a frame is accepted
        self._last_error: Optional[Exception] = None
        self._eos: bool = False

        self._running = True
        self._eos = False
        self._last_error = None

        self._base_time: Optional[float] = None

        self._set_state_playing()

    def is_opened(self) -> bool:
        return self._running

    def stop(self) -> None:
        # Wake any waiters so they don't hang forever
        with self._cv:
            self._cv.notify_all()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame = self.wait_for_sample()
        if frame is not None:
            return True, frame.arr()
        else:
            return False, None

    def get_latest_sample(self) -> Optional[GstSampleData]:
        if not self._keep_latest:
            return None
        with self._latest_lock:
            return self._latest

    def get_frame_seq(self) -> int:
        """Current frame sequence counter (increments on every new accepted sample)."""
        with self._cv:
            return self._frame_seq

    def get_base_time(self) -> int:
        return self._pipeline.get_base_time()

    def wait_for_sample(
        self,
        *,
        after_seq: Optional[int] = None,
        timeout_s: Optional[float] = None,
        raise_on_error: bool = True,
    ) -> Optional[GstSampleData]:
        """
        Block until a new frame arrives.

        Returns:
          (seq, sample) when a new frame becomes available.
          None on timeout.

        Parameters:
          after_seq:
            - If provided, waits until internal seq > after_seq.
            - If None, waits until the next frame after *current* seq at call time.
          timeout_s:
            - None means wait indefinitely.
          raise_on_error:
            - If True, raises on pipeline error/EOS/stop.
        """
        with self._cv:
            if not self._running:
                if raise_on_error:
                    raise RuntimeError("Pipeline is not running")
                return None

            target = self._frame_seq if after_seq is None else after_seq

            ok = self._cv.wait_for(
                lambda: (not self._running)
                or (self._last_error is not None)
                or self._eos
                or (self._frame_seq > target),
                timeout=timeout_s,
            )

            if not ok:
                return None  # timeout

            if not self._running:
                if raise_on_error:
                    raise RuntimeError("Pipeline stopped while waiting for frame")
                return None

            if self._last_error is not None:
                if raise_on_error:
                    raise self._last_error
                return None

            if self._eos:
                if raise_on_error:
                    raise RuntimeError("Pipeline reached EOS")
                return None

            # New frame available: return the latest + seq atomically enough
            seq = self._frame_seq
            with self._latest_lock:
                sample = self._latest

            return sample

    def _set_state_playing(self) -> bool:
        try:
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
        except Exception as e:
            self._running = False
            self._emit_error(e)
        return False

    def _on_new_sample(self, sink: Any) -> Gst.FlowReturn:
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            data = self._sample_to_data(sample)

            if self._keep_latest:
                with self._latest_lock:
                    self._latest = data

            # Notify waiters that a new frame is available
            with self._cv:
                self._frame_seq += 1
                self._cv.notify_all()

            if self._on_sample is not None:
                self._on_sample(data)

        except Exception as e:
            self._emit_error(e)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        try:
            mtype = message.type

            if mtype == Gst.MessageType.ERROR:
                err, dbg = message.parse_error()
                raise RuntimeError(f"GStreamer ERROR: {err}; debug={dbg}")

            if mtype == Gst.MessageType.EOS:
                with self._cv:
                    self._eos = True
                    self._cv.notify_all()
                return

            if mtype == Gst.MessageType.STATE_CHANGED and message.src == self._pipeline:
                old, new, pending = message.parse_state_changed()
                if self._on_state is not None:
                    self._on_state(old, new, pending)

        except Exception as e:
            self._emit_error(e)

    def _sample_to_data(self, sample: Gst.Sample) -> GstSampleData:
        buf: Gst.Buffer = sample.get_buffer()
        caps: Gst.Caps = sample.get_caps()
        s = caps.get_structure(0)

        width = int(s.get_value("width"))
        height = int(s.get_value("height"))
        fmt = str(s.get_value("format"))

        info = GstVideoInfo(width=width, height=height, format=fmt)

        pts = int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else None
        dts = int(buf.dts) if buf.dts != Gst.CLOCK_TIME_NONE else None
        dur = int(buf.duration) if buf.duration != Gst.CLOCK_TIME_NONE else None

        timestamp_ns = self._buffer_pts_to_gst_clock_time_ns(sample, buf)

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Failed to map Gst.Buffer for reading")

        try:
            data = bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)

        return GstSampleData(
            info=info,
            pts_ns=pts,
            dts_ns=dts,
            duration_ns=dur,
            data=data,
            timestamp_ns=timestamp_ns,
        )

    def _buffer_pts_to_gst_clock_time_ns(self, sample: Gst.Sample, buf: Gst.Buffer) -> int | None:
        if buf is None:
            return None

        pts = buf.pts

        seg = sample.get_segment()
        if seg is None:
            # Fallback: many pipelines have pts already as running-time, but not guaranteed
            running_time = pts
        else:
            running_time = seg.to_running_time(Gst.Format.TIME, pts)
            if running_time == Gst.CLOCK_TIME_NONE:
                return None

        if self._base_time is None:
            self._base_time = self._pipeline.get_base_time()
        return self._base_time + running_time

    def _emit_error(self, e: Exception) -> None:
        # Store error + wake waiters
        with self._cv:
            self._last_error = e
            self._cv.notify_all()

        if self._on_error is not None:
            try:
                self._on_error(e)
                return
            except Exception:
                pass

        print(f"[GstAppSinkGrabber] ERROR: {e}")
        try:
            self.stop()
        except Exception:
            pass
