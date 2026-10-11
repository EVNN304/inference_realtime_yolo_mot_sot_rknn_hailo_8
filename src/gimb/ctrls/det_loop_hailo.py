"""Hailo adapter for the existing source rings and MOT -> SOT controller."""
from src.framework.hailo.performance import get_stats
from collections import deque
from dataclasses import dataclass
import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
import numpy as np
import config.config as config
from src.framework.loop.loop import ILoop
from src.framework.core.box import BoxTwoCorners
from src.framework.det.detection import Detection
from src.framework.mot.boxmot_wrapper import BoxMotWrapperBase


@dataclass
class Packet:
    index: int
    timestamp_ns: int
    image: np.ndarray
    detections: np.ndarray
    queued_at: float


class HailoDetectionLoop(ILoop):
    def __init__(self, state, tracking_controller, settings, backend_factory=None, cam_id=None):
        super().__init__()
        if cam_id is None:
            from src.gimb.domain.cameras import TrOsCamId
            cam_id = TrOsCamId.TV_N
        if settings.mot_queue_size < 1 or settings.mot_max_wait_ms < 0 or settings.max_reid_objects < 0:
            raise ValueError('Invalid Hailo MOT queue, wait limit or detection cap')
        self.state, self.controller, self.settings = state, tracking_controller, settings
        self.cam_id = cam_id
        self.stats = get_stats(state)
        self.factory = backend_factory
        self.backend = None
        self.worker = None
        self.stop_event = Event()
        self.queue = Queue(maxsize=settings.mot_queue_size)
        self.errors = Queue()
        self.last_timestamp = -1
        self.index = 0
        self.logger = logging.getLogger('hailo_det')
        self.mot_dropped = 0
        self.done = Event()
        self.last_report = time.monotonic()
        self.det_count = self.mot_count = 0
        self.staged_frames = []
        self.pending_frames = deque()
        self.staged_at = 0
        # Bound source backlog while retaining frames that arrive during submission.
        self.input_max_age_ms = max(1.0, float(getattr(
            getattr(config, "Hailo", None), "input_max_age_ms", 250.0)))
        self.use_sahi = bool(config.Detector.use_sahi)
        self.async_full_frame = getattr(config, "detector_backend", "rknn").lower() == "hailo" and not self.use_sahi

    def on_start(self):
        if self.backend is not None:
            return
        try:
            if self.factory is None:
                from src.framework.hailo.backend import HailoBackend
                self.factory = HailoBackend
            self.backend = self.factory(self.settings, self.use_sahi)
            self.stats.reset()
            self.last_report = time.monotonic()
            self.worker = Thread(target=self._mot_worker, name='hailo-mot', daemon=True)
            self.worker.start()
            self.logger.info('Accelerator backend ready; source timestamps and SOT backend preserved')
        except BaseException:
            self.is_active = False
            self.on_stop()
            raise

    def _mot_worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    packet = self.queue.get(timeout=0.05)
                except Empty:
                    continue
                age_ms = (time.monotonic()-packet.queued_at)*1000
                if self.settings.mot_max_wait_ms and age_ms > self.settings.mot_max_wait_ms:
                    self.mot_dropped += 1
                    continue
                if not config.Detector.enable_local_detector:
                    continue
                start = time.perf_counter_ns()
                tracks = self.backend.track(packet, self.stop_event)
                if self.stop_event.is_set():
                    break
                result = BoxMotWrapperBase._from_boxmot_outputs(tracks)
                self.controller.on_mot_update(self.cam_id, result, packet.timestamp_ns)
                self.state.mot_durations[self.cam_id].enqueue(
                    time.perf_counter_ns()-start, packet.timestamp_ns)
                self.mot_count += 1
                self.stats.tick("mot")
                self.stats.observe("mot_ms", (time.perf_counter_ns()-start)/1e6)
        except BaseException as error:
            if not self.stop_event.is_set():
                self.errors.put(error)
                self.stop_event.set()

    def update(self):
        try:
            if not self.errors.empty():
                raise RuntimeError('Hailo MOT worker failed') from self.errors.get_nowait()
            if self.stop_event.is_set():
                self.is_active = False
                return
            if not config.Detector.enable_local_detector:
                self.staged_frames.clear()
                self.last_timestamp = -1
                # Drain completed device jobs without publishing old detections on re-enable.
                for job, samples, submitted_ns in self.pending_frames:
                    samples.clear()
                while self.pending_frames and self.pending_frames[0][0].done.is_set():
                    job, _, _ = self.pending_frames.popleft()
                    self.backend.collect_frames(job)
                self.stop_event.wait(0.02)
                return
            if self.async_full_frame:
                self._full_frame_update()
            else:
                sample = self.state.cameras[self.cam_id].frames.latest()
                if sample is None or sample.timestamp_ns <= self.last_timestamp:
                    self.stop_event.wait(0.002)
                    return
                self.last_timestamp = sample.timestamp_ns
                start = time.perf_counter_ns()
                rows = self.backend.detect(sample.obj, self.stop_event)
                self._publish(sample, rows, time.perf_counter_ns()-start)
            now = time.monotonic()
            if now-self.last_report >= 2:
                self.logger.info(
                    "perf source=%.1f YOLO=%.1f MOT=%.1f FPS | decode=%.1f ms "
                    "submit=%.1f ms batch=%.1f ms MOT=%.1f ms | real/batch=%.2f "
                    "padding/s=%.1f pending=%d MOT_skipped=%d source_skipped/s=%.1f",
                    self.stats.rate("source"), self.stats.rate("yolo"), self.stats.rate("mot"),
                    self.stats.mean("decode_ms"), self.stats.mean("submit_ms"),
                    self.stats.mean("batch_ms"), self.stats.mean("mot_ms"),
                    self.stats.mean("real_batch"), self.stats.rate("padding"),
                    len(self.pending_frames), self.mot_dropped, self.stats.rate("source_skipped"))
                if self.async_full_frame:
                    self.logger.info(
                        "YOLO stages per batch: prepare=%.1f ms ready_wait=%.1f ms "
                        "submit_to_callback=%.1f ms collect=%.1f ms | "
                        "input_age=%.1f ms backlog_trim_events/s=%.1f",
                        self.stats.mean("yolo_prepare_ms"), self.stats.mean("yolo_ready_ms"),
                        self.stats.mean("yolo_callback_ms"), self.stats.mean("yolo_collect_ms"),
                        self.stats.mean("yolo_input_age_ms"), self.stats.rate("backlog_trim"))
                self.last_report = now
        except BaseException:
            cancelled = not self.is_active and self.stop_event.is_set() and self.errors.empty()
            self.is_active = False
            self.on_stop()
            if not cancelled:
                raise

    def _publish(self, sample, rows, duration_ns):
        if self.stop_event.is_set():
            return
        detections = [Detection(BoxTwoCorners(*map(float, r[:4])), int(r[5]), float(r[4]))
                      for r in rows]
        self.state.detections[self.cam_id].enqueue(detections, sample.timestamp_ns)
        self.state.det_durations[self.cam_id].enqueue(duration_ns, sample.timestamp_ns)
        self.det_count += 1
        self.stats.tick("yolo")
        cap = self.settings.max_reid_objects
        if cap and len(rows) > cap:
            rows = rows[np.argsort(-rows[:, 4])[:cap]]
        packet = Packet(self.index, sample.timestamp_ns, sample.obj,
                        np.array(rows, dtype=np.float32, copy=True), time.monotonic())
        self.index += 1
        try:
            self.queue.put_nowait(packet)
        except Full:
            try:
                self.queue.get_nowait()
                self.mot_dropped += 1
            except Empty:
                pass
            self.queue.put_nowait(packet)

    def _full_frame_update(self):
        # Harvest in submission order, even if device callbacks arrive out of order.
        while self.pending_frames:
            job, samples, submitted_ns = self.pending_frames[0]
            age_ns = time.perf_counter_ns()-submitted_ns
            if not job.done.is_set():
                if age_ns > self.settings.timeout_ms*1_000_000:
                    raise TimeoutError('Hailo YOLO callback timeout')
                break
            self.pending_frames.popleft()
            collect_start = time.perf_counter_ns()
            rows_per_frame = self.backend.collect_frames(job)
            self.stats.observe("yolo_collect_ms", (time.perf_counter_ns()-collect_start)/1e6)
            for name, value in getattr(job, "timings", {}).items():
                self.stats.observe("yolo_"+name, value)
            self.stats.observe("batch_ms", (time.perf_counter_ns()-submitted_ns)/1e6)
            # Existing UI uses inverse mean service time; completed FPS is logged separately.
            duration_ns = (time.perf_counter_ns()-submitted_ns)//max(1, len(samples))
            for sample, rows in zip(samples, rows_per_frame):
                self._publish(sample, rows, duration_ns)
        if not self.backend.can_submit_frames():
            self.stop_event.wait(.001)
            return
        ring = self.state.cameras[self.cam_id].frames
        latest = ring.latest()
        if latest is not None and latest.timestamp_ns > self.last_timestamp:
            # Startup/re-enable begins at the current frame. Subsequently consume
            # retained frames in timestamp order, including arrivals during run_async.
            cutoff = latest.timestamp_ns - int(self.input_max_age_ms*1_000_000)
            stale = len(self.staged_frames)
            self.staged_frames = [s for s in self.staged_frames if s.timestamp_ns >= cutoff]
            if stale != len(self.staged_frames):
                self.stats.tick("backlog_trim")
            if self.last_timestamp < 0:
                first_ts = latest.timestamp_ns
            else:
                first_ts = max(self.last_timestamp+1, cutoff)
                if first_ts > self.last_timestamp+1:
                    self.stats.tick("backlog_trim")
            remaining = self.settings.full_frame_batch-len(self.staged_frames)
            if hasattr(ring, "get_next_by_timestamp"):
                incoming = ring.get_next_by_timestamp(first_ts, count=remaining)
            else:
                incoming = [latest]  # Compatibility with alternate source buffers.
            for sample in incoming:
                if sample.timestamp_ns <= self.last_timestamp:
                    continue
                if not self.staged_frames:
                    self.staged_at = time.monotonic()
                self.last_timestamp = sample.timestamp_ns
                self.staged_frames.append(sample)
                self.stats.observe("yolo_input_age_ms",
                                   max(0, latest.timestamp_ns-sample.timestamp_ns)/1e6)
        expired = (time.monotonic()-self.staged_at)*1000 >= self.settings.full_frame_max_wait_ms
        if self.staged_frames and (len(self.staged_frames) == self.settings.full_frame_batch or expired):
            samples = self.staged_frames
            self.staged_frames = []
            submitted_ns = time.perf_counter_ns()
            job = self.backend.submit_frames([s.obj for s in samples], self.stop_event)
            self.stats.observe("submit_ms", (time.perf_counter_ns()-submitted_ns)/1e6)
            self.stats.observe("real_batch", len(samples))
            self.stats.tick("padding", self.settings.full_frame_batch-len(samples))
            self.pending_frames.append((job, samples, submitted_ns))
        else:
            self.stop_event.wait(.001)

    def request_stop(self):
        self.is_active = False
        self.stop_event.set()

    def on_stop(self):
        self.request_stop()
        # Model waits have a configured timeout. Do not release any device buffers
        # until their owning MOT thread has returned, including on timeout/error.
        try:
            if self.worker is not None:
                if self.worker.ident is not None:
                    self.worker.join()
                self.worker = None
            if self.backend is not None:
                self.backend.close()
                self.backend = None
        finally:
            self.done.set()
