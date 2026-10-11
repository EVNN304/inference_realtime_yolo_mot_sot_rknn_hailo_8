"""Pooled InferModel YOLO with bounded, owned input/output buffers."""
from collections import deque
from dataclasses import dataclass, field
from threading import Event
import numpy as np
import time
from .geometry import decode, fill_full_frame, fill_tile, merge, tile_origins


@dataclass
class Slot:
    images: np.ndarray
    bindings: list


@dataclass
class Job:
    slot: Slot
    transforms: list
    handle: object = None
    error: object = None
    done: object = field(default_factory=Event)
    frames: list = field(default_factory=list)

    timings: dict = field(default_factory=dict)
    submitted_ns: int = 0

    def callback(self, completion_info):
        if self.submitted_ns:
            self.timings["callback_ms"] = (time.perf_counter_ns()-self.submitted_ns)/1e6
        self.error = completion_info.exception
        self.done.set()


class HailoYOLO:
    def __init__(self, device, stack, settings, use_sahi):
        from hailo_platform import FormatType
        from hailo_platform.pyhailort.pyhailort import FormatOrder
        self.sahi = bool(use_sahi)
        self.batch = settings.sahi_batch if self.sahi else settings.full_frame_batch
        self.timeout = settings.timeout_ms
        self.conf, self.iou, self.overlap = settings.confidence, settings.iou, settings.overlap
        if min(self.batch, settings.inflight, self.timeout) < 1:
            raise ValueError('YOLO batch/inflight/timeout must be positive')
        if not (0 <= self.conf < 1 and 0 < self.iou <= 1 and 0 <= self.overlap < 1):
            raise ValueError('Invalid confidence, IoU or overlap')
        self.model = device.create_infer_model(settings.yolo_hef)
        if len(self.model.inputs) != 1 or len(self.model.outputs) != 1:
            raise ValueError('Expected one RGB input and one NMS output')
        inp, out = self.model.input(), self.model.output()
        shape = tuple(inp.shape)
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f'Expected HWC RGB input, got {shape}')
        if not out.is_nms or out.format.order != FormatOrder.HAILO_NMS_BY_CLASS:
            raise ValueError('YOLO HEF must have HAILO_NMS_BY_CLASS output')
        self.height, self.width = shape[:2]
        inp.set_format_type(FormatType.UINT8)
        inp.set_format_order(FormatOrder.NHWC)
        out.set_format_type(FormatType.FLOAT32)
        self.model.set_batch_size(self.batch)
        self.configured = stack.enter_context(self.model.configure())
        if self.batch > self.configured.get_async_queue_size():
            raise ValueError('YOLO batch exceeds HailoRT async queue capacity')
        self.slots = []
        for _ in range(settings.inflight):
            images = np.empty((self.batch, *shape), np.uint8)
            bindings = []
            for image in images:
                binding = self.configured.create_bindings(
                    output_buffers={out.name: np.empty(out.shape, np.float32)})
                binding.input().set_buffer(image)
                bindings.append(binding)
            self.slots.append(Slot(images, bindings))
        self.free_frames = deque(self.slots)

    def shutdown(self):
        self.configured.shutdown()

    def can_submit_frames(self):
        return bool(self.free_frames)

    def submit_frames(self, frames, stop):
        if self.sahi or not 1 <= len(frames) <= self.batch:
            raise ValueError('Invalid full-frame batch')
        if stop.is_set():
            raise RuntimeError('YOLO cancelled')
        slot = self.free_frames.popleft()
        transforms = []
        prepare_start = time.perf_counter_ns()
        try:
            for j, frame in enumerate(frames):
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError('Expected uint8 BGR frame')
                transforms.append((0, 0, fill_full_frame(slot.images[j], frame)))
            if len(frames) < self.batch:
                slot.images[len(frames):] = 114
            prepare_ms = (time.perf_counter_ns()-prepare_start)/1e6
            ready_start = time.perf_counter_ns()
            self.configured.wait_for_async_ready(timeout_ms=self.timeout, frames_count=self.batch)
            job = Job(slot, transforms, frames=list(frames))
            job.timings["prepare_ms"] = prepare_ms
            job.timings["ready_ms"] = (time.perf_counter_ns()-ready_start)/1e6
            job.submitted_ns = time.perf_counter_ns()
            job.handle = self.configured.run_async(slot.bindings, job.callback)
            return job
        except BaseException:
            self.shutdown()
            raise

    def collect_frames(self, job):
        try:
            job.handle.wait(self.timeout)
            if job.error is not None:
                raise RuntimeError('YOLO inference failed') from job.error
            results = []
            for binding, frame, (_, _, transform) in zip(job.slot.bindings, job.frames, job.transforms):
                h, w = frame.shape[:2]
                chunks = decode(binding.output().get_buffer(), 0, 0, self.width, self.height,
                                w, h, self.conf, transform)
                results.append(merge(chunks, self.conf, self.iou))
            self.free_frames.append(job.slot)
            return results
        except BaseException:
            self.shutdown()
            raise

    def infer(self, image, stop):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Source frame must be uint8 BGR HWC')
        h, w = image.shape[:2]
        origins = tile_origins(w, h, self.width, self.height, self.overlap, self.sahi)
        free, pending = deque(self.slots), deque()
        chunks, next_tile = [], 0
        try:
            while next_tile < len(origins) or pending:
                if stop.is_set():
                    raise RuntimeError('YOLO cancelled')
                while free and next_tile < len(origins):
                    slot = free.popleft()
                    transforms = []
                    count = min(self.batch, len(origins)-next_tile)
                    for j, (x, y) in enumerate(origins[next_tile:next_tile+count]):
                        if self.sahi:
                            fill_tile(slot.images[j], image, x, y)
                            letterbox = None
                        else:
                            letterbox = fill_full_frame(slot.images[j], image)
                        transforms.append((x, y, letterbox))
                    if count < self.batch:
                        slot.images[count:] = 114
                    self.configured.wait_for_async_ready(
                        timeout_ms=self.timeout, frames_count=self.batch)
                    job = Job(slot, transforms)
                    job.handle = self.configured.run_async(slot.bindings, job.callback)
                    pending.append(job)
                    next_tile += count
                job = pending.popleft()
                job.handle.wait(self.timeout)
                if job.error is not None:
                    raise RuntimeError('YOLO inference failed') from job.error
                for binding, (x, y, letterbox) in zip(job.slot.bindings, job.transforms):
                    chunks.extend(decode(binding.output().get_buffer(), x, y,
                                         self.width, self.height, w, h, self.conf, letterbox))
                free.append(job.slot)  # decode owns its arrays before reuse.
        except BaseException:
            self.shutdown()
            raise
        return merge(chunks, self.conf, self.iou)
