"""OSNet InferModel backend. Uses the SAME VDevice as YOLO; no hidden CPU model."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def l2_normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms < 1e-12):
        raise ValueError('Invalid/zero OSNet embedding; check HEF output and preprocessing')
    return np.ascontiguousarray(features / norms)


def crop_bounds(box, width, height):
    # Same integer truncation as the local BoxMOT crop backend.
    x1, y1, x2, y2 = np.asarray(box[:4]).astype(int)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid clipped ReID crop: {box}')
    return int(x1), int(y1), int(x2), int(y2)


@dataclass
class ReIDSlot:
    images: np.ndarray
    outputs: list
    bindings: list


@dataclass
class ReIDJob:
    slot: ReIDSlot
    start: int
    count: int
    error: object = None
    handle: object = None

    def callback(self, completion_info):
        # HailoRT calls this with the keyword completion_info, not just positionally.
        self.error = completion_info.exception


class HailoReID:
    def __init__(self, device, stack, hef_path, manifest_path, batch=1, inflight=2,
                 timeout_ms=10000):
        from hailo_platform import FormatType
        from hailo_platform.pyhailort.pyhailort import FormatOrder
        if min(batch, inflight, timeout_ms) < 1:
            raise ValueError('Invalid ReID batch/inflight/timeout')
        hef = Path(hef_path)
        contract = json.loads(Path(manifest_path).read_text())
        if contract.get('input_contract') != 'RGB_UINT8_0_255_NORMALIZATION_IN_HEF':
            raise ValueError('Unsupported ReID contract; use the supplied compiler/manifest')
        if hashlib.sha256(hef.read_bytes()).hexdigest() != contract.get('hef_sha256'):
            raise ValueError('ReID manifest does not match HEF SHA256')
        self.model = device.create_infer_model(str(hef))
        if len(self.model.inputs) != 1 or len(self.model.outputs) != 1:
            raise ValueError('OSNet requires exactly one input and one embedding output')
        inp, out = self.model.input(), self.model.output()
        shape = tuple(inp.shape)
        expected = (contract['height'], contract['width'], 3)
        self.dim = int(contract['embedding_dim'])
        if shape != expected or out.is_nms or int(np.prod(out.shape)) != self.dim:
            raise ValueError(f'OSNet shape mismatch: {shape}, {out.shape}; expected {expected}, {self.dim}')
        self.height, self.width = shape[:2]
        inp.set_format_type(FormatType.UINT8)
        inp.set_format_order(FormatOrder.NHWC)
        out.set_format_type(FormatType.FLOAT32)
        self.model.set_batch_size(batch)
        self.configured = stack.enter_context(self.model.configure())
        if batch > self.configured.get_async_queue_size():
            raise ValueError('ReID batch exceeds runtime queue size')
        self.batch, self.timeout_ms = batch, timeout_ms
        self.slots = []
        for _ in range(inflight):
            images = np.empty((batch, *shape), np.uint8)
            outputs, bindings = [], []
            for image in images:
                output = np.empty(out.shape, np.float32)
                binding = self.configured.create_bindings(output_buffers={out.name: output})
                binding.input().set_buffer(image)
                outputs.append(output)
                bindings.append(binding)
            self.slots.append(ReIDSlot(images, outputs, bindings))
        print(f'ReID: {hef.name}, input={shape}, embedding={self.dim}, batch={batch}, slots={inflight}')

    def shutdown(self):
        self.configured.shutdown()

    def get_features(self, boxes, image, metrics=None, stop=None):
        import cv2
        count = len(boxes)
        if not count:
            return np.empty((0, self.dim), np.float32)
        features = np.empty((count, self.dim), np.float32)
        free, pending = deque(self.slots), deque()
        next_index = 0
        started = time.perf_counter()
        try:
            while next_index < count or pending:
                if stop is not None and stop.is_set():
                    raise RuntimeError('ReID cancelled')
                while next_index < count and free:
                    slot = free.popleft()
                    used = min(self.batch, count-next_index)
                    t = time.perf_counter()
                    for j in range(used):
                        x1,y1,x2,y2 = crop_bounds(boxes[next_index+j], image.shape[1], image.shape[0])
                        resized = cv2.resize(image[y1:y2, x1:x2], (self.width, self.height),
                                             interpolation=cv2.INTER_LINEAR)
                        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=slot.images[j])
                    if used < self.batch:
                        slot.images[used:] = slot.images[used-1]  # Padding is never returned as a feature.
                    if metrics:
                        metrics.add('reid_prepare_s', time.perf_counter()-t)
                        metrics.add('reid_padding', self.batch-used)
                    self.configured.wait_for_async_ready(timeout_ms=self.timeout_ms, frames_count=self.batch)
                    job = ReIDJob(slot, next_index, used)
                    job.handle = self.configured.run_async(slot.bindings, job.callback)
                    pending.append(job)
                    next_index += used
                job = pending.popleft()
                job.handle.wait(self.timeout_ms)
                if job.error is not None:
                    raise RuntimeError('OSNet async inference failed') from job.error
                # Copy BEFORE recycling the slot.
                for j in range(job.count):
                    features[job.start+j] = job.slot.outputs[j].reshape(-1)
                free.append(job.slot)
        except BaseException:
            # No subsequent call may reuse buffers while a failed/timed-out job still runs.
            self.shutdown()
            raise
        if metrics:
            metrics.add('reid_s', time.perf_counter()-started)
            metrics.add('reid_crops', count)
        return l2_normalize(features)
