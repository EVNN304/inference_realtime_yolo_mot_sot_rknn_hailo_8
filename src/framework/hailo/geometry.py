"""RGB preprocessing and class-wise NMS from validated standalone pipeline."""
import math
import threading
from functools import lru_cache
import numpy as np

def starts(length, tile, overlap):
    if not 0 <= overlap < 1 or min(length, tile) <= 0:
        raise ValueError('Invalid slicing geometry')
    if length <= tile:
        return (0,)
    count = math.ceil((length-tile)/(tile*(1-overlap)))
    return tuple(round(i*(length-tile)/count) for i in range(count+1))

@lru_cache(maxsize=32)
def tile_origins(frame_w, frame_h, input_w, input_h, overlap, use_sahi):
    if not use_sahi:
        return ((0, 0),)
    return tuple((x,y) for x in starts(frame_w,input_w,overlap)
                 for y in starts(frame_h,input_h,overlap))

def decode(per_class, x, y, tile_w, tile_h, frame_w, frame_h, threshold, letterbox=None):
    chunks = []
    for cls, detections in enumerate(per_class):
        det = np.asarray(detections, dtype=np.float32)
        if not det.size:
            continue
        if det.ndim != 2 or det.shape[1] != 5:
            raise ValueError(f'Expected NMS class array [N,5], got {det.shape}')
        det = det[np.isfinite(det).all(axis=1) & (det[:, 4] >= threshold)]
        if not len(det):
            continue
        # Hailo: ymin,xmin,ymax,xmax,score -> original-image xyxy,score,class.
        out = np.empty((len(det), 6), np.float32)
        out[:, :4] = det[:, [1, 0, 3, 2]]
        out[:, :4] *= [tile_w, tile_h, tile_w, tile_h]
        if letterbox is None:
            out[:, :4] += [x, y, x, y]
            out[:, [0, 2]] = np.clip(out[:, [0, 2]], x, min(x + tile_w, frame_w))
            out[:, [1, 3]] = np.clip(out[:, [1, 3]], y, min(y + tile_h, frame_h))
        else:
            scale_x, scale_y, left, top = letterbox
            out[:, :4] -= [left, top, left, top]
            out[:, :4] /= [scale_x, scale_y, scale_x, scale_y]
            out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, frame_w)
            out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, frame_h)
        out[:, 4] = det[:, 4]
        out[:, 5] = cls
        out = out[(out[:, 2] - out[:, 0] > 1) & (out[:, 3] - out[:, 1] > 1)]
        if len(out):
            chunks.append(out)
    return chunks

def merge(chunks, threshold, iou):
    import cv2
    if not chunks:
        return np.empty((0, 6), np.float32)
    boxes = np.concatenate(chunks)
    keep = []
    for cls in np.unique(boxes[:, 5]):
        ix = np.flatnonzero(boxes[:, 5] == cls)
        xywh = boxes[ix, :4].copy()
        xywh[:, 2:4] -= xywh[:, :2]
        selected = cv2.dnn.NMSBoxes(xywh.tolist(), boxes[ix, 4].tolist(), threshold, iou)
        keep.extend(ix[np.asarray(selected, dtype=int).reshape(-1)].tolist())
    kept = boxes[keep]
    return kept[np.argsort(-kept[:, 4])]

class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}

    def add(self, name, value=1):
        with self.lock:
            self.values[name] = self.values.get(name, 0) + value

    def snapshot(self):
        with self.lock:
            return dict(self.values)

def fill_tile(destination, frame, x, y):
    import cv2
    h, w = destination.shape[:2]
    crop = frame[y:y+h, x:x+w]
    ch, cw = crop.shape[:2]
    if (ch, cw) == (h, w):
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB, dst=destination)
    else:
        destination.fill(114)
        destination[:ch, :cw] = crop[:, :, ::-1]

def letterbox_geometry(frame_w, frame_h, input_w, input_h):
    scale = min(input_w/frame_w, input_h/frame_h)
    new_w = max(1, min(input_w, round(frame_w*scale)))
    new_h = max(1, min(input_h, round(frame_h*scale)))
    left, top = (input_w-new_w)//2, (input_h-new_h)//2
    return new_w, new_h, left, top

def fill_full_frame(destination, frame):
    import cv2
    ih, iw = destination.shape[:2]
    fh, fw = frame.shape[:2]
    nw, nh, left, top = letterbox_geometry(fw, fh, iw, ih)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    destination.fill(114)
    cv2.cvtColor(resized, cv2.COLOR_BGR2RGB,
                 dst=destination[top:top+nh, left:left+nw])
    # Use actual rounded dimensions for an accurate inverse transform.
    return nw/fw, nh/fh, left, top
