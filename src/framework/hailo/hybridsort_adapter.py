"""Adapter for the user's local BoxMOT 16.0.8 HybridSort external-embedding API."""
import inspect
import time
import numpy as np


class HybridSORTAdapter:
    def __init__(self, reid, options, torch_threads=1):
        import torch
        import boxmot
        from boxmot import HybridSort
        print(f'BoxMOT version: {getattr(boxmot, "__version__", "unknown")}; source: {boxmot.__file__}')
        params = inspect.signature(HybridSort.update).parameters
        # BoxMOT 16.0.8 setup_decorator exposes only *args/**kwargs.
        accepts_keywords = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if 'embs' not in params and not accepts_keywords:
            raise RuntimeError('This adapter needs HybridSort.update(dets, img, embs=...). '
                               'Use the existing compatible BoxMOT, do not silently upgrade it.')
        torch.set_num_threads(torch_threads)
        self.tracker = HybridSort(reid_weights=None, device=torch.device('cpu'), half=False,
                                  with_reid=True, **options)
        if getattr(self.tracker, 'model', None) is not None:
            raise RuntimeError('Unexpected CPU ReID model; refusing duplicate inference')
        self.reid = reid
        self.last_index = -1

    def update(self, state, detections, metrics, stop=None):
        if state.index <= self.last_index:
            raise ValueError('HybridSORT updates must have strictly increasing frame indices')
        if self.last_index >= 0:
            metrics.add('mot_source_gap', max(0, state.index-self.last_index-1))
        self.last_index = state.index
        dets = np.ascontiguousarray(detections, dtype=np.float32).reshape(-1, 6)
        features = self.reid.get_features(dets[:, :4], state.image, metrics, stop)
        t = time.perf_counter()
        tracks = self.tracker.update(dets, state.image, embs=features)
        metrics.add('association_s', time.perf_counter()-t)
        tracks = np.asarray(tracks)
        if not tracks.size:
            return np.empty((0, 8), np.float64)
        if tracks.ndim != 2 or tracks.shape[1] != 8 or not np.isfinite(tracks).all():
            raise ValueError(f'Unexpected HybridSORT output: {tracks.shape}')
        # Do not retain a potentially mutable view of tracker state across threads.
        return np.array(tracks, dtype=np.float64, copy=True)
