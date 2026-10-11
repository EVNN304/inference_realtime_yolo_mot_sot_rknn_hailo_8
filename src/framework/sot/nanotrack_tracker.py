import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, List

from src.framework.core.box import Box, BoxTopLeft
from src.framework.sot.single_object_tracker import ISingleObjectTracker
from src.framework.sot.single_track import SingleTrack



class HybridNanoTrackTracker:
    def __init__(
        self,
        rknn_backbone_path: Optional[str] = None,
        onnx_head_path: Optional[str] = None,
        score_thresh: float = 0.35,
        core_mask: int = 1,
        num_threads: int = 4
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if rknn_backbone_path is None or not os.path.exists(rknn_backbone_path):
            candidates = [
                rknn_backbone_path,
                os.path.join(base_dir, "models", "nanotrackv3_backbone_dyn.rknn"),
                os.path.join(base_dir, "models", "nanotrackv3_backbone_dyn_sim.onnx"),
                "models/nanotrackv3_backbone_dyn.rknn",
                "models/nanotrackv3_backbone_dyn_sim.onnx"
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    rknn_backbone_path = c
                    break

        if onnx_head_path is None or not os.path.exists(onnx_head_path):
            candidates = [
                onnx_head_path,
                os.path.join(base_dir, "models", "nanotrackv3_head_fp16.onnx"),
                os.path.join(base_dir, "models", "nanotrackv3_head_sim.onnx"),
                "models/nanotrackv3_head_fp16.onnx",
                "models/nanotrackv3_head_sim.onnx"
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    onnx_head_path = c
                    break

        self.rknn_backbone_path = rknn_backbone_path
        self.onnx_head_path = onnx_head_path
        self.score_thresh = score_thresh
        self.core_mask = core_mask

        self._backbone_backend = "unknown"
        self._rknn_backbone = None
        self._onnx_backbone_session = None

        self._init_backbone()
        self._init_head(num_threads)

        self.exemplar_size = 127
        self.instance_size = 255
        self.score_size = 15
        self.stride = 16
        self.context_amount = 0.5
        self.window_influence = 0.455
        self.penalty_k = 0.138
        self.lr = 0.348

        hanning = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning).flatten()

        ori = - (self.score_size // 2) * self.stride
        x, y = np.meshgrid(
            [ori + self.stride * dx for dx in range(self.score_size)],
            [ori + self.stride * dy for dy in range(self.score_size)]
        )
        self.points = np.zeros((self.score_size * self.score_size, 2), dtype=np.float32)
        self.points[:, 0] = x.astype(np.float32).flatten()
        self.points[:, 1] = y.astype(np.float32).flatten()

        self.center_pos: Optional[np.ndarray] = None
        self.size: Optional[np.ndarray] = None
        self.channel_average: Optional[np.ndarray] = None
        self.zf: Optional[np.ndarray] = None
        self.is_initialized: bool = False
        self.last_score: float = 0.0

    def _init_backbone(self):
        if self.rknn_backbone_path and self.rknn_backbone_path.endswith(".rknn"):
            try:
                try:
                    from rknnlite.api import RKNNLite
                    is_lite = True
                except ImportError:
                    from rknn.api import RKNN as RKNNLite
                    is_lite = False

                rknn = RKNNLite()
                ret = rknn.load_rknn(self.rknn_backbone_path)
                if ret == 0:
                    ret_rt = rknn.init_runtime(core_mask=self.core_mask) if is_lite else rknn.init_runtime()
                    if ret_rt == 0:
                        self._rknn_backbone = rknn
                        self._backbone_backend = "rknn_npu"
                        print(f"[NanoTrack] ✅ Backbone loaded on NPU (core {self.core_mask})")
                        return
            except Exception as e:
                print(f"[NanoTrack] ⚠️ RKNN init failed: {e}. Falling back to ONNX.")

        onnx_dyn_path = self.rknn_backbone_path if (self.rknn_backbone_path and self.rknn_backbone_path.endswith(".onnx")) else "models/nanotrackv3_backbone_dyn_sim.onnx"
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.exists(onnx_dyn_path):
            onnx_dyn_path = os.path.join(base_dir, "models", "nanotrackv3_backbone_dyn_sim.onnx")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._onnx_backbone_session = ort.InferenceSession(onnx_dyn_path, opts, providers=['CPUExecutionProvider'])
        self._onnx_backbone_in = self._onnx_backbone_session.get_inputs()[0].name
        self._onnx_backbone_out = self._onnx_backbone_session.get_outputs()[0].name
        self._backbone_backend = "onnx_cpu_fallback"
        print(f"[NanoTrack] ⚠️ Using ONNX fallback: {onnx_dyn_path}")

    def _init_head(self, num_threads: int):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.head_session = ort.InferenceSession(self.onnx_head_path, opts, providers=['CPUExecutionProvider'])
        inputs = self.head_session.get_inputs()
        self.head_in_zf = inputs[0].name
        self.head_in_xf = inputs[1].name
        self.head_dtype = inputs[0].type
        print(f"[NanoTrack] ✅ Head model loaded: {self.onnx_head_path}")

    def _forward_backbone(self, img_patch: np.ndarray) -> np.ndarray:
        if self._backbone_backend == "rknn_npu" and self._rknn_backbone is not None:

            img_patch_rknn = img_patch.transpose(0, 2, 3, 1)
            out = self._rknn_backbone.inference(inputs=[img_patch_rknn])
            feat = out[0].astype(np.float32)
            # Ensure output feature map is in NCHW format: (1, 96, H_out, W_out)
            if feat.ndim == 4 and feat.shape[-1] == 96:
                feat = feat.transpose(0, 3, 1, 2)
            return feat
        else:
            out = self._onnx_backbone_session.run(
                [self._onnx_backbone_out],
                {self._onnx_backbone_in: img_patch.astype(np.float32)}
            )
            return out[0].astype(np.float32)

    def _forward_head(self, zf: np.ndarray, xf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if "float16" in self.head_dtype:
            zf_in = zf.astype(np.float16)
            xf_in = xf.astype(np.float16)
        else:
            zf_in = zf.astype(np.float32)
            xf_in = xf.astype(np.float32)

        outs = self.head_session.run(
            None,
            {self.head_in_zf: zf_in, self.head_in_xf: xf_in}
        )
        return outs[0].astype(np.float32), outs[1].astype(np.float32)

    def _get_subwindow(
        self,
        im: np.ndarray,
        pos: np.ndarray,
        model_sz: int,
        original_sz: float,
        avg_chans: np.ndarray
    ) -> np.ndarray:
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1.0) / 2.0

        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1

        left_pad = int(max(0.0, -context_xmin))
        top_pad = int(max(0.0, -context_ymin))
        right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

        context_xmin = int(context_xmin + left_pad)
        context_xmax = int(context_xmax + left_pad)
        context_ymin = int(context_ymin + top_pad)
        context_ymax = int(context_ymax + top_pad)

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (im_sz[0] + top_pad + bottom_pad, im_sz[1] + left_pad + right_pad, 3)
            te_im = np.zeros(size, dtype=np.uint8)
            te_im[top_pad:top_pad + im_sz[0], left_pad:left_pad + im_sz[1], :] = im

            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + im_sz[1], :] = avg_chans
            if bottom_pad:
                te_im[im_sz[0] + top_pad:, left_pad:left_pad + im_sz[1], :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, im_sz[1] + left_pad:, :] = avg_chans

            im_patch = te_im[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
        else:
            im_patch = im[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]

        if im_patch.shape[0] != model_sz or im_patch.shape[1] != model_sz:
            im_patch = cv2.resize(im_patch, (model_sz, model_sz), interpolation=cv2.INTER_LINEAR)

        im_patch = im_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        return im_patch

    def init(self, frame: np.ndarray, bbox: np.ndarray) -> bool:
        x1, y1, x2, y2 = bbox
        w = max(2.0, float(x2 - x1))
        h = max(2.0, float(y2 - y1))
        cx = float(x1) + (w - 1.0) * 0.5
        cy = float(y1) + (h - 1.0) * 0.5

        self.center_pos = np.array([cx, cy], dtype=np.float32)
        self.size = np.array([w, h], dtype=np.float32)

        w_z = self.size[0] + self.context_amount * np.sum(self.size)
        h_z = self.size[1] + self.context_amount * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        self.channel_average = np.mean(frame, axis=(0, 1)).astype(np.uint8)
        z_crop = self._get_subwindow(frame, self.center_pos, self.exemplar_size, s_z, self.channel_average)

        self.zf = self._forward_backbone(z_crop)
        self.is_initialized = True
        self.last_score = 1.0
        return True

    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.is_initialized or self.zf is None:
            return np.empty(0, dtype=np.float32), 0.0

        w_z = self.size[0] + self.context_amount * np.sum(self.size)
        h_z = self.size[1] + self.context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.exemplar_size / s_z
        s_x = s_z * (self.instance_size / self.exemplar_size)

        x_crop = self._get_subwindow(frame, self.center_pos, self.instance_size, round(s_x), self.channel_average)
        xf = self._forward_backbone(x_crop)
        cls_logits, loc_pred = self._forward_head(self.zf, xf)

        cls_flat = cls_logits[0].reshape(2, -1).T
        exp_cls = np.exp(cls_flat - np.max(cls_flat, axis=1, keepdims=True))
        softmax_cls = exp_cls / np.sum(exp_cls, axis=1, keepdims=True)
        score = softmax_cls[:, 1]

        loc_flat = loc_pred[0].reshape(4, -1)
        x1 = self.points[:, 0] - loc_flat[0, :]
        y1 = self.points[:, 1] - loc_flat[1, :]
        x2 = self.points[:, 0] + loc_flat[2, :]
        y2 = self.points[:, 1] + loc_flat[3, :]

        pred_cx = (x1 + x2) * 0.5
        pred_cy = (y1 + y2) * 0.5
        pred_w = x2 - x1
        pred_h = y2 - y1

        def change(r):
            return np.maximum(r, 1.0 / np.maximum(r, 1e-6))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt(np.maximum(1e-6, (w + pad) * (h + pad)))

        s_c = change(sz(pred_w, pred_h) / sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / max(1e-6, self.size[1])) / (pred_w / np.maximum(1e-6, pred_h)))
        penalty = np.exp(-(r_c * s_c - 1.0) * self.penalty_k)

        pscore = penalty * score
        pscore = pscore * (1.0 - self.window_influence) + self.window * self.window_influence
        best_idx = np.argmax(pscore)

        target_score = float(score[best_idx])
        self.last_score = target_score

        best_cx = pred_cx[best_idx] / scale_z
        best_cy = pred_cy[best_idx] / scale_z
        best_w = pred_w[best_idx] / scale_z
        best_h = pred_h[best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * self.lr

        res_cx = self.center_pos[0] + best_cx
        res_cy = self.center_pos[1] + best_cy
        res_w = self.size[0] * (1.0 - lr) + best_w * lr
        res_h = self.size[1] * (1.0 - lr) + best_h * lr

        h_img, w_img = frame.shape[:2]
        res_cx = np.clip(res_cx, 0, w_img - 1)
        res_cy = np.clip(res_cy, 0, h_img - 1)
        res_w = np.clip(res_w, 4, w_img)
        res_h = np.clip(res_h, 4, h_img)

        self.center_pos = np.array([res_cx, res_cy], dtype=np.float32)
        self.size = np.array([res_w, res_h], dtype=np.float32)

        out_x1 = max(0.0, res_cx - res_w * 0.5)
        out_y1 = max(0.0, res_cy - res_h * 0.5)
        out_x2 = min(float(w_img), res_cx + res_w * 0.5)
        out_y2 = min(float(h_img), res_cy + res_h * 0.5)
        out_bbox = np.array([out_x1, out_y1, out_x2, out_y2], dtype=np.float32)

        return out_bbox, target_score

    def reset(self):
        self.center_pos = None
        self.size = None
        self.channel_average = None
        self.zf = None
        self.is_initialized = False
        self.last_score = 0.0



class NanoTrackTracker(ISingleObjectTracker):
    def __init__(
        self,
        rknn_backbone_path: str,
        onnx_head_path: str,
        core_mask: int = 1,
        **kwargs 
    ):

        self.model = HybridNanoTrackTracker(
            rknn_backbone_path=rknn_backbone_path,
            onnx_head_path=onnx_head_path,
            core_mask=core_mask
        )

    def initialize(self, frame: np.ndarray, init_box: Box) -> bool:
        try:

            x1, y1, w, h = init_box.as_top_left().as_tuple()
            bbox = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
            return self.model.init(frame, bbox)
        except Exception as e:
            print(f"[NanoTrack] Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def track(self, frame: np.ndarray, search_area: Optional[Box] = None) -> SingleTrack:
        try:
            bbox, score = self.model.update(frame)
            if len(bbox) == 0 or score < 0.1:
                return SingleTrack(box=BoxTopLeft(0, 0, 10, 10), conf=0.0)
            
            x1, y1, x2, y2 = bbox
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            return SingleTrack(box=BoxTopLeft(x1, y1, w, h), conf=float(score))
        except Exception as e:
            print(f"[NanoTrack] Track failed: {e}")
            import traceback
            traceback.print_exc()
            return SingleTrack(box=BoxTopLeft(0, 0, 10, 10), conf=0.0)


def get_tracker_class():
    return NanoTrackTracker
