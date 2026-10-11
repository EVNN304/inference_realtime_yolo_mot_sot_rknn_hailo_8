import numpy as np
import torch
import torchvision.ops as ops
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from src.framework.core.box import BoxTwoCorners
from src.framework.core.slice import ImageSlice
from src.framework.det.detection import IDetection, Detection


class YoloSAHI:
    """
    SAHI-style batch inference with pre-bound YOLO kwargs and NMS.

    - `model` is a ready ultralytics.YOLO instance.
    - YOLO inference kwargs are provided once in ctor and bound via `functools.partial`.
    - NMS is configured once in ctor and bound via `functools.partial`.
    - `infer()` only takes slices (no kwargs).

    Output:
        list[ScoresDetection]
    """

    NMS_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "classic": {"iou_threshold": 0.6},
        "soft": {"iou_thr": 0.5, "sigma": 0.5, "score_threshold": 0.1},
        "wbf": {"iou_thr": 0.5, "skip_box_thr": 0.0001},
        "diou": {"iou_threshold": 0.5, "beta": 1.0},
        "adaptive": {"iou_thr_min": 0.3, "iou_thr_max": 0.7},
        "cluster": {"iou_threshold": 0.5},
        "nmm": {"iou_threshold": 0.15, "merge_method": "weighted"},
        "greedynmm": {"iou_threshold": 0.05, "score_threshold": 0.01},
    }

    def __init__(
        self,
        model: "YOLO",
        *,
        nms: str = "classic",
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = model

        # self.class_names = dict(getattr(model, "names", {}) or {})
        # self._num_classes = len(self.class_names) if self.class_names else None

        if runtime_args is None:
            runtime_args = {}

        self._yolo_call = partial(self._model, **runtime_args)

        nms_impls: Dict[str, Callable[..., List[List[float]]]] = {
            "classic": self._nms_classic,
            "soft": self._nms_soft,
            "wbf": self._nms_wbf,
            "diou": self._nms_diou,
            "nmm": self._nms_nmm,
            "greedynmm": self._nms_greedynmm,
            "adaptive": self._nms_adaptive,  # stub passthrough
            "cluster": self._nms_cluster,  # stub passthrough
        }

        if nms not in self.NMS_DEFAULTS:
            raise ValueError(f"Unknown NMS type: {nms}")
        if nms not in nms_impls:
            raise ValueError(f"NMS type has defaults but no implementation: {nms}")

        params = dict(self.NMS_DEFAULTS[nms])

        self._apply_nms = partial(nms_impls[nms], **params)

    def infer(self, slices: List[ImageSlice]) -> List[IDetection]:
        if not slices:
            return []

        batch_imgs = [s.image for s in slices]
        offsets = [(s.coords.x_offset, s.coords.y_offset) for s in slices]
        results = self._yolo_call(batch_imgs)

        all_data = []
        for i, r in enumerate(results):
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy()

            x_offset, y_offset = offsets[i]

            boxes_glob = boxes.copy()
            boxes_glob[:, [0, 2]] += float(x_offset)
            boxes_glob[:, [1, 3]] += float(y_offset)

            data = np.column_stack([clss, confs, boxes_glob])
            all_data.append(data)
        

        if not all_data:
            return []

        all_boxes_glob = np.vstack(all_data).tolist()
        #all_boxes_glob = self._apply_nms(all_boxes_glob)
        return self._to_detections(all_boxes_glob)

    @staticmethod
    def _nms_classic(all_boxes_glob, iou_threshold: float = 0.5):
        if not all_boxes_glob:
            return []

        boxes = torch.tensor(
            [[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob],
            dtype=torch.float32,
        )
        scores = torch.tensor(
            [conf for _, conf, _, _, _, _ in all_boxes_glob], dtype=torch.float32
        )

        keep = ops.nms(boxes, scores, float(iou_threshold)).tolist()
        return [all_boxes_glob[i] for i in keep]

    @staticmethod
    def _nms_soft(
        all_boxes_glob,
        iou_thr: float = 0.5,
        sigma: float = 0.5,
        score_threshold: float = 0.1,
    ):
        if not all_boxes_glob:
            return []

        boxes = np.array(
            [[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob],
            dtype=np.float32,
        )
        scores = np.array(
            [conf for _, conf, _, _, _, _ in all_boxes_glob], dtype=np.float32
        )

        keep = []
        while scores.size > 0:
            idx_max = int(np.argmax(scores))
            keep.append(idx_max)

            xx1 = np.maximum(boxes[idx_max][0], boxes[:, 0])
            yy1 = np.maximum(boxes[idx_max][1], boxes[:, 1])
            xx2 = np.minimum(boxes[idx_max][2], boxes[:, 2])
            yy2 = np.minimum(boxes[idx_max][3], boxes[:, 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_current = (boxes[idx_max][2] - boxes[idx_max][0]) * (
                boxes[idx_max][3] - boxes[idx_max][1]
            )
            area_rest = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            union = area_current + area_rest - inter

            iou = inter / (union + 1e-9)
            scores = scores * np.exp(-(iou * iou) / float(sigma))

            mask = scores > float(score_threshold)
            boxes = boxes[mask]
            scores = scores[mask]

        return [all_boxes_glob[i] for i in keep]

    @staticmethod
    def _nms_diou(all_boxes_glob, iou_threshold: float = 0.5, beta: float = 1.0):
        if not all_boxes_glob:
            return []

        boxes = np.array(
            [[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob],
            dtype=np.float32,
        )
        scores = np.array(
            [conf for _, conf, _, _, _, _ in all_boxes_glob], dtype=np.float32
        )

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []
        while scores.size > 0:
            idx_max = int(np.argmax(scores))
            keep.append(idx_max)

            xx1 = np.maximum(x1[idx_max], x1)
            yy1 = np.maximum(y1[idx_max], y1)
            xx2 = np.minimum(x2[idx_max], x2)
            yy2 = np.minimum(y2[idx_max], y2)

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            union = areas[idx_max] + areas - inter
            iou = inter / (union + 1e-9)

            center_x1 = (x1[idx_max] + x2[idx_max]) * 0.5
            center_y1 = (y1[idx_max] + y2[idx_max]) * 0.5
            center_x2 = (x1 + x2) * 0.5
            center_y2 = (y1 + y2) * 0.5
            center_dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

            c_x1 = np.minimum(x1[idx_max], x1)
            c_y1 = np.minimum(y1[idx_max], y1)
            c_x2 = np.maximum(x2[idx_max], x2)
            c_y2 = np.maximum(y2[idx_max], y2)
            diag_dist = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

            diou = iou - (center_dist / (diag_dist + 1e-9)) ** float(beta)

            mask = diou <= float(iou_threshold)
            boxes = boxes[mask]
            scores = scores[mask]
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

        return [all_boxes_glob[i] for i in keep]

    @staticmethod
    def _nms_wbf(all_boxes_glob, iou_thr: float = 0.5, skip_box_thr: float = 0.0001):
        if not all_boxes_glob:
            return []

        cls_groups = {}
        for box in all_boxes_glob:
            cls = box[0]
            if cls not in cls_groups:
                cls_groups[cls] = []
            cls_groups[cls].append(box)

        result = []
        for _, boxes_list in cls_groups.items():
            boxes_arr = np.array(boxes_list, dtype=np.float32)
            scores = boxes_arr[:, 1]

            mask = scores >= float(skip_box_thr)
            boxes_arr = boxes_arr[mask]
            if boxes_arr.size == 0:
                continue

            bboxes = boxes_arr[:, 2:6]
            scores = boxes_arr[:, 1]

            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            keep = []
            while scores.size > 0:
                idx_max = int(np.argmax(scores))
                keep.append(idx_max)

                xx1 = np.maximum(x1[idx_max], x1)
                yy1 = np.maximum(y1[idx_max], y1)
                xx2 = np.minimum(x2[idx_max], x2)
                yy2 = np.minimum(y2[idx_max], y2)

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                union = areas[idx_max] + areas - inter
                iou = inter / (union + 1e-9)

                mask = iou <= float(iou_thr)
                x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
                scores = scores[mask]
                areas = areas[mask]

            result.extend([boxes_list[i] for i in keep])

        return result

    @staticmethod
    def _nms_nmm(
        all_boxes_glob,
        iou_threshold: float = 0.15,
        merge_method: str = "weighted",
    ):
        if not all_boxes_glob:
            return []

        boxes = np.array(
            [[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob],
            dtype=np.float32,
        )
        scores = np.array(
            [conf for _, conf, _, _, _, _ in all_boxes_glob], dtype=np.float32
        )
        classes = np.array(
            [cls for cls, _, _, _, _, _ in all_boxes_glob], dtype=np.float32
        )

        keep = []
        merged = np.zeros(scores.size, dtype=bool)

        for i in range(scores.size):
            if merged[i]:
                continue

            idxs = np.where(classes == classes[i])[0]
            idxs = idxs[~merged[idxs]]

            if idxs.size == 1:
                keep.append(i)
                merged[i] = True
                continue

            current_boxes = boxes[idxs]
            current_scores = scores[idxs]

            x1, y1, x2, y2 = (
                current_boxes[:, 0],
                current_boxes[:, 1],
                current_boxes[:, 2],
                current_boxes[:, 3],
            )
            areas = (x2 - x1) * (y2 - y1)

            xx1 = np.maximum(x1[:, None], x1[None, :])
            yy1 = np.maximum(y1[:, None], y1[None, :])
            xx2 = np.minimum(x2[:, None], x2[None, :])
            yy2 = np.minimum(y2[:, None], y2[None, :])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            union = areas[:, None] + areas[None, :] - inter
            iou = inter / (union + 1e-9)

            to_merge = iou >= float(iou_threshold)
            merged_local = np.zeros(idxs.size, dtype=bool)

            merged_idx = int(idxs[0])
            for j in range(idxs.size):
                if merged_local[j]:
                    continue

                merged_local[to_merge[j]] = True
                sub_boxes = current_boxes[to_merge[j]]
                sub_scores = current_scores[to_merge[j]]

                if merge_method == "weighted":
                    x1_avg = np.average(sub_boxes[:, 0], weights=sub_scores)
                    y1_avg = np.average(sub_boxes[:, 1], weights=sub_scores)
                    x2_avg = np.average(sub_boxes[:, 2], weights=sub_scores)
                    y2_avg = np.average(sub_boxes[:, 3], weights=sub_scores)
                else:
                    x1_avg = float(np.mean(sub_boxes[:, 0]))
                    y1_avg = float(np.mean(sub_boxes[:, 1]))
                    x2_avg = float(np.mean(sub_boxes[:, 2]))
                    y2_avg = float(np.mean(sub_boxes[:, 3]))

                merged_idx = int(idxs[j])
                all_boxes_glob[merged_idx][2] = float(x1_avg)
                all_boxes_glob[merged_idx][3] = float(y1_avg)
                all_boxes_glob[merged_idx][4] = float(x2_avg)
                all_boxes_glob[merged_idx][5] = float(y2_avg)

            merged[idxs[merged_local]] = True
            keep.append(merged_idx)

        return [all_boxes_glob[i] for i in keep]

    @staticmethod
    def _nms_greedynmm(
        all_boxes_glob, iou_threshold: float = 0.05, score_threshold: float = 0.01
    ):
        if not all_boxes_glob:
            return []

        boxes = np.array(
            [[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in all_boxes_glob],
            dtype=np.float32,
        )
        scores = np.array(
            [conf for _, conf, _, _, _, _ in all_boxes_glob], dtype=np.float32
        )
        classes = np.array(
            [cls for cls, _, _, _, _, _ in all_boxes_glob], dtype=np.float32
        )

        keep = []
        merged = np.zeros(scores.size, dtype=bool)

        for i in range(scores.size):
            if merged[i]:
                continue
            if scores[i] < float(score_threshold):
                continue

            idxs = np.where(classes == classes[i])[0]
            idxs = idxs[scores[idxs] >= float(score_threshold)]
            idxs = idxs[~merged[idxs]]

            if idxs.size == 1:
                keep.append(i)
                merged[i] = True
                continue

            current_boxes = boxes[idxs]
            current_scores = scores[idxs]

            x1, y1, x2, y2 = (
                current_boxes[:, 0],
                current_boxes[:, 1],
                current_boxes[:, 2],
                current_boxes[:, 3],
            )
            areas = (x2 - x1) * (y2 - y1)

            xx1 = np.maximum(x1[:, None], x1[None, :])
            yy1 = np.maximum(y1[:, None], y1[None, :])
            xx2 = np.minimum(x2[:, None], x2[None, :])
            yy2 = np.minimum(y2[:, None], y2[None, :])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            union = areas[:, None] + areas[None, :] - inter
            iou = inter / (union + 1e-9)

            to_merge = iou >= float(iou_threshold)
            merged_local = np.zeros(idxs.size, dtype=bool)

            merged_idx = int(idxs[0])
            for j in range(idxs.size):
                if merged_local[j]:
                    continue

                merged_local[to_merge[j]] = True
                sub_boxes = current_boxes[to_merge[j]]
                sub_scores = current_scores[to_merge[j]]

                x1_avg = np.average(sub_boxes[:, 0], weights=sub_scores)
                y1_avg = np.average(sub_boxes[:, 1], weights=sub_scores)
                x2_avg = np.average(sub_boxes[:, 2], weights=sub_scores)
                y2_avg = np.average(sub_boxes[:, 3], weights=sub_scores)

                merged_idx = int(idxs[j])
                all_boxes_glob[merged_idx][2] = float(x1_avg)
                all_boxes_glob[merged_idx][3] = float(y1_avg)
                all_boxes_glob[merged_idx][4] = float(x2_avg)
                all_boxes_glob[merged_idx][5] = float(y2_avg)

            merged[idxs[merged_local]] = True
            keep.append(merged_idx)

        return [all_boxes_glob[i] for i in keep]

    @staticmethod
    def _nms_adaptive(
        all_boxes_glob, iou_thr_min: float = 0.3, iou_thr_max: float = 0.7
    ):
        return all_boxes_glob

    @staticmethod
    def _nms_cluster(all_boxes_glob, iou_threshold: float = 0.5):
        return all_boxes_glob

    @staticmethod
    def _to_detections(detections_xyxy):
        out = []
        for cls_f, conf_f, x1, y1, x2, y2 in detections_xyxy:
            out.append(
                Detection(
                    box=BoxTwoCorners(x1, y1, x2, y2),
                    cls=int(cls_f),
                    conf=float(conf_f)
                    # t_ns=timestamp_ns,
                )
            )

        return out
