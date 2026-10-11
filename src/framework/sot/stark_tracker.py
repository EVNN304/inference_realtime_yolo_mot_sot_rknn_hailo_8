from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import yaml
from easydict import EasyDict

from src.framework.core.box import Box, BoxTopLeft
from src.framework.sot.single_object_tracker import ISingleObjectTracker
from src.framework.sot.single_track import SingleTrack


class StarkTracker(ISingleObjectTracker):
    class Config:
        def __init__(
            self,
            yaml_path: str | Path,
            model_path: str | Path,
            onnx_template_path: Optional[str] = None,
            onnx_complete_path: Optional[str] = None,
            save_all_boxes: bool = False,
        ) -> None:
            with open(yaml_path, "r") as f:
                self.cfg = EasyDict(yaml.safe_load(f))
            self.model_path = self.checkpoint = model_path
            self.template_factor = self.cfg.TEST.TEMPLATE_FACTOR
            self.template_size = self.cfg.TEST.TEMPLATE_SIZE
            self.search_factor = self.cfg.TEST.SEARCH_FACTOR
            self.search_size = self.cfg.TEST.SEARCH_SIZE
            self.onnx_template_path = onnx_template_path
            self.onnx_complete_path = onnx_complete_path
            self.save_all_boxes = save_all_boxes

    def __init__(self, model_name: str, config: Config) -> None:
        if model_name == "stark_s":
            from stark.test.tracker.stark_s import STARK_S

            self.model = STARK_S(config, "video")
        elif model_name == "stark_lightning_X_trt":
            from stark.test.tracker.stark_lightning_X_trt import STARK_LightningXtrt

            self.model = STARK_LightningXtrt(config, "video")
        elif model_name == "stark_lightning_X_trt_onnx":
            from stark.test.tracker.stark_lightning_X_trt import (
                STARK_LightningXtrt_onnx,
            )

            self.model = STARK_LightningXtrt_onnx(config, "video")
        else:
            raise NotImplementedError(f"Model {model_name} for stark is not supported")

    def initialize(self, frame: np.ndarray, init_box: Box) -> bool:
        try:
            self.model.initialize(frame, {"init_bbox": self._to_stark_box(init_box)})
            return True
        except Exception:
            return False

    def track(
        self, frame: np.ndarray, search_area: Optional[Box] = None
    ) -> SingleTrack:
        result = self.model.track(frame)
        return SingleTrack(box=self._to_framework_box(result["target_bbox"]), conf=1.0)

    @staticmethod
    def _to_stark_box(box: Box) -> List[float]:
        return list(box.as_top_left().as_tuple())

    @staticmethod
    def _to_framework_box(box: list[float]) -> Box:
        return BoxTopLeft(*box)
