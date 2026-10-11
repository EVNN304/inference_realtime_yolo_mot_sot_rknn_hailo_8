from pathlib import Path
from typing import Optional, Union

#import torch

from src.framework.mot.boxmot_wrapper import BoxMotWrapperBase


class HybridSortTracker(BoxMotWrapperBase):
    def __init__(
        self,
        *,
        reid_weights: Optional[Union[str, Path]] = None,
        device = 0,
        half: bool = True,
        cmc_method: str = "ecc",
        custom_reid_extractor=None,
        with_reid: bool = True,
        #
        low_thresh: float = 0.1,
        delta_t: int = 3,
        inertia: float = 0.05,
        use_byte: bool = True,
        #
        use_custom_kf: bool = True,
        longterm_bank_length: int = 30,
        alpha: float = 0.9,
        adapfs: bool = False,
        track_thresh: float = 0.5,
        #
        EG_weight_high_score: float = 4.6,
        EG_weight_low_score: float = 1.3,
        #
        TCM_first_step: bool = True,
        TCM_byte_step: bool = True,
        TCM_byte_step_weight: float = 1.0,
        high_score_matching_thresh: float = 0.7,
        #
        with_longterm_reid: bool = True,
        longterm_reid_weight: float = 0.0,
        with_longterm_reid_correction: bool = True,
        longterm_reid_correction_thresh: float = 0.4,
        longterm_reid_correction_thresh_low: float = 0.4,
        dataset: str = "",
        det_thresh: float = 0.5,
        max_age: int = 30,
        max_obs: int = 35,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        nr_classes: int = 80,
        asso_func: str = "iou",
        is_obb: bool = False,
    ) -> None:
        import logging

        # Restore standard names before BoxMOT/PyTorch imports: an earlier
        # dependency may have replaced the process-wide logging level mapping.
        for name, value in (
            ("NOTSET", 0),
            ("DEBUG", 10),
            ("INFO", 20),
            ("WARN", 30),
            ("WARNING", 30),
            ("ERROR", 40),
            ("FATAL", 50),
            ("CRITICAL", 50),
        ):
            logging.addLevelName(value, name)

        from boxmot import HybridSort as _HybridSort

        #dev = torch.device(device) if isinstance(device, str) else device
        dev = "cpu"

        tracker = _HybridSort(
            reid_weights=reid_weights,
            device=dev,
            half=half,
            cmc_method=cmc_method,
            with_reid=with_reid,
            #
            low_thresh=low_thresh,
            delta_t=delta_t,
            inertia=inertia,
            use_byte=use_byte,
            #
            use_custom_kf=use_custom_kf,
            longterm_bank_length=longterm_bank_length,
            alpha=alpha,
            adapfs=adapfs,
            track_thresh=track_thresh,
            #
            EG_weight_high_score=EG_weight_high_score,
            EG_weight_low_score=EG_weight_low_score,
            #
            TCM_first_step=TCM_first_step,
            TCM_byte_step=TCM_byte_step,
            TCM_byte_step_weight=TCM_byte_step_weight,
            high_score_matching_thresh=high_score_matching_thresh,
            #
            with_longterm_reid=with_longterm_reid,
            longterm_reid_weight=longterm_reid_weight,
            with_longterm_reid_correction=with_longterm_reid_correction,
            longterm_reid_correction_thresh=longterm_reid_correction_thresh,
            longterm_reid_correction_thresh_low=longterm_reid_correction_thresh_low,
            #
            dataset=dataset,
            #
            det_thresh=det_thresh,
            max_age=max_age,
            max_obs=max_obs,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            per_class=per_class,
            nr_classes=nr_classes,
            asso_func=asso_func,
            is_obb=is_obb,
        )

        super().__init__(tracker)
