from src.framework.mot.boxmot_wrapper import BoxMotWrapperBase


class ByteTrackTracker(BoxMotWrapperBase):
    def __init__(
        self,
        *,
        min_conf: float = 0.1,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 25,
        frame_rate: int = 30,
        per_class: bool = False,
        nr_classes: int = 1,
        asso_func: str = "iou",
        is_obb: bool = False,
        **kwargs,
    ) -> None:
        from boxmot import ByteTrack as _BoxmotByteTrack

        tracker = _BoxmotByteTrack(
            min_conf=min_conf,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            per_class=per_class,
            nr_classes=nr_classes,
            asso_func=asso_func,
            is_obb=is_obb,
            **kwargs,
        )
        super().__init__(tracker)
