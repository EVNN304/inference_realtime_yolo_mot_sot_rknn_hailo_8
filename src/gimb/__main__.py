import logging
from src.gimb.controllers import TrOsControllers
from src.gimb.ctrls.detection_backend import create_detection_loop, selected_backends
from src.gimb.ctrls.tracking_loop import TrackingLoop
from src.gimb.ctrls.video_images_loop import TrOsVideoImagesLoop
from src.gimb.domain.cameras import TrOsCamId, TrOsChannel
from src.gimb.state import TrOsState
import config.config as config
from src.framework.loop.loop import (
    SimpleSeparateThreadLoopRunner,
    SimpleSameThreadLoopRunner,
    TimeSyncSeparateThreadLoopRunner,
)

if __name__ == "__main__":
    if config.use_faulthandler:
        import faulthandler
        faulthandler.enable(all_threads=True)

    state = TrOsState()
    controllers = TrOsControllers(state)

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for log_name, log_level in config.logging.items():
        logging.getLogger(log_name).setLevel(log_level)

    if config.frame_source == "camera":
        from src.gimb.ctrls.camera_images_loop import CameraImagesLoop
        image_loop = SimpleSameThreadLoopRunner(CameraImagesLoop(state))
    elif config.frame_source == "shmem":
        from src.gimb.ctrls.shmem_images_loop import ShmemImagesLoop
        image_loop = SimpleSameThreadLoopRunner(ShmemImagesLoop(state))
    elif config.frame_source == "rtsp":
        from src.gimb.ctrls.rtsp_images_loop import RtspImagesLoop
        image_loop = SimpleSameThreadLoopRunner(RtspImagesLoop(state))
    elif config.frame_source == "folder":
        from src.gimb.ctrls.dataset_images_loop import TrOsDatasetImagesLoop
        image_loop = SimpleSameThreadLoopRunner(TrOsDatasetImagesLoop(state))
    else:
        image_loop = SimpleSameThreadLoopRunner(TrOsVideoImagesLoop(state))


    detection = create_detection_loop(state, controllers.tracking_controller)
    using_hailo = "hailo" in selected_backends(config)
    det_loop_tv = SimpleSeparateThreadLoopRunner(detection)
    
    tracking_loop_tv = SimpleSeparateThreadLoopRunner(
        TrackingLoop(state, controllers.tracking_controller)
    )

    det_started = tracking_started = False
    display_runner = None
    display_started = False
    try:
        if using_hailo:
            # Validate models and external embeddings before opening the source.
            detection.on_start()
        det_loop_tv.start()
        det_started = True
        tracking_loop_tv.start()
        tracking_started = True

        if config.enable_rendering:
            from src.gimb.ctrls.render_loop import TrOsRenderLoop
            display_runner = SimpleSeparateThreadLoopRunner(TrOsRenderLoop(state))
        else:
            from src.gimb.ctrls.fps_reporter_loop import TrOsFpsReporterLoop
            display_runner = TimeSyncSeparateThreadLoopRunner(
                TrOsFpsReporterLoop(state), iter_duration=1
            )
        display_runner.start()
        display_started = True
        image_loop.start()
    finally:
        if using_hailo:
            detection.request_stop()
            if det_started:
                det_loop_tv.stop()
                detection.done.wait()  # Device buffers outlive both inference owners.
            else:
                detection.on_stop()
            if tracking_started:
                tracking_loop_tv.stop()
            if display_started:
                display_runner.stop()
