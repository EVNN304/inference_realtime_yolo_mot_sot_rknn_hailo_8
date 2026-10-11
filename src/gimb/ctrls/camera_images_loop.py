"""Direct USB camera -> existing frame ring, without a fixed 30 FPS sleep."""
import logging
import time
import config.config as config
from src.framework.loop.loop import ILoop


class CameraImagesLoop(ILoop):
    def __init__(self, state):
        super().__init__()
        from src.gimb.domain.cameras import TrOsCamId
        self.state, self.cam_id, self.cap = state, TrOsCamId.TV_N, None

    def on_start(self):
        import cv2
        settings = config.CameraCapture
        self.cap = cv2.VideoCapture(settings.device, cv2.CAP_V4L2)
        try:
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open camera: {settings.device}')
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
            self.cap.set(cv2.CAP_PROP_FPS, settings.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this request.
            logging.getLogger('camera').info('Camera reports %.0fx%.0f at %.2f FPS (requested %s)',
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                self.cap.get(cv2.CAP_PROP_FPS), settings.fps)
        except BaseException:
            self.on_stop()
            raise

    def update(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError('Camera read failed')
        cam = self.state.cameras[self.cam_id]
        cam.frames.enqueue(frame, time.monotonic_ns())
        if cam.resolution is None:
            cam.resolution = frame.shape[:2]

    def on_stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
