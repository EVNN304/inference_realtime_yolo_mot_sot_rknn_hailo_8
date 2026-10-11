import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from src.gimb.ctrls.camera_images_loop import CameraImagesLoop


class CameraTests(unittest.TestCase):
    def test_frames_enter_existing_ring_and_capture_closes(self):
        frames=[]; settings={}; released=[]
        image=np.zeros((12,20,3),np.uint8)
        cap=NS(isOpened=lambda:True,set=lambda k,v:settings.update({k:v}),
               get=lambda k:settings.get(k,0),read=lambda:(True,image),release=lambda:released.append(True))
        cv=NS(VideoCapture=lambda *args:cap,CAP_V4L2=1,VideoWriter_fourcc=lambda *a:123,
              CAP_PROP_FOURCC=10,CAP_PROP_FRAME_WIDTH=11,CAP_PROP_FRAME_HEIGHT=12,
              CAP_PROP_FPS=13,CAP_PROP_BUFFERSIZE=14)
        modules={'cv2':cv,'src.gimb.domain.cameras':NS(TrOsCamId=NS(TV_N=0))}
        camera=NS(frames=NS(enqueue=lambda img,ts:frames.append((img,ts))),resolution=None)
        with patch.dict(sys.modules,modules):
            loop=CameraImagesLoop(NS(cameras=[camera]));loop.on_start()
            loop.update();loop.on_stop()
        self.assertEqual(settings[cv.CAP_PROP_FPS],60)
        self.assertEqual(camera.resolution,(12,20))
        self.assertIs(frames[0][0],image)
        self.assertGreater(frames[0][1],0)
        self.assertEqual(released,[True])

if __name__ == '__main__':unittest.main()
