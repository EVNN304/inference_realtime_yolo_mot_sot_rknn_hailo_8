import numpy as np
import pytest

from src.framework.cameras import sensor_lib
from src.framework.cameras.camera import Camera
from src.framework.cameras.alignment import CameraIntrinsics
from app_rook.model_msg import pack_target


def test_pack_target_frame_center():
    camera = Camera(sensor_lib.IRIS_17, 75)
    camera_intr = CameraIntrinsics.from_camera(camera)
    target = pack_target(camera_intr, bbox=np.array([310, 235, 20, 10]))

    assert target.delta_azimuth == 0
    assert target.delta_elevation == 0
    assert target.x == 310
    assert target.y == 235
    assert target.z == 20
    assert target.vx == 10


def test_pack_target_frame_top_left_corner():
    camera = Camera(sensor_lib.IRIS_17, 75)
    # print(camera.fov)
    # print(np.degrees(camera.fov))
    camera_intr = CameraIntrinsics.from_camera(camera)
    target = pack_target(camera_intr, bbox=np.array([0, 0, 0.01, 0.01]))

    assert target.delta_azimuth == pytest.approx(-4148, abs=20)
    assert target.delta_elevation == pytest.approx(3114, abs=10)


def test_pack_target_frame_bottom_right_corner():
    camera = Camera(sensor_lib.IRIS_17, 75)
    # print(camera.fov)
    # print(np.degrees(camera.fov))
    camera_intr = CameraIntrinsics.from_camera(camera)
    target = pack_target(camera_intr, bbox=np.array([640, 480, 0.01, 0.01]))

    assert target.delta_azimuth == pytest.approx(4148, abs=20)
    assert target.delta_elevation == pytest.approx(-3114, abs=10)
