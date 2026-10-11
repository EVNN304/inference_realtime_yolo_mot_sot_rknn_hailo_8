import cv2
import time
import multiprocessing as mp
import numpy as np
from pathlib import Path
from boxmot import (
    BoostTrack,
    BotSort,
    StrongSort,
    DeepOcSort,
    ByteTrack,
    HybridSort,
    OcSort,
)
from src.framework.mot.track import SimpleTrack


class MOT:
    def __init__(self, method_mot=5):
        self.mot_method = method_mot
        self.mot = None
        self.path_reid = "osnet_x0_25_msmt17.pt"
        # self.path_reid = '/path/to/local-file'
        self.flag_show_track = False
        self.half = True
        self.device = 0

    def set_mot_method(self, val: int):
        self.mot_method = val

    def set_path_weights_reid(self, val: str):
        self.path_reid = val
        print(f"Choice_weights_reid:{self.path_reid}")

    def set_flag_half(self, val: bool):
        self.half = val

    def set_device(self, val: int):
        self.device = val

    def set_flag_show_track(self, val: bool):
        self.flag_show_track = val

    def covert_to_simpleDetection(self, arr):

        lst_obj = []
        time_stmp = time.time()
        for i, k in enumerate(arr):
            lst_obj.append(
                SimpleTrack(
                    box=[float(k[0]), float(k[1]), float(k[2]), float(k[3])],
                    cls=int(k[-2]),
                    conf=float(k[-3]),
                    track_id=int(k[-4]),
                    t_ns=int(time_stmp),
                )
            )
        return lst_obj

    def choice_mot(self):
        # OK                                                                                                                       #no                                                                                                              #OK                                                                                                              ###  ok                                                                                                                                             ## ok                                                                                       ### ok                                                                                                                   ### ok
        dict_mot = {
            0: (
                BoostTrack(
                    reid_weights=Path(self.path_reid),
                    device=self.device,
                    half=self.half,
                ),
                "Choice method mot BoostTrack!",
            ),
            1: (
                BotSort(
                    reid_weights=Path(self.path_reid),
                    device=self.device,
                    half=self.half,
                ),
                "Choice method mot BotSort!",
            ),
            2: (
                StrongSort(
                    reid_weights=Path(self.path_reid),
                    device=self.device,
                    half=self.half,
                ),
                "Choice method mot StrongSort!",
            ),
            3: (
                DeepOcSort(
                    reid_weights=Path(self.path_reid),
                    device=self.device,
                    half=self.half,
                ),
                "Choice method mot DeepOcSort!",
            ),
            4: (
                ByteTrack(
                    reid_weights=Path(self.path_reid),
                    device=self.device,
                    half=self.half,
                ),
                "Choice method mot ByteTrack!",
            ),
            5: (
                HybridSort(
                    reid_weights=Path(self.path_reid),
                    device=self.device,
                    half=self.half,
                ),
                "Choice method mot HybridSort!",
            ),
            6: (
                OcSort(
                    reid_weights=Path(self.path_reid),
                    device=self.device,
                    half=self.half,
                ),
                "Choice method mot OcSort!",
            ),
        }
        try:
            self.mot, st = dict_mot[self.mot_method]
            print(st)
            return self.mot, True
        except Exception as e:
            print(f"Errrr_choice_mot:{e.args}")
            self.mot, st = dict_mot[5]
            print(f"Default_mot_choice:", st)
            return self.mot, True

    def main_worker(self, frame, detections):

        frame_count = 0
        start_time = time.time()
        H, W = frame.shape[:2]
        try:
            lst = []
            for i, k in enumerate(detections):
                try:
                    x1, y1, x2, y2 = (
                        float(k.box.x1),
                        float(k.box.y1),
                        float(k.box.x2),
                        float(k.box.y2),
                    )
                    conf, cls = float(k.conf), int(k.cls)
                except:
                    continue

                x1, y1, x2, y2 = (
                    max(0, int(x1)),
                    max(0, int(y1)),
                    min(W, int(x2)),
                    min(H, int(y2)),
                )
                w, h = x2 - x1, y2 - y1
                if conf >= 0.2 and w >= 2 and h >= 2:
                    lst.append([x1, y1, x2, y2, conf, cls])

            res = self.mot.update(np.array(lst), frame)
            obb = self.covert_to_simpleDetection(res)
            self.mot.plot_results(frame, show_trajectories=self.flag_show_track)

            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            return frame, res

        except Exception as e:
            print(f"Errrrrrrrrrrr_MOT: {e.args}")
            return frame, []
