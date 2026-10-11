import time

from src.framework.frame.shm.reader import SharedMemoryCapReader
import cv2


def main():
    frames_to_capture = 30
    reader = SharedMemoryCapReader("/dev/shm/tv")
    reader.start()
    n_frames = 0
    while n_frames < frames_to_capture:
        ret, frame = reader.read_no_wait(True)
        if ret:
            cv2.imwrite(f"/tmp/frame_{n_frames}.jpg", frame)
            print(f"Saved frame {n_frames}.jpg")
            n_frames += 1
        else:
            time.sleep(0.02)
            continue


if __name__ == '__main__':
    main()
