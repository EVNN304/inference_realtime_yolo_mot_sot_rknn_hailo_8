import threading
from types import SimpleNamespace as NS
import numpy as np
import unittest
import test_integration as fixtures
from src.framework.core.thread_circular_buffer import ThreadCircularBuffer

class FeedTests(unittest.TestCase):
    make_loop = fixtures.LoopTests.make_loop
    def setup_feed(self):
        loop, _, state, _ = self.make_loop()
        ring = ThreadCircularBuffer(32)
        state.cameras[0].frames = ring
        loop.async_full_frame = True
        loop.settings.full_frame_batch = 2
        loop.settings.full_frame_max_wait_ms = 10000
        loop.settings.timeout_ms = 1000
        jobs = []
        loop.backend.can_submit_frames = lambda: len(loop.pending_frames) < 2
        def submit(images, stop):
            job = NS(done=threading.Event(), images=images)
            jobs.append(job)
            return job
        loop.backend.submit_frames = submit
        loop.backend.collect_frames = lambda job: [np.empty((0,6),np.float32) for _ in job.images]
        return loop, ring, state, jobs

    def test_retains_frames_arriving_between_updates(self):
        loop, ring, state, jobs = self.setup_feed()
        image = np.zeros((10,10,3),np.uint8)
        ring.enqueue(image, 100_000_000); loop.update()
        ring.enqueue(image, 133_000_000); loop.update()
        # Multiple source frames while detector is preparing/submitting/collecting.
        ring.enqueue(image, 166_000_000)
        ring.enqueue(image, 199_000_000)
        loop.update()
        self.assertEqual(len(jobs), 2)
        for job in jobs: job.done.set()
        loop.update()
        self.assertEqual([s.timestamp_ns for s in state.detections[0].items],
                         [100_000_000,133_000_000,166_000_000,199_000_000])

    def test_discards_old_backlog_after_long_stall(self):
        loop, ring, state, jobs = self.setup_feed()
        image = np.zeros((10,10,3),np.uint8)
        ring.enqueue(image, 100_000_000); loop.update()
        ring.enqueue(image, 200_000_000)
        ring.enqueue(image, 900_000_000)
        ring.enqueue(image, 1_000_000_000)
        loop.update()
        self.assertEqual([s.timestamp_ns for s in loop.pending_frames[0][1]],
                         [900_000_000,1_000_000_000])
