# Performance notes

## Target and supplied measurements

Development target: Orange Pi 5 / RK3588, Hailo-8 M.2 over PCIe. The reported link was PCIe Gen2 ×1. HailoRT and firmware were 4.24.0. The YOLO HEF exposed a UINT8 NHWC 640×640×3 input and six-class HAILO_NMS_BY_CLASS output.

The optimized two-context YOLO11s HEF reported:

- Hardware-only throughput: 52.2269 FPS.
- Streaming throughput: 52.1222 FPS.
- Reported hardware latency: 96.347 ms.

The benchmark batch and all command-line options were not retained with this snapshot. Throughput and latency are different metrics; asynchronous batches prevent interpreting reciprocal latency as FPS.

## Integrated file playback

Final development log: 1920×1080 video at 29.97 FPS, full-frame YOLO on Hailo, RKNN ReID, NanoTrack active, batch 4, two Hailo slots. After startup, rolling detector measurements were 28–30.7 FPS with a source around 29–30 FPS. Startup and occasional source skips remain visible.

Typical per-batch values in that log:

| Stage | Observed range |
|---|---:|
| Preparation | 44–58 ms |
| Waiting for async readiness | 0.1–0.4 ms |
| Submission to callback | 113–139 ms |
| Collection and result processing | 5–15 ms |

These are host wall-clock measurements, not a device-only profiler. They overlap across jobs and must not be summed to infer maximum throughput. Preparation includes time lost to host scheduling. Short-window FPS can exceed source FPS as a batch completes.

MOT was about 4–6 FPS in this run. This is explicitly not a claim of 30 FPS end-to-end tracking.

## Repeatable measurement procedure

1. Record board/OS, camera or video properties, model SHA-256, HailoRT/firmware versions, PCIe link, batch, inflight slots, and whether MOT/SOT/rendering are active.
2. Run the same clip for at least 60 seconds; exclude startup from the steady-state summary.
3. Save `perf` and `YOLO stages per batch` lines. Retain source drops and latency alongside FPS.
4. For maximum detector throughput, use a source delivering more frames than the target detector rate. A paced 30 FPS file cannot establish 50 FPS detector throughput.
5. For camera tests, verify actual delivered FPS at the chosen format and resolution. The current `source` counter covers video-file publication only.
6. Compare the same configuration and clip before/after a change. Do not compare standalone HEF throughput to full application FPS as if they were interchangeable.

## Next measurements

- Integrated detector with a confirmed 60 FPS source.
- CPU preparation and decode under the same MOT/SOT load.
- Latency percentiles and sustained drop counts, not only average FPS.
- Tracking metrics on labeled sequences before claiming identity-quality improvements.
