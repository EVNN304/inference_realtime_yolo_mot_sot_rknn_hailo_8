# Architecture and invariants

## Existing application contract

Sources publish images with monotonic timestamps to `ThreadCircularBuffer`. Controllers consume detections and tracks through application state and timestamped callbacks. The Hailo adapter retains those interfaces. Renaming the application to `src.gimb` does not change coordinate conventions or the MOT → SOT handoff.

## Full-frame Hailo path

1. Read newly available frames in timestamp order from the existing source ring.
2. Collect up to `full_frame_batch` frames, or flush at `full_frame_max_wait_ms`.
3. Letterbox each frame to the model input; retain its inverse coordinate transform.
4. Submit using a reusable async slot. Padding inputs do not produce application results.
5. Consume completed jobs in submission order. A slot remains owned until output decoding finishes.
6. Publish detections with each original frame timestamp; send the matching image and detections to the bounded MOT queue.

The default source-backlog age limit is 250 ms relative to the latest source frame (`Hailo.input_max_age_ms` can override it). This does not bound total capture-to-display latency: batching and pending device work add their own delay.

## Tiled path

Tile origins are cached for a given frame geometry and overlap. Tiles are grouped into batches, translated back into source coordinates, and merged with class-wise NMS. Six tiles mean six detector inputs per source frame, so detector inputs/s and video frames/s differ.

## MOT and SOT

The Hailo integration has a single MOT worker to preserve ordered tracker state. A bounded queue drops stale work under overload. OSNet features may come from Hailo or RKNN, but HybridSORT's association runs on CPU. When YOLO and OSNet both use Hailo, the backend creates one shared VDevice.

NanoTrack's backbone and head retain the existing RKNN/ONNX Runtime route. The controller handles target selection and handoff. The source ring is shared; model-buffer ownership is not shared between asynchronous jobs.

## Compatibility boundaries

- RKNN/RKNN selection follows the existing detection loop.
- A Hailo selection uses the integration adapter and bridges to an RKNN component when requested.
- Existing source, controller, protocol and tracking algorithms are retained.
- Packaging changes centralize NanoTrack paths and make STARK import lazy; STARK is still an optional external dependency.
- Older examples are retained as reference material and are not covered by the hardware-free adapter suite.
