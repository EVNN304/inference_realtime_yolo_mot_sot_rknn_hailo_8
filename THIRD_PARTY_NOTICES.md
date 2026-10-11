# Third-party provenance and license status

This source snapshot contains integration code and existing third-party code. It is not presented as an original implementation of the underlying detection and tracking algorithms.

- `boxmot/` is the local copy supplied with the original project. Its `__init__.py` reports version 16.0.8 and contains a BoxMOT AGPL-3.0 attribution. Original source notices are retained. It includes local changes; no claim is made that it matches an upstream release byte for byte. Personal absolute paths in optional modules were replaced with placeholders during packaging.
- YOLO/Ultralytics, HybridSORT, OSNet, NanoTrack, STARK, HailoRT and RKNN are external projects or vendor components. Their models, SDKs and distributions have separate terms and provenance.
- The original project metadata contained an MIT label but no accompanying project LICENSE file was present. That label has not been promoted into a blanket license for this mixed source tree. A new project-wide license was not selected automatically.
- No model weights, datasets or vendor SDK binaries are included. Their absence is not a statement about permission to redistribute them separately.

Before distributing this snapshot, the repository owner needs to resolve the intended project license and the notices/license texts appropriate to the included local third-party sources. This file records what was found in the supplied project; it does not substitute for those licenses.
