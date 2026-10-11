# Setup on Orange Pi / RK3588

## 1. Platform runtimes

Use Linux with Python 3.10 or newer. Python 3.10 is the application target used in the supplied board logs. Install the runtime wheels and drivers that match your operating system and kernel before installing application dependencies.

- **Hailo:** the development target used HailoRT 4.24.0, its Python binding, matching PCIe driver and firmware. Verify `hailortcli scan` and `hailortcli fw-control identify` before launching the application.
- **Rockchip:** the supplied logs used RKNN Toolkit Lite2/runtime 2.3.2 on RK3588. Install the appropriate aarch64 runtime wheel from your Rockchip SDK distribution. NanoTrack still needs this runtime even if detector and ReID both use Hailo.
- **ONNX Runtime:** needed for NanoTrack's head.
- **Camera:** OpenCV must support your chosen capture backend. Use a build with a GUI backend if rendering is enabled.
- **RTSP/GStreamer:** optional sources additionally need system GStreamer plugins and Python GI bindings. They are not installed by the base pip requirements.

Keep a working board environment if you already have one. Do not replace vendor wheels with arbitrary packages to satisfy a generic installation command.

## 2. Python dependencies

From the repository root, in your environment:

```bash
python -m pip install -r requirements.txt
```

The requirements are a starting dependency set, not a lockfile of the board image. Torch/torchvision must be a compatible pair available for aarch64. The local `boxmot/` directory is the project's existing modified copy and takes precedence when running from the repository root; installing BoxMOT also supplies its external dependencies. Replacing the local copy needs a separate compatibility test.

STARK is optional. Its former private dependency URL is deliberately absent. Supply your compatible `stark` package and weights separately if selecting that tracker. The default NanoTrack path does not import it during startup.

## 3. Models and local settings

No weights are included. Place your models in `models/`, or configure absolute paths in `config/local.py`:

```bash
cp config/local.example.py config/local.py
```

Required artifacts depend on the selected path:

| Component | Files |
|---|---|
| Hailo YOLO | HEF with the expected RGB input and HAILO_NMS_BY_CLASS output |
| RKNN YOLO | Compatible RKNN detector model |
| Hailo OSNet | HEF plus matching export/preprocessing manifest JSON |
| RKNN OSNet | Compatible dynamic-batch RKNN ReID model |
| NanoTrack | RKNN backbone plus ONNX head |

The ReID JSON must come from the same OSNet build; it is not an arbitrary label file. The runtime checks the manifest and model contract. Keep your original model/preprocessing artifacts together.

`config/local.py` runs after the defaults and can modify existing classes without duplicating their definitions. Keep confidence and model-specific settings consistent with your existing setup. Personal paths remain outside Git.

## 4. Run

```bash
python -m src.gimb
```

The previous application directory was renamed to `src/gimb`; use this command in service files and launch shortcuts too. The main configuration module is `config/config.py` (imported as `config.config`).

To use a camera, set `frame_source = "camera"` and its device, width, height, FPS and FOURCC in local configuration. Use a real supported camera mode. Keep batch 4 / inflight 2 for the initial comparison; increasing batch trades latency against throughput.

## 5. Local regression tests

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests/hailo_backend -v
```

No accelerator or models are needed for these tests. Hardware runtime, GUI, legacy examples, and tracking accuracy are outside this test suite.

## Preparing a GitHub repository

The snapshot excludes model binaries, recordings, caches, logs and original Git history. Review `THIRD_PARTY_NOTICES.md` before choosing a repository license. Then initialize a new repository from this folder and inspect the staged file list before your first commit. Publishing is a separate action; this preparation does not create a remote repository or upload files.
