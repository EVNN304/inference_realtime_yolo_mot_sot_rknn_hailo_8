# Changelog

## 0.2.0 — GitHub source snapshot

- Rename the main configuration to `config/config.py` and update imports, tests and documentation.
- Rename the application package to `src.gimb` and update application imports and launch commands.
- Include the final timestamp-ordered full-frame feed fix, bounded backlog age and per-batch Hailo timing instrumentation.
- Include clock-paced file playback and completed-result FPS counters.
- Retain the standard logging-level restoration before BoxMOT import.
- Preserve independent detector/ReID backend selection and existing MOT → SOT contracts.
- Move NanoTrack model paths into configuration; load STARK only when selected.
- Add ignored local configuration overrides, portable public defaults and a configuration example.
- Add architecture, setup, measured-performance and Russian documentation plus adapter-test CI.
- Exclude weights, recordings, caches, logs, IDE configuration and original Git history.

Hardware tests of this renamed snapshot remain to be run on the board. Historical performance measurements refer to the prior application name with the same detector fixes.
