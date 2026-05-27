# RVM Web — AI Video Background Removal

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/badge/install%20with-uv-purple)](https://github.com/astral-sh/uv)

**One-command web UI for [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting).**  
Remove video backgrounds with state-of-the-art AI — no green screen needed.

![screenshot](docs/screenshot.png)

---

## Features

- Web GUI with light theme, all parameters configurable via dropdowns
- GPU accelerated (CUDA) with automatic CPU fallback
- Two models: **MobileNetV3** (fast) and **ResNet50** (accurate)
- Export as MP4 composition, alpha mask, or PNG sequence with real RGBA transparency
- Installable in one command via `uv`

---

## Quick Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install git+https://github.com/<your-username>/rvm-web
rvm-web
```

A browser window opens automatically at `http://localhost:7860`.

### CUDA (NVIDIA GPU) — recommended

```bash
uv tool install git+https://github.com/<your-username>/rvm-web --extra cuda
```

---

## Parameters

| Parameter | Options | Default |
|---|---|---|
| Model | mobilenetv3, resnet50 | mobilenetv3 |
| Output type | video, png_sequence | video |
| Downsample ratio | auto, 0.25, 0.5, 0.75, 1.0 | auto |
| Bitrate (Mbps) | 1, 2, 4, 8, 16 | 4 |
| Device | auto, cpu, cuda | auto |
| Seq chunk | 1, 2, 4, 8 | 1 |
| Workers | 0, 1, 2, 4 | 0 |

---

## Known Fix — `av` Compatibility

RVM's `inference_utils.py` passes frame rate as a string to `av.add_stream()`.  
Newer `av` versions require a `Fraction`. The fix is applied automatically on first run.

---

## Credits

- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) by Peter Lin et al.
- Web UI built with [FastAPI](https://fastapi.tiangolo.com/)

---

## License

MIT
