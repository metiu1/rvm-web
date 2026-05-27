"""RobustVideoMatting inference wrapper."""
from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import torch

_model_cache: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


def _patch_rvm_utils() -> None:
    """Fix av API incompatibility: add_stream expects Fraction, not str."""
    try:
        import importlib, sys
        cache_dirs = [p for p in sys.path if "RobustVideoMatting" in p]
        if not cache_dirs:
            import torch.hub as hub
            import os
            hub_dir = hub.get_dir()
            for d in Path(hub_dir).glob("PeterL1n_RobustVideoMatting*"):
                sys.path.insert(0, str(d))

        import inference_utils
        src = Path(inference_utils.__file__).read_text()
        if "f'{frame_rate:.4f}'" in src:
            patched = src.replace(
                "rate=f'{frame_rate:.4f}'",
                "rate=Fraction(frame_rate).limit_denominator(65535)"
            )
            if "from fractions import Fraction" not in patched:
                patched = "from fractions import Fraction\n" + patched
            Path(inference_utils.__file__).write_text(patched)
            importlib.reload(inference_utils)
    except Exception:
        pass


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    key = f"{model_name}_{device}"
    with _lock:
        if key not in _model_cache:
            _patch_rvm_utils()
            model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                model_name,
                pretrained=True,
                trust_repo=True,
            )
            _model_cache[key] = model.to(device).eval()
        return _model_cache[key]


def process_video(params: dict, progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """
    Run RVM background removal.

    params keys:
      model_name       : "mobilenetv3" | "resnet50"
      input_source     : str path
      output_composition: str path or None
      output_alpha     : str path or None
      output_foreground: str path or None
      output_type      : "video" | "png_sequence"
      downsample_ratio : float or None
      output_video_mbps: float
      seq_chunk        : int
      num_workers      : int
      device           : "auto" | "cpu" | "cuda"
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    device_str = params.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    log(f"Loading model '{params['model_name']}' on {device}…")
    model = load_model(params["model_name"], device)

    log("Loading converter…")
    convert_video = torch.hub.load(
        "PeterL1n/RobustVideoMatting", "converter", trust_repo=True
    )

    output_composition = params.get("output_composition") or None
    output_alpha       = params.get("output_alpha") or None
    output_foreground  = params.get("output_foreground") or None

    if not any([output_composition, output_alpha, output_foreground]):
        raise ValueError("At least one output path must be specified.")

    downsample = params.get("downsample_ratio")
    if downsample is not None:
        downsample = float(downsample)

    log("Processing video… (this may take a while)")
    convert_video(
        model,
        input_source=params["input_source"],
        output_type=params.get("output_type", "video"),
        output_composition=output_composition,
        output_alpha=output_alpha,
        output_foreground=output_foreground,
        output_video_mbps=float(params.get("output_video_mbps", 4)),
        downsample_ratio=downsample,
        seq_chunk=int(params.get("seq_chunk", 1)),
        num_workers=int(params.get("num_workers", 0)),
        device=device.type,
    )
    log("Done.")
