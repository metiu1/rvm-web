"""RobustVideoMatting inference wrapper + upscaling utilities."""
from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import torch

_model_cache: dict[str, torch.nn.Module] = {}
_converter_cache: dict[str, Callable] = {}
_lock = threading.Lock()


def load_converter() -> Callable:
    with _lock:
        if "converter" not in _converter_cache:
            _converter_cache["converter"] = torch.hub.load(
                "PeterL1n/RobustVideoMatting", "converter", trust_repo=True
            )
        return _converter_cache["converter"]


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
    convert_video = load_converter()

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


# ---------------------------------------------------------------------------
# Upscaling — built-in Real-ESRGAN (RRDBNet), no external packages needed
# ---------------------------------------------------------------------------

import math
import torch.nn as nn
import torch.nn.functional as F


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.rdb3(self.rdb2(self.rdb1(x)))) * 0.2 + x


class _RRDBNet(nn.Module):
    """RRDBNet — same architecture as Real-ESRGAN x2plus / x4plus models."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_block: int = 23,
                 num_grow_ch: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        in_ch = num_in_ch * 4 if scale == 2 else num_in_ch
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class _RealESRGANUpsampler:
    """Thin wrapper: downloads weights, runs tiled inference."""

    _MODELS = {
        2: ("RealESRGAN_x2plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
        4: ("RealESRGAN_x4plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    }

    def __init__(self, scale: int, tile: int = 512, tile_pad: int = 10):
        import urllib.request
        net_scale = 2 if scale <= 2 else 4
        fname, url = self._MODELS[net_scale]
        cache = Path.home() / ".cache" / "rvm_web" / "realesrgan"
        cache.mkdir(parents=True, exist_ok=True)
        weights = cache / fname
        if not weights.exists():
            urllib.request.urlretrieve(url, weights)

        model = _RRDBNet(scale=net_scale)
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("params_ema") or state.get("params") or state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.net_scale = net_scale
        self.tile = tile
        self.tile_pad = tile_pad

    def enhance(self, img_rgb: "numpy.ndarray") -> "numpy.ndarray":
        """HxWx3 uint8 RGB in → HxWx3 uint8 RGB out."""
        import numpy as np
        t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, h, w = t.shape
        with torch.no_grad():
            if self.tile == 0 or (h <= self.tile and w <= self.tile):
                out = self.model(t)
            else:
                out = self._tiled(t)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1)
        return (out.cpu().numpy() * 255.0).round().astype(np.uint8)

    def _tiled(self, img: torch.Tensor) -> torch.Tensor:
        s = self.net_scale
        _, c, h, w = img.shape
        tile, pad = self.tile, self.tile_pad
        out = torch.zeros(1, c, h * s, w * s, dtype=img.dtype, device=img.device)
        for yi in range(math.ceil(h / tile)):
            for xi in range(math.ceil(w / tile)):
                x1 = max(xi * tile - pad, 0)
                x2 = min((xi + 1) * tile + pad, w)
                y1 = max(yi * tile - pad, 0)
                y2 = min((yi + 1) * tile + pad, h)
                patch = self.model(img[:, :, y1:y2, x1:x2])
                ox1, ox2 = xi * tile * s, min((xi + 1) * tile, w) * s
                oy1, oy2 = yi * tile * s, min((yi + 1) * tile, h) * s
                px1 = (xi * tile - x1) * s
                py1 = (yi * tile - y1) * s
                out[:, :, oy1:oy2, ox1:ox2] = patch[:, :, py1:py1 + (oy2 - oy1), px1:px1 + (ox2 - ox1)]
        return out


_upsampler_cache: dict[int, "_RealESRGANUpsampler"] = {}
_upsampler_lock = threading.Lock()


def _get_realesrgan_upsampler(scale: int) -> "_RealESRGANUpsampler":
    with _upsampler_lock:
        if scale not in _upsampler_cache:
            _upsampler_cache[scale] = _RealESRGANUpsampler(scale)
        return _upsampler_cache[scale]


def upscale_image(
    input_path: Path,
    output_path: Path,
    method: str = "lanczos",
    scale: int = 2,
) -> None:
    import numpy as np
    from PIL import Image  # type: ignore

    img = Image.open(str(input_path))
    w, h = img.size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "realesrgan":
        upsampler = _get_realesrgan_upsampler(scale)
        rgb = np.array(img.convert("RGB"))
        result = Image.fromarray(upsampler.enhance(rgb))
        result.save(str(output_path))
    else:
        img.resize((w * scale, h * scale), Image.LANCZOS).save(str(output_path))


def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable: system PATH first, then imageio-ffmpeg bundle."""
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        return get_ffmpeg_exe()
    except Exception:
        return None


def _mux_audio(video_path: str, audio_src: str, output_path: str,
               log: Callable[[str], None]) -> bool:
    """Mux audio from audio_src onto video_path → output_path. Returns True on success."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        log("WARN: ffmpeg non trovato (pip install imageio-ffmpeg) — output senza audio")
        return False

    import subprocess

    cmd = [
        ffmpeg, "-y",
        "-i", video_path,        # 0: upscaled video (no audio)
        "-i", audio_src,         # 1: original (for audio)
        "-map", "0:v:0",
        "-map", "1:a:0?",        # optional audio — won't fail if source has none
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("Audio unito al video.")
        return True
    except Exception as exc:
        log(f"WARN: unione audio fallita ({exc}) — output senza audio")
        return False


def upscale_video(params: dict, progress_cb: Optional[Callable[[str], None]] = None) -> None:
    import os
    import av  # type: ignore
    from PIL import Image  # type: ignore

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    input_path = str(params["input_path"])
    output_path = str(params["output_path"])
    method = params.get("method", "lanczos")
    scale = int(params.get("scale", 2))
    crf = int(params.get("crf", 16))

    upsampler = None
    if method == "realesrgan":
        log("Caricamento modello Real-ESRGAN…")
        upsampler = _get_realesrgan_upsampler(scale)
        log(f"Modello caricato su {upsampler.device}.")

    def process_frame(pil_frame: "Image.Image") -> "Image.Image":
        if upsampler is not None:
            import numpy as np
            return Image.fromarray(upsampler.enhance(np.array(pil_frame.convert("RGB"))))
        w, h = pil_frame.size
        return pil_frame.resize((w * scale, h * scale), Image.LANCZOS)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # Upscale into a video-only temp file, then mux original audio back with ffmpeg.
    # This avoids the brittle, version-dependent PyAV audio stream-copy API.
    temp_video = str(out_dir / f"{Path(output_path).stem}__videotmp.mp4")

    has_audio = False
    with av.open(input_path) as in_c:
        in_v = in_c.streams.video[0]
        fps = in_v.average_rate
        total = in_v.frames or 0
        orig_w, orig_h = in_v.width, in_v.height
        has_audio = len(in_c.streams.audio) > 0
        log(f"Risoluzione: {orig_w}x{orig_h} → {orig_w * scale}x{orig_h * scale}, {float(fps):.2f} fps")

        with av.open(temp_video, "w") as out_c:
            # Create the encoder lazily from the first processed frame so the
            # stream dimensions always match the real output (robust against any
            # Real-ESRGAN rounding). Let the encoder generate timestamps from the
            # stream rate — copying the input pts/time_base produces invalid,
            # non-monotonic DTS and the muxer rejects packets (errno 22 EINVAL).
            out_v = None
            frame_idx = 0
            for frame in in_c.decode(video=0):
                upscaled = process_frame(frame.to_image())
                if out_v is None:
                    out_v = out_c.add_stream("libx264", rate=fps)
                    out_v.width, out_v.height = upscaled.size
                    out_v.pix_fmt = "yuv420p"
                    # Lower CRF = higher quality (default libx264 would
                    # re-compress heavily and throw away upscaled detail).
                    out_v.options = {"crf": str(crf), "preset": "slow"}
                out_frame = av.VideoFrame.from_image(upscaled)
                for pkt in out_v.encode(out_frame):
                    out_c.mux(pkt)
                frame_idx += 1
                if frame_idx == 1 or frame_idx % 25 == 0:
                    progress = f"{frame_idx}/{total}" if total else str(frame_idx)
                    log(f"Frame {progress} elaborati…")

            if out_v is not None:
                for pkt in out_v.encode():
                    out_c.mux(pkt)

    # Attach audio
    if has_audio:
        log("Unione audio…")
        if _mux_audio(temp_video, input_path, output_path, log):
            try:
                os.remove(temp_video)
            except OSError:
                pass
        else:
            os.replace(temp_video, output_path)
    else:
        os.replace(temp_video, output_path)

    log("Done.")
