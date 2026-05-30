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
# Upscaling
# ---------------------------------------------------------------------------

def _build_realesrgan(scale: int):
    """Return a RealESRGANer for the given scale, downloading weights on first use."""
    from realesrgan import RealESRGANer  # type: ignore
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

    if scale <= 2:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        netscale = 2
    else:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        netscale = 4

    return RealESRGANer(
        scale=netscale, model_path=url, model=net,
        tile=512, tile_pad=10, pre_pad=0, half=False,
    )


def upscale_image(
    input_path: Path,
    output_path: Path,
    method: str = "lanczos",
    scale: int = 2,
) -> None:
    from PIL import Image  # type: ignore

    img = Image.open(str(input_path))
    w, h = img.size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "realesrgan":
        try:
            import numpy as np  # type: ignore
            upsampler = _build_realesrgan(scale)
            bgr = np.array(img.convert("RGB"))[:, :, ::-1]
            out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
            result = Image.fromarray(out_bgr[:, :, ::-1].copy())
            result.save(str(output_path))
        except ImportError:
            raise RuntimeError(
                "Real-ESRGAN non installato. Usa: pip install realesrgan"
            )
    else:
        out = img.resize((w * scale, h * scale), Image.LANCZOS)
        out.save(str(output_path))


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
        try:
            log("Caricamento modello Real-ESRGAN…")
            upsampler = _build_realesrgan(scale)
            log("Modello caricato.")
        except ImportError:
            log("WARN: realesrgan non trovato — uso Lanczos classico")

    def process_frame(pil_frame: "Image.Image") -> "Image.Image":
        if upsampler is not None:
            import numpy as np
            bgr = np.array(pil_frame.convert("RGB"))[:, :, ::-1]
            out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
            return Image.fromarray(out_bgr[:, :, ::-1].copy())
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
