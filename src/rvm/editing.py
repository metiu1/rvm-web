"""Image & video editing engine.

Images: Pillow pipeline (fast, in-process).
Video : bundled ffmpeg (imageio-ffmpeg) driven via subprocess with live
        progress parsed from ``-progress pipe:1``. No system install needed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# ffmpeg discovery (shared with processing.py logic, kept local to stay decoupled)
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> Optional[str]:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        return get_ffmpeg_exe()
    except Exception:
        return None


# ===========================================================================
# IMAGE EDITING  (Pillow)
# ===========================================================================
def edit_image(params: dict) -> str:
    """Apply an ordered pipeline of edits to an image.

    params keys (all optional unless noted):
      input_path   : str  (required)
      output_path  : str  (required)
      rotate       : int  degrees counter-clockwise (0/90/180/270 or any)
      flip_h       : bool horizontal mirror
      flip_v       : bool vertical mirror
      crop         : [x, y, w, h] in pixels
      resize_w     : int  target width  (0 = keep / derive from height)
      resize_h     : int  target height (0 = keep / derive from width)
      brightness   : float 1.0 = unchanged
      contrast     : float 1.0 = unchanged
      saturation   : float 1.0 = unchanged
      sharpness    : float 1.0 = unchanged
      filter       : "none"|"blur"|"sharpen"|"grayscale"|"sepia"|"invert"|"auto"
      fmt          : output format override ("png"|"jpg"|"webp"|...) — else from ext
      quality      : int 1-100 (jpg/webp)
    Returns the output path.
    """
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    inp = Path(params["input_path"])
    out = Path(params["output_path"])
    if not inp.exists():
        raise FileNotFoundError(f"File non trovato: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(str(inp))
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")

    # --- geometry ---
    crop = params.get("crop")
    if crop and len(crop) == 4:
        x, y, w, h = (int(v) for v in crop)
        img = img.crop((x, y, x + w, y + h))

    rotate = int(params.get("rotate", 0) or 0)
    if rotate % 360:
        img = img.rotate(rotate, expand=True)

    if params.get("flip_h"):
        img = ImageOps.mirror(img)
    if params.get("flip_v"):
        img = ImageOps.flip(img)

    rw = int(params.get("resize_w", 0) or 0)
    rh = int(params.get("resize_h", 0) or 0)
    if rw or rh:
        ow, oh = img.size
        if rw and not rh:
            rh = round(oh * rw / ow)
        elif rh and not rw:
            rw = round(ow * rh / oh)
        img = img.resize((max(1, rw), max(1, rh)), Image.LANCZOS)

    # --- enhancements ---
    for key, enhancer in (
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
        ("saturation", ImageEnhance.Color),
        ("sharpness", ImageEnhance.Sharpness),
    ):
        val = params.get(key)
        if val is not None and abs(float(val) - 1.0) > 1e-3:
            img = enhancer(img).enhance(float(val))

    # --- filter ---
    filt = params.get("filter", "none")
    if filt == "blur":
        img = img.filter(ImageFilter.GaussianBlur(3))
    elif filt == "sharpen":
        img = img.filter(ImageFilter.SHARPEN)
    elif filt == "grayscale":
        img = ImageOps.grayscale(img).convert(img.mode)
    elif filt == "invert":
        alpha = img.split()[-1] if img.mode == "RGBA" else None
        rgb = ImageOps.invert(img.convert("RGB"))
        img = rgb if alpha is None else Image.merge("RGBA", (*rgb.split(), alpha))
    elif filt == "sepia":
        gray = ImageOps.grayscale(img)
        sepia = ImageOps.colorize(gray, black="#2b1d0e", white="#ffe9c8")
        img = sepia.convert(img.mode)
    elif filt == "auto":
        img = ImageOps.autocontrast(img.convert("RGB")).convert(img.mode)

    # --- format / save ---
    fmt = (params.get("fmt") or out.suffix.lstrip(".") or "png").lower()
    if fmt in ("jpg", "jpeg"):
        img = img.convert("RGB")  # JPEG has no alpha
        save_kwargs = {"quality": int(params.get("quality", 92)), "optimize": True}
        pil_fmt = "JPEG"
    elif fmt == "webp":
        save_kwargs = {"quality": int(params.get("quality", 90))}
        pil_fmt = "WEBP"
    else:
        save_kwargs = {}
        pil_fmt = {"png": "PNG", "bmp": "BMP", "tiff": "TIFF", "gif": "GIF"}.get(fmt, "PNG")

    if out.suffix.lstrip(".").lower() != fmt:
        out = out.with_suffix("." + ("jpg" if fmt == "jpeg" else fmt))
    img.save(str(out), pil_fmt, **save_kwargs)
    return str(out)


# ===========================================================================
# VIDEO EDITING  (ffmpeg)
# ===========================================================================
def _probe_duration(path: str) -> float:
    """Total duration in seconds (0.0 if unknown). Uses PyAV (already a dep)."""
    try:
        import av  # type: ignore
        with av.open(path) as c:
            if c.duration:
                return float(c.duration) / 1_000_000.0
            v = c.streams.video[0]
            if v.duration and v.time_base:
                return float(v.duration * v.time_base)
    except Exception:
        pass
    return 0.0


def _build_video_args(params: dict, inp: str, out: str) -> tuple[list[str], list[str]]:
    """Return (input_args, output_args) for ffmpeg given a single operation."""
    op = params.get("operation", "convert")
    in_args: list[str] = []
    out_args: list[str] = []
    vf: list[str] = []
    af: list[str] = []

    if op == "trim":
        start = float(params.get("start", 0) or 0)
        end = params.get("end")
        if start > 0:
            in_args += ["-ss", f"{start}"]
        if end:
            out_args += ["-to", f"{max(0.0, float(end) - start)}"]
        out_args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "aac"]

    elif op == "crop":
        x, y, w, h = (int(params.get(k, 0) or 0) for k in ("x", "y", "w", "h"))
        vf.append(f"crop={w}:{h}:{x}:{y}")
        out_args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "copy"]

    elif op == "resize":
        w = int(params.get("w", 0) or -2)
        h = int(params.get("h", 0) or -2)
        w = w if w > 0 else -2
        h = h if h > 0 else -2
        vf.append(f"scale={w}:{h}")
        out_args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "copy"]

    elif op == "speed":
        factor = max(0.1, float(params.get("factor", 1.0) or 1.0))
        vf.append(f"setpts={1.0/factor:.6f}*PTS")
        # atempo only handles 0.5–2.0; chain for extremes
        remaining = factor
        tempo_chain = []
        while remaining > 2.0:
            tempo_chain.append("atempo=2.0"); remaining /= 2.0
        while remaining < 0.5:
            tempo_chain.append("atempo=0.5"); remaining /= 0.5
        tempo_chain.append(f"atempo={remaining:.6f}")
        af.extend(tempo_chain)
        out_args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]

    elif op == "extract_audio":
        out_args += ["-vn", "-q:a", "0"]  # codec inferred from output extension

    elif op == "mute":
        out_args += ["-an", "-c:v", "copy"]

    elif op == "reverse":
        vf.append("reverse")
        af.append("areverse")
        out_args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]

    elif op == "gif":
        fps = int(params.get("fps", 12) or 12)
        width = int(params.get("w", 480) or 480)
        vf.append(
            f"fps={fps},scale={width}:-1:flags=lanczos,"
            f"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
        )
        # palettegen/use is a complex graph → use -vf as filtergraph, no audio
        out_args += ["-an"]

    elif op == "compress":
        crf = int(params.get("crf", 26) or 26)
        out_args += ["-c:v", "libx264", "-crf", f"{crf}", "-preset", "slow", "-c:a", "aac", "-b:a", "128k"]

    else:  # convert — re-encode to target container with sane defaults
        out_args += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-c:a", "aac"]

    if vf:
        out_args = ["-vf", ",".join(vf)] + out_args
    if af:
        out_args = ["-af", ",".join(af)] + out_args
    return in_args, out_args


def edit_video(params: dict, progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """Run a single ffmpeg-based video edit with live progress.

    params: input_path, output_path, operation, + per-op fields (see _build_video_args).
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg non trovato (imageio-ffmpeg dovrebbe fornirlo).")

    inp = str(params["input_path"])
    out = str(params["output_path"])
    if not Path(inp).exists():
        raise FileNotFoundError(f"File non trovato: {inp}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    op = params.get("operation", "convert")
    duration = _probe_duration(inp)
    in_args, out_args = _build_video_args(params, inp, out)

    cmd = [ffmpeg, "-y", "-hide_banner", *in_args, "-i", inp,
           *out_args, "-progress", "pipe:1", "-nostats", out]

    log(f"Operazione: {op}")
    if duration:
        log(f"Durata sorgente: {duration:.1f}s")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1,
    )
    last_pct = -1
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            try:
                cur = int(line.split("=", 1)[1]) / 1_000_000.0
                if duration > 0:
                    pct = min(99, int(cur / duration * 100))
                    if pct != last_pct and pct % 2 == 0:
                        last_pct = pct
                        log(f"Avanzamento: {pct}% ({cur:.1f}/{duration:.1f}s)")
            except (ValueError, ZeroDivisionError):
                pass
        elif line and not line.startswith(("frame=", "fps=", "bitrate=", "total_size=",
                                            "out_time=", "dup_frames=", "drop_frames=",
                                            "speed=", "progress=", "stream_")):
            tail.append(line)
            tail[:] = tail[-8:]  # keep last 8 informative lines for error context

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError("ffmpeg fallito:\n" + "\n".join(tail[-6:]))
    log("Done.")
