"""FastAPI server + CLI entry point for rvm-web."""
from __future__ import annotations

import asyncio
import json
import queue
import tempfile
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

from .processing import process_video

app = FastAPI(title="rvm-web")

_job_queue: queue.Queue[str] = queue.Queue()
_job_running = threading.Event()


class ProcessRequest(BaseModel):
    model_name: str = "mobilenetv3"
    input_source: str = ""
    output_composition: str = ""
    output_alpha: str = ""
    output_foreground: str = ""
    output_type: str = "video"
    downsample_ratio: str = "none"
    output_video_mbps: float = 4.0
    seq_chunk: int = 1
    num_workers: int = 0
    device: str = "auto"


class ImageRequest(BaseModel):
    input_path: str
    output_path: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    html = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    original_name = Path(file.filename).name if file.filename else "input.mp4"
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".mp4"

    # Save to ~/Downloads/rvm-uploads/ — works on Windows & macOS
    upload_dir = Path.home() / "Downloads" / "rvm-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / original_name
    # Avoid overwriting: append counter if exists
    counter = 1
    while dest.exists():
        dest = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    content = await file.read()
    dest.write_bytes(content)

    output_path = str(upload_dir / f"{dest.stem}_output{suffix}")
    return {"ok": True, "path": str(dest), "output_path": output_path}


@app.post("/api/process")
async def api_process(req: ProcessRequest) -> dict:
    if _job_running.is_set():
        return {"ok": False, "error": "A job is already running."}

    params = req.model_dump()
    params["downsample_ratio"] = None if params["downsample_ratio"] == "none" else float(params["downsample_ratio"])

    def run() -> None:
        _job_running.set()
        try:
            process_video(params, progress_cb=lambda msg: _job_queue.put(msg))
        except Exception as exc:
            _job_queue.put(f"ERROR: {exc}")
        finally:
            _job_running.clear()
            _job_queue.put("__done__")

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@app.get("/api/events")
async def api_events() -> StreamingResponse:
    async def generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: _job_queue.get(timeout=0.3))
                yield f"data: {json.dumps({'msg': msg})}\n\n"
                if msg == "__done__":
                    break
            except Exception:
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/status")
async def api_status() -> dict:
    return {"running": _job_running.is_set()}


@app.post("/api/upload-image")
async def api_upload_image(file: UploadFile = File(...)) -> dict:
    original_name = Path(file.filename).name if file.filename else "input.png"
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".png"

    upload_dir = Path.home() / "Downloads" / "rvm-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / original_name
    counter = 1
    while dest.exists():
        dest = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    content = await file.read()
    dest.write_bytes(content)

    output_path = str(upload_dir / f"{dest.stem}_nobg.png")
    return {"ok": True, "path": str(dest), "output_path": output_path}


@app.post("/api/process-image")
async def api_process_image(req: ImageRequest) -> dict:
    from rembg import remove

    input_path = Path(req.input_path)
    output_path = Path(req.output_path)

    if not input_path.exists():
        return {"ok": False, "error": f"File non trovato: {input_path}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "rb") as f:
        result = remove(f.read())

    output_path.write_bytes(result)
    return {"ok": True, "output_path": str(output_path)}


@app.get("/api/preview-image")
async def api_preview_image(path: str) -> FileResponse:
    p = Path(path)
    if not p.exists():
        from fastapi import HTTPException
        raise HTTPException(404, "File not found")
    return FileResponse(str(p), media_type="image/png")


def main() -> None:
    def open_browser() -> None:
        import time
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:7860")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")


if __name__ == "__main__":
    main()
