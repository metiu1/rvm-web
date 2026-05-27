"""FastAPI server + CLI entry point for rvm-web."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import webbrowser
from importlib.resources import files
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
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


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    html = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


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


def main() -> None:
    def open_browser() -> None:
        import time
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:7860")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")


if __name__ == "__main__":
    main()
