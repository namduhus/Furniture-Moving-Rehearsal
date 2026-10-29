"""Read-only, loopback camera observation over the existing SSH TCP connection."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import select
import socket
import threading
import time
from urllib.parse import urlsplit


MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
TERMINAL_STATES = {"finished", "failed", "interrupted"}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path, limit: int = MAX_SNAPSHOT_BYTES) -> dict:
    try:
        with path.open("rb") as source:
            payload = source.read(limit + 1)
        if len(payload) > limit:
            return {}
        value = json.loads(payload)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class ObservationPublisher:
    """Publish one atomic frame/metadata snapshot, independent of browser clients."""

    def __init__(self, directory: Path, max_fps: float = 10, source: str = "live"):
        if not math.isfinite(max_fps) or not 1 <= max_fps <= 30 or source not in {"live", "replay"}:
            raise ValueError("Invalid observation publication settings")
        self.directory = directory
        self.max_fps = max_fps
        self.source = source
        self.sequence = 0
        self.last_publish = -math.inf
        self.disabled = False
        self.phase("initializing")

    def phase(self, phase: str) -> None:
        try:
            atomic_json(self.directory / "observation-phase.json", {"phase": phase})
        except OSError as error:
            self._disable(error)

    def _disable(self, error: Exception) -> None:
        self.disabled = True
        message = f"Camera preview disabled; physics/recording continue: {error}"
        print(message, flush=True)
        try:
            (self.directory / "observation-error.log").write_text(message + "\n")
        except OSError:
            pass

    def publish_image(self, image, *, force: bool = False, **metadata) -> bool:
        if self.disabled or (not force and time.monotonic() - self.last_publish < 1 / self.max_fps):
            return False
        try:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            return self.publish_jpeg(buffer.getvalue(), force=force, **metadata)
        except (OSError, ValueError) as error:
            self._disable(error)
            return False

    def publish_jpeg(self, jpeg: bytes, *, step: int, sim_time_seconds: float, resets: int,
                     action_mode: str = "zero_action_runtime_check", force: bool = False) -> bool:
        if self.disabled or (not force and time.monotonic() - self.last_publish < 1 / self.max_fps):
            return False
        try:
            if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                raise ValueError("Invalid JPEG observation frame")
            if step < 0 or resets < 0 or not math.isfinite(sim_time_seconds) or sim_time_seconds < 0:
                raise ValueError("Invalid observation timestamp or state")
            self.sequence += 1
            snapshot = {"sequence": self.sequence, "step": step, "sim_time_seconds": sim_time_seconds,
                        "resets": resets, "action_mode": action_mode, "source": self.source,
                        "captured_unix_seconds": time.time(),
                        "jpeg_base64": base64.b64encode(jpeg).decode("ascii")}
            if len(json.dumps(snapshot)) > MAX_SNAPSHOT_BYTES:
                raise ValueError("Observation frame exceeds its bounded snapshot size")
            atomic_json(self.directory / "observation.json", snapshot)
            self.last_publish = time.monotonic()
            return True
        except (OSError, ValueError) as error:
            self._disable(error)
            return False


PAGE = b'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 Motion Observation</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111820;color:#e8eef4;font:16px system-ui,sans-serif}
main{max-width:1050px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{font-size:22px;margin:0}#state{color:#a7d5ff;font-size:14px}figure{margin:20px 0 12px;background:#202b36;border-radius:12px;overflow:hidden;min-height:220px}
img{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;color:#b8c7d6}
figcaption{padding:12px 16px;background:#1a2530;line-height:1.7;font-variant-numeric:tabular-nums}
#notice{color:#a8b8c8;min-height:24px}footer{color:#9baabd;font-size:13px;line-height:1.6;margin-top:24px}
.replay{color:#ffd58a!important}@media(max-width:600px){main{padding:14px}header{display:block}#state{margin-top:10px}}
</style>
<main><header><h1>G1 Motion Observation</h1><div id="state">Connecting...</div></header>
<figure><img id="camera" src="/stream.mjpg" alt="Waiting for the first camera frame">
<figcaption id="caption">The camera will appear after Isaac initialization.</figcaption></figure>
<div id="notice" role="status">Checking the run...</div>
<footer>Observation only. Closing this page does not stop the run or release the GPU.<br>
This viewer cannot confirm billing state. Save and stop the Modal session separately when work is finished.</footer></main>
<script>
const state=document.querySelector('#state'),caption=document.querySelector('#caption'),notice=document.querySelector('#notice'),camera=document.querySelector('#camera');
let terminal=false,streamBroken=false;
camera.onerror=()=>{streamBroken=true;};
async function poll(){
 const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),2500);
 try{
  const response=await fetch('/status.json',{cache:'no-store',signal:controller.signal});
  if(!response.ok)throw Error('Viewer response failed');
  const data=await response.json(),frame=data.frame||{};
  const replay=frame.source==='replay';state.className=replay?'replay':'';
  const labels={initializing:'Initializing Isaac',checking_gpu:'Checking GPU',running:'Receiving camera frames',stale:'Camera frame delayed',encoding:'Finalizing recording',closing:'Closing simulation',finished:'Run finished',failed:'Run failed',interrupted:'Run interrupted',preview_failed:'Preview unavailable'};
  state.textContent=(replay?'REPLAY - no live GPU | ':'')+(labels[data.view_status]||data.view_status);
  if(frame.sequence){
   const mode=frame.action_mode==='zero_action_runtime_check'?'ZERO ACTION - runtime check':frame.action_mode;
   caption.textContent=`${mode} | step ${frame.step} | simulated ${Number(frame.sim_time_seconds).toFixed(2)} s | resets ${frame.resets} | frame ${frame.sequence}`;
   notice.textContent=`Last captured ${Number(data.frame_age_seconds).toFixed(1)} s ago. `+(data.view_status==='finished'?'The run ended; the GPU session may still be active.':'');
  }else{notice.textContent='Waiting for camera initialization. Simulation does not require this browser to stay connected.';}
  if(data.preview_error)notice.textContent='Preview unavailable. Check observation-error.log; inspect the run result separately.';
  terminal=['finished','failed','interrupted'].includes(data.view_status);
  if(streamBroken&&!terminal&&!data.preview_error&&frame.sequence){camera.src='/stream.mjpg?retry='+Date.now();streamBroken=false;}
 }catch(error){state.textContent=terminal?'Viewer closed - run already ended':'Viewer disconnected';notice.textContent='Connection lost. This does not confirm that the run or GPU session stopped.';}
 finally{clearTimeout(timer);setTimeout(poll,500);}
}
poll();
</script></html>'''


class ObservationServer:
    def __init__(self, directory: Path, port: int = 8765, max_fps: float = 10,
                 stale_seconds: float = 3):
        self.directory = directory.resolve()
        if not self.directory.is_dir() or not 0 <= port <= 65535 or not 1 <= max_fps <= 30:
            raise ValueError("Invalid observation server configuration")
        if not math.isfinite(stale_seconds) or stale_seconds <= 0:
            raise ValueError("Invalid observation freshness threshold")
        self.max_fps = max_fps
        self.stale_seconds = stale_seconds
        self.stop = threading.Event()
        self.stream_slots = threading.BoundedSemaphore(4)
        self.connections_seen = 0
        self.frames_sent = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _headers(self, status, mime, length=None):
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def do_GET(self):
                path = urlsplit(self.path).path
                try:
                    if path == "/":
                        self._headers(200, "text/html; charset=utf-8", len(PAGE))
                        self.wfile.write(PAGE)
                    elif path == "/status.json":
                        payload = json.dumps(owner.state(), allow_nan=False).encode()
                        self._headers(200, "application/json", len(payload))
                        self.wfile.write(payload)
                    elif path == "/stream.mjpg":
                        self._stream()
                    elif path == "/frame.jpg":
                        snapshot = owner.snapshot()
                        if not snapshot:
                            self._headers(503, "text/plain", 0)
                        else:
                            jpeg = base64.b64decode(snapshot["jpeg_base64"], validate=True)
                            self._headers(200, "image/jpeg", len(jpeg))
                            self.wfile.write(jpeg)
                    else:
                        self._headers(404, "text/plain", 0)
                except (OSError, ValueError):
                    pass  # A viewer disconnect must not affect simulation.

            def _stream(self):
                if not owner.stream_slots.acquire(blocking=False):
                    self._headers(503, "text/plain", 0)
                    return
                owner.connections_seen += 1
                try:
                    self.connection.settimeout(2)
                    self._headers(200, "multipart/x-mixed-replace; boundary=frame")
                    sequence, last_send = -1, -math.inf
                    while not owner.stop.is_set():
                        readable, _, _ = select.select([self.connection], [], [], 0)
                        if readable and not self.connection.recv(1, socket.MSG_PEEK):
                            break  # Also release disconnected clients before the first frame.
                        snapshot = owner.snapshot()
                        if snapshot and (snapshot["sequence"] != sequence or time.monotonic() - last_send >= 1):
                            jpeg = base64.b64decode(snapshot["jpeg_base64"], validate=True)
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\nX-Frame-ID: "
                                             + str(snapshot["sequence"]).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                            owner.frames_sent += 1
                            sequence, last_send = snapshot["sequence"], time.monotonic()
                        if owner.state()["view_status"] in TERMINAL_STATES:
                            self.wfile.write(b"--frame--\r\n")
                            break
                        owner.stop.wait(1 / owner.max_fps)
                finally:
                    owner.stream_slots.release()

        self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.http.daemon_threads = True
        self.port = self.http.server_port
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def snapshot(self) -> dict:
        value = read_json(self.directory / "observation.json")
        try:
            if (type(value["sequence"]) is not int or value["sequence"] < 1
                    or not math.isfinite(value["captured_unix_seconds"])):
                return {}
            jpeg = base64.b64decode(value["jpeg_base64"], validate=True)
            if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                return {}
            return value
        except (KeyError, ValueError, TypeError):
            return {}

    def state(self) -> dict:
        summary = read_json(self.directory / "summary.json", 65536)
        snapshot = self.snapshot()
        phase = read_json(self.directory / "observation-phase.json", 65536).get("phase")
        run_status = summary.get("status", "running")
        age = max(0, time.time() - snapshot["captured_unix_seconds"]) if snapshot else None
        preview_error = (self.directory / "observation-error.log").is_file()
        if run_status.startswith("passed"):
            status = "finished"
        elif run_status in {"failed", "interrupted"}:
            status = run_status
        elif preview_error:
            status = "preview_failed"
        elif summary.get("current_stage") == "gpu":
            status = "checking_gpu"
        elif phase in {"encoding", "closing"}:
            status = phase
        elif not snapshot:
            status = "initializing"
        else:
            status = "stale" if age > self.stale_seconds else "running"
        return {"run_id": self.directory.name, "run_status": run_status, "view_status": status,
                "frame_age_seconds": age, "preview_error": preview_error,
                "frame": {key: value for key, value in snapshot.items() if key != "jpeg_base64"},
                "stream_connections_seen": self.connections_seen, "frames_sent": self.frames_sent}

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=3)

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 1800:
        parser.error("Viewer lifetime must be 1-1800 seconds")
    with ObservationServer(args.run_dir, args.port) as server:
        print(f"Observer: http://127.0.0.1:{server.port}; viewing does not stop/start a GPU session", flush=True)
        try:
            server.stop.wait(args.seconds)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
