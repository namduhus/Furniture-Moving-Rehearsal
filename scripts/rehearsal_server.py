"""Local server for the interactive rehearsal: serves the viewer, checks a user's move, runs it on the Modal session.

- GET  /api/scenes, /api/scene?id=   scene specs under configs/scenes/ and their simulation room for the viewer
- POST /api/feasible                 {scene, target, corrections, previous, alike}: destinations that pass the rehearsal
                                     layout checks, after the moves already queued in `previous` (checked requests), with
                                     the corrected size also given to the `alike` items
- POST /api/check                    {scene, target, corrections, destination, previous}: the first strategy (carry, auto,
                                     push) whose rehearsal layout is valid, and optionally Muse Spark's plan for it
- POST /api/rehearse                 a checked request, or {moves: [checked requests]} for several moves in one room
                                     (scripts/rehearsal_sequence.py): written to configs/requests/<job>.json, copied to
                                     the running Modal session and started there over SSH (one job at a time)
- GET  /api/job?id=                  running (with the phases recorded so far) or done (result downloaded and exported
                                     to web/demos/<job>/ for replay)
- POST /api/voice                    {scene, audio_base64}: a spoken move (WAV) -> Muse Voice Transcribe text -> Muse Spark
                                     picks the item and where it goes relative to another item -> a spot in the room
- GET /api/upload-options            call/cost estimate and door-reference defaults for browser image conversion
- POST /api/convert-room             confirmed image + reference -> a locally registered, editable 3D room estimate

The Meta API key never leaves the Mac except as the remote run's standard input, so it is not in any command line or
log. Bind to localhost only.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
from functools import partial
import http.server
import json
from pathlib import Path
import re
import shlex
import subprocess
from urllib.parse import parse_qs, urlparse

import export_replay
import isaac_g1_rehearsal as rehearsal
import meta_models
import rehearsal_advisor as advisor
import rehearsal_sequence as sequence
import room_upload

ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/workspace/repo/Furniture-Moving-Rehearsal"
REMOTE_PYTHON = "/opt/isaaclab/.venv/bin/python"
STRATEGIES = ("carry", "auto", "push")


def latest_ssh_config():
    configs = sorted((ROOT / "outputs/modal").glob("*/ssh_config"), key=lambda p: p.stat().st_mtime)
    if not configs:
        raise FileNotFoundError("No Modal ssh_config under outputs/modal/; start the Modal session first")
    return configs[-1]


class Remote:
    """Commands on the running Modal session through the launcher's `fmr-modal` SSH alias (the newest session's
    ssh_config unless one is given)."""

    def __init__(self, config=None, run=subprocess.run):
        self.config, self.run = config, run

    def ssh(self, command):
        return ["ssh", "-F", str(self.config or latest_ssh_config()), "fmr-modal", command]

    def __call__(self, command, stdin=b"", timeout=120):
        return self.run(self.ssh(command), input=stdin, capture_output=True, timeout=timeout, check=True).stdout.decode()

    def fetch(self, name, out):
        """Download outputs/isaac/<name> (without frames) into `out`."""
        archive = self.run(self.ssh(f"tar -C {REMOTE_REPO}/outputs/isaac -czf - --exclude=frames {name}"),
                           capture_output=True, timeout=600, check=True).stdout
        self.run(["tar", "-xzf", "-", "-C", str(out)], input=archive, capture_output=True, timeout=600, check=True)


def scene_path(scene_id):
    if not isinstance(scene_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", scene_id):
        raise ValueError(f"Bad scene id: {scene_id}")
    path = ROOT / "configs/scenes" / f"{scene_id}.json"
    if not path.is_file():
        raise ValueError(f"Unknown scene: {scene_id}")
    return path


def scene_view(scene_id, config, trainings):
    """What the viewer needs to draw a scene and let the user pick an item and a destination (simulation frame)."""
    scene = json.loads(scene_path(scene_id).read_text())
    spec = scene["scene"]
    width, depth = spec["room"]["width_m"], spec["room"]["depth_m"]
    room, ids = rehearsal.spec_room(scene)
    walls = trainings["approach"][0]["room"]
    carry_object = trainings["carry"][0]["object"]["size_m"]
    return {"id": scene_id, "photo": f"/{scene['image']}" if (ROOT / scene["image"]).is_file() else None,
            "room": {"width_m": width, "depth_m": depth, "boxes": room["boxes"].tolist(), "ids": ids},
            "walls": {"thickness_m": walls["wall_thickness_m"], "height_m": walls["wall_height_m"]},
            "furniture": [{**{k: f[k] for k in ("id", "category", "size_m", "estimated_mass_kg", "yaw_deg", "center_m")},
                           "sim": [*rehearsal.spec_to_sim(f["center_m"], width, depth).tolist(), rehearsal.facing_yaw(f["yaw_deg"])]}
                          for f in spec["furniture"]],
            "start": rehearsal.entry_start(scene, config["start_inside_door_m"], walls["start_clearance_m"]).tolist(),
            "spec_transform": {"x": [1.0, width/2], "y": [-1.0, depth/2]},
            # Spec order [width along the front, depth, height] of the trained cabinet, and the carry mass range.
            "carry_tier": {"size_m": [carry_object[1], carry_object[0], carry_object[2]], "tolerance_m": 0.02,
                           "mass_kg": trainings["carry"][0]["mass_kg"]},
            "push_distance_m": config["push"]["distance_m"],
            # SAM 3.1 detections, the door used as the scale reference, for the viewer's "how Meta built this room".
            "detections": [{k: d[k] for k in ("id", "phrase", "box_px", "image_px", "confidence")} for d in scene.get("detections", [])],
            "door": spec["door"], "reference": scene.get("reference"),
            "detection_ids": {f["id"]: f["detection_id"] for f in spec["furniture"]},
            "notes": spec.get("notes", ""), "warnings": scene.get("checks_failed", []), "uploaded": scene_id.startswith("upload-"),
            "appearance": scene.get("appearance") or export_replay.appearance(scene, meta_models.MetaModels(transport=export_replay.cache_only))}


def choose_strategy(request, scene, config, trainings):
    """The first of carry, auto (try to lift, then push), push whose rehearsal layout is valid, with the reasons the
    others were rejected; (None, reasons) when none is."""
    reasons = {}
    for strategy in STRATEGIES:
        try:
            rehearsal.rehearsal_layout(scene, {**request, "strategy": strategy}, trainings, config["start_inside_door_m"],
                                       0.411, config["push"])
            return strategy, reasons
        except ValueError as error:
            reasons[strategy] = str(error)
    return None, reasons


def move_request(body):
    return {"scene": f"configs/scenes/{body['scene']}.json", "target": body["target"],
            "corrections": {"size_m": [float(v) for v in body["corrections"]["size_m"]],
                            "mass_kg": float(body["corrections"]["mass_kg"])}}


def static_sizes(requests, targets):
    """Corrected sizes the user gave to items that are not moved (the same size as an item of their kind), from the
    requests' furniture overrides."""
    return {item: {"size_m": v["size_m"]} for r in requests for item, v in (r.get("furniture") or {}).items()
            if item not in targets and "size_m" in v}


def context(body, config, trainings):
    """Request fields (without destination and strategy) and scene for the user's next move. Items listed in
    body["alike"] take the corrected size. After the moves queued in body["previous"] (checked requests, each assumed to
    land on its destination, as rehearsal_sequence.plan does), the items they moved stand there with their corrected
    sizes and the robot starts where the last one should leave it."""
    request, previous = move_request(body), body.get("previous") or []
    scene = json.loads(scene_path(body["scene"]).read_text())
    targets = {m["target"] for m in previous}
    if request["target"] in targets:
        raise ValueError(f"{request['target']} is already in the queued moves")
    alike = {"furniture": {item: {"size_m": request["corrections"]["size_m"]} for item in body.get("alike") or []}}
    furniture = static_sizes(previous + [alike], targets | {request["target"]})
    if previous:
        queued = {"scene": request["scene"], "moves": [{k: m[k] for k in sequence.MOVE_KEYS} for m in previous],
                  "scope": "Try it", "furniture": furniture}
        _, (moved, start) = sequence.plan(queued, config, trainings)
        furniture = {**furniture, **{m["target"]: {"size_m": m["corrections"]["size_m"], **moved[m["target"]]} for m in previous}}
        request = {**request, "start": start}
    return ({**request, "furniture": furniture} if furniture else request), rehearsal.with_furniture(scene, furniture)


def feasible(body, config, trainings, step=0.15):
    """Destinations the item can be rehearsed to, each with its strategy: within the carry tier, floor points on a
    `step` grid with the grip frame's x axis pointing away from the room centre, squared to the walls (the robot sets it
    down from the open side; the heading is turned back by the side carry_side grips it from; a turned item's
    axis-aligned obstacle box would grow and crowd the next move); otherwise points straight along its four sides at the
    push distances, keeping its heading."""
    import numpy as np
    base, scene = context(body, config, trainings)
    spec = scene["scene"]
    width, depth = spec["room"]["width_m"], spec["room"]["depth_m"]
    item = next(f for f in spec["furniture"] if f["id"] == body["target"])
    settings = json.loads(advisor.DEFAULT_CONFIG.read_text())
    candidates = []
    if advisor.carry_rule(base["corrections"]["size_m"], base["corrections"]["mass_kg"], settings)[0]:
        a = trainings["approach"][0]
        cabinet = np.r_[rehearsal.spec_to_sim(item["center_m"], width, depth), rehearsal.facing_yaw(item["yaw_deg"])]
        start = (np.asarray(base["start"]) if "start" in base else
                 rehearsal.entry_start(scene, config["start_inside_door_m"], a["room"]["start_clearance_m"]))
        side = rehearsal.carry_side(rehearsal.spec_room(scene, body["target"])[0], start, cabinet, a["object"]["size_m"], 0.411, a)
        turn = float(side[0][2] - cabinet[2]) if side is not None else 0.0
        for x in np.arange(step/2, width, step):
            for y in np.arange(step/2, depth, step):
                sim = rehearsal.spec_to_sim([x, y], width, depth)
                yaw = float(round(np.arctan2(sim[1], sim[0])/(np.pi/2))*np.pi/2) - turn
                candidates.append(([float(x), float(y)], float(np.degrees(np.arctan2(-np.cos(yaw), np.sin(yaw))))))
    else:
        low, high = config["push"]["distance_m"]
        start, theta = rehearsal.spec_to_sim(item["center_m"], width, depth), rehearsal.facing_yaw(item["yaw_deg"])
        for k in range(4):
            u = np.array([np.cos(theta + k*np.pi/2), np.sin(theta + k*np.pi/2)])
            for d in np.arange(low, high + 1e-9, 0.1):
                sim = start + d*u
                candidates.append(([float(sim[0] + width/2), float(depth/2 - sim[1])], float(item["yaw_deg"])))
    points, reasons = [], {}
    for centre, yaw_deg in candidates:
        strategy, why = choose_strategy({**base, "destination": {"center_m": centre, "yaw_deg": yaw_deg}}, scene, config, trainings)
        if strategy:
            points.append({"center_m": centre, "yaw_deg": yaw_deg, "strategy": strategy})
        elif not reasons:
            reasons = why                          # why the first candidate failed, shown when none passes
    return {"points": points, "candidates": len(candidates), "reasons": {} if points else reasons}


VOICE_SCHEMA = {
    "title": "MoveCommand", "type": "object", "additionalProperties": False,
    "required": ["target_id", "anchor_id", "relation", "explanation"],
    "properties": {"target_id": {"type": "string"}, "anchor_id": {"type": "string"},
                   "relation": {"type": "string", "enum": ["next_to", "in_front_of", "behind", "left_of", "right_of", "room_center"]},
                   "explanation": {"type": "string"}}}
VOICE_PROMPT = """Someone who cannot move furniture themselves asked a home robot, in their own words: "{text}"
Furniture in the room (id: kind, centre [x, y] m, size [width, depth, height] m). x runs to the right along the back
wall, y toward the camera; "in front of" an item is toward the camera, "behind" toward the back wall, left and right as
seen from the camera.
{listing}
The room is {width} m wide and {depth} m deep. Pick the item to move (target_id): if the request does not say which of
several similar items, pick the one easiest for a robot to reach (farthest from the walls and other furniture). Pick
where it should go: relative to another item (anchor_id with next_to, in_front_of, behind, left_of or right_of) or the
middle of the room (relation room_center, anchor_id ""). Use only ids from the list. explanation: one short sentence in
the language the person used, saying what will be moved where."""
# Recognition hints: the viewer is used in Korean and English (without them a Korean request came back as French).
VOICE_LANGUAGES = ["Korean", "English"]


def check_wav(data):
    """The clip Muse Voice Transcribe accepts: RIFF/WAVE, mono 16-bit PCM at 16 or 24 kHz, at most 10 minutes."""
    import struct
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE" or data[12:16] != b"fmt ":
        raise ValueError("The recording is not a WAV file")
    fmt, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[20:36])
    if (fmt, channels, bits) != (1, 1, 16) or rate not in (16000, 24000):
        raise ValueError("The recording must be mono 16-bit PCM at 16 or 24 kHz")
    if len(data) - 44 > 10*60*rate*2:
        raise ValueError("The recording is longer than 10 minutes")


def voice_spot(spec, command):
    """Spec point for a spoken destination: past the anchor's footprint (its rotated extent) plus 0.45 m in the named
    direction, the anchor's centre for next_to (the viewer snaps to the nearest checked destination around it), or the
    room centre."""
    import numpy as np
    width, depth = spec["room"]["width_m"], spec["room"]["depth_m"]
    items = {f["id"]: f for f in spec["furniture"]}
    anchor = items.get(command["anchor_id"])
    if command["relation"] == "room_center" or anchor is None:
        return [width/2, depth/2]
    theta = np.radians(anchor["yaw_deg"])
    half = [(abs(np.cos(theta))*anchor["size_m"][0] + abs(np.sin(theta))*anchor["size_m"][1])/2,
            (abs(np.sin(theta))*anchor["size_m"][0] + abs(np.cos(theta))*anchor["size_m"][1])/2]
    dx, dy = {"next_to": (0, 0), "in_front_of": (0, 1), "behind": (0, -1), "left_of": (-1, 0), "right_of": (1, 0)}[command["relation"]]
    x, y = anchor["center_m"]
    return [float(np.clip(x + dx*(half[0] + 0.45), 0.0, width)), float(np.clip(y + dy*(half[1] + 0.45), 0.0, depth))]


def voice_move(body, models=None):
    """A spoken move: Muse Voice Transcribe turns the clip into text, Muse Spark picks the item and where it should go
    relative to another item, and voice_spot turns that into a point of the room."""
    wav = base64.b64decode(body["audio_base64"])
    check_wav(wav)
    models = models or meta_models.MetaModels()
    text = models.transcribe(wav, language=VOICE_LANGUAGES)["text"].strip()
    if not text:
        raise ValueError("No speech was recognised; try again a little closer to the microphone")
    spec = json.loads(scene_path(body["scene"]).read_text())["scene"]
    listing = "\n".join(f"- {f['id']}: {f['category']}, centre {f['center_m']}, size {f['size_m']}" for f in spec["furniture"])
    settings = json.loads(advisor.DEFAULT_CONFIG.read_text())
    prompt = VOICE_PROMPT.format(text=text, listing=listing, width=spec["room"]["width_m"], depth=spec["room"]["depth_m"])
    # A short choice: low reasoning effort answered the same in 6.9 s instead of 17 s (2026-11-02).
    command = models.chat(settings["chat_model"], [{"role": "user", "content": prompt}], schema=VOICE_SCHEMA,
                          reasoning_effort="low")["content"]
    if command["target_id"] not in {f["id"] for f in spec["furniture"]}:
        raise ValueError(f"Could not tell which item to move from: {text}")
    return {"transcript": text, **command, "center_m": voice_spot(spec, command)}


def check(body, config, trainings, models=None):
    """Validate a user's move; with `ask_meta`, add Muse Spark's plan for a valid one (rules accept or override its
    choice)."""
    base, scene = context(body, config, trainings)
    request = {**base, "destination": {"center_m": [float(v) for v in body["destination"]["center_m"]],
                                       "yaw_deg": float(body["destination"]["yaw_deg"])}}
    strategy, reasons = choose_strategy(request, scene, config, trainings)
    result = {"ok": strategy is not None, "strategy": strategy, "reasons": reasons,
              "request": {**request, "strategy": strategy} if strategy else None}
    if body.get("ask_meta") and strategy:
        try:
            settings = json.loads(advisor.DEFAULT_CONFIG.read_text())
            result["advice"] = advisor.advise(models or meta_models.MetaModels(), settings, request, scene)
        except Exception as error:                 # the plan is optional; report why it is missing
            result["advice"] = {"error": f"{type(error).__name__}: {error}"}
    return result


class Jobs:
    """Rehearsals started on the Modal session from this server (one at a time)."""

    def __init__(self, remote, key=lambda: ""):
        self.remote, self.key, self.jobs = remote, key, {}

    def start(self, request):
        if any(job["status"] == "running" for job in self.jobs.values()):
            raise RuntimeError("A rehearsal is already running")
        moves = request.get("moves") if "moves" in request else None
        if any(m.get("strategy") not in STRATEGIES for m in (moves or [request])) or moves == []:
            raise ValueError("Check the move first")
        job = "web-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if moves is not None:
            # Several moves: a sequence spec; each move runs from the measured end of the previous one.
            request = {"scene": moves[0]["scene"], "moves": [{k: m[k] for k in sequence.MOVE_KEYS} for m in moves],
                       "scope": f"Try it: {len(moves)} moves queued in the viewer",
                       "furniture": static_sizes(moves, {m["target"] for m in moves})}
        path = ROOT / "configs/requests" / f"{job}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request, indent=1) + "\n")
        relative = f"configs/requests/{job}.json"
        # Uploaded scenes are local/ignored; send their immutable spec and photo with the requested rehearsal.
        scene_relative = request.get("scene", "")
        if re.fullmatch(r"configs/scenes/upload-[a-f0-9]{32}\.json", scene_relative):
            uploaded = ROOT / scene_relative
            spec = json.loads(uploaded.read_text())
            image_relative = spec["image"]
            if not re.fullmatch(r"outputs/uploads/upload-[a-f0-9]{32}/photo\.png", image_relative):
                raise ValueError("Invalid uploaded scene image")
            for name in (scene_relative, image_relative):
                parent = shlex.quote(f"{REMOTE_REPO}/{Path(name).parent}")
                destination = shlex.quote(f"{REMOTE_REPO}/{name}")
                self.remote(f"mkdir -p {parent} && cat > {destination}", (ROOT / name).read_bytes())
        self.remote(f"mkdir -p {REMOTE_REPO}/configs/requests && cat > {REMOTE_REPO}/{relative}", path.read_bytes())
        runner = (f"scripts/rehearsal_sequence.py --sequence {relative}" if moves is not None else
                  f"scripts/isaac_g1_rehearsal.py --request {relative}")
        # The key arrives on standard input and is exported only into the run's environment.
        self.remote(f"cd {REMOTE_REPO} && read -r META_API_KEY; export META_API_KEY; nohup {REMOTE_PYTHON} "
                    f"{runner} > outputs/isaac/{job}.out 2>&1 < /dev/null &", (self.key() + "\n").encode())
        # Run folders are named by their UTC start; only those from this job on count (30 s for clock skew).
        since = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime("%Y%m%dT%H%M%S")
        self.jobs[job] = {"status": "running", "request": request, "started": datetime.now().isoformat(),
                          "moves": len(moves) if moves is not None else 1, "since": since}
        return job

    def status(self, job):
        if job not in self.jobs:
            raise ValueError(f"Unknown job: {job}")
        state = self.jobs[job]
        if state["status"] != "running":
            return state
        last = self.remote(f"tail -n 1 {REMOTE_REPO}/outputs/isaac/{job}.out 2>/dev/null || true").strip()
        try:
            summary = json.loads(last)
        except ValueError:
            summary = None
        done = "sequence_dir" if "moves" in state["request"] else "run_dir"
        if not isinstance(summary, dict) or done not in summary:
            # No result and no process left: the run ended early (e.g. the session has no Isaac runtime); show its log.
            alive = self.remote(f"pgrep -f 'requests/{job}.json' >/dev/null && echo alive || echo stopped").strip()
            if alive == "stopped":
                log = self.remote(f"tail -n 5 {REMOTE_REPO}/outputs/isaac/{job}.out 2>/dev/null || true").strip()
                state.update(status="failed", error=log or "the remote run stopped without output")
                return state
            # This job's newest run (inside its sequence folder for several moves) and the phases it has recorded; an
            # older run (the previous job's) is not shown while Isaac starts. Folder names start with the UTC time.
            def newest(pattern):
                found = [line.strip() for line in self.remote(f"ls -1d {pattern} 2>/dev/null || true").splitlines()]
                found = [path for path in found if path and Path(path).name >= state.get("since", "")]
                return max(found, key=lambda path: Path(path).name) if found else ""
            folder = newest(f"{REMOTE_REPO}/outputs/isaac/*-sequence") if done == "sequence_dir" else None
            newest = (newest(f"{folder}/*/*-rehearsal") if folder else "") if done == "sequence_dir" else \
                newest(f"{REMOTE_REPO}/outputs/isaac/*-rehearsal")
            worker = self.remote(f"cat {newest}/rehearsal-worker.json 2>/dev/null || true") if newest else ""
            try:
                phases = list(json.loads(worker).get("phases", {})) if worker.strip() else []
            except ValueError:
                phases = []
            move = int(Path(newest).parent.name) + 1 if done == "sequence_dir" and Path(newest).parent.name.isdigit() else 1
            return {**state, "phases": phases, "move": move}
        name = Path(summary[done]).name
        self.remote.fetch(name, ROOT / "outputs/isaac")
        summary.setdefault("status", "unknown")
        state.update(status="done", summary=summary, run=name)
        local = ROOT / "outputs/isaac" / name
        if done == "sequence_dir" and any(r.get("run") for r in json.loads((local / "sequence.json").read_text())["runs"]):
            export_replay.export_sequence(local, job)
            state["demo"] = job
        elif done == "run_dir" and (local / "replay.json").is_file():
            export_replay.export(local, job)
            state["demo"] = job
        return state


def handler(config, trainings, jobs):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def reply(self, payload, code=200):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def api(self, method):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                size = int(self.headers.get("Content-Length", 0))
                if size < 0 or size > (9 * 1024 * 1024 if url.path == "/api/convert-room" else 40 * 1024 * 1024):
                    return self.reply({"error": "Request is too large"}, 413)
                if url.path == "/api/convert-room" and self.headers.get("Origin"):
                    origin = urlparse(self.headers["Origin"])
                    if origin.scheme != "http" or origin.netloc != self.headers.get("Host"):
                        return self.reply({"error": "Open the upload form on this local server"}, 403)
                body = json.loads(self.rfile.read(size) or b"{}") if method == "POST" else {}
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if url.path == "/api/scenes":
                    return self.reply([p.stem for p in sorted((ROOT / "configs/scenes").glob("*.json"))])
                if url.path == "/api/scene":
                    return self.reply(scene_view(query.get("id"), config, trainings))
                if url.path == "/api/upload-options" and method == "GET":
                    return self.reply(room_upload.estimate())
                if url.path == "/api/convert-room" and method == "POST":
                    return self.reply(room_upload.convert(body))
                if url.path == "/api/feasible" and method == "POST":
                    return self.reply(feasible(body, config, trainings))
                if url.path == "/api/check" and method == "POST":
                    return self.reply(check(body, config, trainings))
                if url.path == "/api/rehearse" and method == "POST":
                    return self.reply({"job": jobs.start(body)})
                if url.path == "/api/voice" and method == "POST":
                    return self.reply(voice_move(body))
                if url.path == "/api/job":
                    return self.reply(jobs.status(query.get("id")))
                return self.reply({"error": "not found"}, 404)
            except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
                return self.reply({"error": str(error)}, 400)

        def do_GET(self):
            if self.path.startswith("/api/"):
                return self.api("GET")
            return super().do_GET()

        def do_POST(self):
            return self.api("POST")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    config, trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)

    def key():
        try:
            return meta_models.api_key("META_API_KEY")
        except Exception:
            return ""

    jobs = Jobs(Remote(), key)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port),
                                             partial(handler(config, trainings, jobs), directory=str(ROOT)))
    print(f"Open http://127.0.0.1:{args.port}/web/  (Ctrl+C to stop)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
