"""Local server for the interactive rehearsal: serves the viewer, checks a user's move, runs it on the Modal session.

- GET  /api/scenes, /api/scene?id=   scene specs under configs/scenes/ and their simulation room for the viewer
- POST /api/feasible                 {scene, target, corrections}: destinations that pass the rehearsal layout checks
- POST /api/check                    {scene, target, corrections, destination}: the first strategy (carry, auto, push)
                                     whose rehearsal layout is valid, and optionally Muse Spark's plan for it
- POST /api/rehearse                 the checked request: written to configs/requests/<job>.json, copied to the running
                                     Modal session and started there over SSH (one job at a time)
- GET  /api/job?id=                  running (with the phases recorded so far) or done (result downloaded and exported
                                     to web/demos/<job>/ for replay)

The Meta API key never leaves the Mac except as the remote run's standard input, so it is not in any command line or
log. Bind to localhost only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
import http.server
import json
from pathlib import Path
import subprocess
from urllib.parse import parse_qs, urlparse

import export_replay
import isaac_g1_rehearsal as rehearsal
import meta_models
import rehearsal_advisor as advisor

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
    if not scene_id or "/" in scene_id or scene_id.startswith("."):
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
            "start": rehearsal.door_start(scene, config["start_inside_door_m"]).tolist(),
            "spec_transform": {"x": [1.0, width/2], "y": [-1.0, depth/2]},
            # Spec order [width along the front, depth, height] of the trained cabinet, and the carry mass range.
            "carry_tier": {"size_m": [carry_object[1], carry_object[0], carry_object[2]], "tolerance_m": 0.02,
                           "mass_kg": trainings["carry"][0]["mass_kg"]},
            "push_distance_m": config["push"]["distance_m"],
            "appearance": export_replay.appearance(scene, meta_models.MetaModels(transport=export_replay.cache_only))}


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


def feasible(body, config, trainings, step=0.15):
    """Destinations the item can be rehearsed to, each with its strategy: within the carry tier, floor points on a
    `step` grid with the grip frame's x axis pointing away from the room centre (the robot sets it down from the open
    side; the heading is turned back by the side carry_side grips it from); otherwise points straight along its four
    sides at the push distances, keeping its heading."""
    import numpy as np
    scene = json.loads(scene_path(body["scene"]).read_text())
    spec, base = scene["scene"], move_request(body)
    width, depth = spec["room"]["width_m"], spec["room"]["depth_m"]
    item = next(f for f in spec["furniture"] if f["id"] == body["target"])
    settings = json.loads(advisor.DEFAULT_CONFIG.read_text())
    candidates = []
    if advisor.carry_rule(base["corrections"]["size_m"], base["corrections"]["mass_kg"], settings)[0]:
        a = trainings["approach"][0]
        cabinet = np.r_[rehearsal.spec_to_sim(item["center_m"], width, depth), rehearsal.facing_yaw(item["yaw_deg"])]
        side = rehearsal.carry_side(rehearsal.spec_room(scene, body["target"])[0],
                                    rehearsal.door_start(scene, config["start_inside_door_m"]), cabinet, a["object"]["size_m"],
                                    0.411, a)
        turn = float(side[0][2] - cabinet[2]) if side is not None else 0.0
        for x in np.arange(step/2, width, step):
            for y in np.arange(step/2, depth, step):
                sim = rehearsal.spec_to_sim([x, y], width, depth)
                yaw = float(np.arctan2(sim[1], sim[0])) - turn
                candidates.append(([float(x), float(y)], float(np.degrees(np.arctan2(-np.cos(yaw), np.sin(yaw))))))
    else:
        low, high = config["push"]["distance_m"]
        start, theta = rehearsal.spec_to_sim(item["center_m"], width, depth), rehearsal.facing_yaw(item["yaw_deg"])
        for k in range(4):
            u = np.array([np.cos(theta + k*np.pi/2), np.sin(theta + k*np.pi/2)])
            for d in np.arange(low, high + 1e-9, 0.1):
                sim = start + d*u
                candidates.append(([float(sim[0] + width/2), float(depth/2 - sim[1])], float(item["yaw_deg"])))
    points = []
    for centre, yaw_deg in candidates:
        strategy, _ = choose_strategy({**base, "destination": {"center_m": centre, "yaw_deg": yaw_deg}}, scene, config, trainings)
        if strategy:
            points.append({"center_m": centre, "yaw_deg": yaw_deg, "strategy": strategy})
    return {"points": points, "candidates": len(candidates)}


def check(body, config, trainings, models=None):
    """Validate a user's move; with `ask_meta`, add Muse Spark's plan for a valid one (rules accept or override its
    choice)."""
    request = {**move_request(body), "destination": {"center_m": [float(v) for v in body["destination"]["center_m"]],
                                                     "yaw_deg": float(body["destination"]["yaw_deg"])}}
    scene = json.loads(scene_path(body["scene"]).read_text())
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
        if request.get("strategy") not in STRATEGIES:
            raise ValueError("Check the move first")
        job = "web-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        path = ROOT / "configs/requests" / f"{job}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request, indent=1) + "\n")
        relative = f"configs/requests/{job}.json"
        self.remote(f"mkdir -p {REMOTE_REPO}/configs/requests && cat > {REMOTE_REPO}/{relative}", path.read_bytes())
        # The key arrives on standard input and is exported only into the run's environment.
        self.remote(f"cd {REMOTE_REPO} && read -r META_API_KEY; export META_API_KEY; nohup {REMOTE_PYTHON} "
                    f"scripts/isaac_g1_rehearsal.py --request {relative} > outputs/isaac/{job}.out 2>&1 < /dev/null &",
                    (self.key() + "\n").encode())
        self.jobs[job] = {"status": "running", "request": request, "started": datetime.now().isoformat()}
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
        if not isinstance(summary, dict) or "run_dir" not in summary:
            newest = self.remote(f"ls -t {REMOTE_REPO}/outputs/isaac | grep -- '-rehearsal$' | head -1").strip()
            worker = self.remote(f"cat {REMOTE_REPO}/outputs/isaac/{newest}/rehearsal-worker.json 2>/dev/null || true")
            try:
                phases = list(json.loads(worker).get("phases", {})) if worker.strip() else []
            except ValueError:
                phases = []
            return {**state, "phases": phases}
        name = Path(summary["run_dir"]).name
        self.remote.fetch(name, ROOT / "outputs/isaac")
        state.update(status="done", summary=summary, run=name)
        if (ROOT / "outputs/isaac" / name / "replay.json").is_file():
            export_replay.export(ROOT / "outputs/isaac" / name, job)
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
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}") if method == "POST" else {}
            try:
                if url.path == "/api/scenes":
                    return self.reply([p.stem for p in sorted((ROOT / "configs/scenes").glob("*.json"))])
                if url.path == "/api/scene":
                    return self.reply(scene_view(query.get("id"), config, trainings))
                if url.path == "/api/feasible" and method == "POST":
                    return self.reply(feasible(body, config, trainings))
                if url.path == "/api/check" and method == "POST":
                    return self.reply(check(body, config, trainings))
                if url.path == "/api/rehearse" and method == "POST":
                    return self.reply({"job": jobs.start(body)})
                if url.path == "/api/job":
                    return self.reply(jobs.status(query.get("id")))
                return self.reply({"error": "not found"}, 404)
            except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
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
