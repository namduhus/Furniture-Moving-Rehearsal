"""Several moves in one room, one after another (for example three pedestals carried to new places).

Each move is its own rehearsal run (scripts/isaac_g1_rehearsal.py, a fresh Isaac process). The first starts inside the
door, or at the sequence's optional `start` (root pose [x, y, yaw], simulation frame); each later one starts at the root pose where the previous run left the robot, in a room where every item already
moved stands where it was measured after its set-down, and every item of the sequence has the user's corrected size
(request keys `start` and `furniture`; the sequence's optional `furniture` gives sizes of items it does not move). The
sequence stops at the first move that does not complete. Runs are written
under outputs/isaac/<time>-sequence/<k>/ next to sequence.json; scripts/export_replay.py merges them into one WebGL demo.
--dry-run plans every move on the Mac, assuming each item lands on its destination and the robot ends 0.5 m behind it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import isaac_g1_rehearsal as rehearsal

ROOT = Path(__file__).resolve().parents[1]
MOVE_KEYS = ("target", "corrections", "destination", "strategy")
CLEARANCE_M = 0.5                  # where the rehearsal's final step back leaves the robot (room inflation + 5 cm)


def load_sequence(path):
    sequence = json.loads(Path(path).read_text())
    moves = sequence.get("moves")
    if (not {"scene", "moves", "scope"} <= set(sequence) <= {"scene", "moves", "scope", "start", "furniture"} or
            not isinstance(moves, list) or not 1 <= len(moves) <= 4 or
            any(not isinstance(m, dict) or set(m) != set(MOVE_KEYS) for m in moves) or
            len({m["target"] for m in moves}) != len(moves)):
        raise ValueError("A sequence needs scene, scope and 1-4 moves with distinct targets, each with "
                         "target, corrections, destination and strategy (optional: start, furniture)")
    rehearsal.repo_path(sequence["scene"], "configs/scenes/", ".json")
    return sequence


def spec_yaw(sim_yaw):
    """Inverse of rehearsal.facing_yaw: simulation yaw of the item's x axis -> spec yaw in degrees."""
    import numpy as np
    return float((np.degrees(sim_yaw) - 90.0 + 180.0) % 360.0 - 180.0)


def move_request(sequence, k, moved, start):
    """Request of move k: corrected sizes of every item in the sequence, the measured poses of the items already moved
    and, after the first move, the robot's carried-over start."""
    furniture = {item: dict(v) for item, v in sequence.get("furniture", {}).items()}
    furniture.update({m["target"]: {"size_m": m["corrections"]["size_m"]} for m in sequence["moves"]})
    for target, pose in moved.items():
        furniture[target].update(pose)
    request = {"scene": sequence["scene"], **{key: sequence["moves"][k][key] for key in MOVE_KEYS}, "furniture": furniture}
    start = sequence.get("start") if k == 0 else start
    if start is not None:
        request["start"] = [float(v) for v in start]
    return request


def settled(run_dir):
    """Spec pose of the item as measured after its set-down, and the robot's last root pose (simulation frame)."""
    import numpy as np
    import isaac_g1_ppo as ppo
    placement = json.loads((run_dir / "rehearsal-worker.json").read_text())["placement"]
    replay = json.loads((run_dir / "replay.json").read_text())
    root = np.asarray(replay["frames"][-1]["bodies"][replay["bodies"].index("pelvis")], dtype=float)
    pose = {"center_m": [round(float(v), 4) for v in placement["cabinet_spec_xy_m"]],
            "yaw_deg": round(spec_yaw(placement["yaw_rad"]), 2)}
    return pose, [float(root[0]), float(root[1]), float(ppo.yaw_of(root[None, 3:])[0])]


def plan(sequence, config, trainings):
    """Layouts of every move, assuming each item lands on its destination and the robot ends CLEARANCE_M behind it,
    and the state after the last move: (items moved -> spec pose, robot start)."""
    import numpy as np
    base = json.loads(rehearsal.repo_path(sequence["scene"], "configs/scenes/", ".json").read_text())
    moved, start, layouts = {}, None, []
    for k, move in enumerate(sequence["moves"]):
        request = move_request(sequence, k, moved, start)
        scene = rehearsal.with_furniture(base, request["furniture"])
        layout = rehearsal.rehearsal_layout(scene, request, trainings, config["start_inside_door_m"], 0.411, config["push"])
        layouts.append(layout)
        moved[move["target"]] = dict(move["destination"])
        x, y, yaw = layout["placed"]
        behind = layout["size_m"][0]/2 + CLEARANCE_M
        start = [float(x - behind*np.cos(yaw)), float(y - behind*np.sin(yaw)), float(yaw)]
    return layouts, (moved, start)


def write_request(request, name):
    path = ROOT / "configs/requests" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=1) + "\n")
    return path


def rehearse(request_path, output_root):
    """One rehearsal run in a fresh process; its log goes to output_root/rehearsal.out."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "rehearsal.out").open("w") as log:
        return subprocess.run([sys.executable, str(ROOT / "scripts/isaac_g1_rehearsal.py"), "--request",
                               str(request_path.relative_to(ROOT)), "--output-root", str(output_root)],
                              stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def run(sequence, directory, runner=rehearse):
    moved, start, runs = {}, None, []
    record = {"status": "running", "sequence": sequence, "runs": runs}
    for k, move in enumerate(sequence["moves"]):
        path = write_request(move_request(sequence, k, moved, start), f"{directory.name}-{k}")
        code = runner(path, directory / str(k))
        found = sorted((directory / str(k)).glob("*-rehearsal"))
        summary = json.loads((found[-1] / "summary.json").read_text()) if found and (found[-1] / "summary.json").exists() else {}
        runs.append({"target": move["target"], "run": str(found[-1].relative_to(directory)) if found else None,
                     "exit_code": code, "status": summary.get("status", "failed_before_run"),
                     "placement": summary.get("placement")})
        if runs[-1]["status"] != "completed_rehearsal":
            record["status"] = f"stopped_at_move_{k}"
            break
        moved[move["target"]], start = settled(found[-1])
        runs[-1]["settled"] = {"item": moved[move["target"]], "robot_start_next": start}
        (directory / "sequence.json").write_text(json.dumps(record, indent=1) + "\n")
    else:
        record["status"] = "completed_sequence"
    (directory / "sequence.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=str, required=True, help="configs/... sequence JSON")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    args = parser.parse_args(argv)
    sequence = load_sequence(rehearsal.repo_path(args.sequence, "configs/", ".json"))
    config, trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
    layouts, _ = plan(sequence, config, trainings)
    if args.dry_run:
        import numpy as np
        print(json.dumps({"status": "local_plan_only", "moves": [
            {"target": m["target"], "strategy": m["strategy"], **{k: np.round(layout[k], 3).tolist() for k in
                                                                    ("start", "cabinet", "placed")}}
            for m, layout in zip(sequence["moves"], layouts)]}, indent=1))
        return 0
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-sequence")
    directory.mkdir(parents=True)
    record = run(sequence, directory)
    print(json.dumps({"status": record["status"], "sequence_dir": str(directory),
                      "runs": [{k: r[k] for k in ("target", "status", "placement")} for r in record["runs"]]}), flush=True)
    return 0 if record["status"] == "completed_sequence" else 1


if __name__ == "__main__":
    raise SystemExit(main())
