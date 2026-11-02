"""Copy a finished rehearsal run into web/demos/<name>/ for the WebGL viewer.

Writes replay.json (per-decision link and item poses, unchanged), rehearsal.json (request, layout, phase outcomes, lift
attempt, Meta advice, placement and the wall dimensions), scene.json (the scene spec), photo.png (the source room image
the scene was built from) and appearance.json: each detected furniture item's SAM 3.1 mask, taken from the cached Meta
API responses of the scene step (never a new call) and reduced to a grid over its pixel box, so the viewer can colour the
item from the photo pixels inside its mask. A sequence folder (scripts/rehearsal_sequence.py) is merged into one demo:
its runs' frames back to back, each tagged with the index of the item it moves, and rehearsal.json listing every move
over a display room without the moved items. Runs and images under outputs/ are Git-ignored; the exported demo is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import isaac_g1_rehearsal as rehearsal
import isaac_g1_room as room
import meta_models

ROOT = Path(__file__).resolve().parents[1]
WORKER_KEYS = ("status", "request", "mass_kg", "friction", "checkpoints", "phases", "grasp", "lift", "advice", "placement",
               "placement_goal", "key_frames")


GRID = 24


def cache_only(url, headers, body, stream):
    raise LookupError("not in the Meta API response cache")


def appearance(scene, models):
    """SAM 3.1 mask of each furniture item that has a detection, as GRID x GRID rows of 0/1 over its pixel box; items
    whose response is not cached are left out."""
    image = ROOT / scene["image"]
    detections = {d["id"]: d for d in scene.get("detections", [])}
    items = {}
    for item in scene["scene"]["furniture"]:
        detection = detections.get(item["detection_id"])
        if detection is None or not image.is_file():
            continue
        try:
            records = models.segment(image, detection["phrase"])["records"]
        except (LookupError, RuntimeError):
            continue
        prefix = detection["phrase"].replace(" ", "_")
        record = next((r for r in records if f"{prefix}_{r['id']}" == detection["id"] and "mask" in r), None)
        if record is None:
            continue
        mask = meta_models.mask_raster(record)
        h, w = mask.shape
        rows = ["".join(str(int(mask[int((i + 0.5)*h/GRID), int((j + 0.5)*w/GRID)])) for j in range(GRID))
                for i in range(GRID)]
        items[item["id"]] = {"detection_id": detection["id"], "box_px": detection["box_px"], "grid": rows}
    return {"image_px": next(iter(detections.values()))["image_px"] if detections else None, "items": items}


def grip_frame_heading(worker, scene):
    """Auto runs recorded before the grip-frame heading fix (2026-11-02) stored the destination heading in the item's
    front frame while the box was described in the grip frame (90 degrees apart), so their placement heading error was
    off by about 90 degrees. Rotate the destination into the grip frame and recompute the heading error and tolerance."""
    import numpy as np
    request, layout, placement = worker["request"], worker["layout"], worker.get("placement")
    if request["strategy"] != "auto":
        return worker
    target = next(f for f in scene["scene"]["furniture"] if f["id"] == request["target"])
    front = rehearsal.facing_yaw(request["destination"]["yaw_deg"])
    if abs(float(room.wrap(layout["placed"][2] - front))) > 1e-6:
        return worker                                        # already in the grip frame
    placed = [*layout["placed"][:2], float(room.wrap(front + layout["cabinet"][2] - rehearsal.facing_yaw(target["yaw_deg"])))]
    worker = {**worker, "layout": {**layout, "placed": placed, "heading_frame_corrected": True}}
    if placement:
        tolerance = room.load_config(ROOT / "configs/isaac_g1_room_carry.json")[0]["goal"]
        yaw = float(np.degrees(abs(room.wrap(placement["yaw_rad"] - placed[2]))))
        worker["placement"] = {**placement, "yaw_deg": yaw,
                               "within_tolerance": bool(placement["position_m"] <= tolerance["position_tolerance_m"] and
                                                        yaw <= tolerance["yaw_tolerance_deg"])}
    return worker


def export(run, name, out=ROOT / "web/demos", models=None):
    scene = json.loads((run / "scene.json").read_text())
    worker = grip_frame_heading(json.loads((run / "rehearsal-worker.json").read_text()), scene)
    walls = room.load_config(ROOT / "configs/isaac_g1_room_approach.json")[0]["room"]
    target = out / name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "replay.json", target / "replay.json")
    (target / "scene.json").write_text(json.dumps(scene, indent=1) + "\n")
    rehearsal = {"run": run.name, **{k: worker[k] for k in WORKER_KEYS if k in worker}, "layout": worker["layout"],
                 "walls": {"thickness_m": walls["wall_thickness_m"], "height_m": walls["wall_height_m"]}}
    (target / "rehearsal.json").write_text(json.dumps(rehearsal, indent=1, ensure_ascii=False) + "\n")
    photo = ROOT / scene["image"]
    if photo.is_file():
        shutil.copyfile(photo, target / "photo.png")
    looks = appearance(scene, models or meta_models.MetaModels(transport=cache_only))
    (target / "appearance.json").write_text(json.dumps(looks) + "\n")
    return target


def export_sequence(directory, name, out=ROOT / "web/demos", models=None, completed_only=False):
    """Merge a sequence's runs; with completed_only, only its completed moves (a stopped sequence's last attempt is left
    out and its item stays where it stood)."""
    record = json.loads((directory / "sequence.json").read_text())
    walls = room.load_config(ROOT / "configs/isaac_g1_room_approach.json")[0]["room"]
    frames, moves = [], []
    kept = [r for r in record["runs"] if r["run"] and (r.get("status") == "completed_rehearsal" or not completed_only)]
    for k, entry in enumerate(kept):
        run = directory / entry["run"]
        replay = json.loads((run / "replay.json").read_text())
        worker = json.loads((run / "rehearsal-worker.json").read_text())
        if k == 0:
            scene, first = json.loads((run / "scene.json").read_text()), worker
        start = len(frames)
        frames += [{**frame, "t": round((start + i)*replay["decision_dt"], 3), "item": k} for i, frame in enumerate(replay["frames"])]
        moves.append({"run": run.name, "target": entry["target"], "status": worker["status"], "request": worker["request"],
                      **{key: worker[key] for key in ("phases", "lift", "advice", "placement") if key in worker},
                      "layout": {key: worker["layout"][key] for key in ("cabinet", "placed", "size_m")},
                      "size_m": replay["cabinet_size_m"], "frames": [start, len(frames) - 1]})
    # Display room: the first run's room (corrected sizes, original poses) without the items the sequence moves.
    moved = {m["target"] for m in moves}
    display, _ = rehearsal.spec_room({**scene, "scene": {**scene["scene"], "furniture": [
        f for f in scene["scene"]["furniture"] if f["id"] not in moved]}})
    display = {**display, "boxes": display["boxes"].tolist()}
    target = out / name
    target.mkdir(parents=True, exist_ok=True)

    def write(file, data):
        (target / file).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")

    write("replay.json", {**{k: replay[k] for k in ("bodies", "decision_dt", "frame", "spec_transform")},
                          "cabinet_size_m": moves[0]["size_m"], "room": display, "items": [m["target"] for m in moves],
                          "frames": frames})
    # The spec as Meta estimated it (the runs' scene carries the corrected sizes and moved poses), for the side panel.
    spec = record.get("sequence", {}).get("scene")
    write("scene.json", json.loads((ROOT / spec).read_text()) if spec and (ROOT / spec).is_file() else scene)
    status = f"completed_{len(moves)}_of_{len(record['sequence']['moves'])}_moves" if completed_only else record["status"]
    # A left-out move is reported with how its last attempt ended (what the rehearsal found), not hidden.
    left_out = []
    for entry in record["runs"]:
        if entry in kept or not entry.get("run"):
            continue
        worker = json.loads((directory / entry["run"] / "rehearsal-worker.json").read_text())
        left_out.append({"target": entry["target"], "status": entry.get("status"), "attempts": len(entry.get("attempts") or [entry]),
                         "carry": (worker["phases"].get("carry") or {}).get("outcome"), "lift": worker.get("lift")})
    write("rehearsal.json", {"run": directory.name, "status": status, "request": moves[0]["request"], "left_out": left_out,
                             "moves": moves, "layout": {**first["layout"], "room": display, "placed": moves[0]["layout"]["placed"]},
                             "walls": {"thickness_m": walls["wall_thickness_m"], "height_m": walls["wall_height_m"]}})
    photo = ROOT / scene["image"]
    if photo.is_file():
        shutil.copyfile(photo, target / "photo.png")
    write("appearance.json", appearance(scene, models or meta_models.MetaModels(transport=cache_only)))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="outputs/isaac/<run>-rehearsal or outputs/isaac/<run>-sequence")
    parser.add_argument("name", help="demo folder name under web/demos/")
    parser.add_argument("--completed-only", action="store_true", help="sequence: only the moves that completed")
    args = parser.parse_args(argv)
    if (args.run / "sequence.json").exists():
        target = export_sequence(args.run, args.name, completed_only=args.completed_only)
    else:
        target = export(args.run, args.name)
    print(json.dumps({"demo": str(target), "files": sorted(p.name for p in target.iterdir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
