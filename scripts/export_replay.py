"""Copy a finished rehearsal run into web/demos/<name>/ for the WebGL viewer.

Writes replay.json (per-decision link and item poses, unchanged), rehearsal.json (request, layout, phase outcomes, lift
attempt, Meta advice, placement and the wall dimensions), scene.json (the scene spec) and photo.png (the source room
image the scene was built from). Runs and images under outputs/ are Git-ignored; the exported demo is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import isaac_g1_room as room

ROOT = Path(__file__).resolve().parents[1]
WORKER_KEYS = ("status", "request", "mass_kg", "friction", "checkpoints", "phases", "grasp", "lift", "advice", "placement",
               "placement_goal", "key_frames")


def export(run, name, out=ROOT / "web/demos"):
    worker = json.loads((run / "rehearsal-worker.json").read_text())
    scene = json.loads((run / "scene.json").read_text())
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
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="outputs/isaac/<run>-rehearsal")
    parser.add_argument("name", help="demo folder name under web/demos/")
    args = parser.parse_args(argv)
    target = export(args.run, args.name)
    print(json.dumps({"demo": str(target), "files": sorted(p.name for p in target.iterdir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
