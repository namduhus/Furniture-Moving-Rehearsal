"""Check scene loading, CPU physics, contact, and offscreen RGB rendering."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import sys
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_png(path, frame):
    """Write MuJoCo's uint8 RGB array without an image-library dependency."""
    height, width, channels = frame.shape
    if channels != 3 or frame.dtype.name != "uint8":
        raise ValueError("Expected a uint8 RGB frame")

    def chunk(kind, payload):
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    scanlines = b"".join(b"\x00" + row.tobytes() for row in frame)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/smoke")
    parser.add_argument("--physics-only", action="store_true", help="Skip rendering for diagnosis")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.output_dir.resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "failed",
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "physics_device": "cpu",
        "render_backend_env": os.environ.get("MUJOCO_GL", "default"),
        "physics_only": args.physics_only,
        "checks": {},
    }
    try:
        config = json.loads(args.config.read_text())
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        scene = ROOT / config["scene"]
        shutil.copyfile(scene, run_dir / "scene.xml")
        summary["scene_sha256"] = hashlib.sha256(scene.read_bytes()).hexdigest()
        summary["seed"] = config["seed"]

        import mujoco

        summary["mujoco"] = mujoco.__version__
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = config["timestep_s"]
        model.vis.global_.offwidth = config["width"]
        model.vis.global_.offheight = config["height"]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        summary["checks"]["scene_loaded"] = True
        initial_height = float(data.body("box").xpos[2])
        floor_id = model.geom("floor").id
        box_id = model.geom("box_geom").id
        contact_seen = False
        finite = True
        warnings = 0
        with (run_dir / "trajectory.jsonl").open("w") as trajectory:
            for step in range(config["steps"]):
                mujoco.mj_step(model, data)
                finite = finite and all(math.isfinite(float(v)) for v in data.qpos)
                finite = finite and all(math.isfinite(float(v)) for v in data.qvel)
                contact_seen = contact_seen or any(
                    {int(c.geom1), int(c.geom2)} == {floor_id, box_id}
                    for c in data.contact
                )
                warnings = max(warnings, sum(int(w.number) for w in data.warning))
                trajectory.write(json.dumps({
                    "step": step + 1,
                    "time_s": float(data.time),
                    "qpos": data.qpos.tolist(),
                    "qvel": data.qvel.tolist(),
                    "contacts": int(data.ncon),
                }) + "\n")

        mujoco.mj_forward(model, data)
        final_height = float(data.body("box").xpos[2])
        expected_height = float(model.geom("box_geom").size[2])
        max_velocity = max(abs(float(v)) for v in data.qvel)
        summary.update({
            "steps": config["steps"],
            "simulation_time_s": float(data.time),
            "initial_box_height_m": initial_height,
            "final_box_height_m": final_height,
            "expected_rest_height_m": expected_height,
            "max_final_generalized_velocity": max_velocity,
            "warning_count": warnings,
        })
        summary["checks"].update({
            "finite_states": finite,
            "no_simulation_warnings": warnings == 0,
            "time_advanced": math.isclose(
                float(data.time), config["steps"] * config["timestep_s"], abs_tol=1e-8
            ),
            "box_fell": final_height < initial_height - 0.1,
            "floor_contact_observed": contact_seen,
            "box_resting_on_floor": abs(final_height - expected_height) < config["height_tolerance_m"],
            "box_settled": max_velocity < config["velocity_tolerance"],
        })

        if not args.physics_only:
            initial_data = mujoco.MjData(model)
            mujoco.mj_forward(model, initial_data)
            hashes = []
            with mujoco.Renderer(model, height=config["height"], width=config["width"]) as renderer:
                for label, state in (("initial", initial_data), ("final", data)):
                    renderer.update_scene(state, camera=config["camera"])
                    frame = renderer.render()
                    summary["checks"][f"{label}_frame_valid"] = (
                        frame.shape == (config["height"], config["width"], 3)
                        and int(frame.max()) > int(frame.min())
                    )
                    write_png(run_dir / f"{label}.png", frame)
                    hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            summary["checks"]["frames_changed"] = hashes[0] != hashes[1]

        if not all(summary["checks"].values()):
            raise RuntimeError("One or more smoke checks failed; inspect summary.json")
        summary["status"] = "passed_physics_only" if args.physics_only else "passed"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        (run_dir / "error.log").write_text(traceback.format_exc())
        print(summary["error"], file=sys.stderr)
    finally:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"status": summary["status"], "run_dir": str(run_dir)}))
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
