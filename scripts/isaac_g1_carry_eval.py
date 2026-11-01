"""Evaluate a trained corridor carry checkpoint on held-out corridors and record pursuit-vs-learned showcase rollouts.

Two isolated Isaac processes run in order:
- `held_out`: corridors and cabinet masses drawn from a seed not used in training (same ranges); the pure-pursuit
  command and the checkpoint run one episode each from the same seeded starts, with the robot's walking speed logged.
- `showcase`: chosen corridors, each built twice, one driven by pursuit and one by the checkpoint; an oblique camera per
  environment records its first episode at 10 fps with a per-decision trace.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys
import traceback

import isaac_g1_carry as carry
import isaac_g1_ppo as ppo
import isaac_g1_ppo_eval as ppo_eval
import isaac_g1_rl as rl
from isaac_g1_wbc import ROOT, trial_contract
from isaac_preflight import check_versions, run_stage, write_json
from isaac_video import encode_recording, file_sha256, require_ffmpeg


DEFAULT_CONFIG = ROOT / "configs/isaac_g1_carry_eval.json"
STAGES = ("held_out", "showcase")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"training_config", "checkpoint", "held_out", "showcase", "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed carry evaluation configuration keys")
    training, profile, asset, runtime = carry.load_config(ROOT / config["training_config"])
    checkpoint = config["checkpoint"]
    if (not isinstance(checkpoint, str) or not checkpoint.startswith("outputs/isaac/") or
            not checkpoint.endswith(".pt") or ".." in Path(checkpoint).parts):
        raise ValueError("Checkpoint must be a run artifact under outputs/isaac/")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    def vector(value):
        return isinstance(value, list) and len(value) == 3 and all(type(v) in (int, float) for v in value)

    h, s, c, m = config["held_out"], config["showcase"], training["corridor"], training["mass_kg"]
    if (set(h) != {"num_envs", "corridor_seed", "start_seed"} or not all(type(v) is int for v in h.values()) or
            not 1 <= h["num_envs"] <= 4096 or h["corridor_seed"] == training["seed"]):
        raise ValueError("Held-out evaluation needs its own corridor seed")
    if (set(s) != {"cases", "start_seed", "env_spacing_m", "eye_from_corner_m", "target_from_corner_m"} or
            type(s["start_seed"]) is not int or not number(s["env_spacing_m"], training["env_spacing_m"], 20.0) or
            not isinstance(s["cases"], list) or not 1 <= len(s["cases"]) <= 4 or
            any(set(k) != {"width_m", "corner_m", "turn", "mass_kg"} or not number(k["width_m"], *c["width_m"]) or
                not number(k["corner_m"], *c["corner_m"]) or k["turn"] not in ("left", "right") or
                not number(k["mass_kg"], *m) for k in s["cases"]) or
            not vector(s["eye_from_corner_m"]) or not vector(s["target_from_corner_m"])):
        raise ValueError("Showcase corridors and masses must lie inside the training ranges")
    if type(config["stage_timeout_seconds"]) is not int or not 60 <= config["stage_timeout_seconds"] <= 3600:
        raise ValueError("Invalid stage_timeout_seconds")
    return config, training, profile, asset, runtime


def showcase_layout(cases):
    """Each case twice in a row: even environments follow pursuit, odd ones the checkpoint."""
    import numpy as np
    rows = [case for case in cases for _ in range(2)]
    corridors = {"width_m": np.array([k["width_m"] for k in rows], dtype=float),
                 "corner_m": np.array([k["corner_m"] for k in rows], dtype=float),
                 "turn": np.array([1.0 if k["turn"] == "left" else -1.0 for k in rows])}
    return corridors, np.array([k["mass_kg"] for k in rows], dtype=float), np.arange(len(rows)) % 2 == 1


def camera_views(origins, corridors, settings):
    """Camera eye and target per environment, given relative to its corridor corner with y mirrored for right turns."""
    import numpy as np

    def place(offset):
        o = np.asarray(offset, dtype=float)
        return origins + np.c_[corridors["corner_m"] + o[0], corridors["turn"]*o[1], np.full(len(origins), o[2])]

    return place(settings["eye_from_corner_m"]), place(settings["target_from_corner_m"])


def record_showcase(env, act, learned, settings, profile, asset, directory):
    """Run each environment's first episode on camera without auto-reset; one MP4 and trace per environment."""
    import numpy as np
    from PIL import Image
    s, n = env.s, env.num_envs
    encoder = require_ffmpeg(directory)
    camera, (width, height) = s.scene["camera"], (profile["camera"]["width"], profile["camera"]["height"])
    eyes, targets = camera_views(s.origins, env.corridors, settings)
    camera.set_world_poses_from_view(s.device(eyes), s.device(targets))
    env.reset_on_done = False
    env.reset_envs(np.arange(n), np.random.default_rng(settings["start_seed"]))
    env.obs = env.observe()
    cases = [directory / f"case_{i // 2}_{'learned' if learned[i] else 'pursuit'}" for i in range(n)]
    for case in cases:
        (case / "frames").mkdir(parents=True)
    frames, done = np.zeros(n, dtype=int), {}

    def capture(ids):
        for _ in range(2):
            s.sim.render()
            camera.update(asset["physics_dt"], force_recompute=True)
        rgb = s.batch(camera.data.output["rgb"])[..., :3]
        for i in ids:
            if rgb[i].shape != (height, width, 3) or float(rgb[i].std()) < 1:
                raise RuntimeError("Invalid showcase camera frame")
            Image.fromarray(rgb[i].astype(np.uint8)).save(cases[i] / "frames" / f"{frames[i]:06d}.png")
            frames[i] += 1

    capture(range(n))
    traces = [(case / "trace.jsonl").open("w") for case in cases]
    commands, speeds, clearance = [[] for _ in range(n)], [[] for _ in range(n)], np.full(n, np.inf)
    try:
        for _ in range(env.max_episode_length):
            active = [i for i in range(n) if i not in done]
            env.step(act(env.obs))
            robot = s.batch(s.robot.data.root_link_pose_w)
            cabinet = s.batch(s.box.data.root_link_pose_w)
            robot_yaw, cabinet_yaw = ppo.yaw_of(robot[:, 3:]), ppo.yaw_of(cabinet[:, 3:])
            task = env.last_task
            for i in active:
                clearance[i] = min(clearance[i], task["clearance_m"][i])
                if task["active"][i]:
                    commands[i].append(env.last_command[i])
                    speeds[i].append(task["speed_mps"][i])
                traces[i].write(json.dumps({
                    "time_s": float(env.length[i]*env.decision_dt), "lift_prefix": not bool(task["active"][i]),
                    "command": dict(zip(env.action_keys, map(float, env.last_command[i]))),
                    "speed_mps": float(task["speed_mps"][i]), "wall_clearance_m": float(task["clearance_m"][i]),
                    "cabinet_tilt_deg": float(np.degrees(task["tilt_rad"][i])),
                    "progress_m": float(env.previous_progress[i] - env.start_progress[i]),
                    "robot_xy_yaw": [*map(float, robot[i, :2] - s.origins[i, :2]), float(robot_yaw[i])],
                    "cabinet_xy_yaw": [*map(float, cabinet[i, :2] - s.origins[i, :2]), float(cabinet_yaw[i])]}) + "\n")
                if i in env.last_completed:
                    done[i] = env.last_completed[i]
            capture(active)
            if len(done) == n:
                break
    finally:
        for trace in traces:
            trace.close()
    results = []
    for i, case in enumerate(cases):
        video = encode_recording(case, binary=encoder, count=int(frames[i]), fps=1/Fraction(str(env.decision_dt)),
                                 width=width, height=height, require_motion=False)
        c = np.asarray(commands[i], dtype=float).reshape(-1, env.num_actions)
        results.append({"case": case.name, "policy": "learned" if learned[i] else "pursuit",
                        "width_m": float(env.corridors["width_m"][i]), "corner_m": float(env.corridors["corner_m"][i]),
                        "turn": "left" if env.corridors["turn"][i] > 0 else "right", "mass_kg": float(env.masses[i]),
                        **done[i], "min_wall_clearance_m": float(clearance[i]),
                        "speed_mps": {"mean": float(np.mean(speeds[i])), "max": float(np.max(speeds[i]))} if speeds[i] else None,
                        "commands": {"forward_mean_mps": float(c[:, 0].mean()) if len(c) else None,
                                     "yaw_rate_abs_mean_rps": float(np.abs(c[:, 1]).mean()) if len(c) else None,
                                     **({"lean_mean_rad": float(c[:, 2].mean()) if len(c) else None}
                                        if env.num_actions > 2 else {})},
                        "frames": int(frames[i]), "video": {k: video.get(k) for k in ("status", "decoded_frames", "sha256")}})
    return results


def run_worker(stage, config, training, profile, asset, runtime, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp

    contract, reference = trial_contract(profile)
    policy = ppo.gr00t_policy(contract, model_paths)
    checkpoint, showcase = ROOT / config["checkpoint"], stage == "showcase"
    if showcase:
        corridors, masses, learned = showcase_layout(config["showcase"]["cases"])
        start_seed, spacing = config["showcase"]["start_seed"], config["showcase"]["env_spacing_m"]
    else:
        h = config["held_out"]
        # Same sampling procedure as training (corridors, then masses), from a new seed.
        rng = np.random.default_rng(h["corridor_seed"])
        corridors = carry.sample_corridors(training["corridor"], h["num_envs"], rng)
        masses = rng.uniform(*training["mass_kg"], h["num_envs"])
        start_seed, spacing = h["start_seed"], training["env_spacing_m"]
    count = len(masses)
    torch.manual_seed(training["seed"])
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation

    scene_profile, built = carry.carry_scene(training, profile, asset, runtime, contract, reference, masses,
                                             num_envs=count, env_spacing=spacing, with_camera=showcase)
    evidence = directory / "eval-worker.json"
    result = {"status": "running", "stage": stage, "num_envs": count, "checkpoint": config["checkpoint"],
              "checkpoint_sha256": file_sha256(checkpoint), "start_seed": start_seed, "masses_kg": masses.tolist(),
              "corridors": {k: v.tolist() for k, v in corridors.items()}}
    write_json(evidence, result)
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "enable_cameras": showcase, "device": runtime["device"]}):
        try:
            env = carry.prepare_carry_env(built, scene_profile, training, profile, asset, runtime, contract, reference,
                                          policy, corridors, masses, directory, result, evidence)
            actor, result["checkpoint_iteration"] = ppo_eval.load_policy(env, training, checkpoint, runtime["device"])

            def pursuit(obs):
                return carry.pursuit_actions(obs["policy"].detach().cpu().numpy(), training)

            if showcase:
                def mixed(obs):
                    chosen = np.asarray(actor(obs).detach().cpu().numpy(), dtype=float)
                    return torch.as_tensor(np.where(learned[:, None], chosen, pursuit(obs)), dtype=torch.float32)

                with torch.inference_mode():
                    result["showcase"] = record_showcase(env, mixed, learned, config["showcase"], profile, asset, directory)
            else:
                result["scripted_eval"] = carry.evaluate(env, lambda obs: torch.as_tensor(pursuit(obs), dtype=torch.float32),
                                                         start_seed)
                write_json(evidence, result)
                with torch.inference_mode():
                    result["learned_eval"] = carry.evaluate(env, actor, start_seed)
            result["status"] = f"passed_{stage}"
            write_json(evidence, result)
        except BaseException as error:
            result.update(status="failed", error=str(error) or repr(error), error_type=type(error).__name__)
            write_json(evidence, result)
            (directory / "eval-error.log").write_text(traceback.format_exc())
            raise


def brief(evidence):
    """Headline numbers for the printed summary; full per-environment records stay in eval-worker.json."""
    if evidence.get("stage") == "showcase":
        return [{k: c.get(k) for k in ("case", "turn", "width_m", "mass_kg", "outcome", "seconds", "min_wall_clearance_m")} |
                {"max_speed_mps": (c.get("speed_mps") or {}).get("max")} for c in evidence.get("showcase", [])]
    out = {}
    for name in ("scripted_eval", "learned_eval"):
        e = evidence.get(name)
        if e:
            out[name] = {"success_rate": e["success_rate"], "outcome_rates": e["outcome_rates"],
                         "median_success_seconds": e["median_success_seconds"], "walking_speed_mps": e["walking_speed_mps"],
                         "group_success": {k: g["success_rate"] for k, g in e["groups"].items()}}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config, training, profile, asset, runtime = load_config(args.config)
    contract, _ = trial_contract(profile)
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "checkpoint": config["checkpoint"],
                          "held_out_envs": config["held_out"]["num_envs"], "showcase_cases": config["showcase"]["cases"],
                          "showcase_envs": 2*len(config["showcase"]["cases"]), "training": False}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(runtime))
        run_worker(args.worker, config, training, profile, asset, runtime, ppo.gr00t_models(contract), args.run_dir)
        return 0
    if not (ROOT / config["checkpoint"]).is_file():
        parser.error(f"Missing checkpoint: {config['checkpoint']}")
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-carry-eval")
    directory.mkdir(parents=True)
    for name, value in (("eval-config.json", config), ("carry-config.json", training), ("profile.json", profile),
                        ("runtime-config.json", runtime), ("controller-contract.json", contract)):
        write_json(directory / name, value)
    ppo.gr00t_models(contract)
    stages = {}
    for stage in STAGES:
        stage_dir = directory / stage
        stage_dir.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                   "--worker", stage, "--run-dir", str(stage_dir)]
        code = run_stage(command, stage_dir / "worker.log", config["stage_timeout_seconds"])
        path = stage_dir / "eval-worker.json"
        evidence = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        stages[stage] = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code,
                         "error": evidence.get("error"), "results": brief(evidence)}
    passed = all(stages[stage]["status"] == f"passed_{stage}" for stage in STAGES)
    summary = {"status": "passed_carry_evaluation" if passed else "failed", "run_dir": str(directory), **stages}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
