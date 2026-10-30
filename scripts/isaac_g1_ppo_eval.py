"""Evaluate a trained PPO push checkpoint on held-out boxes and record showcase rollouts (RL stage 4 evidence).

Two isolated Isaac processes run in order:
- `held_out`: environments whose box mass/friction come from a seed not used in training; the scripted command and
  the checkpoint run one episode each from the same seeded starts.
- `showcase`: chosen boxes with one camera each; every high-level command, the robot and box poses and the sensed
  hand push are logged per decision, and each environment's first episode is encoded to MP4.
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

import isaac_g1_ppo as ppo
import isaac_g1_rl as rl
from isaac_g1_wbc import ROOT, trial_contract
from isaac_preflight import check_versions, run_stage, write_json
from isaac_video import encode_recording, file_sha256, require_ffmpeg


DEFAULT_CONFIG = ROOT / "configs/isaac_g1_ppo_eval.json"
STAGES = ("held_out", "showcase")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"training_config", "checkpoint", "held_out", "showcase", "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed PPO evaluation configuration keys")
    training, profile, asset, runtime = ppo.load_config(ROOT / config["training_config"])
    checkpoint = config["checkpoint"]
    if (not isinstance(checkpoint, str) or not checkpoint.startswith("outputs/isaac/") or
            not checkpoint.endswith(".pt") or ".." in Path(checkpoint).parts):
        raise ValueError("Checkpoint must be a run artifact under outputs/isaac/")
    h, s, r = config["held_out"], config["showcase"], training["randomization"]
    if (set(h) != {"num_envs", "box_seed", "start_seed"} or not all(type(v) is int for v in h.values()) or
            not 1 <= h["num_envs"] <= 4096 or h["box_seed"] == training["seed"]):
        raise ValueError("Held-out evaluation needs its own box seed")
    if (set(s) != {"cases", "start_seed", "env_spacing_m", "eye_offset_m", "target_offset_m"} or
            type(s["start_seed"]) is not int or type(s["env_spacing_m"]) not in (int, float) or
            not training["env_spacing_m"] <= s["env_spacing_m"] <= 20.0 or
            not isinstance(s["cases"], list) or not 1 <= len(s["cases"]) <= 8 or
            any(set(c) != {"mass_kg", "friction"} or not r["mass_kg"][0] <= c["mass_kg"] <= r["mass_kg"][1] or
                not r["friction"][0] <= c["friction"] <= r["friction"][1] for c in s["cases"]) or
            any(not isinstance(s[k], list) or len(s[k]) != 3 or any(type(v) not in (int, float) for v in s[k])
                for k in ("eye_offset_m", "target_offset_m"))):
        raise ValueError("Showcase boxes must lie inside the training ranges")
    if type(config["stage_timeout_seconds"]) is not int or not 60 <= config["stage_timeout_seconds"] <= 3600:
        raise ValueError("Invalid stage_timeout_seconds")
    return config, training, profile, asset, runtime


def load_policy(env, training, checkpoint, device):
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, ppo.train_config(training), log_dir=None, device=device)
    runner.load(str(checkpoint), map_location=device)
    return runner.get_inference_policy(device=device), runner.current_learning_iteration


def record_showcase(env, act, settings, profile, asset, directory):
    """Run each environment's first episode on camera without auto-reset; one MP4 and trace per environment."""
    import numpy as np
    from PIL import Image
    s, n = env.s, env.num_envs
    encoder = require_ffmpeg(directory)
    camera, (width, height) = s.scene["camera"], (profile["camera"]["width"], profile["camera"]["height"])
    camera.set_world_poses_from_view(s.device(s.origins + settings["eye_offset_m"]),
                                     s.device(s.origins + settings["target_offset_m"]))
    env.reset_on_done = False
    env.reset_envs(np.arange(n), np.random.default_rng(settings["start_seed"]))
    env.obs = env.observe()
    cases = [directory / f"case_{i}" for i in range(n)]
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
    commands, pushes = [[] for _ in range(n)], [[] for _ in range(n)]
    try:
        for _ in range(env.max_episode_length):
            active = [i for i in range(n) if i not in done]
            env.step(act(env.obs))
            push = ppo.hand_push(env)
            robot = s.batch(s.robot.data.root_link_pose_w)
            box = s.batch(s.box.data.root_link_pose_w)
            robot_yaw, box_yaw = ppo.yaw_of(robot[:, 3:]), ppo.yaw_of(box[:, 3:])
            for i in active:
                commands[i].append(env.last_command[i])
                pushes[i].append(push[i])
                traces[i].write(json.dumps({
                    "time_s": float(env.length[i]*env.decision_dt),
                    "command": dict(zip(ppo.ACTION_KEYS, map(float, env.last_command[i]))),
                    "hand_push_n": float(push[i]), "box_progress_m": float(box[i, 0] - env.start_x[i]),
                    "robot_xy_yaw": [*map(float, robot[i, :2] - s.origins[i, :2]), float(robot_yaw[i])],
                    "box_xy_yaw": [*map(float, box[i, :2] - s.origins[i, :2]), float(box_yaw[i])]}) + "\n")
            for i, record in env.last_completed.items():
                done.setdefault(i, record)
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
        results.append({"case": case.name, "mass_kg": float(env.masses[i]), "friction": float(env.frictions[i]),
                        "slide_force_n": float(env.masses[i]*env.frictions[i]*9.81), **done[i],
                        "commands": ppo.command_summary(commands[i], pushes[i]), "frames": int(frames[i]),
                        "video": {k: video.get(k) for k in ("status", "decoded_frames", "sha256")}})
    return results


def run_worker(stage, config, training, profile, asset, runtime, model_path, directory):
    import numpy as np
    import torch
    import warp as wp
    from g1_wbc import BalancePolicy

    contract, reference = trial_contract(profile)
    policy = BalancePolicy(model_path, contract)
    checkpoint, r = ROOT / config["checkpoint"], training["randomization"]
    if stage == "held_out":
        count, start_seed = config["held_out"]["num_envs"], config["held_out"]["start_seed"]
        rng = np.random.default_rng(config["held_out"]["box_seed"])
        masses, frictions = rng.uniform(*r["mass_kg"], count), rng.uniform(*r["friction"], count)
    else:
        cases = config["showcase"]["cases"]
        count, start_seed = len(cases), config["showcase"]["start_seed"]
        masses, frictions = np.array([c["mass_kg"] for c in cases]), np.array([c["friction"] for c in cases])
    torch.manual_seed(training["seed"])
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation

    showcase = stage == "showcase"
    # Showcase cameras stand 4 m beside their robot, so those environments are spaced wider apart.
    spacing = config["showcase"]["env_spacing_m"] if showcase else training["env_spacing_m"]
    built = rl.scene_config(profile, asset, runtime, contract, reference, with_camera=showcase, num_envs=count,
                            env_spacing=spacing, replicate_physics=False)
    evidence = directory / "eval-worker.json"
    result = {"status": "running", "stage": stage, "num_envs": count, "checkpoint": config["checkpoint"],
              "checkpoint_sha256": file_sha256(checkpoint), "start_seed": start_seed,
              "box_properties": {"masses_kg": masses.tolist(), "frictions": frictions.tolist()}}
    write_json(evidence, result)
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "enable_cameras": showcase, "device": runtime["device"]}):
        try:
            s = rl.prepare_push_scene(built, count, profile, asset, runtime, contract, reference, directory, result,
                                      evidence, (masses, frictions))
            from tensordict import TensorDict
            env = ppo.PushTaskEnv(s, policy, contract, profile, asset, runtime, training, masses, frictions,
                                  lambda groups: TensorDict(groups, batch_size=[count]))
            actor, result["checkpoint_iteration"] = load_policy(env, training, checkpoint, runtime["device"])
            if showcase:
                with torch.inference_mode():
                    result["showcase"] = record_showcase(env, actor, config["showcase"], profile, asset, directory)
            else:
                scripted = ppo.normalized_action([training["scripted_action"][k] for k in ppo.ACTION_KEYS],
                                                 training["action_bounds"])
                result["scripted_eval"] = ppo.evaluate(
                    env, lambda obs: torch.as_tensor(np.tile(scripted, (count, 1)), dtype=torch.float32), start_seed)
                write_json(evidence, result)
                with torch.inference_mode():
                    result["learned_eval"] = ppo.evaluate(env, actor, start_seed)
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
        return [{k: c.get(k) for k in ("case", "slide_force_n", "outcome", "progress_m", "seconds")} |
                {k: c["commands"][k] for k in ("forward_max_mps", "lean_mean_rad", "hand_push_mean_during_contact_n")}
                for c in evidence.get("showcase", [])]
    out = {}
    for name in ("scripted_eval", "learned_eval"):
        e = evidence.get(name)
        if e:
            out[name] = {"success_rate": e["success_rate"], "median_success_seconds": e["median_success_seconds"],
                         "by_slide_force": {k: g["success_rate"] for k, g in e["by_slide_force"].items()},
                         "forward_command": e["forward_command"]}
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
                          "policy_role": contract["policy_role"], "training": False}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    from g1_wbc import prepare_model
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(runtime))
        run_worker(args.worker, config, training, profile, asset, runtime, prepare_model(contract), args.run_dir)
        return 0
    if not (ROOT / config["checkpoint"]).is_file():
        parser.error(f"Missing checkpoint: {config['checkpoint']}")
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-ppo-eval")
    directory.mkdir(parents=True)
    for name, value in (("eval-config.json", config), ("ppo-config.json", training), ("profile.json", profile),
                        ("runtime-config.json", runtime), ("controller-contract.json", contract)):
        write_json(directory / name, value)
    prepare_model(contract)
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
    summary = {"status": "passed_ppo_evaluation" if passed else "failed", "run_dir": str(directory), **stages}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
