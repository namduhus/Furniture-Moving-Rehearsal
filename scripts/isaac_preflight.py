"""Bounded CUDA/Vulkan and Isaac Lab G1 checks for the pinned Linux runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/isaac_smoke.json"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config["physics"] != "isaacsim_physx" or config["device"] != "cuda:0":
        raise ValueError("This preflight requires Isaac Sim PhysX on CUDA device 0")
    if config["num_envs"] != 1:
        raise ValueError("The first G1 smoke check uses exactly one environment")
    for key, limit in (("steps", 256), ("stage_timeout_seconds", 1200)):
        if type(config[key]) is not int or not 1 <= config[key] <= limit:
            raise ValueError(f"Invalid {key}")
    for key in ("width", "height"):
        if type(config["camera"][key]) is not int or not 16 <= config["camera"][key] <= 640:
            raise ValueError("Invalid smoke camera size")
    if not re.fullmatch(r"[0-9a-f]{40}", config["isaaclab_commit"]):
        raise ValueError("Invalid source pin")
    return config


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def driver_version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)+", value.strip()):
        raise ValueError("Invalid NVIDIA driver version")
    return tuple(int(part) for part in value.strip().split("."))


def validate_vulkan(text: str) -> None:
    required = ("NVIDIA", "L40S", "VK_KHR_acceleration_structure", "VK_KHR_ray_tracing_pipeline")
    missing = [term for term in required if term.lower() not in text.lower()]
    if missing:
        raise RuntimeError("Vulkan NVIDIA/RTX support unavailable: " + ", ".join(missing)
                           + ". CUDA visibility alone is insufficient; inspect the worker's graphics runtime.")


def check_versions(config: dict) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Run inside the Modal Linux x86_64 Isaac image")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Isaac Lab 3.0 requires Python 3.12")
    actual = {name: importlib.metadata.version(name) for name in config["versions"]}
    for name, wanted in config["versions"].items():
        if actual[name].split("+", 1)[0] != wanted:
            raise RuntimeError(f"Expected {name}=={wanted}, found {actual[name]}")
    checkout = Path(config["runtime_path"])
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if commit != config["isaaclab_commit"]:
        raise RuntimeError("The Isaac Lab checkout differs from the reviewed commit")
    digest = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
    if digest != config["isaaclab_lock_sha256"]:
        raise RuntimeError("The upstream dependency lock was modified")
    return {"packages": actual, "python": sys.version, "isaaclab_commit": commit,
            "upstream_lock_sha256": digest}


def gpu_check(config: dict, directory: Path) -> dict:
    import torch

    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    (directory / "gpu_inventory.txt").write_text(inventory + "\n")
    rows = inventory.splitlines()
    if len(rows) != 1 or "L40S" not in rows[0]:
        raise RuntimeError("Expected exactly one visible L40S")
    driver = rows[0].split(",")[1].strip()
    if driver_version(driver) < driver_version(config["minimum_driver"]):
        raise RuntimeError("Worker driver is below the reviewed release requirement")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one usable CUDA device")
    if torch.version.cuda != config["torch_cuda"]:
        raise RuntimeError(f"Expected CUDA {config['torch_cuda']} PyTorch wheels")
    if "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Unexpected CUDA device")
    matrix = torch.ones((32, 32), device="cuda:0")
    product = matrix @ matrix
    torch.cuda.synchronize()
    if not torch.all(product == 32).item():
        raise RuntimeError("CUDA matrix computation failed")
    result = subprocess.run(["vulkaninfo"], capture_output=True, text=True, timeout=45)
    (directory / "vulkaninfo.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("vulkaninfo failed; inspect vulkaninfo.txt. No driver changes were attempted.")
    validate_vulkan(result.stdout)
    return {"driver": driver, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "cuda_matmul": "passed", "vulkan_nvidia_rtx": "passed"}


def g1_check(config: dict, directory: Path) -> None:
    import torch
    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401: register the pinned release's task IDs
    from isaaclab_tasks.utils import parse_env_cfg
    from PIL import Image

    torch.manual_seed(config["seed"])
    cfg = parse_env_cfg(config["task"], device=config["device"], num_envs=1,
                        overrides=["physics=isaacsim_physx"])
    cfg.seed = config["seed"]
    if not isinstance(cfg.sim.physics, PhysxCfg):
        raise RuntimeError("Task configuration did not select Isaac Sim PhysX")
    cfg.scene.fmr_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FmrCamera", update_period=0,
        width=config["camera"]["width"], height=config["camera"]["height"],
        data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                       clipping_range=(0.1, 100.0)),
    )
    evidence = {"status": "running", "task": config["task"], "physics": config["physics"],
                "seed": config["seed"], "num_envs": 1, "steps_completed": 0}
    write_json(directory / "g1-worker.json", evidence)

    def tensor(value):
        return value.torch if hasattr(value, "torch") else value

    def check_finite(value):
        if isinstance(value, Mapping):
            for child in value.values():
                check_finite(child)
        else:
            value = tensor(value)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not torch.isfinite(value).all().item():
                    raise RuntimeError("Non-finite G1 state, observation or reward")

    def save_frame(camera, filename: str):
        rgb = tensor(camera.data.output["rgb"])[0, :, :, :3].detach().cpu().numpy()
        if rgb.shape != (config["camera"]["height"], config["camera"]["width"], 3):
            raise RuntimeError("Unexpected camera frame shape")
        if float(rgb.std()) < 1.0:
            raise RuntimeError("Rendered frame is blank or near-uniform")
        Image.fromarray(rgb.astype("uint8")).save(directory / filename)
        return {"shape": list(rgb.shape), "std": float(rgb.std())}

    # The selected physics and camera explicitly require Kit; no Newton fallback.
    with launch_simulation(cfg, {"device": "cuda:0", "visualizer": [],
                                 "enable_cameras": True, "kit_args": "--allow-root"}):
        env = gym.make(config["task"], cfg=cfg)
        try:
            observation, _ = env.reset(seed=config["seed"])
            check_finite(observation)
            backend = env.unwrapped.sim.physics_manager
            if not backend.__module__.startswith("isaaclab_physx."):
                raise RuntimeError("The running physics manager is not Isaac Sim PhysX")
            evidence["physics_manager"] = backend.__module__ + "." + backend.__name__
            robot = env.unwrapped.scene["robot"]
            if not tensor(robot.data.joint_pos).is_cuda:
                raise RuntimeError("Robot joint state is not on CUDA")
            camera = env.unwrapped.scene["fmr_camera"]
            camera.set_world_poses_from_view(
                torch.tensor([[2.5, 2.5, 1.7]], device="cuda:0"),
                torch.tensor([[0.0, 0.0, 0.7]], device="cuda:0"),
            )
            for _ in range(8):
                env.unwrapped.sim.render()
                camera.update(cfg.sim.dt, force_recompute=True)
            evidence["initial_frame"] = save_frame(camera, "initial.png")
            actions = torch.zeros(env.action_space.shape, device="cuda:0")
            resets = 0
            with (directory / "trajectory.jsonl").open("w") as trajectory:
                for step in range(config["steps"]):
                    observation, reward, terminated, truncated, _ = env.step(actions)
                    check_finite(observation)
                    check_finite(reward)
                    check_finite(robot.data.joint_pos)
                    root_state = tensor(robot.data.root_state_w)
                    check_finite(root_state)
                    resets += int(torch.logical_or(terminated, truncated).sum().item())
                    trajectory.write(json.dumps({"step": step + 1,
                        "root_state": root_state.detach().cpu().tolist(),
                        "reward": reward.detach().cpu().tolist(), "resets_total": resets}) + "\n")
                    evidence["steps_completed"] = step + 1
            torch.cuda.synchronize()
            for _ in range(4):
                env.unwrapped.sim.render()
                camera.update(cfg.sim.dt, force_recompute=True)
            evidence["final_frame"] = save_frame(camera, "final.png")
            evidence["resets"] = resets
            evidence["interpretation"] = "Zero actions validate runtime/finite states, not learned walking."
        finally:
            env.close()
        # Kit may exit from its close method. Parent still requires a clean process exit.
        evidence["status"] = "checks_completed_before_kit_shutdown"
        write_json(directory / "g1-worker.json", evidence)


def run_worker(stage: str, config: dict, directory: Path) -> int:
    try:
        versions = check_versions(config)
        write_json(directory / "versions.json", versions)
        if stage == "gpu":
            evidence = gpu_check(config, directory)
            write_json(directory / "gpu-worker.json", {"status": "checks_completed", **evidence})
        else:
            g1_check(config, directory)
        return 0
    except BaseException as error:
        # Some Kit versions raise SystemExit(0) during close(). Accept it only
        # after the full G1 evidence has already been written and checked.
        if stage == "g1" and isinstance(error, SystemExit) and error.code in (None, 0):
            try:
                evidence = json.loads((directory / "g1-worker.json").read_text())
                validate_evidence(stage, evidence, directory, config)
            except (OSError, ValueError, RuntimeError):
                pass
            else:
                return 0
        (directory / f"{stage}-error.log").write_text(traceback.format_exc())
        write_json(directory / f"{stage}-worker.json", {"status": "failed", "error": str(error)})
        raise


def run_stage(command: list[str], log_path: Path, timeout: int) -> int:
    """Bound the entire Kit process group, including child processes and shutdown."""
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                import os
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
            raise


def validate_evidence(stage: str, evidence: dict, directory: Path, config: dict) -> None:
    wanted = "checks_completed" if stage == "gpu" else "checks_completed_before_kit_shutdown"
    if evidence.get("status") != wanted:
        raise RuntimeError(f"{stage} exited without complete evidence")
    if stage == "g1":
        if evidence.get("steps_completed") != config["steps"]:
            raise RuntimeError("G1 exited before the requested number of steps")
        for name in ("initial.png", "final.png"):
            if not (directory / name).is_file() or (directory / name).stat().st_size < 50:
                raise RuntimeError(f"Missing G1 rendering evidence: {name}")
        with (directory / "trajectory.jsonl").open() as trajectory:
            if sum(1 for _ in trajectory) != config["steps"]:
                raise RuntimeError("Incomplete G1 trajectory evidence")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("all", "gpu", "g1"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/isaac"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "config": config,
                          "stages": ["gpu", "g1"] if args.stage == "all" else [args.stage]}, indent=2))
        return 0
    if args.worker:
        if args.stage == "all" or args.run_dir is None:
            parser.error("Internal workers require one stage and a run directory")
        return run_worker(args.stage, config, args.run_dir)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Use --dry-run on Mac; execute inside the Modal Isaac image")
    directory = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    summary = {"status": "running", "stages": {}, "run_dir": str(directory)}
    write_json(directory / "summary.json", summary)
    try:
        # Even a G1-only request must first validate CUDA and Vulkan in this process environment.
        stages = ["gpu", "g1"] if args.stage in ("all", "g1") else ["gpu"]
        for stage in stages:
            print(f"Running {stage}; log: {directory / (stage + '.log')}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--stage", stage,
                       "--config", str(args.config.resolve()), "--run-dir", str(directory)]
            code = run_stage(command, directory / f"{stage}.log", config["stage_timeout_seconds"])
            worker = directory / f"{stage}-worker.json"
            evidence = json.loads(worker.read_text()) if worker.exists() else {}
            if code != 0:
                raise RuntimeError(f"{stage} failed (exit {code}); inspect {stage}.log and worker evidence")
            validate_evidence(stage, evidence, directory, config)
            summary["stages"][stage] = "passed"
            write_json(directory / "summary.json", summary)
        summary["status"] = "passed_g1_smoke" if "g1" in stages else "passed_gpu_graphics_smoke"
        return 0
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error_type"] = type(error).__name__
        (directory / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        write_json(directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
