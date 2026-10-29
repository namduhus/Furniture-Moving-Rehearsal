"""Inspect the floating-base G1 asset before connecting a pretrained controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import sys
import traceback

from isaac_preflight import ROOT, check_versions, gpu_check, load_config, run_stage, write_json


DEFAULT_PROFILE = ROOT / "configs/isaac_g1_asset.json"


def load_profile(path: Path) -> tuple[dict, dict]:
    profile = json.loads(path.read_text())
    if profile["asset_config"] != "G1_29DOF_CFG":
        raise ValueError("Only the reviewed floating-base G1 candidate is supported")
    groups = profile["joint_groups"]
    if set(groups) != {"lower_body", "arms", "hands"}:
        raise ValueError("Expected lower-body, arm and hand groups")
    for key, count in (("lower_body", 15), ("arms", 14), ("hands", 14)):
        if len(groups[key]) != count:
            raise ValueError(f"Invalid {key} joint count")
    names = sum(groups.values(), [])
    if len(set(names)) != 43 or any(not re.fullmatch(r"[a-z0-9_]+", n) for n in names):
        raise ValueError("Joint names must be valid and unique")
    controller = profile["controller_reference"]
    if controller["joint_order"] != groups["lower_body"] + groups["arms"]:
        raise ValueError("Controller body order differs from the reviewed robot order")
    if (controller["action_dim"], controller["single_observation_dim"],
            controller["history_length"], controller["observation_dim"]) != (15, 86, 6, 516):
        raise ValueError("Unexpected controller observation/action contract")
    if controller["weights_loaded"] is not False:
        raise ValueError("This inventory profile does not load controller weights")
    if profile["physics_dt"] != 0.005 or profile["control_decimation"] != 4:
        raise ValueError("The reviewed controller uses 200 Hz physics and 50 Hz control")
    if type(profile["stage_timeout_seconds"]) is not int or not 1 <= profile["stage_timeout_seconds"] <= 600:
        raise ValueError("Asset inspection must be bounded to at most ten minutes per stage")
    runtime_path = (ROOT / profile["runtime_config"]).resolve()
    if runtime_path != ROOT / "configs/isaac_smoke.json":
        raise ValueError("Use the existing reviewed Isaac runtime configuration")
    return profile, load_config(runtime_path)


def validate_inventory(inventory: dict, profile: dict) -> dict:
    """Reject structural mismatches without claiming controller or skill success."""
    expected = sum(profile["joint_groups"].values(), [])
    names = inventory["joint_names"]
    if len(names) != len(set(names)):
        raise RuntimeError("The asset contains duplicate joint names")
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    if missing or extra:
        raise RuntimeError(f"Asset joint mismatch: missing={missing}, extra={extra}")
    if inventory["is_fixed_base"] is not False:
        raise RuntimeError("A fixed-base asset cannot validate mobile G1 control")
    if inventory["gravity_disabled"] is not False:
        raise RuntimeError("Gravity must be enabled for the mobile asset")
    if inventory["joint_state_on_cuda"] is not True:
        raise RuntimeError("The initialized joint state is not on CUDA")
    if not inventory["physics_manager"].startswith("isaaclab_physx."):
        raise RuntimeError("Expected the reviewed Isaac Sim PhysX backend")
    for field in ("joint_positions", "default_joint_positions"):
        values = inventory[field]
        if len(values) != len(names) or not all(math.isfinite(v) for v in values):
            raise RuntimeError(f"Invalid {field}")
    limits = inventory["joint_position_limits"]
    if len(limits) != len(names):
        raise RuntimeError("Incomplete joint limit evidence")
    for name, value, bounds in zip(names, inventory["default_joint_positions"], limits, strict=True):
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
            raise RuntimeError(f"Invalid position limits for {name}")
        if not bounds[0] - 1e-5 <= value <= bounds[1] + 1e-5:
            raise RuntimeError(f"Default position outside limits for {name}")
    return {
        "status": "passed_asset_inventory",
        "source_to_sim_joint_indices": [names.index(n) for n in expected],
        "controller_body_to_sim_joint_indices": [names.index(n) for n in profile["controller_reference"]["joint_order"]],
        "joint_group_sizes": {k: len(v) for k, v in profile["joint_groups"].items()},
        "controller_inference_verified": False,
        "balance_or_walking_verified": False,
        "interpretation": "Names, limits, CUDA storage and mobile-base structure only; joint axes, dynamics and policy execution still require validation.",
    }


def inspect_asset(profile: dict, runtime: dict, directory: Path) -> None:
    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass, instantiate, replace
    from isaaclab_assets import G1_29DOF_CFG
    from isaaclab_physx.physics import PhysxCfg

    @configclass
    class AssetSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        robot = replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")

    scene_cfg = AssetSceneCfg(num_envs=1, env_spacing=3.0)
    if scene_cfg.robot.spawn.fix_root_link is not False:
        raise RuntimeError("The selected asset configuration does not declare a mobile base")
    if scene_cfg.robot.spawn.rigid_props.disable_gravity is not False:
        raise RuntimeError("The selected asset configuration disables gravity")
    sim_cfg = sim_utils.SimulationCfg(
        device=runtime["device"], dt=profile["physics_dt"], physics=PhysxCfg(),
    )
    with launch_simulation(sim_cfg, {"headless": True, "device": runtime["device"],
                                    "kit_args": "--allow-root"}):
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        sim.reset()
        robot = scene["robot"]

        def tensor(value):
            return value.torch if hasattr(value, "torch") else value

        def row(value):
            return tensor(value)[0].detach().cpu().tolist()

        inventory = {
            "asset_config": profile["asset_config"],
            "usd_path": scene_cfg.robot.spawn.usd_path,
            "configured_root_position": list(scene_cfg.robot.init_state.pos),
            "configured_root_rotation_wxyz": list(scene_cfg.robot.init_state.rot),
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "is_fixed_base": bool(robot.is_fixed_base),
            "gravity_disabled": scene_cfg.robot.spawn.rigid_props.disable_gravity,
            "joint_state_on_cuda": bool(tensor(robot.data.joint_pos).is_cuda),
            "physics_manager": sim.physics_manager.__module__ + "." + sim.physics_manager.__name__,
            "joint_positions": row(robot.data.joint_pos),
            "default_joint_positions": row(robot.data.default_joint_pos),
            "joint_position_limits": row(robot.data.joint_pos_limits),
            "actuator_configuration": {k: v.to_dict() for k, v in scene_cfg.robot.actuators.items()},
            "simulation_steps_requested": 0,
            "controller_weights_loaded": False,
        }
        # Write evidence before validation so mismatched USDs remain diagnosable.
        write_json(directory / "asset-inventory.json", inventory)
        result = validate_inventory(inventory, profile)
        write_json(directory / "asset-worker.json", result)


def validate_worker(stage: str, directory: Path, profile: dict) -> None:
    evidence = json.loads((directory / f"{stage}-worker.json").read_text())
    wanted = "checks_completed" if stage == "gpu" else "passed_asset_inventory"
    if evidence.get("status") != wanted:
        raise RuntimeError(f"{stage} exited without complete evidence")
    if stage == "asset":
        inventory = json.loads((directory / "asset-inventory.json").read_text())
        if validate_inventory(inventory, profile) != evidence:
            raise RuntimeError("Asset validation evidence differs from inventory")


def worker(stage: str, profile: dict, runtime: dict, directory: Path) -> int:
    try:
        write_json(directory / "versions.json", check_versions(runtime))
        if stage == "gpu":
            write_json(directory / "gpu-worker.json", {"status": "checks_completed", **gpu_check(runtime, directory)})
        else:
            inspect_asset(profile, runtime, directory)
        return 0
    except BaseException as error:
        if stage == "asset" and isinstance(error, SystemExit) and error.code in (None, 0):
            try:
                validate_worker(stage, directory, profile)
            except (OSError, ValueError, RuntimeError):
                pass
            else:
                return 0
        (directory / f"{stage}-error.log").write_text(traceback.format_exc())
        write_json(directory / f"{stage}-worker.json", {"status": "failed", "error": str(error)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", choices=("gpu", "asset"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    profile, runtime = load_profile(args.profile)
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "asset": profile["asset_config"],
                          "stages": ["gpu", "asset"], "joint_group_sizes": {
                              k: len(v) for k, v in profile["joint_groups"].items()},
                          "output_root": str(args.output_root.expanduser().resolve()),
                          "controller_reference": profile["controller_reference"],
                          "scope": "Initialize and inspect one floating-base asset; no policy inference, control rollout or training."}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Use --dry-run on Mac; execute inspection inside the existing Modal Isaac image")
    if args.worker:
        if args.run_dir is None:
            parser.error("Internal workers require --run-dir")
        return worker(args.worker, profile, runtime, args.run_dir)
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-asset")
    directory.mkdir(parents=True)
    write_json(directory / "asset-profile.json", profile)
    write_json(directory / "runtime-config.json", runtime)
    summary = {"status": "running", "stages": {}, "run_dir": str(directory)}
    write_json(directory / "summary.json", summary)
    try:
        for stage in ("gpu", "asset"):
            print(f"Running {stage}; log: {directory / (stage + '.log')}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--profile", str(args.profile.resolve()),
                       "--worker", stage, "--run-dir", str(directory)]
            code = run_stage(command, directory / f"{stage}.log", profile["stage_timeout_seconds"])
            if code:
                worker_path = directory / f"{stage}-worker.json"
                evidence = json.loads(worker_path.read_text()) if worker_path.exists() else {}
                cause = f": {evidence['error']}" if evidence.get("error") else ""
                raise RuntimeError(f"{stage} failed (exit {code}){cause}; inspect {stage}.log and worker evidence")
            validate_worker(stage, directory, profile)
            summary["stages"][stage] = "passed"
            write_json(directory / "summary.json", summary)
        summary["status"] = "passed_asset_inventory"
        return 0
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=str(error), error_type=type(error).__name__)
        (directory / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        write_json(directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
