"""Parallel G1 push environments driving the frozen GR00T walk policy (RL stage 2: setup and throughput).

Every environment receives the qualified single-trial setup (fixed hands, palm inertia, source foot contact,
implicit arm hold, crouch start, box). Per-environment physics is parsed separately so that session-layer
edits apply to each copy; runtime buffers are checked for all environments before any control step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace

from isaac_g1_wbc import (ROOT, apply_fixed_hands, apply_palm_override, apply_source_contact, load_plan,
                          read_collision_geometry, read_joint_frames, read_material, scene_config, trial_contract)
from isaac_preflight import check_versions, run_stage, write_json


DEFAULT_CONFIG = ROOT / "configs/isaac_g1_rl_throughput.json"


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"push_profile", "num_envs", "control_steps", "env_spacing_m", "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed RL environment configuration keys")
    for key, (low, high) in {"num_envs": (1, 4096), "control_steps": (4, 1000), "stage_timeout_seconds": (60, 600)}.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid {key}")
    if type(config["env_spacing_m"]) not in (int, float) or not 2.0 <= config["env_spacing_m"] <= 10.0:
        raise ValueError("Invalid env_spacing_m")
    profile, asset, runtime = load_plan(ROOT / config["push_profile"])
    if profile.get("functional_case") != "push" or profile.get("arm_hold") != "implicit":
        raise ValueError("RL environments must start from a reviewed implicit-arm push profile")
    return config, profile, asset, runtime


def verify_parallel_setup(robot, box, contract, asset, profile, hands, palms, contacts, numpy, box_masses):
    """All environments must carry the qualified per-copy physics before control."""
    from g1_wbc import load_contact_reference, palm_inertial_reference, source_sphere_center
    np = numpy

    def batch(value):
        value = value.torch if hasattr(value, "torch") else value
        return value.detach().cpu().numpy().copy()

    names, bodies = list(robot.joint_names), list(robot.body_names)
    lower = asset["joint_groups"]["lower_body"]
    if len(names) != 29 or set(names) != set(asset["controller_reference"]["joint_order"]):
        raise RuntimeError("Parallel robots did not resolve the reviewed 29 active joints")
    if any(h["status"] != "authored_fixed_hands_before_reset" or len(h["joints"]) != 14 for h in hands):
        raise RuntimeError("Fixed hands were not authored in every environment")
    masses = batch(robot.data.body_mass)
    if not np.allclose(masses, masses[0], atol=1e-6, rtol=0):
        raise RuntimeError("Body masses differ between parallel environments")
    palm = palm_inertial_reference(contract)
    if not np.allclose(masses[:, bodies.index("left_hand_palm_link")], palm["mass_kg"], atol=1e-6, rtol=0):
        raise RuntimeError("Left-palm inertial override is missing in some environments")
    reference = load_contact_reference()
    for report in contacts:
        for name, foot in report["feet"].items():
            if not np.isclose(masses[0, bodies.index(name)], foot["authored_mass_kg"], atol=1e-6, rtol=0):
                raise RuntimeError("Foot mass changed with collider geometry")
            for sphere in foot["spheres"]:
                if not np.allclose(sphere["after_center_body_m"], source_sphere_center(sphere["center_body_m"], reference), atol=1e-9, rtol=0):
                    raise RuntimeError(f"Source foot sphere missing: {sphere['path']}")
    if any(p["status"] != "authored_before_physics_reset" for p in palms):
        raise RuntimeError("Palm override was not authored in every environment")
    drive = np.array([[0.0, 0.0] if n in lower else [100.0, 0.5] for n in names])
    solver = np.stack([batch(robot.data.joint_stiffness), batch(robot.data.joint_damping)], axis=2)
    if not np.allclose(solver, drive[None], atol=1e-6, rtol=0):
        raise RuntimeError("Arm-hold PhysX drive gains differ in some environments")
    for value, wanted in ((robot.data.joint_armature, 0.01), (robot.data.joint_friction_coeff, 0.1),
                          (robot.data.joint_dynamic_friction_coeff, 0.1), (robot.data.joint_viscous_friction_coeff, 0.001)):
        if not np.allclose(batch(value), wanted, atol=1e-7, rtol=0):
            raise RuntimeError("Armature or passive joint terms differ in some environments")
    if not np.allclose(batch(box.data.body_mass)[:, 0], box_masses, atol=1e-4, rtol=0):
        raise RuntimeError("Box mass differs in some environments")
    return {"status": "verified_parallel_setup", "environments": len(masses), "active_joints": 29,
            "left_palm_mass_kg": float(masses[0, bodies.index("left_hand_palm_link")])}


LAUNCH_SETTINGS = {"headless": True, "enable_cameras": False, "kit_args": "--allow-root"}


def apply_box_properties(stage, box, env, mass, friction):
    """Author one environment's box mass and floor friction in the session layer before physics reset."""
    from pxr import Usd, UsdPhysics

    if box.is_initialized:
        raise RuntimeError("Box properties must precede physics initialization")
    body = stage.GetPrimAtPath(f"{env}/Furniture")
    material = stage.GetPrimAtPath(f"{env}/Furniture/geometry/physicsMaterial")
    if not body or not body.HasAPI(UsdPhysics.MassAPI) or not material:
        raise RuntimeError(f"Expected the spawned box mass and material: {env}")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        written = [UsdPhysics.MassAPI(body).GetMassAttr().Set(float(mass))]
        written += [material.GetAttribute(name).Set(float(friction))
                    for name in ("physics:staticFriction", "physics:dynamicFriction")]
    if not all(written):
        raise RuntimeError(f"Unable to author box properties: {env}")


def prepare_push_scene(built, count, profile, asset, runtime, contract, reference, directory, result, evidence,
                       box_properties=None, extra_edits=None, check_box_start=True):
    """Instantiate the parallel push scene with every qualified edit, verify it and place the crouch start.

    evidence: JSON path that receives `result` once materials are read, before they are checked.
    box_properties: optional per-environment (masses, frictions) authored before reset; defaults to the profile box.
    extra_edits: optional callable(stage, robot, box, env) for further session-layer edits before reset.
    check_box_start: require the box at its push spawn after reset; callers that place the box themselves skip it.
    """
    import numpy as np
    import torch
    from g1_wbc import check_frames, collision_clearance, finite, joint_anchor_poses, matrix_quat_xyzw, placement_shift
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    push = profile["push_object"]
    if box_properties is None:
        masses, frictions = np.full(count, float(push["mass_kg"])), np.full(count, float(push["friction"]))
    else:
        masses, frictions = (np.asarray(v, dtype=float) for v in box_properties)
    sim = sim_utils.SimulationContext(built.cfg.sim)
    scene = instantiate(built.cfg.scene)
    robot, box = scene["robot"], scene["furniture"]
    hands, palms, contacts = [], [], []
    setup = directory / "env0-setup"
    setup.mkdir()
    for i in range(count):
        env, record = f"/World/envs/env_{i}", setup if i == 0 else None
        hands.append(apply_fixed_hands(sim.stage, robot, record, env=env))
        palms.append(apply_palm_override(sim.stage, robot, contract, record, env=env))
        contacts.append(apply_source_contact(sim.stage, robot, record, env=env))
        if box_properties is not None:
            apply_box_properties(sim.stage, box, env, masses[i], frictions[i])
        if extra_edits is not None:
            extra_edits(sim.stage, robot, box, env)
    sim.reset()

    def batch(value):
        value = value.torch if hasattr(value, "torch") else value
        return value.detach().cpu().numpy().copy()

    def device(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=runtime["device"])

    result["setup"] = verify_parallel_setup(robot, box, contract, asset, profile, hands, palms, contacts, np, masses)
    boxes = [read_material(sim.stage, f"/World/envs/env_{i}/Furniture/geometry/physicsMaterial") for i in range(count)]
    surface = {"default": read_material(sim.stage, "/physicsScene/defaultMaterial"),
               "ground": read_material(sim.stage, "/World/Ground/physicsMaterial"), "box_env0": boxes[0]}
    result["materials"] = surface
    write_json(evidence, result)
    # USD stores friction as float32, so compare with the single-run object tolerance; PhysX must hold the same values.
    authored = np.array([[b["static_friction"], b["dynamic_friction"]] for b in boxes], dtype=float)
    physx = np.asarray(box.root_view.get_material_properties().numpy(), dtype=float)[:, 0, :2]
    if (any(m["friction_combine_mode"] != "multiply" for m in [*surface.values(), *boxes]) or
            not np.allclose(authored, frictions[:, None], atol=1e-6, rtol=0) or
            not np.allclose(physx, frictions[:, None], atol=1e-6, rtol=0)):
        raise RuntimeError("Push friction materials differ from the reviewed profile")

    names, bodies = list(robot.joint_names), list(robot.body_names)
    lower = asset["joint_groups"]["lower_body"]
    indices = [names.index(n) for n in asset["controller_reference"]["joint_order"]]
    lower_indices = [names.index(n) for n in lower]
    limits = batch(robot.data.joint_pos_limits)[0][lower_indices]
    defaults = np.zeros(29)
    defaults[lower_indices] = contract["default_lower_angles"]
    origins = batch(scene.env_origins)

    # Kinematic frame calibration on environment 0, as in the single trial.
    q0 = batch(robot.data.joint_pos)[0]
    poses0 = dict(zip(bodies, finite(batch(robot.data.body_link_pose_w)[0], (len(bodies), 7)), strict=True))
    frames = check_frames(reference, dict(zip(names, q0, strict=True)),
                          joint_anchor_poses(reference, poses0, read_joint_frames(sim.stage, reference)),
                          profile["frame_position_tolerance_m"], profile["frame_angle_tolerance_rad"])
    rotation = np.asarray(frames["source_to_asset_rotation"])

    # Crouch start: default joints from the policy height, lifted to the reviewed foot clearance.
    root = np.zeros((count, 7))
    root[:, :3] = origins + [0.0, 0.0, contract["height_command"]]
    root[:, 3:] = matrix_quat_xyzw(rotation.T)

    def place(lift):
        pose = root.copy()
        pose[:, 2] += lift
        robot.write_root_pose_to_sim_index(root_pose=device(pose))
        robot.write_root_velocity_to_sim_index(root_velocity=device(np.zeros((count, 6))))
        robot.write_joint_position_to_sim_index(position=device(np.tile(defaults, (count, 1))))
        robot.write_joint_velocity_to_sim_index(velocity=device(np.zeros((count, 29))))
        sim.forward()
        scene.update(0.0)
        feet = profile["diagnostics"]["foot_body_names"]
        poses = dict(zip(bodies, batch(robot.data.body_link_pose_w)[0], strict=True))
        return collision_clearance(read_collision_geometry(sim.stage, {f: poses[f].tolist() for f in feet}))

    before = place(0.0)
    lift = placement_shift(before["minimum_clearance_m"], profile["initialization"])
    after = place(lift)
    root[:, 2] += lift
    scene.reset()
    robot.set_joint_position_target_index(target=device(np.tile(defaults, (count, 1))))
    robot.set_joint_velocity_target_index(target=device(np.zeros((count, 29))))
    if abs(after["minimum_clearance_m"] - profile["initialization"]["minimum_clearance_m"]) > 1e-5:
        raise RuntimeError("Crouch-start clearance was not established")
    box_start = batch(box.data.root_link_pose_w)
    wanted_x = built.hand_reach + push["gap_m"] + push["size_m"][0]/2
    if check_box_start and not np.allclose(box_start[:, :2] - origins[:, :2], [wanted_x, 0.0], atol=0.01, rtol=0):
        raise RuntimeError("Box placement differs between environments")
    result["initialization"] = {"lift_m": lift, "spawn_height_m": contract["height_command"] + lift,
                                "clearance_after_m": after["minimum_clearance_m"],
                                "source_to_asset_rotation": rotation.tolist()}
    return SimpleNamespace(sim=sim, scene=scene, robot=robot, box=box, batch=batch, device=device, indices=indices,
                           lower_indices=lower_indices, limits=limits, defaults=defaults, origins=origins,
                           rotation=rotation, root=root, box_start=box_start, hand_reach=built.hand_reach,
                           names=names, bodies=bodies)


def run_worker(config, profile, asset, runtime, model_path, directory):
    import numpy as np
    import torch
    import warp as wp
    from g1_wbc import BalancePolicy, BatchHistory, finite, quat_xyzw_matrices

    contract, reference = trial_contract(profile)
    policy = BalancePolicy(model_path, contract)
    count, steps = config["num_envs"], config["control_steps"]
    torch.manual_seed(runtime["seed"])
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation

    built = scene_config(profile, asset, runtime, contract, reference, with_camera=False, num_envs=count,
                         env_spacing=config["env_spacing_m"], replicate_physics=False)
    result = {"status": "running", "num_envs": count, "control_steps": steps, "training": False}
    write_json(directory / "rl-worker.json", result)
    with launch_simulation(built.cfg, {**LAUNCH_SETTINGS, "device": runtime["device"]}):
        try:
            s = prepare_push_scene(built, count, profile, asset, runtime, contract, reference, directory, result,
                                   directory / "rl-worker.json")
            scene, robot, box, batch, device = s.scene, s.robot, s.box, s.batch, s.device
            history = BatchHistory(contract, count)
            commands = np.tile(profile["command"], (count, 1))
            rpys = np.tile(profile.get("rpy_command", [0, 0, 0]), (count, 1))
            targets = np.tile(s.defaults, (count, 1))
            fallen = np.zeros(count, dtype=bool)
            invalid = np.zeros(count, dtype=bool)
            clipped = np.zeros(count, dtype=int)
            physics_dt = asset["physics_dt"]
            # Host-side phase timing; GPU work not finished by sim.step surfaces at the next device-to-host read.
            phases = dict.fromkeys(("write", "step", "update", "read_observe", "policy", "command"), 0.0)

            def timed(phase, call, *args, **kwargs):
                tic = time.perf_counter()
                value = call(*args, **kwargs)
                phases[phase] += time.perf_counter() - tic
                return value

            start = time.perf_counter()
            for _ in range(steps):
                # Source schedule: default targets for the first interval, inference after each physics interval.
                for _ in range(asset["control_decimation"]):
                    timed("write", scene.write_data_to_sim)
                    timed("step", s.sim.step, render=False)
                    timed("update", scene.update, physics_dt)
                tic = time.perf_counter()
                pose = finite(batch(robot.data.root_link_pose_w), (count, 7))
                observation = history.observe(batch(robot.data.joint_pos)[:, s.indices], batch(robot.data.joint_vel)[:, s.indices],
                                              pose[:, 3:], batch(robot.data.root_link_vel_w)[:, 3:], s.rotation, commands, rpys)
                phases["read_observe"] += time.perf_counter() - tic
                actions = timed("policy", policy.batch, observation)
                tic = time.perf_counter()
                lower_targets, clip_count, bad = history.targets(actions, s.limits)
                targets[:, s.lower_indices] = lower_targets
                robot.set_joint_position_target_index(target=device(targets))
                phases["command"] += time.perf_counter() - tic
                clipped += clip_count
                invalid |= bad
                gravity = -(quat_xyzw_matrices(pose[:, 3:]) @ s.rotation)[:, 2, :]
                tilt = np.arccos(np.clip(-gravity[:, 2], -1, 1))
                fallen |= (pose[:, 2] < profile["minimum_base_height_m"]) | (tilt > profile["maximum_tilt_rad"])
            wall = time.perf_counter() - start
            box_end = finite(batch(box.data.root_link_pose_w), (count, 7))
            progress = box_end[:, 0] - s.box_start[:, 0]
            result.update(status="passed_rl_throughput", wall_seconds=wall, policy_seconds=phases["policy"],
                          phase_seconds=phases, seconds_per_physics_step=wall/(steps*asset["control_decimation"]),
                          env_steps_per_second=count*steps/wall,
                          realtime_factor=count*steps*physics_dt*asset["control_decimation"]/wall,
                          simulated_seconds_per_env=steps*physics_dt*asset["control_decimation"],
                          fallen_envs=int(fallen.sum()), invalid_action_envs=int(invalid.sum()),
                          target_clip_events=int(clipped.sum()),
                          box_progress_m={"min": float(progress.min()), "median": float(np.median(progress)),
                                          "max": float(progress.max())},
                          single_env_baseline={"run": "20261030T153248.725956Z-wbc", "box_progress_m": 0.2017})
            write_json(directory / "rl-worker.json", result)
        except BaseException as error:
            result.update(status="failed", error=str(error) or repr(error), error_type=type(error).__name__)
            write_json(directory / "rl-worker.json", result)
            (directory / "rl-error.log").write_text(traceback.format_exc())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config, profile, asset, runtime = load_config(args.config)
    contract, _ = trial_contract(profile)
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "num_envs": config["num_envs"], "control_steps": config["control_steps"],
                          "env_steps": config["num_envs"]*config["control_steps"], "policy_role": contract["policy_role"],
                          "push_object": profile["push_object"], "replicate_physics": False, "training": False}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    from g1_wbc import prepare_model
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(runtime))
        run_worker(config, profile, asset, runtime, prepare_model(contract), args.run_dir)
        return 0
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-rl")
    directory.mkdir(parents=True)
    for name, value in (("rl-config.json", config), ("profile.json", profile), ("runtime-config.json", runtime),
                        ("controller-contract.json", contract)):
        write_json(directory / name, value)
    prepare_model(contract)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--worker", "--run-dir", str(directory)]
    code = run_stage(command, directory / "rl.log", config["stage_timeout_seconds"])
    evidence_path = directory / "rl-worker.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {"status": "missing"}
    summary = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code, "run_dir": str(directory),
               **{k: evidence.get(k) for k in ("env_steps_per_second", "realtime_factor", "phase_seconds", "fallen_envs",
                                               "invalid_action_envs", "box_progress_m", "error")}}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "passed_rl_throughput" else 1


if __name__ == "__main__":
    raise SystemExit(main())
