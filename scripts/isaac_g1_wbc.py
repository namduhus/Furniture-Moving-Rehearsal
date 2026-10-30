"""Verify the pinned balance policy and run one bounded Isaac G1 standing trial."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys
import traceback

from isaac_g1_asset import load_profile as load_asset_profile, validate_inventory
from isaac_preflight import ROOT, check_versions, gpu_check, run_stage, write_json
from isaac_video import encode_recording, file_sha256, require_ffmpeg


DEFAULT_PROFILE = ROOT / "configs/isaac_g1_balance.json"


@contextmanager
def record_worker_failure(directory, result):
    """Persist the original cause before the enclosing Kit launcher can exit."""
    try:
        yield
    except BaseException as error:
        message = result.get("error") or str(error) or repr(error)
        result.update(status="failed", error=message, error_type=type(error).__name__)
        write_json(directory / "balance-worker.json", result)
        (directory / "balance-error.log").write_text(traceback.format_exc())
        raise


def read_joint_frames(stage, reference):
    """Read authored joint pivots/axes; do not infer offsets from calibration."""
    import numpy as np
    from pxr import UsdPhysics
    from g1_wbc import quat_xyzw_matrix

    frames = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in reference.limits or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        bodies = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if any(len(paths) != 1 for paths in bodies):
            raise RuntimeError(f"Invalid USD joint body bindings: {name}")
        paths = [str(paths[0]) for paths in bodies]
        if not all(path.startswith("/World/envs/env_0/Robot/") for path in paths):
            continue
        if name in frames:
            raise RuntimeError(f"Duplicate USD joint: {name}")
        quaternion = joint.GetLocalRot1Attr().Get()
        xyzw = [*quaternion.GetImaginary(), quaternion.GetReal()]
        axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[str(joint.GetAxisAttr().Get())]]
        frames[name] = {"parent": paths[0].rsplit("/", 1)[-1],
                        "child": paths[1].rsplit("/", 1)[-1],
                        "position_in_child": list(joint.GetLocalPos1Attr().Get()),
                        "axis_in_child": (quat_xyzw_matrix(xyzw) @ axis).tolist(),
                        "joint_rotation_in_child_xyzw": xyzw, "usd_path": str(prim.GetPath())}
    return frames


def stage_failure(stage, code, directory):
    """Keep failure evidence visible even when Kit exits with an empty error."""
    path = directory / f"{stage}-worker.json"
    try:
        evidence = json.loads(path.read_text())
        detail = evidence.get("error") or evidence.get("error_type")
    except (OSError, ValueError, AttributeError):
        detail = None
    message = f"{stage} failed (exit {code}): {detail or 'no worker error message'}"
    for name in (f"{stage}-error.log", f"{stage}.log"):
        log = directory / name
        try:
            with log.open("rb") as source:
                source.seek(0, 2)
                source.seek(max(0, source.tell() - 16384))
                tail = "\n".join(source.read().decode(errors="replace").splitlines()[-50:])
        except OSError:
            continue
        if tail:
            return RuntimeError(f"{message}\nLog tail ({log}):\n{tail}")
    return RuntimeError(f"{message}; inspect {directory / (stage + '.log')}")


def load_plan(path):
    profile = json.loads(path.read_text())
    asset, runtime = load_asset_profile(ROOT / profile["asset_profile"])
    if type(profile["control_steps"]) is not int or not 4 <= profile["control_steps"] <= 500:
        raise ValueError("Standing trial needs 4-500 control steps")
    if type(profile["stage_timeout_seconds"]) is not int or not 1 <= profile["stage_timeout_seconds"] <= 600:
        raise ValueError("Standing stage timeout must not exceed ten minutes")
    camera = profile["camera"]
    stride = camera["capture_every_control_steps"]
    if type(stride) is not int or not 1 <= stride <= 10 or profile["control_steps"] % stride:
        raise ValueError("Invalid camera capture cadence")
    for key in ("width", "height"):
        if type(camera[key]) is not int or not 64 <= camera[key] <= 640 or camera[key] % 2:
            raise ValueError("Invalid recording camera size")
    bounds = {"frame_position_tolerance_m": (0, 0.005), "frame_angle_tolerance_rad": (0, 0.02),
              "initial_joint_tolerance_rad": (0, 0.001), "initial_pose_tolerance": (0, 0.001),
              "minimum_base_height_m": (0.5, 0.7), "maximum_tilt_rad": (0.01, 0.6)}
    for name, (low, high) in bounds.items():
        value = profile[name]
        if type(value) not in (float, int) or not low <= value <= high or (low == 0 and value == 0):
            raise ValueError(f"Unreviewed transfer guard: {name}")
    return profile, asset, runtime


def policy_check(contract, model_path):
    import numpy as np
    from g1_wbc import BalancePolicy, ObservationHistory

    policy = BalancePolicy(model_path, contract)
    history = ObservationHistory(contract)
    standing = history.observe(np.r_[contract["default_lower_angles"], np.zeros(14)],
                               np.zeros(29), [0, 0, 0, 1], np.zeros(3), np.eye(3))
    outputs = [policy(np.zeros(516)), policy(standing)]
    return {"status": "passed_policy_contract", **policy.metadata,
            "probe_max_abs_outputs": [float(np.max(np.abs(a))) for a in outputs],
            "balance_or_walking_verified": False}


def balance_check(profile, asset, runtime, contract, reference, model_path, directory):
    import numpy as np
    import torch
    import warp as wp
    from g1_wbc import (BalancePolicy, ObservationHistory, FrameMismatchError, actuator_limits, check_frames,
                        finite, joint_anchor_poses, matrix_quat_xyzw, quat_xyzw_matrix)

    encoder = require_ffmpeg(directory)
    torch.manual_seed(runtime["seed"])
    policy = BalancePolicy(model_path, contract)
    history = ObservationHistory(contract)
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass, instantiate, replace
    from isaaclab_assets import G1_29DOF_CFG
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from PIL import Image

    groups = asset["joint_groups"]
    expected = sum(groups.values(), [])
    lower = groups["lower_body"]
    kp = {n: contract["kp"][i] for i, n in enumerate(lower)}
    kd = {n: contract["kd"][i] for i, n in enumerate(lower)}
    for key, prefix in (("arms", "upper"), ("hands", "hand")):
        kp.update({n: contract[f"{prefix}_hold_kp"] for n in groups[key]})
        kd.update({n: contract[f"{prefix}_hold_kd"] for n in groups[key]})
    candidate = replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    motor_limits = actuator_limits(contract, reference)
    candidate.actuators = {"reference_pd": IdealPDActuatorCfg(
        joint_names_expr=expected, stiffness=kp, damping=kd,
        actuator_effort_limit={n: motor_limits[n]["effort"] for n in expected},
        joint_effort_limit={n: motor_limits[n]["effort"] for n in expected},
        joint_velocity_limit={n: motor_limits[n]["velocity"] for n in expected},
        armature=0.0, friction=0.0, dynamic_friction=0.0, viscous_friction=0.0,
    )}

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500))
        robot = candidate
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera", update_period=0,
            width=profile["camera"]["width"], height=profile["camera"]["height"],
            data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24, horizontal_aperture=20.955,
                                           clipping_range=(0.1, 100)),
        )

    @configclass
    class LaunchCfg:
        sim = sim_utils.SimulationCfg(device=runtime["device"], dt=asset["physics_dt"], physics=PhysxCfg())
        scene = SceneCfg(num_envs=1, env_spacing=3.0)

    cfg = LaunchCfg()
    result = {"status": "running", "steps_completed": 0, "resets": 0,
              "controller": policy.metadata, "mode": "stand_only_zero_navigation",
              "controller_inference_verified": False, "walking_verified": False}
    write_json(directory / "balance-worker.json", result)
    write_json(directory / "actuators.json", {"kp": kp, "kd": kd,
               "configuration": candidate.actuators["reference_pd"].to_dict(),
               "limit_origin": contract["actuator_limit_origin"],
               "upper_hold_gain_origin": contract["hold_gain_origin"]})
    with launch_simulation(cfg, {"headless": True, "device": runtime["device"],
                                 "enable_cameras": True, "kit_args": "--allow-root"}), record_worker_failure(directory, result):
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        sim.reset()
        robot, camera = scene["robot"], scene["camera"]

        def tensor(value):
            return value.torch if hasattr(value, "torch") else value

        def array(value):
            return tensor(value)[0].detach().cpu().numpy().copy()

        def device(value):
            return torch.as_tensor(value, dtype=torch.float32, device=runtime["device"])[None]

        names = list(robot.joint_names)
        inventory = {"joint_names": names, "is_fixed_base": bool(robot.is_fixed_base),
                     "gravity_disabled": candidate.spawn.rigid_props.disable_gravity,
                     "joint_state_on_cuda": bool(tensor(robot.data.joint_pos).is_cuda),
                     "physics_manager": sim.physics_manager.__module__ + "." + sim.physics_manager.__name__,
                     "joint_positions": array(robot.data.joint_pos).tolist(),
                     "default_joint_positions": array(robot.data.default_joint_pos).tolist(),
                     "joint_position_limits": array(robot.data.joint_pos_limits).tolist()}
        write_json(directory / "asset-inventory.json", inventory)
        validate_inventory(inventory, asset)
        if set(reference.limits) != set(names):
            raise RuntimeError("Runtime model differs from the pinned URDF")
        indices = [names.index(n) for n in asset["controller_reference"]["joint_order"]]
        lower_indices = [names.index(n) for n in lower]
        limits = array(robot.data.joint_pos_limits)
        for i, n in enumerate(names):
            if not np.allclose(limits[i], [reference.limits[n]["lower"], reference.limits[n]["upper"]], atol=1e-4, rtol=0):
                raise RuntimeError(f"Runtime joint limits differ from the reference: {n}")
        joint_frames = read_joint_frames(sim.stage, reference)
        write_json(directory / "joint-frames.json", joint_frames)

        def frames():
            poses = finite(array(robot.data.body_link_pose_w), (len(robot.body_names), 7))
            return dict(zip(robot.body_names, poses, strict=True))

        def verify_frames(rotation=None, label="calibration"):
            q = finite(array(robot.data.joint_pos), (43,))
            positions = dict(zip(names, q, strict=True))
            poses = frames()
            write_json(directory / f"frame-input-{label}.json", {
                "joint_positions": positions, "body_poses_xyzw": {n: p.tolist() for n, p in poses.items()}})
            anchors = joint_anchor_poses(reference, poses, joint_frames)
            try:
                report = check_frames(reference, positions, anchors,
                                      profile["frame_position_tolerance_m"], profile["frame_angle_tolerance_rad"], rotation)
            except FrameMismatchError as error:
                write_json(directory / f"frame-comparison-{label}.json", error.report)
                raise
            write_json(directory / f"frame-comparison-{label}.json", report)
            return report

        # Calibration uses measured link poses; it does not assume the asset's
        # configured wxyz root rotation equals an API xyzw root pose.
        frame_evidence = {"calibration": verify_frames(), "probes": []}
        rotation = np.asarray(frame_evidence["calibration"]["source_to_asset_rotation"])
        defaults = np.zeros(43)
        defaults[lower_indices] = contract["default_lower_angles"]

        def write_joints(q):
            robot.write_joint_position_to_sim_index(position=device(q))
            robot.write_joint_velocity_to_sim_index(velocity=device(np.zeros(43)))
            sim.forward()
            scene.update(0.0)

        for probe_index, delta in enumerate((-0.04, 0.04)):
            write_joints(np.clip(defaults+delta, limits[:, 0], limits[:, 1]))
            probe = verify_frames(rotation, label=f"probe-{probe_index}")
            frame_evidence["probes"].append(probe)
        write_json(directory / "frames.json", {"status": "passed_kinematic_probes", **frame_evidence})

        root_pose = np.r_[0.0, 0.0, contract["height_command"], matrix_quat_xyzw(rotation.T)]
        robot.write_root_pose_to_sim_index(root_pose=device(root_pose))
        robot.write_root_velocity_to_sim_index(root_velocity=device(np.zeros(6)))
        write_joints(defaults)
        scene.reset()
        robot.set_joint_position_target_index(target=device(defaults))
        robot.set_joint_velocity_target_index(target=device(np.zeros(43)))
        sim.forward()
        scene.update(0.0)
        pose = finite(array(robot.data.root_link_pose_w), (7,))
        q = finite(array(robot.data.joint_pos), (43,))
        velocity = finite(array(robot.data.root_link_vel_w), (6,))
        dq = finite(array(robot.data.joint_vel), (43,))
        source_rotation = quat_xyzw_matrix(pose[3:]) @ rotation
        initial = {"status": "initialization_mismatch", "wanted_root_pose_xyzw": root_pose.tolist(),
                   "wanted_joint_positions": defaults.tolist(), "root_pose_xyzw": pose.tolist(),
                   "source_rotation_world": source_rotation.tolist(), "joint_positions": q.tolist(),
                   "root_vel_world": velocity.tolist(), "joint_velocities": dq.tolist()}
        write_json(directory / "initial-state.json", initial)
        if (np.max(np.abs(q-defaults)) > profile["initial_joint_tolerance_rad"] or
                np.max(np.abs(pose[:3]-root_pose[:3])) > profile["initial_pose_tolerance"] or
                not np.allclose(source_rotation, np.eye(3), atol=profile["initial_pose_tolerance"]) or
                np.max(np.abs(velocity)) > 1e-4 or np.max(np.abs(dq)) > 1e-4):
            raise RuntimeError("Explicit initial joint/root state was not established")
        initial["status"] = "passed_explicit_initialization"
        write_json(directory / "initial-state.json", initial)
        history.reset()
        camera.set_world_poses_from_view(device([2.5, 2.5, 1.7]), device([0.0, 0.0, 0.7]))
        frame_count = 0
        stride = profile["camera"]["capture_every_control_steps"]
        dt = asset["physics_dt"]*asset["control_decimation"]
        (directory / "frames").mkdir()

        def capture(path):
            for _ in range(2):
                sim.render()
                camera.update(asset["physics_dt"], force_recompute=True)
            rgb = tensor(camera.data.output["rgb"])[0, :, :, :3].detach().cpu().numpy()
            if rgb.shape != (profile["camera"]["height"], profile["camera"]["width"], 3) or float(rgb.std()) < 1:
                raise RuntimeError("Invalid balance camera evidence")
            Image.fromarray(rgb.astype(np.uint8)).save(path)

        capture(directory / "initial.png")
        error = None
        try:
            with (directory / "trajectory.jsonl").open("w") as trajectory:
                for step in range(profile["control_steps"]):
                    q, dq = array(robot.data.joint_pos), array(robot.data.joint_vel)
                    pose, vel = array(robot.data.root_link_pose_w), array(robot.data.root_link_vel_w)
                    observation = history.observe(q[indices], dq[indices], pose[3:], vel[3:], rotation)
                    action = policy(observation)
                    target, clipped = history.targets(action, limits[lower_indices])
                    targets = defaults.copy()
                    targets[lower_indices] = target
                    robot.set_joint_position_target_index(target=device(targets))
                    for _ in range(asset["control_decimation"]):
                        scene.write_data_to_sim()
                        sim.step(render=False)
                        scene.update(asset["physics_dt"])
                    pose = finite(array(robot.data.root_link_pose_w), (7,))
                    q = finite(array(robot.data.joint_pos), (43,))
                    dq = finite(array(robot.data.joint_vel), (43,))
                    velocity = finite(array(robot.data.root_link_vel_w), (6,))
                    gravity = (quat_xyzw_matrix(pose[3:]) @ rotation).T @ np.array([0, 0, -1])
                    tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
                    record = {"step": step+1, "sim_time_seconds": (step+1)*dt,
                              "root_height_m": float(pose[2]), "tilt_rad": tilt,
                              "action": action.tolist(), "clipped_joint_targets": clipped,
                              "root_pose_xyzw": pose.tolist(), "root_velocity_world": velocity.tolist(),
                              "joint_positions": q.tolist(), "joint_velocities": dq.tolist()}
                    trajectory.write(json.dumps(record)+"\n")
                    result.update(steps_completed=step+1, controller_inference_verified=True)
                    if (step+1) % stride == 0:
                        capture(directory / "frames" / f"{frame_count:06d}.png")
                        frame_count += 1
                    if pose[2] < profile["minimum_base_height_m"] or tilt > profile["maximum_tilt_rad"]:
                        raise RuntimeError(f"Balance guard failed at step {step+1}: height={pose[2]:.3f}, tilt={tilt:.3f}")
            result["status"] = "passed_balance_rollout"
        except BaseException as failure:
            error = failure
            result.update(status="failed", error=str(failure))
            raise
        finally:
            try:
                capture(directory / "final.png")
                if frame_count >= 2:
                    result["video"] = encode_recording(
                        directory, binary=encoder, count=frame_count, fps=1/(Fraction(str(dt))*stride),
                        width=profile["camera"]["width"], height=profile["camera"]["height"], require_motion=False)
            except BaseException as video_error:
                (directory / "video-error.log").write_text(traceback.format_exc())
                result.update(status="failed", video_error=str(video_error))
                if error is None:
                    raise
            finally:
                write_json(directory / "balance-worker.json", result)


def validate_stage(stage, directory, profile):
    evidence = json.loads((directory / f"{stage}-worker.json").read_text())
    wanted = {"policy": "passed_policy_contract", "gpu": "checks_completed", "balance": "passed_balance_rollout"}[stage]
    if evidence.get("status") != wanted:
        raise RuntimeError(f"Incomplete {stage} evidence")
    if stage == "balance":
        import numpy as np
        from g1_wbc import check_frames, finite, joint_anchor_poses, load_contract, quat_xyzw_matrix
        if evidence["steps_completed"] != profile["control_steps"] or evidence["resets"] != 0 or not evidence["controller_inference_verified"]:
            raise RuntimeError("Incomplete or reset balance rollout")
        records = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        if len(records) != profile["control_steps"]:
            raise RuntimeError("Incomplete balance trajectory")
        frames = json.loads((directory / "frames.json").read_text())
        source_to_asset = finite(frames["calibration"]["source_to_asset_rotation"], (3, 3))
        for step, item in enumerate(records, 1):
            if (item["step"] != step or not profile["minimum_base_height_m"] <= item["root_height_m"] < 2 or
                    not 0 <= item["tilt_rad"] <= profile["maximum_tilt_rad"] or
                    not np.isclose(item["sim_time_seconds"], step*0.02, atol=1e-8)):
                raise RuntimeError("Trajectory violates balance guards")
            finite(item["joint_positions"], (43,))
            finite(item["joint_velocities"], (43,))
            finite(item["root_velocity_world"], (6,))
            finite(item["action"], (15,))
            pose = finite(item["root_pose_xyzw"], (7,))
            gravity = (quat_xyzw_matrix(pose[3:]) @ source_to_asset).T @ np.array([0, 0, -1])
            tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
            if not np.isclose(pose[2], item["root_height_m"], atol=1e-6) or not np.isclose(tilt, item["tilt_rad"], atol=1e-6):
                raise RuntimeError("Trajectory metrics disagree with measured root poses")
        for name, status in (("frames.json", "passed_kinematic_probes"), ("initial-state.json", "passed_explicit_initialization")):
            if json.loads((directory / name).read_text())["status"] != status:
                raise RuntimeError("Missing initialization/frame evidence")
        if len(frames["probes"]) != 2:
            raise RuntimeError("Incomplete kinematic probes")
        _, reference = load_contract(ROOT / profile["controller_contract"])
        joint_frames = json.loads((directory / "joint-frames.json").read_text())
        for label in ("calibration", "probe-0", "probe-1"):
            measured = json.loads((directory / f"frame-input-{label}.json").read_text())
            anchors = joint_anchor_poses(reference, measured["body_poses_xyzw"], joint_frames)
            check_frames(reference, measured["joint_positions"], anchors,
                         profile["frame_position_tolerance_m"], profile["frame_angle_tolerance_rad"], source_to_asset)
        for frame in [frames["calibration"], *frames["probes"]]:
            rotation = finite(frame["source_to_asset_rotation"], (3, 3))
            if (frame["links_checked"] < 30 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
                    not np.isclose(np.linalg.det(rotation), 1) or
                    not 0 <= frame["maximum_position_error_m"] <= profile["frame_position_tolerance_m"] or
                    not 0 <= frame["maximum_angle_error_rad"] <= profile["frame_angle_tolerance_rad"]):
                raise RuntimeError("Invalid kinematic evidence")
        initial = json.loads((directory / "initial-state.json").read_text())
        pose = finite(initial["root_pose_xyzw"], (7,))
        wanted_pose = finite(initial["wanted_root_pose_xyzw"], (7,))
        if (np.max(np.abs(pose[:3]-wanted_pose[:3])) > profile["initial_pose_tolerance"] or
                np.max(np.abs(finite(initial["joint_positions"], (43,))-finite(initial["wanted_joint_positions"], (43,)))) > profile["initial_joint_tolerance_rad"] or
                not np.allclose(finite(initial["source_rotation_world"], (3, 3)), np.eye(3), atol=profile["initial_pose_tolerance"]) or
                np.max(np.abs(finite(initial["root_vel_world"], (6,)))) > 1e-4 or
                np.max(np.abs(finite(initial["joint_velocities"], (43,)))) > 1e-4):
            raise RuntimeError("Invalid initial-state evidence")
        video = evidence["video"]
        expected_count = profile["control_steps"] // profile["camera"]["capture_every_control_steps"]
        duration = profile["control_steps"]*0.02
        fps = 1/(0.02*profile["camera"]["capture_every_control_steps"])
        if (video["status"] != "passed_video_decode" or video["decoded_frames"] != expected_count or
                video["width"] != profile["camera"]["width"] or video["height"] != profile["camera"]["height"] or
                not np.isclose(float(Fraction(video["fps"])), fps) or
                not np.isclose(video["duration_seconds"], duration) or
                video["sha256"] != file_sha256(directory / "rollout.mp4")):
            raise RuntimeError("Incomplete balance video")
        for name in ("initial.png", "final.png"):
            if not (directory / name).is_file():
                raise RuntimeError("Missing balance camera evidence")


def worker(stage, profile, asset, runtime, contract_path, directory):
    try:
        write_json(directory / "versions.json", check_versions(runtime))
        from g1_wbc import load_contract, prepare_model
        contract, reference = load_contract(contract_path)
        if stage == "gpu":
            write_json(directory / "gpu-worker.json", {"status": "checks_completed", **gpu_check(runtime, directory)})
        else:
            path = prepare_model(contract)
            if stage == "policy":
                write_json(directory / "policy-worker.json", policy_check(contract, path))
            else:
                balance_check(profile, asset, runtime, contract, reference, path, directory)
        return 0
    except BaseException as error:
        if stage == "balance" and isinstance(error, SystemExit) and error.code in (None, 0):
            try:
                validate_stage(stage, directory, profile)
            except (OSError, ValueError, RuntimeError, KeyError):
                pass
            else:
                return 0
        path = directory / f"{stage}-worker.json"
        evidence = json.loads(path.read_text()) if path.exists() else {}
        message = (evidence.get("error") if isinstance(error, SystemExit) else None) or str(error) or repr(error)
        write_json(path, {**evidence, "status": "failed", "error": message,
                          "error_type": type(error).__name__})
        (directory / f"{stage}-error.log").write_text(traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--stage", choices=("policy", "balance"), default="policy")
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", choices=("policy", "gpu", "balance"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    profile, asset, runtime = load_plan(args.profile)
    stages = ["policy"] if args.stage == "policy" else ["policy", "gpu", "balance"]
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "stages": stages, "asset": asset["asset_config"],
                          "control_steps": profile["control_steps"], "simulated_seconds": profile["control_steps"]*0.02,
                          "model_download_bytes": 1886682 if args.prepare_model else 0,
                          "runtime_image_change": False, "training": False}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    contract_path = ROOT / profile["controller_contract"]
    if args.worker:
        if args.run_dir is None:
            parser.error("Internal worker requires --run-dir")
        return worker(args.worker, profile, asset, runtime, contract_path, args.run_dir)
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-wbc")
    directory.mkdir(parents=True)
    write_json(directory / "profile.json", profile)
    write_json(directory / "asset-profile.json", asset)
    write_json(directory / "runtime-config.json", runtime)
    write_json(directory / "controller-contract.json", json.loads(contract_path.read_text()))
    summary = {"status": "running", "stages": {}, "run_dir": str(directory)}
    write_json(directory / "summary.json", summary)
    try:
        from g1_wbc import load_contract, prepare_model
        contract, _ = load_contract(contract_path)
        if args.prepare_model:
            print("Preparing pinned 1,886,682-byte balance checkpoint and verifying SHA-256...", flush=True)
        prepare_model(contract, args.prepare_model)
        for stage in stages:
            print(f"Running {stage}; log: {directory / (stage+'.log')}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--profile", str(args.profile.resolve()),
                       "--worker", stage, "--run-dir", str(directory)]
            timeout = profile["stage_timeout_seconds"] if stage == "balance" else (120 if stage == "gpu" else 60)
            code = run_stage(command, directory / f"{stage}.log", timeout)
            if code:
                raise stage_failure(stage, code, directory)
            validate_stage(stage, directory, profile)
            summary["stages"][stage] = "passed"
            write_json(directory / "summary.json", summary)
        summary["status"] = "passed_policy_contract" if args.stage == "policy" else "passed_balance_rollout"
        return 0
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        (directory / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        write_json(directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
