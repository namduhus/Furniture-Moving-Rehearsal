"""Verify the pinned balance policy and run one bounded Isaac G1 standing trial."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
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


def read_contact_materials(stage, feet):
    """Read bound collision materials without changing their coefficients."""
    from pxr import Usd, UsdPhysics, UsdShade

    scopes = {"ground": "/World/Ground", **{n: f"/World/envs/env_0/Robot/{n}" for n in feet}}
    report = {}
    for name, path in scopes.items():
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Missing diagnostic material scope: {path}")
        shapes = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            if not material:
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            values = {"collision_path": str(prim.GetPath()), "material_path": None,
                      "physics_material_found": False}
            enabled = prim.GetAttribute("physics:collisionEnabled")
            values["collision_enabled"] = enabled.Get() if enabled else None
            if material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                material_prim = material.GetPrim()
                values.update(material_path=str(material_prim.GetPath()), physics_material_found=True)
                for attribute in ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution",
                                  "physxMaterial:frictionCombineMode", "physxMaterial:restitutionCombineMode"):
                    attr = material_prim.GetAttribute(attribute)
                    value = attr.Get() if attr else None
                    values[attribute] = value if value is None or isinstance(value, (float, int, str, bool)) else str(value)
            shapes.append(values)
        if not shapes:
            raise RuntimeError(f"No collision shapes found in diagnostic material scope: {path}")
        report[name] = shapes
    return {"scopes": report, "interpretation": "Authored bound-material values only; unresolved bindings and combine modes are not guessed."}


def analyze_run(directory):
    """Analyze saved inputs on CPU without loading Isaac, ONNX or cloud SDKs."""
    import numpy as np
    from g1_wbc import ObservationHistory, contact_summary, finite, motion_summary

    directory = directory.expanduser().resolve()
    profile = json.loads((directory / "profile.json").read_text())
    initial = json.loads((directory / "initial-state.json").read_text())
    records = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    report = {"status": "motion_only", "motion": motion_summary(initial["root_pose_xyzw"], records),
              "stationary_success_assessed": False, "contacts": None, "policy_inputs_verified": False}
    worker_path = directory / "balance-worker.json"
    report["run_status"] = json.loads(worker_path.read_text()).get("status") if worker_path.exists() else "unknown"
    if not profile.get("diagnostics"):
        report["limitation"] = "This historical run did not record contact forces or policy inputs; foot slip cannot be determined."
        return report
    diagnostics = profile["diagnostics"]
    asset = json.loads((directory / "asset-profile.json").read_text())
    metadata = json.loads((directory / "contact-metadata.json").read_text())
    if set(metadata["sensor_body_names"]) != set(diagnostics["foot_body_names"]):
        raise ValueError("Contact sensor metadata does not match the diagnostic profile")
    samples_path = directory / "contacts.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
    report["contacts"] = contact_summary(samples, diagnostics["foot_body_names"],
        diagnostics["minimum_normal_force_n"], asset["physics_dt"], asset["control_decimation"])
    expected = profile["control_steps"]*asset["control_decimation"]
    report["expected_physics_samples"] = expected
    contract = json.loads((directory / "controller-contract.json").read_text())
    names = json.loads((directory / "asset-inventory.json").read_text())["joint_names"]
    indices = [names.index(n) for n in asset["controller_reference"]["joint_order"]]
    lower_indices = [names.index(n) for n in asset["joint_groups"]["lower_body"]]
    limits = np.asarray(json.loads((directory / "asset-inventory.json").read_text())["joint_position_limits"])[lower_indices]
    rotation = json.loads((directory / "frames.json").read_text())["calibration"]["source_to_asset_rotation"]
    history = ObservationHistory(contract)
    for row in records:
        state = row["policy_input_state"]
        q = finite(state["joint_positions"], (43,))
        dq = finite(state["joint_velocities"], (43,))
        pose = finite(state["root_pose_xyzw"], (7,))
        velocity = finite(state["root_velocity_world"], (6,))
        observation = history.observe(q[indices], dq[indices], pose[3:], velocity[3:], rotation)
        if not np.allclose(observation, finite(row["policy_observation"], (516,)), atol=1e-6, rtol=0):
            raise ValueError("Logged policy observation disagrees with its saved input state/history")
        history.targets(row["action"], limits)
    report["policy_inputs_verified"] = bool(records)
    report["status"] = "diagnostics_complete" if len(samples) == expected and len(records) == profile["control_steps"] else "diagnostics_partial"
    report["limitation"] = "Input-state consistency is checked with the adapter; it does not independently establish equivalence to upstream training observations."
    return report


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
    if "base_profile" in profile:
        base = (ROOT / profile["base_profile"]).resolve()
        if base != DEFAULT_PROFILE or set(profile) - {"base_profile", "diagnostics", "scope"}:
            raise ValueError("Diagnostic profiles may only extend the reviewed standing profile with instrumentation")
        profile = {**json.loads(base.read_text()), **profile}
    if (diagnostics := profile.get("diagnostics")) is not None:
        if (not isinstance(diagnostics, dict) or
                diagnostics.get("foot_body_names") != ["left_ankle_roll_link", "right_ankle_roll_link"] or
                type(diagnostics.get("minimum_normal_force_n")) not in (float, int) or
                not 0.1 <= diagnostics["minimum_normal_force_n"] <= 20):
            raise ValueError("Invalid foot contact diagnostic configuration")
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
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    diagnostics = profile.get("diagnostics")
    if diagnostics:
        from isaaclab.sensors import ContactSensorCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass, instantiate, replace
    from isaaclab_assets import G1_29DOF_CFG
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from PIL import Image

    groups = asset["joint_groups"]
    lower = groups["lower_body"]
    kp = {n: contract["kp"][i] for i, n in enumerate(lower)}
    kd = {n: contract["kd"][i] for i, n in enumerate(lower)}
    for key, prefix in (("arms", "upper"), ("hands", "hand")):
        kp.update({n: contract[f"{prefix}_hold_kp"] for n in groups[key]})
        kd.update({n: contract[f"{prefix}_hold_kd"] for n in groups[key]})
    candidate = replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    if diagnostics:
        candidate.spawn = replace(candidate.spawn, activate_contact_sensors=True)
    motor_limits = actuator_limits(contract, reference)
    upper = groups["arms"] + groups["hands"]
    candidate.actuators = {"reference_pd": IdealPDActuatorCfg(
        joint_names_expr=lower, stiffness={n: kp[n] for n in lower}, damping={n: kd[n] for n in lower},
        actuator_effort_limit={n: motor_limits[n]["effort"] for n in lower},
        joint_effort_limit={n: motor_limits[n]["effort"] for n in lower},
        joint_velocity_limit={n: motor_limits[n]["velocity"] for n in lower},
        armature=contract["lower_body_armature"], friction=0.0, dynamic_friction=0.0, viscous_friction=0.0,
    ), "upper_hold": ImplicitActuatorCfg(
        joint_names_expr=upper, stiffness={n: kp[n] for n in upper}, damping={n: kd[n] for n in upper},
        joint_effort_limit={n: motor_limits[n]["effort"] for n in upper},
        joint_velocity_limit={n: motor_limits[n]["velocity"] for n in upper},
        armature=contract["upper_body_armature"], friction=0.0, dynamic_friction=0.0, viscous_friction=0.0,
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
        if diagnostics:
            feet_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", update_period=0.0,
                history_length=1, track_friction_forces=True, debug_vis=False)

    @configclass
    class LaunchCfg:
        sim = sim_utils.SimulationCfg(device=runtime["device"], dt=asset["physics_dt"], physics=PhysxCfg())
        scene = SceneCfg(num_envs=1, env_spacing=3.0)

    cfg = LaunchCfg()
    result = {"status": "running", "steps_completed": 0, "resets": 0,
              "controller": policy.metadata, "mode": "stand_only_zero_navigation",
              "controller_inference_verified": False, "walking_verified": False}
    write_json(directory / "balance-worker.json", result)
    actuator_evidence = {"kp": kp, "kd": kd,
                         "configuration": {n: cfg.to_dict() for n, cfg in candidate.actuators.items()},
                         "limit_origin": contract["actuator_limit_origin"],
                         "actuation_origin": contract["actuation_origin"],
                         "upper_hold_gain_origin": contract["hold_gain_origin"]}
    write_json(directory / "actuators.json", actuator_evidence)
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

        if diagnostics:
            contact_sensor = scene["feet_contact"]
            feet = diagnostics["foot_body_names"]
            if set(contact_sensor.body_names) != set(feet) or len(contact_sensor.body_names) != 2:
                raise RuntimeError("Diagnostic sensor did not resolve exactly the two reviewed foot bodies")
            foot_body_indices = [robot.body_names.index(name) for name in feet]
            sensor_indices = [contact_sensor.body_names.index(name) for name in feet]
            write_json(directory / "contact-materials.json", read_contact_materials(sim.stage, feet))
            write_json(directory / "contact-metadata.json", {
                "sensor_body_names": list(contact_sensor.body_names), "foot_body_names": feet,
                "physics_dt_seconds": asset["physics_dt"], "sensor_history_length": 1,
                "force_scope": "Aggregate forces from all contact partners in the single-G1 ground-only scene",
                "sampling": "Post-physics-step poses, world twists and forces; explicit efforts are those held over that step",
                "point_velocity_limitation": "Velocity is measured at the rigid-body origin, not a contact point"})

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
        applied_armature = finite(array(robot.data.joint_armature), (43,))
        wanted_armature = np.array([contract["lower_body_armature"] if n in lower
                                   else contract["upper_body_armature"] for n in names])
        actuator_evidence["joint_armature"] = dict(zip(names, applied_armature.tolist(), strict=True))
        write_json(directory / "actuators.json", actuator_evidence)
        if not np.allclose(applied_armature, wanted_armature, rtol=0, atol=1e-7):
            raise RuntimeError("Configured joint armature was not applied to the Isaac articulation")
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
            with (directory / "trajectory.jsonl").open("w") as trajectory, \
                    ((directory / "contacts.jsonl").open("w") if diagnostics else nullcontext(None)) as contact_log:
                for step in range(profile["control_steps"]):
                    q, dq = array(robot.data.joint_pos), array(robot.data.joint_vel)
                    pose, vel = array(robot.data.root_link_pose_w), array(robot.data.root_link_vel_w)
                    observation = history.observe(q[indices], dq[indices], pose[3:], vel[3:], rotation)
                    if diagnostics:
                        input_state = {"joint_positions": q.tolist(), "joint_velocities": dq.tolist(),
                                       "root_pose_xyzw": pose.tolist(), "root_velocity_world": vel.tolist()}
                    action = policy(observation)
                    try:
                        target, clipped = history.targets(action, limits[lower_indices])
                    except (RuntimeError, ValueError):
                        write_json(directory / "rejected-action.json", {
                            "step": step+1, "sim_time_seconds": step*dt, "applied": False,
                            "limit": contract["action_limit"], "maximum_absolute_action": float(np.max(np.abs(action))),
                            "maximum_action_joint": lower[int(np.argmax(np.abs(action)))],
                            "observation": observation.tolist(), "action": action.tolist(),
                            "joint_positions": q.tolist(), "joint_velocities": dq.tolist(),
                            "root_pose_xyzw": pose.tolist(), "root_velocity_world": vel.tolist(),
                            "source_to_asset_rotation": rotation.tolist(),
                            "controller_body_joint_indices": indices, "lower_body_joint_names": lower})
                        raise
                    targets = defaults.copy()
                    targets[lower_indices] = target
                    robot.set_joint_position_target_index(target=device(targets))
                    for substep in range(asset["control_decimation"]):
                        scene.write_data_to_sim()
                        sim.step(render=False)
                        scene.update(asset["physics_dt"])
                        if diagnostics:
                            data = contact_sensor.data
                            normal = finite(array(data.net_normal_forces_w), (2, 3))[sensor_indices]
                            friction = finite(array(data.net_friction_forces_w), (2, 3))[sensor_indices]
                            total = finite(array(data.net_forces_w), (2, 3))[sensor_indices]
                            foot_poses = finite(array(robot.data.body_link_pose_w), (len(robot.body_names), 7))[foot_body_indices]
                            foot_velocities = finite(array(robot.data.body_link_vel_w), (len(robot.body_names), 6))[foot_body_indices]
                            computed = finite(array(robot.actuators.computed_effort), (43,))[lower_indices]
                            applied = finite(array(robot.actuators.applied_effort), (43,))[lower_indices]
                            physics_step = step*asset["control_decimation"]+substep+1
                            contact_log.write(json.dumps({
                                "physics_step": physics_step, "control_step": step+1,
                                "sim_time_seconds": physics_step*asset["physics_dt"],
                                "lower_body_computed_effort": computed.tolist(),
                                "lower_body_applied_effort": applied.tolist(),
                                "feet": {name: {"pose_xyzw": foot_poses[i].tolist(),
                                    "velocity_world": foot_velocities[i].tolist(),
                                    "normal_force_world_n": normal[i].tolist(),
                                    "friction_force_world_n": friction[i].tolist(),
                                    "total_force_world_n": total[i].tolist()} for i, name in enumerate(feet)}
                            }, allow_nan=False)+"\n")
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
                    if diagnostics:
                        record.update(policy_observation=observation.tolist(), policy_input_state=input_state)
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
        contract, reference = load_contract(ROOT / profile["controller_contract"])
        armature = json.loads((directory / "actuators.json").read_text())["joint_armature"]
        asset, _ = load_asset_profile(ROOT / profile["asset_profile"])
        if set(armature) != set(reference.limits) or any(
                not np.isclose(value, contract["lower_body_armature"] if name in asset["joint_groups"]["lower_body"]
                               else contract["upper_body_armature"], rtol=0, atol=1e-7)
                for name, value in armature.items()):
            raise RuntimeError("Incomplete or incorrect applied armature evidence")
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
        if profile.get("diagnostics"):
            analysis = analyze_run(directory)
            if analysis["status"] != "diagnostics_complete":
                raise RuntimeError("Incomplete contact diagnostics")
            if not analysis["contacts"]["force_evidence_present"]:
                raise RuntimeError("Contact diagnostics recorded no supporting foot-force evidence")
            materials = json.loads((directory / "contact-materials.json").read_text())
            if (set(materials["scopes"]) != {"ground", *profile["diagnostics"]["foot_body_names"]} or
                    any(not shapes for shapes in materials["scopes"].values())):
                raise RuntimeError("Incomplete diagnostic material evidence")


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
    parser.add_argument("--analyze-run", type=Path, help="Analyze saved artifacts on CPU without Isaac, ONNX or Modal")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", choices=("policy", "gpu", "balance"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.analyze_run:
        print(json.dumps(analyze_run(args.analyze_run), indent=2))
        return 0
    profile, asset, runtime = load_plan(args.profile)
    stages = ["policy"] if args.stage == "policy" else ["policy", "gpu", "balance"]
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "stages": stages, "asset": asset["asset_config"],
                          "control_steps": profile["control_steps"], "simulated_seconds": profile["control_steps"]*0.02,
                          "model_download_bytes": 1886682 if args.prepare_model else 0,
                          "contact_diagnostics": bool(profile.get("diagnostics")),
                          "physics_samples": profile["control_steps"]*asset["control_decimation"] if args.stage == "balance" and profile.get("diagnostics") else 0,
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
        if profile.get("diagnostics") and (directory / "trajectory.jsonl").is_file():
            try:
                write_json(directory / "diagnostics-summary.json", analyze_run(directory))
            except (OSError, ValueError, RuntimeError, KeyError):
                (directory / "diagnostics-error.log").write_text(traceback.format_exc())
        write_json(directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
