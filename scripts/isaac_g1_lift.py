"""G1 lifts, carries and sets down light furniture held by hand-object attachment (single-environment probe).

Grasp abstraction: two fixed joints, authored before reset, attach the palms to the object at the default arm pose,
and hand/wrist collisions with the object are filtered. The object's weight and contacts reach the robot through the
joints. Arm targets for lifting and lowering come from position-and-orientation IK on the pinned reference model;
GR00T Balance/Walk drive the lower body on the source schedule (Balance whenever the command norm is at most 0.05).
Disabling the joints at runtime (release) is attempted and measured, not assumed.
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
from isaac_g1_wbc import ROOT, load_plan, trial_contract
from isaac_preflight import check_versions, run_stage, write_json
from isaac_video import encode_recording, require_ffmpeg


DEFAULT_CONFIG = ROOT / "configs/isaac_g1_lift.json"
ARM_POSES = ("start", "lift", "place")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"push_profile", "object", "arm_offsets_m", "timeline", "checks", "camera", "stage_timeout_seconds",
                       "scope"}:
        raise ValueError("Unreviewed lift configuration keys")
    profile, asset, runtime = load_plan(ROOT / config["push_profile"])
    if profile.get("arm_hold") != "implicit" or profile.get("initialization", {}).get("mode") != "collision_clearance":
        raise ValueError("Lifting starts from the reviewed implicit-arm crouch-start profile")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    o = config["object"]
    if (set(o) != {"size_m", "mass_kg", "friction", "grasp_below_top_m"} or not isinstance(o["size_m"], list) or
            len(o["size_m"]) != 3 or not all(number(v, 0.1, 1.5) for v in o["size_m"]) or
            not number(o["mass_kg"], 0.1, 3.0) or not number(o["friction"], 0.05, 1.5) or
            not number(o["grasp_below_top_m"], 0.0, 0.1)):
        raise ValueError("Invalid lift object; mass above 3 kg exceeds the 5 Nm wrist pitch limit at this grasp")
    a = config["arm_offsets_m"]
    if set(a) != {"lift", "place"} or not number(a["lift"], 0.0, 0.12) or not number(a["place"], -0.05, 0.0):
        raise ValueError("Invalid vertical palm offsets")
    def arms_valid(arms):
        if isinstance(arms, str):
            return arms in ARM_POSES
        return isinstance(arms, list) and len(arms) == 2 and all(a in ARM_POSES for a in arms)

    for phase in config["timeline"]:
        if (set(phase) - {"release"} != {"name", "seconds", "command", "arms"} or not number(phase["seconds"], 0.1, 20.0) or
                not arms_valid(phase["arms"]) or phase["command"] not in ([0, 0, 0], [0.2, 0, 0]) or
                phase.get("release", False) not in (True, False)):
            raise ValueError("Timeline phases need a reviewed command (stand or 0.2 m/s forward) and arm poses")
    names = [p["name"] for p in config["timeline"]]
    if len(set(names)) != len(names) or not {"carry", "lower", "release", "withdraw"} <= set(names):
        raise ValueError("Timeline needs unique carry, lower, release and withdraw phases")
    c = config["checks"]
    if set(c) != {"attachment_error_m", "carry_clearance_m", "carry_distance_m", "placed_clearance_m", "object_tilt_deg"}:
        raise ValueError("Invalid lift checks")
    cam = config["camera"]
    if set(cam) != {"eye_m", "target_m", "capture_every_control_steps"} or type(cam["capture_every_control_steps"]) is not int:
        raise ValueError("Invalid lift camera")
    if type(config["stage_timeout_seconds"]) is not int or not 60 <= config["stage_timeout_seconds"] <= 1800:
        raise ValueError("Invalid stage_timeout_seconds")
    return config, profile, asset, runtime


def base_pose(reference, asset, contract):
    return {**dict.fromkeys(reference.limits, 0.0), **dict(zip(asset["joint_groups"]["lower_body"], contract["default_lower_angles"]))}


def rotation_vector(rotation):
    import numpy as np
    angle = np.arccos(np.clip((np.trace(rotation) - 1)/2, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    return angle/(2*np.sin(angle))*np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                                             rotation[1, 0] - rotation[0, 1]])


def arm_joints(reference, side):
    return [n for n in reference.limits if n.startswith(side) and any(k in n for k in ("shoulder", "elbow", "wrist"))]


def arm_pose(reference, base, side, dz, iterations=50):
    """Damped least-squares IK: move one palm by dz (pelvis frame) keeping its orientation; returns arm angles."""
    import numpy as np
    joints = arm_joints(reference, side)
    palm = f"{side}_hand_palm_link"
    target = reference.forward(base)[palm].copy()
    target[2, 3] += dz
    q = dict(base)
    for _ in range(iterations):
        current = reference.forward(q)[palm]
        error = np.r_[target[:3, 3] - current[:3, 3], rotation_vector(target[:3, :3] @ current[:3, :3].T)]
        if np.linalg.norm(error) < 1e-8:
            break
        jacobian = np.zeros((6, len(joints)))
        for j, name in enumerate(joints):
            moved = reference.forward({**q, name: q[name] + 1e-6})[palm]
            jacobian[:, j] = np.r_[moved[:3, 3] - current[:3, 3], rotation_vector(moved[:3, :3] @ current[:3, :3].T)]/1e-6
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3*np.eye(6), error)
        q.update({name: q[name] + step[j] for j, name in enumerate(joints)})
    if np.linalg.norm(error) > 1e-6:
        raise RuntimeError(f"Arm IK did not converge for {side} palm offset {dz}")
    for name in joints:
        if not reference.limits[name]["lower"] <= q[name] <= reference.limits[name]["upper"]:
            raise RuntimeError(f"Arm IK left the joint range: {name}")
    return {name: float(q[name]) for name in joints}


def attachment_frames(reference, base, settings, hand_reach):
    """Object pose in the pelvis frame at the default arm pose and each palm joint's frame in object coordinates.

    The near face sits at the fingertip reach, centred between the palms, with its top grasp_below_top_m above them.
    """
    import numpy as np
    fk = reference.forward(base)
    palms = {side: fk[f"{side}_hand_palm_link"] for side in ("left", "right")}
    size = np.asarray(settings["size_m"], dtype=float)
    top = np.mean([palms[side][2, 3] for side in palms]) + settings["grasp_below_top_m"]
    obj = np.eye(4)
    obj[:3, 3] = [hand_reach + size[0]/2, 0.0, top - size[2]/2]
    joints = {side: np.linalg.inv(obj) @ palm for side, palm in palms.items()}
    return {"pelvis_object": obj, "object_palm": joints, "palm_object": {s: np.linalg.inv(t) for s, t in joints.items()}}


def lowest_point_z(pose_xyzw, size):
    """Lowest box corner height for a box pose [x, y, z, qx, qy, qz, qw]."""
    import numpy as np
    from g1_wbc import quat_xyzw_matrix
    half = np.asarray(size, dtype=float)/2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])*half
    return float(np.min((corners @ quat_xyzw_matrix(pose_xyzw[3:]).T)[:, 2] + pose_xyzw[2]))


def tilt_deg(pose_xyzw):
    import numpy as np
    from g1_wbc import quat_xyzw_matrix
    return float(np.degrees(np.arccos(np.clip(quat_xyzw_matrix(pose_xyzw[3:])[2, 2], -1.0, 1.0))))


def apply_hand_attachment(stage, robot, box, env, frames):
    """Author palm-object fixed joints and hand/wrist collision filters in the session layer before reset."""
    from pxr import Gf, Usd, UsdPhysics
    from g1_wbc import matrix_quat_xyzw

    if robot.is_initialized or box.is_initialized:
        raise RuntimeError("Hand attachment must precede physics initialization")
    obj = stage.GetPrimAtPath(f"{env}/Furniture")
    links = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{env}/Robot"))
             if prim.HasAPI(UsdPhysics.RigidBodyAPI) and ("_hand_" in prim.GetName() or "wrist" in prim.GetName())]
    palms = {side: next((p for p in links if p.GetName() == f"{side}_hand_palm_link"), None) for side in ("left", "right")}
    if not obj or any(p is None or p.IsInstanceProxy() for p in palms.values()):
        raise RuntimeError(f"Expected the furniture and non-proxy palm bodies: {env}")
    report = {"joints": {}, "filtered_links": []}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for side, palm in palms.items():
            local = frames["object_palm"][side]
            quat = matrix_quat_xyzw(local[:3, :3])
            joint = UsdPhysics.FixedJoint.Define(stage, f"{env}/HandAttachment/{side}")
            joint.CreateBody0Rel().SetTargets([palm.GetPath()])
            joint.CreateBody1Rel().SetTargets([obj.GetPath()])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, local[:3, 3])))
            joint.CreateLocalRot1Attr(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateJointEnabledAttr(True)
            report["joints"][side] = {"path": str(joint.GetPath()), "palm": str(palm.GetPath()),
                                      "local_pos1_m": local[:3, 3].tolist(), "local_rot1_xyzw": quat.tolist()}
        filtered = [p for p in links if not p.IsInstanceProxy()]
        UsdPhysics.FilteredPairsAPI.Apply(obj).CreateFilteredPairsRel().SetTargets([p.GetPath() for p in filtered])
    report["filtered_links"] = [str(p.GetPath()) for p in filtered]
    report["unfiltered_instance_proxies"] = [str(p.GetPath()) for p in links if p.IsInstanceProxy()]
    return report


def release_attachment(stage, env):
    from pxr import Usd, UsdPhysics
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        return all(UsdPhysics.Joint(stage.GetPrimAtPath(f"{env}/HandAttachment/{side}")).GetJointEnabledAttr().Set(False)
                   for side in ("left", "right"))


def schedule(timeline, control_dt):
    """Per-control-step phase name, command, arm blend (from, to, fraction) and release flag."""
    steps = []
    for phase in timeline:
        count = int(round(phase["seconds"]/control_dt))
        arms = phase["arms"] if isinstance(phase["arms"], list) else [phase["arms"]]*2
        for k in range(count):
            steps.append({"phase": phase["name"], "command": phase["command"], "arms": arms, "fraction": (k + 1)/count,
                          "release": bool(phase.get("release")) and k == 0})
    return steps


def lift_summary(records, checks):
    """Functional checks over the logged control steps; release is reported, not required."""
    import numpy as np

    def phase(*names):
        return [r for r in records if r["phase"] in names]

    attached = [r for r in records if not r["released"]]
    carry, hold = phase("carry"), phase("hold")
    placed = phase("release")[-1] if phase("release") else None
    withdraw = phase("withdraw")
    start_x = carry[0]["object_xyz"][0] if carry else None
    result = {
        "attachment_error_max_m": max(r["attachment_error_m"] for r in attached),
        "carry_clearance_min_m": min(r["object_bottom_m"] for r in carry) if carry else None,
        "carry_distance_m": carry[-1]["object_xyz"][0] - start_x if carry else None,
        "robot_fallen": any(r["fallen"] for r in records),
        "placed_clearance_m": placed["object_bottom_m"] if placed else None,
        "placed_tilt_deg": placed["object_tilt_deg"] if placed else None,
        "hold_clearance_m": hold[-1]["object_bottom_m"] if hold else None}
    result["checks"] = {
        "attachment_held": result["attachment_error_max_m"] <= checks["attachment_error_m"],
        "lifted_during_carry": result["carry_clearance_min_m"] is not None and result["carry_clearance_min_m"] >= checks["carry_clearance_m"],
        "carried_forward": result["carry_distance_m"] is not None and result["carry_distance_m"] >= checks["carry_distance_m"],
        "robot_upright": not result["robot_fallen"],
        "set_down": result["placed_clearance_m"] is not None and result["placed_clearance_m"] <= checks["placed_clearance_m"],
        "object_upright": result["placed_tilt_deg"] is not None and result["placed_tilt_deg"] <= checks["object_tilt_deg"]}
    if withdraw and placed:
        palm_rise = float(np.mean([withdraw[-1]["palm_z"][s] - placed["palm_z"][s] for s in ("left", "right")]))
        object_rise = withdraw[-1]["object_bottom_m"] - placed["object_bottom_m"]
        result["release"] = {"palm_rise_m": palm_rise, "object_rise_m": object_rise,
                             "object_stayed": bool(palm_rise >= 0.01 and object_rise <= 0.3*palm_rise)}
    result["status"] = "passed_lift_probe" if all(result["checks"].values()) else "failed_lift_probe"
    return result


def run_worker(config, profile, asset, runtime, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp
    from PIL import Image
    from g1_wbc import BatchHistory, finite, matrix_quat_xyzw, pose_matrix, quat_xyzw_matrices

    contract, reference = trial_contract(profile)
    policy = ppo.gr00t_policy(contract, model_paths)
    encoder = require_ffmpeg(directory)
    torch.manual_seed(runtime["seed"])
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation

    o = config["object"]
    # The lift object replaces the push box in the shared scene (same spawn rule, its own size, mass and friction).
    scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": o["mass_kg"], "friction": o["friction"],
                                                "gap_m": profile["push_object"]["gap_m"]}}
    built = rl.scene_config(scene_profile, asset, runtime, contract, reference, with_camera=True, num_envs=1,
                            env_spacing=4.0, replicate_physics=False)
    base = base_pose(reference, asset, contract)
    frames = attachment_frames(reference, base, o, built.hand_reach)
    # The start pose is the default arm hold (all arm targets zero).
    arm_angles = {"start": {n: 0.0 for side in ("left", "right") for n in arm_joints(reference, side)}}
    for pose in ("lift", "place"):
        arm_angles[pose] = {n: v for side in ("left", "right")
                            for n, v in arm_pose(reference, base, side, config["arm_offsets_m"][pose]).items()}
    evidence = directory / "lift-worker.json"
    result = {"status": "running", "object": o, "arm_angles_rad": arm_angles,
              "pelvis_object": frames["pelvis_object"].tolist()}
    write_json(evidence, result)
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "enable_cameras": True, "device": runtime["device"]}):
        try:
            attachment = {}
            # The enabled joints pull the object off its push spawn in the reset warmup step; it is placed below.
            s = rl.prepare_push_scene(
                built, 1, scene_profile, asset, runtime, contract, reference, directory, result, evidence,
                extra_edits=lambda stage, robot, box, env: attachment.update(apply_hand_attachment(stage, robot, box, env, frames)),
                check_box_start=False)
            result["attachment"] = attachment
            robot, box, batch, device = s.robot, s.box, s.batch, s.device
            palm_ids = {side: s.bodies.index(f"{side}_hand_palm_link") for side in ("left", "right")}
            arm_ids = {name: s.names.index(name) for name in arm_angles["start"]}

            def palm_poses():
                poses = finite(batch(robot.data.body_link_pose_w)[0], (len(s.bodies), 7))
                return {side: pose_matrix(poses[i]) for side, i in palm_ids.items()}

            # Place the object where both joints expect it at the crouch start, then start stepping.
            palms = palm_poses()
            world_object = palms["left"] @ frames["palm_object"]["left"]
            mismatch = float(np.linalg.norm((palms["right"] @ frames["palm_object"]["right"])[:3, 3] - world_object[:3, 3]))
            object_pose = np.r_[world_object[:3, 3], matrix_quat_xyzw(world_object[:3, :3])]
            box.write_root_pose_to_sim_index(root_pose=device(object_pose[None]))
            box.write_root_velocity_to_sim_index(root_velocity=device(np.zeros((1, 6))))
            start_bottom = lowest_point_z(object_pose, o["size_m"])
            result["start"] = {"object_pose_xyzw": object_pose.tolist(), "object_bottom_m": start_bottom,
                               "palm_frame_mismatch_m": mismatch,
                               "after_reset_offset_m": (s.box_start[0, :3] - s.origins[0]).tolist()}
            write_json(evidence, result)
            if mismatch > 0.002 or not 0.002 <= start_bottom <= 0.06:
                raise RuntimeError("Attached object start pose is inconsistent with the palms or the floor")

            control_dt = asset["physics_dt"]*asset["control_decimation"]
            steps = schedule(config["timeline"], control_dt)
            history = BatchHistory(contract, 1)
            targets = s.defaults[None].copy()
            camera = s.scene["camera"]
            camera.set_world_poses_from_view(device(np.asarray(config["camera"]["eye_m"])[None]),
                                             device(np.asarray(config["camera"]["target_m"])[None]))
            (directory / "frames").mkdir()
            frame_count, released, records = 0, False, []
            width, height = profile["camera"]["width"], profile["camera"]["height"]

            def capture():
                nonlocal frame_count
                for _ in range(2):
                    s.sim.render()
                    camera.update(asset["physics_dt"], force_recompute=True)
                rgb = batch(camera.data.output["rgb"])[0, :, :, :3]
                if rgb.shape != (height, width, 3) or float(rgb.std()) < 1:
                    raise RuntimeError("Invalid lift camera frame")
                Image.fromarray(rgb.astype(np.uint8)).save(directory / "frames" / f"{frame_count:06d}.png")
                frame_count += 1

            capture()
            with (directory / "lift-trajectory.jsonl").open("w") as log:
                for k, step in enumerate(steps):
                    if step["release"]:
                        result["release_authored"] = release_attachment(s.sim.stage, "/World/envs/env_0")
                        released = True
                    for _ in range(asset["control_decimation"]):
                        s.scene.write_data_to_sim()
                        s.sim.step(render=False)
                        s.scene.update(asset["physics_dt"])
                    pose = finite(batch(robot.data.root_link_pose_w), (1, 7))
                    command = np.asarray(step["command"], dtype=float)[None]
                    observation = history.observe(batch(robot.data.joint_pos)[:, s.indices], batch(robot.data.joint_vel)[:, s.indices],
                                                  pose[:, 3:], batch(robot.data.root_link_vel_w)[:, 3:], s.rotation, command,
                                                  np.zeros((1, 3)))
                    lower_targets, _, invalid = history.targets(policy.batch(observation, command), s.limits)
                    if invalid.any():
                        raise RuntimeError("GR00T action beyond the transfer bound")
                    targets[:, s.lower_indices] = lower_targets
                    begin, end = (arm_angles[name] for name in step["arms"])
                    for name, index in arm_ids.items():
                        targets[0, index] = begin[name] + step["fraction"]*(end[name] - begin[name])
                    robot.set_joint_position_target_index(target=device(targets))
                    gravity = -(quat_xyzw_matrices(pose[:, 3:]) @ s.rotation)[0, 2, :]
                    obj = finite(batch(box.data.root_link_pose_w)[0], (7,))
                    palms = palm_poses()
                    predicted = (palms["left"] @ frames["palm_object"]["left"])[:3, 3]
                    record = {"time_s": (k + 1)*control_dt, "phase": step["phase"], "command": step["command"],
                              "released": released, "root_xyz": pose[0, :3].tolist(),
                              "robot_tilt_deg": float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1)))),
                              "fallen": bool(pose[0, 2] < profile["minimum_base_height_m"] or
                                             np.arccos(np.clip(-gravity[2], -1, 1)) > profile["maximum_tilt_rad"]),
                              "object_xyz": obj[:3].tolist(), "object_bottom_m": lowest_point_z(obj, o["size_m"]),
                              "object_tilt_deg": tilt_deg(obj), "attachment_error_m": float(np.linalg.norm(predicted - obj[:3])),
                              "palm_z": {side: float(t[2, 3]) for side, t in palms.items()}}
                    log.write(json.dumps(record) + "\n")
                    records.append(record)
                    if (k + 1) % config["camera"]["capture_every_control_steps"] == 0:
                        capture()
                    if record["fallen"]:
                        break
            result["summary"] = lift_summary(records, config["checks"])
            result["video"] = encode_recording(directory, binary=encoder, count=frame_count,
                                               fps=1/(Fraction(str(control_dt))*config["camera"]["capture_every_control_steps"]),
                                               width=width, height=height, require_motion=False)
            result["status"] = result["summary"]["status"]
            write_json(evidence, result)
        except BaseException as error:
            result.update(status="failed", error=str(error) or repr(error), error_type=type(error).__name__)
            write_json(evidence, result)
            (directory / "lift-error.log").write_text(traceback.format_exc())
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
    contract, reference = trial_contract(profile)
    if args.dry_run:
        base = base_pose(reference, asset, contract)
        poses = {pose: arm_pose(reference, base, "left", config["arm_offsets_m"][pose]) for pose in ("lift", "place")}
        print(json.dumps({"status": "local_plan_only", "object": config["object"], "left_arm_rad": poses,
                          "seconds": sum(p["seconds"] for p in config["timeline"]), "training": False}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(runtime))
        run_worker(config, profile, asset, runtime, ppo.gr00t_models(contract), args.run_dir)
        return 0
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-lift")
    directory.mkdir(parents=True)
    for name, value in (("lift-config.json", config), ("profile.json", profile), ("runtime-config.json", runtime),
                        ("controller-contract.json", contract)):
        write_json(directory / name, value)
    ppo.gr00t_models(contract)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--worker", "--run-dir", str(directory)]
    code = run_stage(command, directory / "lift.log", config["stage_timeout_seconds"])
    path = directory / "lift-worker.json"
    evidence = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
    summary = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code, "run_dir": str(directory),
               "start": evidence.get("start"), "summary": evidence.get("summary"), "error": evidence.get("error")}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "passed_lift_probe" else 1


if __name__ == "__main__":
    raise SystemExit(main())
