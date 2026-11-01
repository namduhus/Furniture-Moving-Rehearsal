"""G1 lifts, carries and sets down light furniture held by hand-object attachment (one environment per mass case).

Grasp abstraction: two fixed joints, authored before reset, attach the palms to the object's side faces at the
grasp start pose (the default arms with both palms moved forward by arm_offsets_m.forward; the palms face each other),
and hand/wrist collisions with the object are filtered. The object's weight and contacts reach the robot through the
joints. Arm targets for lifting and lowering come from position-and-orientation IK on the pinned reference model;
GR00T Balance/Walk drive the lower body on the source schedule (Balance whenever the command norm is at most 0.05).
Enabling the joints at runtime (grasp, when a timeline phase asks for it; the joints then start disabled and the
object rests on the floor) and disabling them (release) are attempted and measured, not assumed. A grasp re-authors
each joint's object frame to the current palm pose first, so enabling should not move the object; an optional reach
phase before it places the cabinet on the floor at the settled grasp pose plus object.approach_error_m (the approach
stop error) and moves each palm to its grasp point on the object by IK. Cases sweep the object mass, with and
without static load feedforward on the arm targets, to measure where lifting fails; arm tracking, the implied PD torque
and the object's contact with the legs and torso are logged to show what limits it.
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
SIDES = ("left", "right")
GRAVITY = 9.81
REACH_LIMIT_M = (0.08, 0.05, 0.05)   # palm reach offsets (forward, left, up) that keep the start-height IK converging


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"push_profile", "object", "cases", "arm_offsets_m", "timeline", "checks", "camera",
                       "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed lift configuration keys")
    profile, asset, runtime = load_plan(ROOT / config["push_profile"])
    if profile.get("arm_hold") != "implicit" or profile.get("initialization", {}).get("mode") != "collision_clearance":
        raise ValueError("Lifting starts from the reviewed implicit-arm crouch-start profile")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    o = config["object"]
    if (set(o) != {"size_m", "friction", "start_clearance_m", "near_face_ahead_of_palm_m", "approach_error_m"} or
            not isinstance(o["size_m"], list) or len(o["size_m"]) != 3 or not all(number(v, 0.1, 1.5) for v in o["size_m"]) or
            not number(o["friction"], 0.05, 1.5) or not number(o["start_clearance_m"], 0.0, 0.02) or
            not number(o["near_face_ahead_of_palm_m"], -0.3, 0.1) or not isinstance(o["approach_error_m"], list) or
            len(o["approach_error_m"]) != 2 or not all(number(v, -0.06, 0.06) for v in o["approach_error_m"])):
        raise ValueError("Invalid lift object")
    cases = config["cases"]
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 8 or
            any(set(c) != {"mass_kg", "feedforward"} or not number(c["mass_kg"], 0.1, 12.0) or
                c["feedforward"] not in (True, False) for c in cases) or
            len({(c["mass_kg"], c["feedforward"]) for c in cases}) != len(cases)):
        raise ValueError("Lift cases need 1-8 unique (mass 0.1-12 kg, feedforward) pairs")
    a = config["arm_offsets_m"]
    if (set(a) != {"forward", "lift", "place"} or not number(a["forward"], 0.0, 0.2) or not number(a["lift"], 0.0, 0.12) or
            not number(a["place"], -0.05, 0.0)):
        raise ValueError("Invalid palm offsets")
    def arms_valid(arms):
        if isinstance(arms, str):
            return arms in ARM_POSES + ("reach",)
        return isinstance(arms, list) and len(arms) == 2 and all(a in ARM_POSES + ("reach",) for a in arms)

    for phase in config["timeline"]:
        if (set(phase) - {"release", "grasp", "reach"} != {"name", "seconds", "command", "arms"} or
                not number(phase["seconds"], 0.1, 20.0) or not arms_valid(phase["arms"]) or
                phase["command"] not in ([0, 0, 0], [0.2, 0, 0]) or
                any(phase.get(flag, False) not in (True, False) for flag in ("release", "grasp", "reach"))):
            raise ValueError("Timeline phases need a reviewed command (stand or 0.2 m/s forward) and arm poses")
    names = [p["name"] for p in config["timeline"]]
    if len(set(names)) != len(names) or not {"carry", "lower", "release", "withdraw"} <= set(names):
        raise ValueError("Timeline needs unique carry, lower, release and withdraw phases")
    grasps = [i for i, p in enumerate(config["timeline"]) if p.get("grasp")]
    if grasps and (len(grasps) > 1 or "lift" not in names or grasps[0] > names.index("lift") or
                   config["timeline"][grasps[0]]["command"] != [0, 0, 0]):
        raise ValueError("At most one standing grasp phase, before the lift")
    reaches = [i for i, p in enumerate(config["timeline"]) if p.get("reach")]
    uses = [i for i, p in enumerate(config["timeline"]) if "reach" in (p["arms"] if isinstance(p["arms"], list) else [p["arms"]])]
    if ((reaches and (len(reaches) > 1 or not grasps or reaches[0] != grasps[0] - 1 or
                      config["timeline"][reaches[0]]["command"] != [0, 0, 0])) or (uses and (not reaches or uses[0] < reaches[0]))):
        raise ValueError("At most one standing reach phase, right before the grasp; the reach pose only from it on")
    c = config["checks"]
    if set(c) != {"attachment_error_m", "carry_clearance_m", "carry_distance_m", "placed_clearance_m", "object_tilt_deg"}:
        raise ValueError("Invalid lift checks")
    cam = config["camera"]
    if (set(cam) != {"eye_offset_m", "target_offset_m", "capture_every_control_steps", "env_spacing_m"} or
            type(cam["capture_every_control_steps"]) is not int or not number(cam["env_spacing_m"], 8.0, 20.0)):
        raise ValueError("Invalid lift camera; environments must be spaced so each camera sees only its own case")
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


def palm_jacobian(reference, q, side):
    """Palm pose and numeric Jacobian over one arm's joints: position and rotation-vector rates in the pelvis frame."""
    import numpy as np
    palm = f"{side}_hand_palm_link"
    current = reference.forward(q)[palm]
    joints = arm_joints(reference, side)
    jacobian = np.zeros((6, len(joints)))
    for j, name in enumerate(joints):
        moved = reference.forward({**q, name: q[name] + 1e-6})[palm]
        jacobian[:, j] = np.r_[moved[:3, 3] - current[:3, 3], rotation_vector(moved[:3, :3] @ current[:3, :3].T)]/1e-6
    return current, jacobian


def arm_pose(reference, base, side, dz, dx=0.0, dy=0.0, iterations=50):
    """Damped least-squares IK: move one palm by (dx, dy, dz) (pelvis frame) keeping its orientation; returns arm angles."""
    import numpy as np
    joints = arm_joints(reference, side)
    target = reference.forward(base)[f"{side}_hand_palm_link"].copy()
    target[0, 3] += dx
    target[1, 3] += dy
    target[2, 3] += dz
    q = dict(base)
    for _ in range(iterations):
        current, jacobian = palm_jacobian(reference, q, side)
        error = np.r_[target[:3, 3] - current[:3, 3], rotation_vector(target[:3, :3] @ current[:3, :3].T)]
        if np.linalg.norm(error) < 1e-8:
            break
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3*np.eye(6), error)
        q.update({name: q[name] + step[j] for j, name in enumerate(joints)})
    if np.linalg.norm(error) > 1e-6:
        raise RuntimeError(f"Arm IK did not converge for {side} palm offset ({dx}, {dy}, {dz})")
    for name in joints:
        if not reference.limits[name]["lower"] <= q[name] <= reference.limits[name]["upper"]:
            raise RuntimeError(f"Arm IK left the joint range: {name}")
    return {name: float(q[name]) for name in joints}


def start_pose(reference, base, offsets):
    """Whole-body pose with both palms moved forward by offsets["forward"] (the grasp start pose)."""
    return {**base, **{n: v for side in SIDES for n, v in arm_pose(reference, base, side, 0.0, offsets["forward"]).items()}}


def attachment_frames(reference, base, settings, height):
    """Object pose in the pelvis frame at the default arm pose and each palm joint's frame in object coordinates.

    Side grasp: the object is centred between the palms with its near face near_face_ahead_of_palm_m ahead of them and
    its bottom start_clearance_m above the floor when the pelvis stands at `height` (the policy height command, not the
    raised crouch start). Its side faces must lie at most 3 cm inside the palm origins and the palms at least 4 cm
    inside the face height.
    """
    import numpy as np
    fk = reference.forward(base)
    palms = {side: fk[f"{side}_hand_palm_link"] for side in ("left", "right")}
    size = np.asarray(settings["size_m"], dtype=float)
    centre = np.mean([palms[side][:3, 3] for side in palms], axis=0)
    obj = np.eye(4)
    obj[:3, 3] = [centre[0] + settings["near_face_ahead_of_palm_m"] + size[0]/2, centre[1],
                  settings["start_clearance_m"] - height + size[2]/2]
    joints = {side: np.linalg.inv(obj) @ palm for side, palm in palms.items()}
    inset = min(abs(t[1, 3]) for t in joints.values()) - size[1]/2
    below_top = size[2]/2 - max(t[2, 3] for t in joints.values())
    if not 0.0 <= inset <= 0.03 or not 0.04 <= below_top <= size[2] - 0.04:
        raise ValueError("The object's side faces must meet the palms and the palms must lie within the face height")
    return {"pelvis_object": obj, "object_palm": joints, "palm_object": {s: np.linalg.inv(t) for s, t in joints.items()},
            "palm_inset_m": float(inset), "grasp_below_top_m": float(below_top)}


def load_feedforward(reference, q, side, frames, mass, kp, efforts):
    """Arm target offsets that let the implicit PD hold half the object's weight at pose q (static, pelvis upright).

    The palm carries (0, 0, -m g/2) at the object's centre; the arm must supply tau = -J^T w, reached at q by offsetting
    the targets by tau/kp. The arm's own weight is not included. Joints whose torque exceeds the effort limit are listed.
    """
    import numpy as np
    palm, jacobian = palm_jacobian(reference, q, side)
    centre = (palm @ frames["palm_object"][side])[:3, 3]
    force = np.array([0.0, 0.0, -mass*GRAVITY/2])
    torque = -jacobian.T @ np.r_[force, np.cross(centre - palm[:3, 3], force)]
    joints = arm_joints(reference, side)
    return {"offset_rad": {n: float(torque[j]/kp) for j, n in enumerate(joints)},
            "torque_nm": {n: float(torque[j]) for j, n in enumerate(joints)},
            "over_limit": [n for j, n in enumerate(joints) if abs(torque[j]) > efforts[n]]}


def arm_plan(reference, base, config, frames, contract):
    """Arm joint order, per-case target arrays for each arm pose and per-case load evidence.

    All three poses keep the palms offsets["forward"] ahead of the default arms. Feedforward cases add the load offsets
    to the lift pose only, so they ramp in while lifting and out while lowering to the floor-supported place pose.
    """
    import numpy as np
    from g1_wbc import actuator_limits
    joints = [n for side in SIDES for n in arm_joints(reference, side)]
    a = config["arm_offsets_m"]
    angles = {pose: {n: v for side in SIDES for n, v in arm_pose(reference, base, side, dz, a["forward"]).items()}
              for pose, dz in (("start", 0.0), ("lift", a["lift"]), ("place", a["place"]))}
    lifted = {**base, **angles["lift"]}
    efforts = {n: v["effort"] for n, v in actuator_limits(contract, reference).items()}
    cases = config["cases"]
    targets = {pose: np.tile([angles[pose][n] for n in joints], (len(cases), 1)) for pose in ARM_POSES}
    loads = []
    for i, case in enumerate(cases):
        load = {side: load_feedforward(reference, lifted, side, frames, case["mass_kg"], contract["upper_hold_kp"], efforts)
                for side in SIDES}
        if case["feedforward"]:
            targets["lift"][i] += [load["left" if n.startswith("left") else "right"]["offset_rad"][n] for n in joints]
        loads.append(load)
    return joints, angles, targets, loads


def reach_plan(reference, base, forward, joints, object_palm, palms, objects, pelvis):
    """Per-case arm targets that move each palm to its grasp point on the object (palms: side -> world palm poses,
    objects: world object poses, pelvis: world pelvis rotations).

    Each palm's offset to its grasp point is measured in the pelvis frame, clipped to REACH_LIMIT_M and added to the
    start pose by IK; this is first order (the model palm is assumed to move like the simulated one). A case whose IK
    fails keeps the start pose and reports the error.
    """
    import numpy as np
    limit = np.asarray(REACH_LIMIT_M)
    targets, report = [], []
    for i, obj in enumerate(objects):
        offsets = {side: pelvis[i].T @ ((obj @ object_palm[side])[:3, 3] - palms[side][i][:3, 3]) for side in SIDES}
        applied = {side: np.clip(offset, -limit, limit) for side, offset in offsets.items()}
        try:
            angles, error = {n: v for side in SIDES for n, v in arm_pose(reference, base, side, applied[side][2],
                                                                         forward + applied[side][0], applied[side][1]).items()}, None
        except RuntimeError as failure:
            angles, error = {n: v for side in SIDES for n, v in arm_pose(reference, base, side, 0.0, forward).items()}, str(failure)
        targets.append([angles[n] for n in joints])
        report.append({"offset_m": {side: offset.tolist() for side, offset in offsets.items()},
                       "applied_m": None if error else {side: value.tolist() for side, value in applied.items()}, "error": error})
    return np.array(targets), report


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


def apply_hand_attachment(stage, robot, box, env, frames, enabled=True):
    """Author palm-object fixed joints (enabled or not) and hand/wrist collision filters in the session layer before
    reset."""
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
            joint.CreateJointEnabledAttr(bool(enabled))
            report["joints"][side] = {"path": str(joint.GetPath()), "palm": str(palm.GetPath()),
                                      "local_pos1_m": local[:3, 3].tolist(), "local_rot1_xyzw": quat.tolist()}
        filtered = [p for p in links if not p.IsInstanceProxy()]
        UsdPhysics.FilteredPairsAPI.Apply(obj).CreateFilteredPairsRel().SetTargets([p.GetPath() for p in filtered])
    report["filtered_links"] = [str(p.GetPath()) for p in filtered]
    report["unfiltered_instance_proxies"] = [str(p.GetPath()) for p in links if p.IsInstanceProxy()]
    return report


def set_attachment(stage, env, enabled, object_palm=None):
    """Enable (grasp) or disable (release) both palm joints at runtime; object_palm (side -> palm pose in object
    coordinates) first re-authors each joint's object frame so the joint holds the current relative pose."""
    from pxr import Gf, Usd, UsdPhysics
    from g1_wbc import matrix_quat_xyzw
    done = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for side in ("left", "right"):
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{env}/HandAttachment/{side}"))
            if object_palm is not None:
                quat = matrix_quat_xyzw(object_palm[side][:3, :3])
                joint.GetLocalPos1Attr().Set(Gf.Vec3f(*map(float, object_palm[side][:3, 3])))
                joint.GetLocalRot1Attr().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
            done.append(joint.GetJointEnabledAttr().Set(bool(enabled)))
    return all(done)


def schedule(timeline, control_dt):
    """Per-control-step phase name, command, arm blend (from, to, fraction) and reach/grasp/release flags."""
    steps = []
    for phase in timeline:
        count = int(round(phase["seconds"]/control_dt))
        arms = phase["arms"] if isinstance(phase["arms"], list) else [phase["arms"]]*2
        for k in range(count):
            steps.append({"phase": phase["name"], "command": phase["command"], "arms": arms, "fraction": (k + 1)/count,
                          "reach": bool(phase.get("reach")) and k == 0, "grasp": bool(phase.get("grasp")) and k == 0,
                          "release": bool(phase.get("release")) and k == 0})
    return steps


def lift_summary(records, checks):
    """Functional checks over the logged control steps; grasp and release are reported, not required."""
    import numpy as np

    def phase(*names):
        return [r for r in records if r["phase"] in names]

    attached = [r for r in records if r["attached"]]
    settle, carry, hold = phase("settle"), phase("carry"), phase("hold")
    held = phase("hold", "carry", "stop")
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
        "settled_clearance_m": settle[-1]["object_bottom_m"] if settle else None,
        "hold_clearance_m": hold[-1]["object_bottom_m"] if hold else None,
        "palm_rise_m": float(np.mean([hold[-1]["palm_z"][s] - settle[-1]["palm_z"][s] for s in SIDES])) if hold and settle else None,
        "held_tilt_max_deg": max(r["object_tilt_deg"] for r in held) if held else None}
    result["checks"] = {
        "attachment_held": result["attachment_error_max_m"] <= checks["attachment_error_m"],
        "lifted_during_carry": result["carry_clearance_min_m"] is not None and result["carry_clearance_min_m"] >= checks["carry_clearance_m"],
        "carried_forward": result["carry_distance_m"] is not None and result["carry_distance_m"] >= checks["carry_distance_m"],
        "robot_upright": not result["robot_fallen"],
        "actions_in_bound": not any(r.get("invalid_action", False) for r in records),
        "set_down": result["placed_clearance_m"] is not None and result["placed_clearance_m"] <= checks["placed_clearance_m"],
        "object_upright": result["placed_tilt_deg"] is not None and result["placed_tilt_deg"] <= checks["object_tilt_deg"]}
    grasp = phase("grasp")
    if grasp:
        before, reach = records[records.index(grasp[0]) - 1], phase("reach")
        result["grasp"] = {"palm_gap_reach_start_m": reach[0]["palm_gap_m"] if reach else None,
                           "palm_gap_at_grasp_m": before["palm_gap_m"],
                           "error_before_m": before["attachment_error_m"], "error_after_m": grasp[-1]["attachment_error_m"],
                           "object_shift_m": float(np.linalg.norm(np.subtract(grasp[0]["object_xyz"], before["object_xyz"])))}
    if withdraw and placed:
        palm_rise = float(np.mean([withdraw[-1]["palm_z"][s] - placed["palm_z"][s] for s in SIDES]))
        object_rise = withdraw[-1]["object_bottom_m"] - placed["object_bottom_m"]
        result["release"] = {"palm_rise_m": palm_rise, "object_rise_m": object_rise,
                             "object_stayed": bool(palm_rise >= 0.01 and object_rise <= 0.3*palm_rise)}
    result["status"] = "passed_lift_probe" if all(result["checks"].values()) else "failed_lift_probe"
    return result


def diagnostics_summary(records, joints, efforts):
    """What limits the lift: arm tracking error and implied PD torque during hold, drive saturation while held, and the
    object's contact force on the legs and torso.

    The torque is the implicit drive law kp (target - q) - kd qd on logged states and the targets applied over the
    preceding interval; a magnitude at or above the effort limit means the drive saturates. Joint friction is excluded.
    """
    import numpy as np
    held = [r for r in records if r["phase"] in ("hold", "carry", "stop")]
    hold = [r for r in records if r["phase"] == "hold"]
    limit = np.asarray(efforts, dtype=float)
    held_ratio = np.abs([r["arm_pd_torque_nm"] for r in held])/limit
    hold_ratio = np.abs([r["arm_pd_torque_nm"] for r in hold])/limit
    error = np.array([r["arm_error_rad"] for r in hold])
    contact = np.array([r["body_contact_n"] for r in held])
    bodies = [r["body_contact_body"] for r in held if r["body_contact_body"]]
    short = [n.removesuffix("_joint") for n in joints]
    return {"hold_left_arm": {short[j].removeprefix("left_"): {"error_rad": float(error[:, j].mean()),
                                                              "torque_ratio": float(hold_ratio[:, j].mean())}
                              for j, n in enumerate(joints) if n.startswith("left")},
            "saturated_joints": [short[j] for j in range(len(joints)) if (held_ratio[:, j] >= 1.0).mean() > 0.5],
            "body_contact": {"held_max_n": float(contact.max()), "held_fraction": float((contact > 1.0).mean()),
                             "most_frequent_body": max(sorted(set(bodies)), key=bodies.count) if bodies else None}}


def run_worker(config, profile, asset, runtime, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp
    from PIL import Image
    from g1_wbc import BatchHistory, actuator_limits, finite, matrix_quat_xyzw, pose_matrix, quat_xyzw_matrices

    contract, reference = trial_contract(profile)
    policy = ppo.gr00t_policy(contract, model_paths)
    encoder = require_ffmpeg(directory)
    torch.manual_seed(runtime["seed"])
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation

    o, cases, cam = config["object"], config["cases"], config["camera"]
    count = len(cases)
    masses = np.array([c["mass_kg"] for c in cases], dtype=float)
    # The lift object replaces the push box in the shared scene (same spawn rule, its own size and friction); each
    # environment authors its case mass.
    scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": float(masses[0]), "friction": o["friction"],
                                                "gap_m": profile["push_object"]["gap_m"]}}
    built = rl.scene_config(scene_profile, asset, runtime, contract, reference, with_camera=True, num_envs=count,
                            env_spacing=cam["env_spacing_m"], replicate_physics=False, with_body_contact=True)
    base = base_pose(reference, asset, contract)
    frames = attachment_frames(reference, start_pose(reference, base, config["arm_offsets_m"]), o, contract["height_command"])
    joints, angles, plan, loads = arm_plan(reference, base, config, frames, contract)
    efforts = [actuator_limits(contract, reference)[n]["effort"] for n in joints]
    kp, kd = contract["upper_hold_kp"], contract["upper_hold_kd"]
    evidence = directory / "lift-worker.json"
    result = {"status": "running", "object": o, "cases": cases, "arm_joints": joints, "arm_effort_nm": efforts,
              "arm_angles_rad": angles,
              "lift_targets_rad": plan["lift"].tolist(), "loads": loads, "pelvis_object": frames["pelvis_object"].tolist(),
              "grasp": {k: frames[k] for k in ("palm_inset_m", "grasp_below_top_m")}}
    write_json(evidence, result)
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "enable_cameras": True, "device": runtime["device"]}):
        try:
            attachment = {}
            # With a grasp phase the joints start disabled and the object rests on the floor until the grasp.
            grasping = any(phase.get("grasp") for phase in config["timeline"])
            result["grasp_phase"] = grasping
            # The enabled joints pull the object off its push spawn in the reset warmup step; it is placed below.
            s = rl.prepare_push_scene(
                built, count, scene_profile, asset, runtime, contract, reference, directory, result, evidence,
                box_properties=(masses, np.full(count, float(o["friction"]))),
                extra_edits=lambda stage, robot, box, env: attachment.update({env: apply_hand_attachment(stage, robot, box, env, frames,
                                                                                                         not grasping)}),
                check_box_start=False)
            result["attachment"] = attachment
            robot, box, batch, device = s.robot, s.box, s.batch, s.device
            palm_ids = {side: s.bodies.index(f"{side}_hand_palm_link") for side in SIDES}
            arm_ids = [s.names.index(name) for name in joints]
            # The arms take the grasp start pose (palms moved forward) before the object is placed at the palms.
            start = np.tile(s.defaults, (count, 1))
            start[:, arm_ids] = plan["start"]
            robot.write_joint_position_to_sim_index(position=device(start))
            robot.write_joint_velocity_to_sim_index(velocity=device(np.zeros((count, 29))))
            robot.set_joint_position_target_index(target=device(start))
            s.sim.forward()
            s.scene.update(0.0)
            sensor = s.scene["body_contact"]
            contact_bodies = list(sensor.body_names)

            def palm_poses():
                poses = finite(batch(robot.data.body_link_pose_w), (count, len(s.bodies), 7))
                return {side: [pose_matrix(p) for p in poses[:, k]] for side, k in palm_ids.items()}

            # Place each object where both joints expect it at the crouch start, then start stepping. The crouch start
            # raises the robot above the policy height, so the object starts that much above its standing clearance.
            palms = palm_poses()
            world = [palms["left"][i] @ frames["palm_object"]["left"] for i in range(count)]
            mismatch = [float(np.linalg.norm((palms["right"][i] @ frames["palm_object"]["right"])[:3, 3] - world[i][:3, 3]))
                        for i in range(count)]
            object_pose = np.array([np.r_[w[:3, 3], matrix_quat_xyzw(w[:3, :3])] for w in world])
            box.write_root_pose_to_sim_index(root_pose=device(object_pose))
            box.write_root_velocity_to_sim_index(root_velocity=device(np.zeros((count, 6))))
            bottoms = [lowest_point_z(pose, o["size_m"]) for pose in object_pose]
            expected_bottom = o["start_clearance_m"] + result["initialization"]["lift_m"]
            result["start"] = {"object_bottom_m": bottoms, "expected_bottom_m": expected_bottom,
                               "palm_frame_mismatch_m": mismatch,
                               "after_reset_offset_m": (s.box_start[:, :3] - s.origins).tolist()}
            write_json(evidence, result)
            if max(mismatch) > 0.002 or max(abs(b - expected_bottom) for b in bottoms) > 0.01:
                raise RuntimeError("Attached object start pose is inconsistent with the palms or the standing height")

            control_dt = asset["physics_dt"]*asset["control_decimation"]
            steps = schedule(config["timeline"], control_dt)
            history = BatchHistory(contract, count)
            targets = start.copy()
            camera = s.scene["camera"]
            camera.set_world_poses_from_view(device(s.origins + cam["eye_offset_m"]), device(s.origins + cam["target_offset_m"]))
            directories = [directory / f"case_{i}" for i in range(count)]
            for case in directories:
                (case / "frames").mkdir(parents=True)
            frame_count, attached, released, records = 0, not grasping, False, [[] for _ in range(count)]
            # Left palm -> object transform each case's attachment error is measured against (re-authored at the grasp).
            palm_object = [frames["palm_object"]["left"]]*count
            width, height = profile["camera"]["width"], profile["camera"]["height"]

            def capture():
                nonlocal frame_count
                for _ in range(2):
                    s.sim.render()
                    camera.update(asset["physics_dt"], force_recompute=True)
                rgb = batch(camera.data.output["rgb"])[..., :3]
                for i, case in enumerate(directories):
                    if rgb[i].shape != (height, width, 3) or float(rgb[i].std()) < 1:
                        raise RuntimeError("Invalid lift camera frame")
                    Image.fromarray(rgb[i].astype(np.uint8)).save(case / "frames" / f"{frame_count:06d}.png")
                frame_count += 1

            # No frame before the first step: the render still shows the warmup-step robot pose then (run 013531).
            logs = [(case / "trajectory.jsonl").open("w") for case in directories]
            try:
                for k, step in enumerate(steps):
                    if step["reach"]:
                        # Probe set-up: the settle drifts the robot, so the cabinet is moved on the floor to the settled
                        # grasp pose plus the approach error (pelvis heading frame) before the palms reach for it.
                        palms = palm_poses()
                        pelvis = quat_xyzw_matrices(finite(batch(robot.data.root_link_pose_w), (count, 7))[:, 3:]) @ s.rotation
                        placed = []
                        for i in range(count):
                            yaw = np.arctan2(pelvis[i][1, 0], pelvis[i][0, 0])
                            heading = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                            xy = (palms["left"][i] @ frames["palm_object"]["left"])[:2, 3] + heading @ o["approach_error_m"]
                            placed.append([*xy, s.origins[i, 2] + o["size_m"][2]/2, 0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)])
                        placed = np.array(placed)
                        box.write_root_pose_to_sim_index(root_pose=device(placed))
                        box.write_root_velocity_to_sim_index(root_velocity=device(np.zeros((count, 6))))
                        plan["reach"], result["reach"] = reach_plan(reference, base, config["arm_offsets_m"]["forward"], joints,
                                                                   frames["object_palm"], palms, [pose_matrix(p) for p in placed], pelvis)
                    if step["grasp"]:
                        # Grasp where the hands are: each joint takes the current palm pose in object coordinates.
                        palms = palm_poses()
                        objects = [pose_matrix(p) for p in finite(batch(box.data.root_link_pose_w), (count, 7))]
                        held = [{side: np.linalg.inv(objects[i]) @ palms[side][i] for side in SIDES} for i in range(count)]
                        result["grasp_authored"] = all([set_attachment(s.sim.stage, f"/World/envs/env_{i}", True, held[i])
                                                        for i in range(count)])
                        result["grasp_frames"] = [
                            {side: {"shift_m": float(np.linalg.norm(h[side][:3, 3] - frames["object_palm"][side][:3, 3])),
                                    "rotation_deg": float(np.degrees(np.linalg.norm(
                                        rotation_vector(h[side][:3, :3] @ frames["object_palm"][side][:3, :3].T))))}
                             for side in SIDES} for h in held]
                        palm_object = [np.linalg.inv(h["left"]) for h in held]
                        attached = True
                    if step["release"]:
                        result["release_authored"] = all([set_attachment(s.sim.stage, f"/World/envs/env_{i}", False)
                                                          for i in range(count)])
                        attached, released = False, True
                    for _ in range(asset["control_decimation"]):
                        s.scene.write_data_to_sim()
                        s.sim.step(render=False)
                        s.scene.update(asset["physics_dt"])
                    pose = finite(batch(robot.data.root_link_pose_w), (count, 7))
                    # Arm tracking against the targets applied over the interval just simulated.
                    arm_error = targets[:, arm_ids] - batch(robot.data.joint_pos)[:, arm_ids]
                    arm_torque = kp*arm_error - kd*batch(robot.data.joint_vel)[:, arm_ids]
                    contact = np.linalg.norm(batch(sensor.data.force_matrix_w), axis=-1).sum(axis=-1)
                    command = np.tile(np.asarray(step["command"], dtype=float), (count, 1))
                    observation = history.observe(batch(robot.data.joint_pos)[:, s.indices], batch(robot.data.joint_vel)[:, s.indices],
                                                  pose[:, 3:], batch(robot.data.root_link_vel_w)[:, 3:], s.rotation, command,
                                                  np.zeros((count, 3)))
                    # An action beyond the transfer bound fails only its own case; the bounded action is applied.
                    lower_targets, _, invalid = history.targets(policy.batch(observation, command), s.limits)
                    targets[:, s.lower_indices] = lower_targets
                    begin, end = (plan[name] for name in step["arms"])
                    targets[:, arm_ids] = begin + step["fraction"]*(end - begin)
                    robot.set_joint_position_target_index(target=device(targets))
                    gravity = -(quat_xyzw_matrices(pose[:, 3:]) @ s.rotation)[:, 2, :]
                    tilt = np.arccos(np.clip(-gravity[:, 2], -1, 1))
                    obj = finite(batch(box.data.root_link_pose_w), (count, 7))
                    palms = palm_poses()
                    for i in range(count):
                        predicted = (palms["left"][i] @ palm_object[i])[:3, 3]
                        grasp_points = {side: (pose_matrix(obj[i]) @ frames["object_palm"][side])[:3, 3] for side in SIDES}
                        record = {"time_s": (k + 1)*control_dt, "phase": step["phase"], "command": step["command"],
                                  "attached": attached, "released": released, "root_xyz": (pose[i, :3] - s.origins[i]).tolist(),
                                  "robot_tilt_deg": float(np.degrees(tilt[i])),
                                  "fallen": bool(pose[i, 2] < profile["minimum_base_height_m"] or tilt[i] > profile["maximum_tilt_rad"]),
                                  "invalid_action": bool(invalid[i]),
                                  "object_xyz": (obj[i, :3] - s.origins[i]).tolist(), "object_bottom_m": lowest_point_z(obj[i], o["size_m"]),
                                  "object_tilt_deg": tilt_deg(obj[i]),
                                  "attachment_error_m": float(np.linalg.norm(predicted - obj[i, :3])),
                                  "palm_gap_m": max(float(np.linalg.norm(grasp_points[side] - palms[side][i][:3, 3])) for side in SIDES),
                                  "palm_z": {side: float(palms[side][i][2, 3]) for side in SIDES},
                                  "arm_error_rad": arm_error[i].tolist(), "arm_pd_torque_nm": arm_torque[i].tolist(),
                                  "body_contact_n": float(contact[i].sum()),
                                  "body_contact_body": contact_bodies[int(np.argmax(contact[i]))] if contact[i].sum() > 1.0 else None}
                        logs[i].write(json.dumps(record) + "\n")
                        records[i].append(record)
                    if (k + 1) % cam["capture_every_control_steps"] == 0:
                        capture()
            finally:
                for log in logs:
                    log.close()
            result["results"] = []
            for i, case in enumerate(cases):
                video = encode_recording(directories[i], binary=encoder, count=frame_count,
                                         fps=1/(Fraction(str(control_dt))*cam["capture_every_control_steps"]),
                                         width=width, height=height, require_motion=False)
                result["results"].append({"case": directories[i].name, **case, "summary": lift_summary(records[i], config["checks"]),
                                          "diagnostics": diagnostics_summary(records[i], joints, efforts), "video": video})
            result["heaviest_passed_kg"] = {
                label: max((r["mass_kg"] for r in result["results"] if r["feedforward"] == flag and
                            r["summary"]["status"] == "passed_lift_probe"), default=None)
                for label, flag in (("uncompensated", False), ("feedforward", True))}
            result["status"] = "completed_lift_sweep"
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
        base, offsets = base_pose(reference, asset, contract), config["arm_offsets_m"]
        poses = {pose: arm_pose(reference, base, "left", dz, offsets["forward"])
                 for pose, dz in (("start", 0.0), ("lift", offsets["lift"]), ("place", offsets["place"]))}
        frames = attachment_frames(reference, start_pose(reference, base, offsets), config["object"], contract["height_command"])
        *_, loads = arm_plan(reference, base, config, frames, contract)
        print(json.dumps({"status": "local_plan_only", "object": config["object"], "left_arm_rad": poses,
                          "grasp": {k: frames[k] for k in ("palm_inset_m", "grasp_below_top_m")},
                          "left_load": [{**case, "wrist_pitch_nm": load["left"]["torque_nm"]["left_wrist_pitch_joint"],
                                         "over_limit": load["left"]["over_limit"]} for case, load in zip(config["cases"], loads)],
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
    cases = [{k: r[k] for k in ("case", "mass_kg", "feedforward")} |
             {"status": r["summary"]["status"], "failed_checks": [c for c, ok in r["summary"]["checks"].items() if not ok]} |
             {k: r["summary"][k] for k in ("palm_rise_m", "carry_clearance_min_m", "held_tilt_max_deg", "carry_distance_m")} |
             {"saturated_joints": r["diagnostics"]["saturated_joints"],
              "body_contact_max_n": r["diagnostics"]["body_contact"]["held_max_n"], "grasp": r["summary"].get("grasp")}
             for r in evidence.get("results", [])]
    summary = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code, "run_dir": str(directory),
               "heaviest_passed_kg": evidence.get("heaviest_passed_kg"), "cases": cases, "error": evidence.get("error")}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "completed_lift_sweep" else 1


if __name__ == "__main__":
    raise SystemExit(main())
