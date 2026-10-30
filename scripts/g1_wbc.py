"""G1 controller data adapters; reference interfaces are attributed in README.

Isaac 3 pose buffers use xyzw. Configuration-file rotations use wxyz.
The math here operates on explicit xyzw poses or rotation matrices only.
"""

from __future__ import annotations

from collections import deque
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "4141c34280abb67c82e115342a8720f4a83d750d"
DYNAMICS_REFERENCE_SHA256 = "c61a85fc44fc74482285fb2efb09e4e0f8918a1a94a722c78bbf22887c0cad9d"
DEPLOYMENT_REFERENCE_SHA256 = "790678175ea58d658f4d501aacf349f275c24dd4eea3a2bc9c3e7deddc602cc6"
CONTACT_REFERENCE_SHA256 = "dc2b3eddaa2b69a4d9c742e5f8e3d72d60983c6312b9bde4866aef5a7725d53e"


def finite(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid finite array: expected {shape}, got {array.shape}")
    return array


def quat_xyzw_matrix(quat):
    q = finite(quat, (4,))
    norm = np.linalg.norm(q)
    if norm < 1e-8 or abs(norm - 1.0) > 0.01:
        raise ValueError("Pose quaternion is not normalized")
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quat_xyzw(rotation):
    r = finite(rotation, (3, 3))
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Expected a proper rotation")
    # Symmetric eigensystem avoids trace singularities at 180-degree rotations.
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], r.trace()],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def axis_rotation(axis, angle):
    a = finite(axis, (3,))
    if np.linalg.norm(a) < 1e-9:
        raise ValueError("Joint axis is zero")
    x, y, z = a / np.linalg.norm(a)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)


def pose_matrix(pose_xyzw):
    pose = finite(pose_xyzw, (7,))
    matrix = np.eye(4)
    matrix[:3, :3] = quat_xyzw_matrix(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix


class RobotReference:
    """Read the pinned URDF as data; its meshes are not loaded by this module."""

    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.joints = []
        self.limits = {}
        children = set()
        links = {x.attrib["name"] for x in root.findall("link")}
        for joint in root.findall("joint"):
            kind = joint.attrib["type"]
            if kind not in ("fixed", "revolute"):
                raise ValueError(f"Unreviewed joint type: {kind}")
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            children.add(child)
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
            rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
            transform = np.eye(4)
            transform[:3, 3] = finite(xyz, (3,))
            transform[:3, :3] = (axis_rotation([0,0,1], rpy[2]) @
                                     axis_rotation([0,1,0], rpy[1]) @ axis_rotation([1,0,0], rpy[0]))
            axis = joint.find("axis")
            axis = [float(v) for v in axis.attrib.get("xyz", "1 0 0").split()] if axis is not None else [1,0,0]
            name = joint.attrib["name"]
            self.joints.append((name, parent, child, kind, transform, axis))
            if kind != "fixed":
                limit = joint.find("limit")
                self.limits[name] = {k: float(limit.attrib[k]) for k in ("lower", "upper", "effort", "velocity")}
        roots = links - children
        if roots != {"pelvis"}:
            raise ValueError(f"Unexpected URDF roots: {roots}")
        self.root = "pelvis"

    def forward(self, positions: dict):
        if set(positions) != set(self.limits):
            raise ValueError("FK positions do not match the reference joint set")
        poses = {self.root: np.eye(4)}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, parent, child, kind, origin, axis in pending:
                if parent not in poses:
                    remaining.append((name, parent, child, kind, origin, axis))
                    continue
                turn = np.eye(4)
                if kind != "fixed":
                    value = float(positions[name])
                    if not math.isfinite(value):
                        raise ValueError("Non-finite FK joint position")
                    turn[:3, :3] = axis_rotation(axis, value)
                poses[child] = poses[parent] @ origin @ turn
            if len(remaining) == len(pending):
                raise ValueError("Disconnected or cyclic reference model")
            pending = remaining
        return poses


def joint_anchor_poses(reference: RobotReference, body_poses: dict, joint_frames: dict):
    """Express USD joint pivots using measured body poses and authored offsets.

    URDF child-link origins are joint pivots; USD rigid-body origins need not be.
    Offsets come from USD physics metadata, never from fitting measured errors.
    Body orientations remain unchanged, so the existing orientation checks apply.
    """
    expected = {name: (parent, child, axis) for name, parent, child, kind, _, axis
                in reference.joints if kind == "revolute"}
    if set(joint_frames) != set(expected):
        raise RuntimeError("Incomplete USD joint-frame metadata")
    result = {reference.root: body_poses[reference.root]}
    for name, (parent, child, axis) in expected.items():
        frame = joint_frames[name]
        if (frame["parent"], frame["child"]) != (parent, child):
            raise RuntimeError(f"USD joint body relationships differ: {name}")
        direction = finite(frame["axis_in_child"], (3,))
        wanted = finite(axis, (3,))
        wanted /= np.linalg.norm(wanted)
        if not np.allclose(direction, wanted, atol=1e-4, rtol=0):
            raise RuntimeError(f"USD joint axis differs from URDF: {name}")
        body = pose_matrix(body_poses[child])
        position = body[:3, 3] + body[:3, :3] @ finite(frame["position_in_child"], (3,))
        result[child] = np.r_[position, finite(body_poses[child], (7,))[3:]]
    return result


class FrameMismatchError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"Reference/asset frames differ: position={report['maximum_position_error_m']:.6g} m "
            f"({report['worst_position_link']}), angle={report['maximum_angle_error_rad']:.6g} rad "
            f"({report['worst_angle_link']})")


def check_frames(reference: RobotReference, joint_positions: dict, world_poses: dict,
                 position_tolerance, angle_tolerance, source_to_asset=None):
    """Fit a single root-frame rotation, then require link poses to agree with FK."""
    if "pelvis" not in world_poses:
        raise ValueError("Missing pelvis pose")
    local = reference.forward(joint_positions)
    names = sorted(set(local) & set(world_poses))
    if len(names) < 30:
        raise RuntimeError("Too few corresponding link frames for calibration")
    root = pose_matrix(world_poses["pelvis"])
    actual = {name: np.linalg.inv(root) @ pose_matrix(world_poses[name]) for name in names}
    if source_to_asset is None:
        if np.linalg.matrix_rank(np.stack([local[n][:3, 3] for n in names])) < 3:
            raise RuntimeError("Degenerate frame calibration geometry")
        # Fit orientations, which are independent of USD body-origin offsets.
        cross = sum(actual[n][:3, :3] @ local[n][:3, :3].T
                    for n in names if n != reference.root)
        u, _, vt = np.linalg.svd(cross)
        source_to_asset = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    rotation = finite(source_to_asset, (3, 3))
    errors = []
    for name in names:
        expected = local[name]
        difference = rotation @ expected[:3, 3] - actual[name][:3, 3]
        angle = 0.0
        # The fitted offset relates the two root frames. The pelvis orientation
        # defines that offset, so it is not an independent child-frame check.
        if name != "pelvis":
            delta = (rotation @ expected[:3, :3]).T @ actual[name][:3, :3]
            angle = float(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
        errors.append({"link": name, "position_error_m": float(np.linalg.norm(difference)),
                       "position_difference_m": difference.tolist(), "angle_error_rad": angle})
    position = max(errors, key=lambda e: e["position_error_m"])
    angle = max(errors, key=lambda e: e["angle_error_rad"])
    report = {"source_to_asset_rotation": rotation.tolist(), "links_checked": len(names),
              "maximum_position_error_m": position["position_error_m"],
              "maximum_angle_error_rad": angle["angle_error_rad"],
              "worst_position_link": position["link"], "worst_angle_link": angle["link"],
              "link_errors": errors}
    if position["position_error_m"] > position_tolerance or angle["angle_error_rad"] > angle_tolerance:
        raise FrameMismatchError(report)
    return report


def load_contract(path: Path):
    contract = json.loads(path.read_text())
    if contract["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Unreviewed controller source revision")
    if (contract["observation_dim"], contract["action_dim"], contract["history_length"]) != (516, 15, 6):
        raise ValueError("Unexpected controller dimensions")
    for name in ("kp", "kd", "default_lower_angles"):
        finite(contract[name], (15,))
    for name in ("lower_body_armature", "upper_body_armature"):
        value = contract[name]
        if type(value) not in (float, int) or not math.isfinite(value) or not 0.001 <= value <= 0.03:
            raise ValueError(f"Unreviewed joint armature: {name}")
    reference = ROOT / contract["reference_urdf"]
    if hashlib.sha256(reference.read_bytes()).hexdigest() != contract["reference_urdf_sha256"]:
        raise ValueError("Reference URDF hash mismatch")
    robot = RobotReference(reference)
    actuator_limits(contract, robot)
    return contract, robot


def load_deployment_reference():
    path = ROOT / "assets/references/g1_wbc_deployment.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DEPLOYMENT_REFERENCE_SHA256:
        raise ValueError("Deployment reference hash mismatch")
    return json.loads(path.read_text())


def load_contact_reference():
    path = ROOT / "assets/references/g1_wbc_contact.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTACT_REFERENCE_SHA256:
        raise ValueError("Contact reference hash mismatch")
    return json.loads(path.read_text())


def source_sphere_center(center, reference):
    """Pair an existing foot sphere with the source sphere in its toe/heel and side quadrant."""
    sources = finite(reference["foot_sphere_centers_m"], (4, 3))
    center = finite(center, (3,))
    split = sources[:, 0].mean()
    matches = [i for i, source in enumerate(sources)
               if (source[0] > split) == (center[0] > split) and np.sign(source[1]) == np.sign(center[1])]
    if len(matches) != 1 or center[1] == 0:
        raise ValueError(f"Foot sphere does not map to one source quadrant: {center.tolist()}")
    return sources[matches[0]]


def contact_override_summary(evidence, placement, runtime, combine_mode=None):
    """Check source sphere centers, unchanged foot inertia and friction materials before control."""
    reference = load_contact_reference()
    if combine_mode is not None:
        # Multiply keeps foot-ground friction at 1.0 x 1.0 while an object keeps its own coefficient.
        reference = {**reference, "isaac_material": {**reference["isaac_material"], "friction_combine_mode": combine_mode}}
    if (evidence["reference_sha256"] != CONTACT_REFERENCE_SHA256 or evidence["edit_layer"] != "session" or
            evidence["applied_before_reset"] is not True or set(evidence["feet"]) != set(reference["foot_bodies"])):
        raise ValueError("Invalid source contact provenance")
    sources = sorted(map(tuple, finite(reference["foot_sphere_centers_m"], (4, 3))))
    for name, foot in evidence["feet"].items():
        targets = set()
        for sphere in foot["spheres"]:
            linear = finite(sphere["linear_body"], (3, 3))
            target = source_sphere_center(sphere["center_body_m"], reference)
            radius = float(finite(sphere["radius"]))*float(np.sqrt(np.trace(linear @ linear.T)/3))
            if (not np.allclose(finite(sphere["after_center_body_m"], (3,)), target, atol=1e-9, rtol=0) or
                    not np.allclose(finite(sphere["after_linear_body"], (3, 3)), linear, atol=1e-12, rtol=0) or
                    not np.isclose(radius, reference["foot_sphere_radius_m"], atol=1e-6, rtol=0)):
                raise ValueError(f"Source foot sphere was not applied: {sphere['path']}")
            targets.add(tuple(target))
        if len(foot["spheres"]) != 4 or len(targets) != 4:
            raise ValueError(f"Foot spheres do not cover the four source centers: {name}")
        shapes = placement["before_geometry"]["feet"][name]["colliders"]
        placed = sorted(tuple(finite(s["body_from_shape"], (4, 4))[:3, 3]) for s in shapes)
        if len(shapes) != 4 or not np.allclose(placed, sources, atol=1e-7, rtol=0):
            raise ValueError(f"Initialized foot geometry differs from the source centers: {name}")
        i = runtime["body_names"].index(name)
        inertia = finite(runtime["body_inertia_backend_kg_m2"][i], (9,)).reshape(3, 3)
        if (not np.isclose(runtime["body_mass_kg"][i], foot["authored_mass_kg"], atol=1e-6, rtol=0) or
                not np.allclose(finite(runtime["body_com_pose_b_xyzw"][i], (7,))[:3], foot["authored_com_b_m"], atol=1e-6, rtol=0) or
                not np.allclose(np.linalg.eigvalsh(inertia), np.sort(foot["authored_principal_inertia_kg_m2"]), atol=1e-9, rtol=0)):
            raise ValueError(f"Foot inertial values changed with collider geometry: {name}")
    wanted = reference["isaac_material"]
    for scope in ("default", "ground"):
        material = evidence["materials"][scope]
        if (material["friction_combine_mode"] != wanted["friction_combine_mode"] or
                not np.isclose(material["static_friction"], wanted["static_friction"], atol=1e-6, rtol=0) or
                not np.isclose(material["dynamic_friction"], wanted["dynamic_friction"], atol=1e-6, rtol=0)):
            raise ValueError(f"Source friction material was not applied: {scope}")
    return {"status": "verified_source_contact", "foot_sphere_centers_m": [list(s) for s in sources],
            "effective_foot_ground_friction": wanted["static_friction"],
            "limitations": ["Composed USD and authored materials before control; PhysX-internal shape poses are not read back.",
                            *reference["residual_differences"]]}


def deployment_contract(contract, reference, command):
    """Build an explicit 29-DOF transfer view; leave the pinned URDF unchanged."""
    data = load_deployment_reference()
    lower = load_dynamics_reference()["lower_body"]
    for key, source_key in (("kp", "kp"), ("kd", "kd"), ("default_lower_angles", "default_angles")):
        if contract[key] != lower[source_key]:
            raise ValueError("Deployment lower-body contract differs from the pinned source")
    command = finite(command, (3,))
    hands = set(data["fixed_hand_joints"])
    if len(reference.limits) != 43 or len(hands) != 14 or not hands <= reference.limits.keys():
        raise ValueError("Unexpected deployment hand topology")
    reference.joints = [(n, p, c, "fixed" if n in hands else kind, origin, axis)
                        for n, p, c, kind, origin, axis in reference.joints]
    reference.limits = {n: v for n, v in reference.limits.items() if n not in hands}
    limits = actuator_limits(contract, RobotReference(ROOT / contract["reference_urdf"]))
    changes = {n: {"effort": data["body_effort_limits_nm"][n], "velocity": limits[n]["velocity"]}
               for n in reference.limits}
    moving = bool(np.linalg.norm(command) > data["policy_switch_command_norm"])
    effective = {**contract, "model": data["walk_model"] if moving else data["balance_model"],
        "policy_role": "walk" if moving else "balance", "deployment_sha256": DEPLOYMENT_REFERENCE_SHA256,
        "lower_body_armature": data["body_armature"], "upper_body_armature": data["body_armature"],
        "upper_hold_kp": data["upper_kp"], "upper_hold_kd": data["upper_kd"],
        "actuator_limit_overrides": changes,
        "actuator_limit_origin": "Pinned deployment effort caps; previous experiment velocity safety caps retained as an explicit residual difference.",
        "actuation_origin": "All 29 body joints use explicit PD and source armature 0.01; finger joints are fixed in the USD session layer.",
        "hold_gain_origin": "Pinned standalone deployment arm PD; contact/passive-friction equivalence is not claimed."}
    return effective, reference


def push_object_summary(evidence, settings):
    """Check the spawned object's size, mass, friction and resting pose before control."""
    size = finite(settings["size_m"], (3,))
    pose = finite(evidence["initial_pose_xyzw"], (7,))
    material = evidence["material"]
    wanted_x = evidence["palm_reach_m"] + settings["gap_m"] + size[0]/2
    if (not np.allclose(finite(evidence["size_m"], (3,)), size, atol=1e-9, rtol=0) or
            not np.isclose(evidence["runtime_mass_kg"], settings["mass_kg"], atol=1e-4, rtol=0) or
            material["friction_combine_mode"] != "multiply" or
            not np.allclose([material["static_friction"], material["dynamic_friction"]], settings["friction"], atol=1e-6, rtol=0) or
            not np.allclose(pose[:3], [wanted_x, 0, size[2]/2], atol=0.01, rtol=0) or
            object_tilt_degrees(pose) > 1.0):
        raise ValueError("Push object differs from the reviewed configuration")
    return {"status": "verified_push_object", "size_m": size.tolist(), "mass_kg": settings["mass_kg"],
            "friction": settings["friction"], "near_face_x_m": float(pose[0]-size[0]/2)}


def object_tilt_degrees(pose):
    return float(np.degrees(np.arccos(np.clip(quat_xyzw_matrix(finite(pose, (7,))[3:])[2, 2], -1, 1))))


def functional_summary(case, initial, records, foot_start=None, samples=(), object_start=None):
    """Project five-second acceptance gates, separate from fall/action guards."""
    motion = motion_summary(initial["root_pose_xyzw"], records)
    if len(records) != 250 or not np.isclose(motion["simulated_seconds"], 5):
        return {"status": "failed_functional_evaluation", "case": case,
                "checks": {"complete_five_second_trial": False},
                "on_failure": "Preserve partial evidence; no automatic retry."}
    velocity = finite([r["root_velocity_world"] for r in records], (250, 6))
    poses = finite([r["root_pose_xyzw"] for r in records], (250, 7))
    delta = poses[-1, :2] - finite(initial["root_pose_xyzw"], (7,))[:2]
    yaw = np.unwrap([np.arctan2(quat_xyzw_matrix(p[3:])[1, 0], quat_xyzw_matrix(p[3:])[0, 0])
                     for p in [initial["root_pose_xyzw"], *poses]])
    tail_vx = float(velocity[-50:, 0].mean())
    tail_yaw_rate = float((yaw[-1] - yaw[-51]) / 1.0)
    checks = {"tilt_within_20_degrees": motion["maximum_tilt_degrees"] <= 20}
    feet = None
    if case == "stand":
        # Pelvis position also moves when the policy squats to its height command; planted feet show standing in place.
        if not foot_start or len(samples) != 1000:
            checks["complete_foot_trace"] = False
        else:
            feet = {n: float(np.max(np.linalg.norm(finite([s["feet"][n]["pose_xyzw"][:2] for s in samples], (1000, 2))
                                                   - finite(xy, (2,)), axis=1))) for n, xy in foot_start.items()}
            checks["foot_displacement_within_0_02m"] = max(feet.values()) <= 0.02
        checks["final_speed_within_0_10m_s"] = motion["final_planar_speed_m_s"] <= 0.10
    elif case == "forward":
        checks.update(forward_progress_at_least_0_30m=float(delta[0]) >= 0.30,
                      lateral_drift_within_0_15m=float(np.max(np.abs(poses[:, 1]-initial["root_pose_xyzw"][1]))) <= 0.15,
                      final_second_vx_in_0_10_to_0_30m_s=0.10 <= tail_vx <= 0.30,
                      heading_change_within_0_20rad=float(np.max(np.abs(yaw-yaw[0]))) <= 0.20)
    elif case == "push":
        start = finite(object_start, (7,))
        objects = finite([r["object_pose_xyzw"] for r in records], (250, 7))
        pushed = {"progress_m": float(objects[-1, 0]-start[0]),
                  "maximum_lateral_drift_m": float(np.max(np.abs(objects[:, 1]-start[1]))),
                  "maximum_tilt_degrees": max(object_tilt_degrees(p) for p in objects)}
        checks.update(object_progress_at_least_0_20m=pushed["progress_m"] >= 0.20,
                      object_lateral_drift_within_0_15m=pushed["maximum_lateral_drift_m"] <= 0.15,
                      object_tilt_within_10_degrees=pushed["maximum_tilt_degrees"] <= 10)
    elif case == "turn":
        checks.update(positive_turn_at_least_0_30rad=float(yaw[-1]-yaw[0]) >= 0.30,
                      displacement_within_0_20m=motion["maximum_xy_displacement_m"] <= 0.20,
                      final_speed_within_0_15m_s=motion["final_planar_speed_m_s"] <= 0.15,
                      final_second_yaw_rate_in_0_10_to_0_30rad_s=0.10 <= tail_yaw_rate <= 0.30)
    else:
        raise ValueError("Unknown functional case")
    return {"status": "passed_functional_evaluation" if all(checks.values()) else "failed_functional_evaluation",
            "case": case, "checks": {k: bool(v) for k, v in checks.items()},
            "final_second_mean_vx_m_s": tail_vx, "final_second_mean_heading_rate_rad_s": tail_yaw_rate,
            "heading_change_rad": float(yaw[-1]-yaw[0]),
            "maximum_pelvis_displacement_m": motion["maximum_xy_displacement_m"], "maximum_foot_displacement_m": feet,
            **({"object": pushed} if case == "push" else {}),
            "criteria_origin": "Project acceptance proposal, not vendor thresholds; one bounded trial is not a robustness claim.",
            "on_failure": "Stop this gate; inspect the whole-contract evidence before proposing another physical change. No automatic retry or framework switch."}


def palm_inertial_reference(contract):
    """Diagonalize the pinned left-palm tensor in its link frame, in SI units."""
    path = ROOT / contract["reference_urdf"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract["reference_urdf_sha256"]:
        raise ValueError("Reference URDF hash mismatch")
    inertial = ET.parse(path).getroot().find("link[@name='left_hand_palm_link']/inertial")
    mass = float(inertial.find("mass").attrib["value"])
    origin = inertial.find("origin")
    com = finite([float(v) for v in origin.attrib["xyz"].split()], (3,))
    rpy = finite([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()], (3,))
    rotation = (axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1]) @
                axis_rotation([1, 0, 0], rpy[0]))
    data = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
    tensor = finite([[data["ixx"], data["ixy"], data["ixz"]],
                     [data["ixy"], data["iyy"], data["iyz"]],
                     [data["ixz"], data["iyz"], data["izz"]]], (3, 3))
    tensor = rotation @ tensor @ rotation.T
    moments, axes = np.linalg.eigh(tensor)
    if not math.isfinite(mass) or mass <= 0 or np.min(moments) <= 0 or moments[-1] > sum(moments[:2]) + 1e-12:
        raise ValueError("Invalid palm mass or physical inertia")
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    quat = matrix_quat_xyzw(axes)
    if not np.allclose(axes @ np.diag(moments) @ axes.T, tensor, atol=1e-12, rtol=0):
        raise ValueError("Palm inertia diagonalization failed")
    return {"body": "left_hand_palm_link", "reference_urdf": contract["reference_urdf"],
            "reference_urdf_sha256": contract["reference_urdf_sha256"], "mass_kg": mass,
            "com_b_m": com.tolist(), "inertia_b_kg_m2": tensor.tolist(),
            "principal_inertia_kg_m2": moments.tolist(), "principal_axes_xyzw": quat.tolist()}


def palm_override_summary(evidence, expected):
    """Validate authored and actual SI values; tensor orientation matters here."""
    saved_reference = evidence["reference"]
    if (any(saved_reference[k] != expected[k] for k in ("body", "reference_urdf", "reference_urdf_sha256")) or
            evidence["applied_before_reset"] is not True or
            evidence["edit_layer"] != "session" or
            evidence["prim_path"] != "/World/envs/env_0/Robot/left_hand/left_hand_palm_link"):
        raise ValueError("Invalid palm inertial override provenance")
    if evidence["before_usd"]["right"] != evidence["after_usd"]["right"]:
        raise ValueError("Right-palm USD properties changed during left-palm override")

    def tensor_from_usd(value):
        rotation = quat_xyzw_matrix(value["principal_axes_xyzw"])
        return rotation @ np.diag(finite(value["principal_inertia_kg_m2"], (3,))) @ rotation.T

    # LAPACK eigenvector signs/roundoff can differ between Mac and Linux.
    # Compare the represented tensor, not an arbitrary eigenvector basis.
    for key, shape in (("mass_kg", ()), ("com_b_m", (3,)), ("inertia_b_kg_m2", (3, 3))):
        if not np.allclose(finite(saved_reference[key], shape), finite(expected[key], shape), atol=1e-12, rtol=0):
            raise ValueError("Saved palm reference differs from the pinned URDF")
    if not np.allclose(tensor_from_usd(saved_reference), expected["inertia_b_kg_m2"], atol=1e-12, rtol=0):
        raise ValueError("Saved palm principal axes do not represent the pinned tensor")

    def compare(actual, wanted, tensor, label):
        errors = {"mass_kg": abs(float(finite(actual["mass_kg"], ()))-wanted["mass_kg"]),
                  "com_m": float(np.max(np.abs(finite(actual["com_b_m"], (3,))-finite(wanted["com_b_m"], (3,))))),
                  "inertia_kg_m2": float(np.max(np.abs(finite(tensor, (3, 3))-finite(wanted["inertia_b_kg_m2"], (3, 3)))))}
        if any(errors[k] > tolerance for k, tolerance in (("mass_kg", 1e-6), ("com_m", 1e-6), ("inertia_kg_m2", 1e-8))):
            raise ValueError(f"Palm inertial override mismatch ({label}): {errors}")
        return errors

    authored = evidence["after_usd"]["left"]
    if authored["has_mass_api"] is not True:
        raise ValueError("Left-palm PhysicsMassAPI was not applied")
    result = {"authored_left_max_absolute_errors": compare(authored, expected, tensor_from_usd(authored), "authored left")}
    right = evidence["after_usd"]["right"]
    right_expected = {**right, "inertia_b_kg_m2": tensor_from_usd(right).tolist()}
    for side, wanted in (("left", expected), ("right", right_expected)):
        actual = evidence["runtime"][side]
        result[f"runtime_{side}_max_absolute_errors"] = compare(actual, wanted, actual["inertia_b_kg_m2"], f"runtime {side}")
    return {"status": "verified_palm_inertial_override", **result,
            "tolerances": {"mass_kg": 1e-6, "com_m": 1e-6, "inertia_kg_m2": 1e-8},
            "tensor_semantics": "Pinned PhysX tensor API: at COM, rigid-body-prim frame, column-major; no world-frame rotation."}


def actuator_limits(contract, reference):
    """Keep experiment motor bounds explicit and separate from FK revision data."""
    limits = {name: {key: values[key] for key in ("effort", "velocity")}
              for name, values in reference.limits.items()}
    for name, changes in contract.get("actuator_limit_overrides", {}).items():
        if name not in limits or not changes or set(changes) - {"effort", "velocity"}:
            raise ValueError("Invalid actuator limit override")
        for key, value in changes.items():
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("Actuator limits must be finite and positive")
            limits[name][key] = value
    return limits


def prepare_model(contract, allow_download=False):
    model = contract["model"]
    path = ROOT / model["cache_path"]
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError("Pinned balance weights are missing; use --prepare-model once")
        expected_prefix = f"https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{SOURCE_COMMIT}/"
        if not model["url"].startswith(expected_prefix):
            raise ValueError("Unreviewed model download source")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            temporary = Path(tmp) / "model.onnx"
            started = time.monotonic()
            with urlopen(model["url"], timeout=30) as source, temporary.open("wb") as target:
                total = 0
                while chunk := source.read(65536):
                    if time.monotonic()-started > 60:
                        raise TimeoutError("Model download exceeded one minute")
                    total += len(chunk)
                    if total > model["bytes"]:
                        raise ValueError("Model download exceeded the reviewed size")
                    target.write(chunk)
            verify_model(temporary, model)
            os.link(temporary, path)  # Atomic and refuses to overwrite a cache entry.
    verify_model(path, model)
    license_target = path.parent / "LICENSE.txt"
    license_bytes = (ROOT / model["license"]).read_bytes()
    if license_target.exists() and license_target.read_bytes() != license_bytes:
        raise ValueError("Cached model license differs from the reviewed notice")
    if not license_target.exists():
        shutil.copyfile(ROOT / model["license"], license_target)
    return path


def verify_model(path, model):
    if path.stat().st_size != model["bytes"]:
        raise ValueError("Checkpoint size differs from the pinned model (possibly an LFS pointer)")
    if hashlib.sha256(path.read_bytes()).hexdigest() != model["sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")


class BalancePolicy:
    def __init__(self, path, contract):
        verify_model(path, contract["model"])
        version = importlib.metadata.version("onnxruntime")
        if version != contract["onnxruntime_version"]:
            raise RuntimeError(f"Expected the existing upstream onnxruntime=={contract['onnxruntime_version']}; found {version}")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        for entries, dim in ((inputs, 516), (outputs, 15)):
            if len(entries) != 1 or entries[0].type != "tensor(float)" or len(entries[0].shape) != 2 or entries[0].shape[1] != dim:
                raise RuntimeError("Checkpoint input/output contract differs from the reviewed policy")
        self.input_name = inputs[0].name
        self.metadata = {"onnxruntime": version, "providers": self.session.get_providers(),
                         "input": inputs[0].shape, "output": outputs[0].shape,
                         "sha256": contract["model"]["sha256"]}

    def __call__(self, observation):
        obs = finite(observation, (516,)).astype(np.float32)
        output = self.session.run(None, {self.input_name: obs[None]})[0]
        return finite(output, (1, 15))[0].astype(np.float32)


class ObservationHistory:
    def __init__(self, contract):
        self.contract = contract
        self.reset()

    def reset(self):
        self.history = deque(maxlen=6)
        self.last_action = np.zeros(15, dtype=np.float32)

    def observe(self, body_positions, body_velocities, root_quat_xyzw, angular_velocity_w,
                source_to_asset, command=(0.0, 0.0, 0.0)):
        c = self.contract
        body = finite(body_positions, (29,))
        velocities = finite(body_velocities, (29,))
        rotation = quat_xyzw_matrix(root_quat_xyzw) @ finite(source_to_asset, (3, 3))
        gyro = rotation.T @ finite(angular_velocity_w, (3,))
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        defaults = np.r_[finite(c["default_lower_angles"], (15,)), np.zeros(14)]
        frame = np.r_[finite(command, (3,))*c["command_scale"], c["height_command"],
                      np.zeros(3), gyro*c["angular_velocity_scale"], gravity,
                      (body-defaults)*c["joint_position_scale"],
                      velocities*c["joint_velocity_scale"], self.last_action].astype(np.float32)
        finite(frame, (86,))
        self.history.append(frame)
        while len(self.history) < 6:
            self.history.appendleft(np.zeros(86, dtype=np.float32))
        return np.concatenate(self.history)

    def targets(self, action, lower_limits):
        c = self.contract
        action = finite(action, (15,))
        if np.max(np.abs(action)) > c["action_limit"]:
            index = int(np.argmax(np.abs(action)))
            raise RuntimeError("Policy action exceeds the configured transfer bound: "
                               f"index={index}, value={action[index]:.6g}, limit={c['action_limit']:.6g}")
        limits = finite(lower_limits, (15, 2))
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Invalid lower-body limits")
        self.last_action = action.astype(np.float32)
        raw = finite(c["default_lower_angles"], (15,)) + c["action_scale"]*action
        return np.clip(raw, limits[:, 0], limits[:, 1]), int(np.count_nonzero((raw < limits[:, 0]) | (raw > limits[:, 1])))


def load_dynamics_reference():
    path = ROOT / "assets/references/g1_wbc_dynamics.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DYNAMICS_REFERENCE_SHA256:
        raise ValueError("Dynamics reference differs from the reviewed source extraction")
    return json.loads(path.read_text())


def dynamics_comparison(measured, reference, source, contract, asset):
    """Compare a static snapshot; differences are not evidence of a drift cause."""
    names, joints = measured["body_names"], measured["joint_names"]
    if measured["status"] != "captured_runtime_dynamics":
        raise ValueError(f"Incomplete runtime dynamics capture at {measured.get('collection_phase', 'unknown phase')}: "
                         f"{measured.get('collection_error', 'collection did not finish')}")
    dofs = len(reference.limits)
    if (not names or len(names) != len(set(names)) or set(joints) != set(reference.limits) or len(joints) != dofs):
        raise ValueError("Invalid dynamics body/joint mapping")
    count = len(names)
    masses = finite(measured["body_mass_kg"], (count,))
    com = finite(measured["body_com_pose_b_xyzw"], (count, 7))
    poses = finite(measured["body_link_pose_w_xyzw"], (count, 7))
    tensors = finite(measured["body_inertia_backend_kg_m2"], (count, 9)).reshape(count, 3, 3)
    if np.any(masses <= 0) or not np.allclose(tensors, tensors.transpose(0,2,1), atol=1e-8, rtol=1e-5):
        raise ValueError("Invalid runtime mass or nonsymmetric inertia")
    principal = np.linalg.eigvalsh(tensors)
    if np.any(principal <= 0):
        raise ValueError("Runtime inertia is not positive definite")
    for key in ("solver_joint_stiffness", "solver_joint_damping", "joint_armature"):
        if np.any(finite(measured[key], (dofs,)) < 0):
            raise ValueError(f"Invalid runtime {key}")
    finite(measured["joint_effort_limits"], (dofs,))
    targets = finite(measured["joint_position_targets"], (dofs,))
    velocity_targets = finite(measured["joint_velocity_targets"], (dofs,))
    q = dict(zip(joints, finite(measured["joint_positions"], (dofs,)), strict=True))
    if source["actuated_joint_names"] != asset["controller_reference"]["joint_order"]:
        raise ValueError("Source dynamics joint order differs from the reviewed policy")
    root = pose_matrix(measured["root_pose_xyzw"])
    rotation = root[:3,:3] @ finite(measured["source_to_asset_rotation"], (3,3))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("Invalid dynamics source-frame rotation")
    # Published fingers are fixed. Evaluate their source pose at zero; retain
    # all actual finger coordinates and the articulation difference in evidence.
    source_q = {n: q[n] if n in source["actuated_joint_names"] else 0.0 for n in q}
    fk = reference.forward(source_q)
    actual_com = np.array([(pose_matrix(p) @ np.r_[c[:3],1])[:3] for p,c in zip(poses,com,strict=True)])
    for c in com: quat_xyzw_matrix(c[3:])
    actual_root_com = (rotation.T @ (actual_com-root[:3,3]).T).T
    source_com = {n:(fk[n] @ np.r_[v["com_position_b_m"],1])[:3] for n,v in source["bodies"].items()}
    source_mass = sum(v["mass_kg"] for v in source["bodies"].values())
    source_total_com = sum(v["mass_kg"]*source_com[n] for n,v in source["bodies"].items())/source_mass
    actual_total_com = np.average(actual_root_com, axis=0, weights=masses)
    rows = []
    for name in sorted(set(names) & source["bodies"].keys()):
        i = names.index(name); s = source["bodies"][name]
        source_principal = np.sort(finite(s["principal_inertia_kg_m2"], (3,)))
        rows.append({"body": name, "source_mass_kg": s["mass_kg"], "runtime_mass_kg": float(masses[i]),
                     "mass_difference_kg": float(masses[i]-s["mass_kg"]),
                     "source_com_in_source_root_m": source_com[name].tolist(),
                     "runtime_com_in_source_root_m": actual_root_com[i].tolist(),
                     "com_difference_norm_m": float(np.linalg.norm(actual_root_com[i]-source_com[name])),
                     "source_principal_inertia_kg_m2": source_principal.tolist(),
                     "runtime_principal_inertia_kg_m2": principal[i].tolist()})
    covered = set(); controls = []
    for group in measured["actuators"].values():
        model_names = group["joint_names"]
        kp, kd = finite(group["kp"], (len(model_names),)), finite(group["kd"], (len(model_names),))
        if (covered & set(model_names) or len(model_names) != len(set(model_names)) or
                np.any(kp < 0) or np.any(kd < 0) or type(group["implicit"]) is not bool):
            raise ValueError("Invalid runtime actuator evidence")
        covered.update(model_names)
        for i,name in enumerate(model_names):
            if name in asset["joint_groups"]["lower_body"]:
                index = asset["joint_groups"]["lower_body"].index(name)
                source_kp, source_kd = source["lower_body"]["kp"][index], source["lower_body"]["kd"][index]
            elif name in source["upper_body"]["joint_names"]:
                source_kp, source_kd = source["upper_body"]["kp"], source["upper_body"]["kd"]
            else:
                source_kp = source_kd = None
            index = joints.index(name)
            controls.append({"joint": name, "source_actuated": source_kp is not None,
                "source_actuator": "explicit_pd" if source_kp is not None else "fixed_joint",
                "source_kp": source_kp, "source_kd": source_kd,
                "source_target_position_rad": float(source["lower_body"]["default_angles"][asset["joint_groups"]["lower_body"].index(name)])
                    if name in asset["joint_groups"]["lower_body"] else (0.0 if source_kp is not None else None),
                "runtime_actuator": "implicit_pd" if group["implicit"] else "explicit_pd",
                "runtime_class": group["class"], "runtime_kp": float(kp[i]), "runtime_kd": float(kd[i]),
                "runtime_parameter_source": group.get("parameter_source", "not_recorded"),
                "runtime_solver_stiffness": measured["solver_joint_stiffness"][index],
                "runtime_solver_damping": measured["solver_joint_damping"][index],
                "runtime_armature": measured["joint_armature"][index],
                "source_joint_defaults": source["joint_defaults"],
                "runtime_target_position_rad": float(targets[index]),
                "runtime_target_velocity_rad_s": float(velocity_targets[index])})
    if covered != set(joints):
        raise ValueError("Runtime actuator evidence does not cover all joints")
    return {"status": "compared_runtime_dynamics", "source_body_count": len(source["bodies"]),
            "runtime_body_count": count, "source_actuated_joint_count": len(source["actuated_joint_names"]),
            "runtime_actuated_joint_count": len(joints),
            "totals": {"source_mass_kg": source_mass, "runtime_mass_kg": float(masses.sum()),
                       "mass_difference_kg": float(masses.sum()-source_mass),
                       "source_com_in_source_root_m": source_total_com.tolist(),
                       "runtime_com_in_source_root_m": actual_total_com.tolist(),
                       "com_difference_m": (actual_total_com-source_total_com).tolist()},
            "source_only_bodies": sorted(source["bodies"].keys()-set(names)),
            "runtime_only_bodies": sorted(set(names)-source["bodies"].keys()),
            "bodies": rows, "controls": sorted(controls, key=lambda r:r["joint"]),
            "balance_or_walking_verified": False,
            "limitations": ["Static comparison at the recorded pose; no causal attribution or controller execution.",
                "Published deployment data is not proof of the training distribution.",
                "Inertia comparison uses eigenvalues only; raw backend tensor orientation is not normalized.",
                "Same-name rows may differ because of split/merged bodies; all bodies contribute to total mass and COM.",
                "Published fingers are fixed; runtime finger coordinates and articulation are retained separately."]}


def collision_clearance(geometry):
    """Measure sphere surfaces against one horizontal plane in meter units."""
    if geometry["up_axis"] != "Z" or geometry["meters_per_unit"] != 1.0:
        raise ValueError("Initial clearance requires a Z-up meter stage")
    plane = geometry["ground"]
    transform = finite(plane["world_from_plane"], (4, 4))
    if (plane["type"] != "Plane" or plane["axis"] not in ("X", "Y", "Z") or
            not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0)):
        raise ValueError("Invalid ground plane geometry")
    normal = np.linalg.solve(transform[:3, :3].T, np.eye(3)["XYZ".index(plane["axis"])])
    normal /= np.linalg.norm(normal)
    if not np.allclose(np.abs(normal), [0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Initial clearance requires a horizontal ground plane")
    height = float(transform[2, 3])
    report = {"ground_height_m": height, "feet": {}}
    if set(geometry["feet"]) != {"left_ankle_roll_link", "right_ankle_roll_link"}:
        raise ValueError("Initial clearance requires exactly the two G1 feet")
    paths = set()
    for name, foot in geometry["feet"].items():
        body = pose_matrix(foot["body_pose_xyzw"])
        colliders = []
        if not foot["colliders"]:
            raise ValueError(f"No enabled foot spheres: {name}")
        for shape in foot["colliders"]:
            path = shape["collision_path"]
            if path in paths or shape["type"] != "Sphere" or not path.startswith(f"/World/envs/env_0/Robot/{name}/"):
                raise ValueError("Invalid or duplicate foot collision sphere")
            paths.add(path)
            local = finite(shape["body_from_shape"], (4, 4))
            radius = float(finite(shape["radius_m"]))
            if not np.allclose(local[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
                raise ValueError(f"Invalid foot sphere transform: {path}")
            if radius <= 0:
                raise ValueError(f"Invalid foot sphere radius: {path}, local_radius={radius:.9g}")
            world = body @ local
            gram = world[:3, :3].T @ world[:3, :3]
            scale = float(np.sqrt(np.trace(gram)/3))
            if scale <= 0 or not np.allclose(gram, np.eye(3)*scale**2, atol=1e-9, rtol=1e-6):
                raise ValueError(f"Nonuniform sphere scaling is not supported by the clearance experiment: {path}")
            radius_world = radius*scale
            # USD may represent a small collider as a scaled unit sphere.
            # Bound its physical size, rather than the untransformed attribute.
            if not np.isfinite(radius_world) or not 0 < radius_world <= 0.1:
                raise ValueError(f"Invalid foot sphere dimensions: {path}, local_radius={radius:.9g}, "
                                 f"world_scale={scale:.9g}, world_radius_m={radius_world:.9g}, maximum_m=0.1")
            bottom = float(world[2, 3]-radius_world)
            colliders.append({"collision_path": path, "center_world_m": world[:3, 3].tolist(),
                              "radius_world_m": radius_world, "bottom_world_z_m": bottom,
                              "clearance_m": bottom-height})
        report["feet"][name] = {"colliders": colliders,
                                "minimum_clearance_m": min(s["clearance_m"] for s in colliders)}
    report["minimum_clearance_m"] = min(f["minimum_clearance_m"] for f in report["feet"].values())
    return report


def placement_shift(clearance, settings):
    shift = max(0.0, settings["minimum_clearance_m"]-float(finite(clearance)))
    if shift > settings["maximum_vertical_shift_m"]:
        raise RuntimeError(f"Initial placement needs {shift:.6g} m lift, above the configured bound")
    return shift


def placement_summary(evidence, settings, policy_height, initial_pose):
    """Revalidate saved geometry and placement without USD or a simulator."""
    def agrees(first, last):
        if isinstance(first, dict):
            return isinstance(last, dict) and first.keys() == last.keys() and all(agrees(v, last[k]) for k,v in first.items())
        if isinstance(first, list):
            return isinstance(last, list) and len(first) == len(last) and all(agrees(a,b) for a,b in zip(first,last,strict=True))
        if isinstance(first, (int, float)):
            return type(last) in (int, float) and bool(np.isclose(first, last, atol=1e-8, rtol=0))
        return first == last

    if evidence["status"] != "passed_initial_placement" or evidence["mode"] != settings["mode"]:
        raise ValueError("Incomplete initial placement evidence")
    before = collision_clearance(evidence["before_geometry"])
    after = collision_clearance(evidence["after_geometry"])
    if not agrees(before, evidence["before_clearance"]) or not agrees(after, evidence["after_clearance"]):
        raise ValueError("Saved clearance disagrees with collision geometry")
    source_pose = settings["mode"] == "source_pose"
    lift = 0.0 if source_pose else placement_shift(before["minimum_clearance_m"], settings)
    wanted = np.array([0.0, 0.0, lift, 0, 0, 0, 0])
    root_before = finite(evidence["before_root_pose_xyzw"], (7,))
    root_after = finite(evidence["after_root_pose_xyzw"], (7,))
    if (not np.isclose(evidence["policy_height_command_m"], policy_height, atol=1e-8, rtol=0) or
            not np.isclose(evidence["vertical_shift_m"], lift, atol=1e-8, rtol=0) or
            not np.allclose(root_before[:3], [0, 0, settings["root_height_m"] if source_pose else policy_height], atol=1e-6, rtol=0) or
            not np.allclose(root_after-root_before, wanted, atol=1e-6, rtol=0) or
            not np.allclose(root_after, finite(initial_pose, (7,)), atol=1e-6, rtol=0)):
        raise ValueError("Initial placement changed more than the declared vertical translation")
    if evidence["before_geometry"]["ground"] != evidence["after_geometry"]["ground"]:
        raise ValueError("Ground geometry changed during initial placement")
    for name in before["feet"]:
        first = evidence["before_geometry"]["feet"][name]
        last = evidence["after_geometry"]["feet"][name]
        shapes_before = {s["collision_path"]: s for s in first["colliders"]}
        shapes_after = {s["collision_path"]: s for s in last["colliders"]}
        if (shapes_before.keys() != shapes_after.keys() or
                not np.allclose(finite(last["body_pose_xyzw"], (7,))-finite(first["body_pose_xyzw"], (7,)),
                                wanted, atol=1e-5, rtol=0)):
            raise ValueError("Foot geometry changed beyond the declared lift")
        for path, shape in shapes_before.items():
            other = shapes_after[path]
            if (shape["type"] != other["type"] or
                    not np.isclose(shape["radius_m"], other["radius_m"], atol=1e-10, rtol=0) or
                    not np.allclose(shape["body_from_shape"], other["body_from_shape"], atol=1e-7, rtol=0)):
                raise ValueError("Foot collider geometry changed during placement")
    if (after["minimum_clearance_m"] < settings["minimum_clearance_m"]-1e-5 or
            not np.isclose(after["minimum_clearance_m"]-before["minimum_clearance_m"], lift, atol=1e-5, rtol=0)):
        raise ValueError("Initial foot clearance was not established: "
                         f"mode={settings['mode']}, before={before['minimum_clearance_m']:.9g} m, "
                         f"after={after['minimum_clearance_m']:.9g} m, "
                         f"required={settings['minimum_clearance_m']:.9g} m, tolerance=1e-5 m, "
                         f"expected_vertical_shift={lift:.9g} m")
    return {"status": "passed_initial_placement", "vertical_shift_m": lift,
            "minimum_clearance_before_m": before["minimum_clearance_m"],
            "minimum_clearance_after_m": after["minimum_clearance_m"],
            "spawn_height_m": float(root_after[2]), "policy_height_command_m": policy_height,
            "interpretation": "Geometric sphere-to-plane clearance before any physics step; contact offsets and dynamic settling are not measured here."}


def motion_summary(initial_pose, records):
    """Report motion without treating a non-fall pass as stationary standing."""
    if not records:
        return {"samples": 0}
    initial = finite(initial_pose, (7,))
    times = finite([r["sim_time_seconds"] for r in records], (len(records),))
    if (times[0] <= 0 or np.any(np.diff(times) <= 0) or
            any(row["step"] != index for index, row in enumerate(records, 1))):
        raise ValueError("Invalid motion sample sequence")
    poses = finite([r["root_pose_xyzw"] for r in records], (len(records), 7))
    velocities = finite([r["root_velocity_world"] for r in records], (len(records), 6))
    xy = np.vstack([initial[:2], poses[:, :2]])
    tilt = finite([r["tilt_rad"] for r in records], (len(records),))
    return {"samples": len(records), "simulated_seconds": float(times[-1]),
            "net_xy_displacement_m": float(np.linalg.norm(xy[-1]-xy[0])),
            "maximum_xy_displacement_m": float(np.linalg.norm(xy-xy[0], axis=1).max()),
            "xy_path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "final_world_position_m": poses[-1, :3].tolist(),
            "final_planar_speed_m_s": float(np.linalg.norm(velocities[-1, :2])),
            "minimum_root_height_m": float(poses[:, 2].min()),
            "maximum_tilt_degrees": float(np.degrees(tilt.max())),
            "maximum_absolute_action": float(np.abs(finite([r["action"] for r in records], (len(records), 15))).max()),
            "target_clips": sum(r["clipped_joint_targets"] for r in records)}


def contact_summary(samples, feet, threshold, physics_dt, decimation):
    """Describe loaded-foot motion; it is not proof of slip at a contact point."""
    collected = {name: [] for name in feet}
    clipping = np.zeros(15, dtype=int)
    support_counts = []
    for index, sample in enumerate(samples, 1):
        if (sample["physics_step"] != index or sample["control_step"] != (index-1)//decimation+1 or
                not np.isclose(sample["sim_time_seconds"], index*physics_dt, atol=1e-9, rtol=0) or
                set(sample["feet"]) != set(feet)):
            raise ValueError("Invalid contact sample sequence or foot mapping")
        computed = finite(sample["lower_body_computed_effort"], (15,))
        applied = finite(sample["lower_body_applied_effort"], (15,))
        clipping += np.abs(computed-applied) > 1e-5
        supported = 0
        for name in feet:
            foot = sample["feet"][name]
            pose = finite(foot["pose_xyzw"], (7,))
            velocity = finite(foot["velocity_world"], (6,))
            normal = finite(foot["normal_force_world_n"], (3,))
            friction = finite(foot["friction_force_world_n"], (3,))
            total = finite(foot["total_force_world_n"], (3,))
            if not np.allclose(total, normal+friction, atol=1e-3, rtol=1e-5):
                raise ValueError("Contact total force differs from normal plus friction")
            force = float(np.linalg.norm(normal))
            loaded = force >= threshold
            supported += loaded
            collected[name].append((loaded, np.linalg.norm(velocity[:2]), force, np.linalg.norm(friction), pose[2]))
        support_counts.append(supported)
    report = {"samples": len(samples), "minimum_normal_force_n": threshold,
              "effort_clipped_samples_by_lower_body_index": clipping.tolist(), "feet": {},
              "effort_telemetry_limitation": "Counts describe differences in recorded computed/applied staging buffers; they do not establish clipping in native actuator execution.",
              "interpretation": "Loaded rigid-body origin speed includes foot rolling; contact-point slip is not established by this metric alone."}
    if not samples:
        report["force_evidence_present"] = False
        return report
    report["force_evidence_present"] = any(any(row[0] for row in values) for values in collected.values())
    report["no_support_fraction"] = float(np.mean(np.array(support_counts) == 0))
    report["double_support_fraction"] = float(np.mean(np.array(support_counts) == 2))
    for name, values in collected.items():
        a = np.asarray(values, dtype=float)
        loaded = a[:, 0] > 0
        report["feet"][name] = {
            "loaded_fraction": float(loaded.mean()),
            "loaded_planar_speed_mean_m_s": float(a[loaded, 1].mean()) if loaded.any() else None,
            "loaded_planar_speed_p95_m_s": float(np.quantile(a[loaded, 1], 0.95)) if loaded.any() else None,
            "maximum_normal_force_n": float(a[:, 2].max()),
            "maximum_friction_force_n": float(a[:, 3].max()),
            "mean_friction_normal_ratio_when_loaded": float((a[loaded, 3]/a[loaded, 2]).mean()) if loaded.any() else None,
            "minimum_body_origin_height_m": float(a[:, 4].min()),
            "maximum_body_origin_height_m": float(a[:, 4].max())}
    return report
