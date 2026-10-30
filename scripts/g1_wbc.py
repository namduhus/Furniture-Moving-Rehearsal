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
