"""Inspect G1 dynamics or verify the pinned policy in a bounded standing trial."""

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
FINGERTIP_ALLOWANCE_M = 0.05


def trial_contract(profile):
    from g1_wbc import load_contract, deployment_contract
    contract, reference = load_contract(ROOT / profile["controller_contract"])
    if not profile.get("deployment"):
        return contract, reference
    contract, reference = deployment_contract(contract, reference, profile["command"])
    if profile.get("arm_hold") == "implicit":
        contract = {**contract, "actuation_origin": "15 policy joints use explicit PD; 14 arm joints hold with implicit PhysX PD at the source gains; all body joints keep source armature 0.01; finger joints are fixed in the USD session layer."}
    return contract, reference


def apply_fixed_hands(stage, robot, directory):
    """Replace joint types in the session layer; preserve all joint frames."""
    from pxr import Usd, UsdPhysics
    from g1_wbc import load_deployment_reference
    expected = set(load_deployment_reference()["fixed_hand_joints"])
    evidence = {"status": "preparing_fixed_hands", "joints": {}, "edit_layer": "session"}
    write_json(directory / "deployment-joints.json", evidence)
    if robot.is_initialized:
        raise RuntimeError("Fixed hands must be authored before reset")
    prims = [p for p in stage.Traverse() if p.GetName() in expected]
    if len(prims) != 14 or {p.GetName() for p in prims} != expected:
        raise RuntimeError("Expected exactly the 14 source hand joints")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in prims:
            if (not prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsInstanceProxy() or
                    not str(prim.GetPath()).startswith("/World/envs/env_0/Robot/")):
                raise RuntimeError("Unexpected source hand joint type/proxy")
            joint = UsdPhysics.Joint(prim)
            before = {"body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                      "body1": [str(p) for p in joint.GetBody1Rel().GetTargets()]}
            if len(before["body0"]) != 1 or len(before["body1"]) != 1:
                raise RuntimeError("Invalid hand joint body relationships")
            fixed = UsdPhysics.FixedJoint.Define(stage, prim.GetPath())
            if not fixed:
                raise RuntimeError("Unable to author fixed hand joint")
            evidence["joints"][prim.GetName()] = {"path": str(prim.GetPath()), **before,
                "authored_type": fixed.GetPrim().GetTypeName()}
            write_json(directory / "deployment-joints.json", evidence)
    evidence["status"] = "authored_fixed_hands_before_reset"
    write_json(directory / "deployment-joints.json", evidence)


def validate_deployment_evidence(directory, asset, contract, implicit_arms=False):
    from g1_wbc import load_deployment_reference, finite
    import numpy as np
    data = load_deployment_reference()
    lower_body = asset["joint_groups"]["lower_body"]
    fixed = json.loads((directory / "deployment-joints.json").read_text())
    if (fixed["status"] != "authored_fixed_hands_before_reset" or fixed["edit_layer"] != "session" or
            set(fixed["joints"]) != set(data["fixed_hand_joints"]) or
            any(j["authored_type"] != "PhysicsFixedJoint" for j in fixed["joints"].values())):
        raise ValueError("Incomplete fixed-hand deployment evidence")
    measured = json.loads((directory / "deployment-runtime.json").read_text())
    if measured["joint_names"] != json.loads((directory / "asset-inventory.json").read_text())["joint_names"]:
        raise ValueError("Deployment joint order differs from the inventory")
    expected = asset["controller_reference"]["joint_order"]
    if len(measured["joint_names"]) != 29 or set(measured["joint_names"]) != set(expected):
        raise ValueError("Deployment did not resolve exactly 29 active body joints")
    covered = []
    for group in measured["actuators"].values():
        covered.extend(group["joint_names"])
        arms = not set(group["joint_names"]) & set(lower_body)
        if group["implicit"] != (implicit_arms and arms):
            raise ValueError("Deployment actuator model differs from the reviewed profile")
        for i, name in enumerate(group["joint_names"]):
            lower = name in lower_body
            index = lower_body.index(name) if lower else 0
            kp = contract["kp"][index] if lower else 100.0
            kd = contract["kd"][index] if lower else 0.5
            if not np.allclose([group["kp"][i], group["kd"][i]], [kp, kd], atol=1e-6, rtol=0):
                raise ValueError("Deployment native PD gains were not applied")
    if len(covered) != 29 or set(covered) != set(expected):
        raise ValueError("Incomplete deployment native actuator coverage")
    # Implicit arm gains live in the PhysX drive; explicit joints must leave it at zero.
    solver = np.stack([finite(measured["solver_joint_stiffness"], (29,)), finite(measured["solver_joint_damping"], (29,))], axis=1)
    drive = [[100.0, 0.5] if implicit_arms and n not in lower_body else [0.0, 0.0] for n in measured["joint_names"]]
    if not np.allclose(solver, drive, atol=1e-6, rtol=0):
        raise ValueError("Deployment PhysX drive gains differ from the reviewed arm hold")
    armature = finite(measured["joint_armature"], (29,))
    limits = finite(measured["joint_effort_limits"], (29,))
    if not np.allclose(armature, 0.01, atol=1e-7, rtol=0) or not np.allclose(
            limits, [data["body_effort_limits_nm"][n] for n in measured["joint_names"]], atol=1e-5, rtol=0):
        raise ValueError("Deployment armature/effort caps differ from the source")
    for key, wanted in (("joint_static_friction_nm", 0.1), ("joint_dynamic_friction_nm", 0.1),
                        ("joint_viscous_friction_nm_s_rad", 0.001)):
        if not np.allclose(finite(measured[key], (29,)), wanted, atol=1e-7, rtol=0):
            raise ValueError("Deployment passive joint terms were not applied")
    return {"status": "verified_deployment_runtime", "active_joints": 29, "fixed_finger_joints": 14,
            "arm_hold": "implicit" if implicit_arms else "explicit",
            "policy_role": contract["policy_role"], "policy_phase": data["policy_phase"],
            "residual_differences": data["residual_differences"]}


def apply_palm_override(stage, robot, contract, directory):
    """Author only the left palm in the session layer before physics reset."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from g1_wbc import palm_inertial_reference

    wanted = palm_inertial_reference(contract)
    path = "/World/envs/env_0/Robot/left_hand/left_hand_palm_link"
    report = {"status": "preparing_palm_inertial_override", "reference": wanted,
              "prim_path": path, "edit_layer": "session", "applied_before_reset": False}
    write_json(directory / "palm-inertial-override.json", report)
    if robot.is_initialized:
        raise RuntimeError("Palm inertial override must precede physics initialization")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0:
        raise RuntimeError("Palm inertial override requires meter/kilogram stage units")

    def read(side):
        body_path = f"/World/envs/env_0/Robot/{side}_hand/{side}_hand_palm_link"
        prim = stage.GetPrimAtPath(body_path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsInstanceProxy():
            raise RuntimeError(f"Expected a non-proxy palm rigid body: {body_path}")
        api = UsdPhysics.MassAPI(prim)
        result = {"has_mass_api": bool(prim.HasAPI(UsdPhysics.MassAPI))}
        for name, attribute in (("mass_kg", api.GetMassAttr()), ("com_b_m", api.GetCenterOfMassAttr()),
                                ("principal_inertia_kg_m2", api.GetDiagonalInertiaAttr()),
                                ("principal_axes_xyzw", api.GetPrincipalAxesAttr())):
            if attribute and attribute.HasAuthoredValueOpinion():
                value = attribute.Get()
                if name == "principal_axes_xyzw":
                    result[name] = [*map(float, value.GetImaginary()), float(value.GetReal())]
                elif name == "mass_kg":
                    result[name] = float(value)
                else:
                    result[name] = list(map(float, value))
        return result

    report["before_usd"] = {side: read(side) for side in ("left", "right")}
    write_json(directory / "palm-inertial-override.json", report)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path))
        quat = wanted["principal_axes_xyzw"]
        for attr, value in ((api.CreateMassAttr(), wanted["mass_kg"]),
                            (api.CreateCenterOfMassAttr(), Gf.Vec3f(*wanted["com_b_m"])),
                            (api.CreateDiagonalInertiaAttr(), Gf.Vec3f(*wanted["principal_inertia_kg_m2"])),
                            (api.CreatePrincipalAxesAttr(), Gf.Quatf(quat[3], Gf.Vec3f(*quat[:3])))):
            if not attr.Set(value):
                raise RuntimeError("Unable to author left-palm inertial property")
    report["after_usd"] = {side: read(side) for side in ("left", "right")}
    report.update(status="authored_before_physics_reset", applied_before_reset=True)
    write_json(directory / "palm-inertial-override.json", report)
    return report


def apply_source_contact(stage, robot, directory):
    """Move only the existing foot sphere centers to the pinned source model before reset."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from g1_wbc import CONTACT_REFERENCE_SHA256, load_contact_reference, source_sphere_center

    reference = load_contact_reference()
    report = {"status": "preparing_source_contact", "reference_sha256": CONTACT_REFERENCE_SHA256,
              "edit_layer": "session", "applied_before_reset": False, "feet": {}}
    write_json(directory / "contact-override.json", report)
    if robot.is_initialized:
        raise RuntimeError("Source contact geometry must precede physics initialization")

    def read(body, prim):
        cache = UsdGeom.XformCache()
        relative = cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()
        return {"radius": float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()),
                "center_body_m": [float(v) for v in relative.ExtractTranslation()],
                "linear_body": [[float(relative[i][j]) for j in range(3)] for i in range(3)]}

    edits = []
    for name in reference["foot_bodies"]:
        body = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI) or not body.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError(f"Expected a rigid foot with authored mass: {name}")
        spheres = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(spheres) != 4 or any(p.GetTypeName() != "Sphere" for p in spheres):
            raise RuntimeError(f"Expected exactly four foot collision spheres: {name}")
        mass = UsdPhysics.MassAPI(body)
        foot = {"authored_mass_kg": float(mass.GetMassAttr().Get()),
                "authored_com_b_m": [float(v) for v in mass.GetCenterOfMassAttr().Get()],
                "authored_principal_inertia_kg_m2": [float(v) for v in mass.GetDiagonalInertiaAttr().Get()],
                "spheres": [{"path": str(p.GetPath()), **read(body, p)} for p in spheres]}
        report["feet"][name] = foot
        edits.append((body, foot))
    write_json(directory / "contact-override.json", report)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for body, foot in edits:
            for sphere in foot["spheres"]:
                # Instance proxies are read-only; de-instance only the collider ancestors.
                parent = stage.GetPrimAtPath(sphere["path"]).GetParent()
                while parent.GetPath() != body.GetPath():
                    if parent.IsInstance():
                        parent.SetInstanceable(False)
                    parent = parent.GetParent()
            for sphere in foot["spheres"]:
                prim = stage.GetPrimAtPath(sphere["path"])
                translate = prim.GetAttribute("xformOp:translate")
                if not prim or prim.IsInstanceProxy() or not translate:
                    raise RuntimeError(f"Foot sphere is not editable: {sphere['path']}")
                if not translate.Set(Gf.Vec3d(*source_sphere_center(sphere["center_body_m"], reference).tolist())):
                    raise RuntimeError(f"Unable to author source foot sphere: {sphere['path']}")
    for body, foot in edits:
        for sphere in foot["spheres"]:
            after = read(body, stage.GetPrimAtPath(sphere["path"]))
            sphere.update(after_center_body_m=after["center_body_m"], after_linear_body=after["linear_body"])
    report.update(status="authored_before_physics_reset", applied_before_reset=True)
    write_json(directory / "contact-override.json", report)
    return report


def read_material(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing physics material: {path}")
    report = {"path": path}
    for name, attribute in (("static_friction", "physics:staticFriction"), ("dynamic_friction", "physics:dynamicFriction"),
                            ("friction_combine_mode", "physxMaterial:frictionCombineMode")):
        attr = prim.GetAttribute(attribute)
        value = attr.Get() if attr else None
        report[name] = value if value is None or isinstance(value, float) else str(value)
    return report


def read_palm_runtime(robot):
    """Read the pinned PhysX body's local COM and full inertia in public order."""
    import numpy as np

    def array(value):
        value = value.torch if hasattr(value, "torch") else value
        return value[0].detach().cpu().numpy().copy()

    masses, poses, inertias = (array(robot.data.body_mass), array(robot.data.body_com_pose_b),
                              array(robot.data.body_inertia))
    result = {}
    for side in ("left", "right"):
        name = f"{side}_hand_palm_link"
        if list(robot.body_names).count(name) != 1:
            raise RuntimeError(f"Expected exactly one runtime palm: {name}")
        i = robot.body_names.index(name)
        result[side] = {"mass_kg": float(masses[i]), "com_b_m": poses[i, :3].tolist(),
                        "com_pose_b_xyzw": poses[i].tolist(),
                        "inertia_b_kg_m2": np.asarray(inertias[i]).reshape(3, 3, order="F").tolist()}
    return result


def analyze_palm_override(directory, profile, contract, measured=None):
    if not profile.get("inertial_override"):
        return None
    from g1_wbc import palm_inertial_reference, palm_override_summary, finite
    import numpy as np

    evidence = json.loads((directory / "palm-inertial-override.json").read_text())
    if evidence["status"] != "passed_palm_inertial_override":
        raise ValueError("Incomplete palm inertial override evidence")
    report = palm_override_summary(evidence, palm_inertial_reference(contract))
    if measured is not None:
        for side in ("left", "right"):
            i = measured["body_names"].index(f"{side}_hand_palm_link")
            actual = evidence["runtime"][side]
            if (abs(actual["mass_kg"]-measured["body_mass_kg"][i]) > 1e-7 or
                    not np.allclose(finite(actual["com_b_m"], (3,)), finite(measured["body_com_pose_b_xyzw"][i], (7,))[:3], atol=1e-7, rtol=0) or
                    not np.allclose(finite(actual["inertia_b_kg_m2"], (3, 3)),
                        finite(measured["body_inertia_backend_kg_m2"][i], (9,)).reshape(3, 3, order="F"), atol=1e-10, rtol=0)):
                raise ValueError("Palm override evidence differs from the static dynamics snapshot")
    return report


def read_runtime_dynamics(robot, source_to_asset, usd_path, *, directory=None, filename="dynamics-runtime.json"):
    """Capture backend-reordered buffers and resolved actuator parameters."""
    from isaaclab.actuators import ActuatorBase, ImplicitActuatorCfg
    from isaaclab.actuators.newton import read_group_parameter
    from g1_wbc import finite

    def array(value):
        value = value.torch if hasattr(value, "torch") else value
        return value[0].detach().cpu().numpy().copy().tolist()

    data = robot.data
    result = {"status": "partial_runtime_dynamics", "asset_usd_path": usd_path,
            "body_names": list(robot.body_names), "joint_names": list(robot.joint_names),
            "body_mass_kg": array(data.body_mass),
            "body_com_pose_b_xyzw": array(data.body_com_pose_b),
            "body_link_pose_w_xyzw": array(data.body_link_pose_w),
            "body_inertia_backend_kg_m2": array(data.body_inertia),
            "inertia_semantics": "Raw pinned PhysX backend tensor; compare eigenvalues only, without assuming world or principal-axis orientation.",
            "root_pose_xyzw": array(data.root_link_pose_w),
            "joint_positions": array(data.joint_pos),
            "joint_position_targets": array(data.joint_pos_target),
            "joint_velocity_targets": array(data.joint_vel_target),
            "solver_joint_stiffness": array(data.joint_stiffness),
            "solver_joint_damping": array(data.joint_damping),
            "joint_armature": array(data.joint_armature),
            "joint_effort_limits": array(data.joint_effort_limits),
            "source_to_asset_rotation": source_to_asset.tolist(), "actuators": {}}

    def save():
        if directory is not None:
            write_json(directory / filename, result)

    save()  # Preserve physical data even if controller introspection fails.
    try:
        if len(robot.joint_names) == 29:
            result["collection_phase"] = "passive_joint_terms"
            save()
            result.update(joint_static_friction_nm=array(data.joint_friction_coeff),
                          joint_dynamic_friction_nm=array(data.joint_dynamic_friction_coeff),
                          joint_viscous_friction_nm_s_rad=array(data.joint_viscous_friction_coeff))
            save()
        for name, owner in robot.actuators.items():
            result["collection_phase"] = f"actuator:{name}"
            save()
            cfg = robot.cfg.actuators[name]
            # Use the same public resolver/order as ActuatorCollection.
            _, names = robot.find_joints(cfg.joint_names_expr)
            if isinstance(owner, ActuatorBase):
                if list(owner.joint_names) != list(names):
                    raise RuntimeError(f"Actuator group joint order differs: {name}")
                implicit = bool(owner.is_implicit_model)
                kp, kd = array(owner.stiffness), array(owner.damping)
                origin = "isaaclab_model"
            else:
                # PhysX can execute native controllers too. The raw owner may
                # be shared between groups or be a tuple of merged actuators.
                if isinstance(cfg, ImplicitActuatorCfg):
                    raise RuntimeError(f"Unexpected native implicit group in the reviewed PhysX path: {name}")
                kp = array(read_group_parameter(robot.actuators, name, "controller", "kp"))
                kd = array(read_group_parameter(robot.actuators, name, "controller", "kd"))
                implicit = False  # Reviewed PhysX native path selects explicit groups only.
                origin = "native_group_parameter_api"
            finite(kp, (len(names),))
            finite(kd, (len(names),))
            owners = owner if isinstance(owner, tuple) else (owner,)
            classes = [type(o).__module__+"."+type(o).__name__ for o in owners]
            result["actuators"][name] = {"joint_names": list(names), "class": ", ".join(classes),
                "owner_classes": classes, "implicit": implicit, "kp": kp, "kd": kd,
                "parameter_source": origin}
            save()
    except Exception as error:
        result["collection_error"] = str(error) or repr(error)
        save()
        raise
    result.pop("collection_phase", None)
    result["status"] = "captured_runtime_dynamics"
    save()
    return result


def analyze_dynamics(directory):
    from g1_wbc import dynamics_comparison, load_dynamics_reference, placement_summary
    profile = json.loads((directory / "profile.json").read_text())
    asset = json.loads((directory / "asset-profile.json").read_text())
    contract, reference = trial_contract(profile)
    if contract != json.loads((directory / "controller-contract.json").read_text()):
        raise ValueError("Saved audit contract differs from the current reviewed contract")
    source = load_dynamics_reference()
    if source != json.loads((directory / "dynamics-reference.json").read_text()):
        raise ValueError("Saved dynamics reference differs from the pinned extraction")
    measured = json.loads((directory / "dynamics-runtime.json").read_text())
    initial = json.loads((directory / "initial-state.json").read_text())
    import numpy as np
    validate_inventory(json.loads((directory / "asset-inventory.json").read_text()), asset)
    if (initial["status"] != "passed_explicit_initialization" or
            not np.allclose(initial["root_vel_world"], np.zeros(6), atol=1e-4, rtol=0) or
            not np.allclose(initial["joint_velocities"], np.zeros(len(reference.limits)), atol=1e-4, rtol=0) or
            not np.allclose(measured["root_pose_xyzw"], initial["root_pose_xyzw"], atol=1e-6, rtol=0) or
            not np.allclose(measured["joint_positions"], initial["joint_positions"], atol=1e-6, rtol=0)):
        raise ValueError("Audit pose differs from the initialized state")
    report = dynamics_comparison(measured, reference, source, contract, asset)
    report["initial_placement"] = placement_summary(json.loads((directory / "initial-placement.json").read_text()),
        profile["initialization"], contract["height_command"], initial["root_pose_xyzw"])
    if profile.get("inertial_override"):
        report["inertial_override"] = analyze_palm_override(directory, profile, contract, measured)
    if profile.get("deployment"):
        report["deployment"] = validate_deployment_evidence(directory, asset, contract, profile.get("arm_hold") == "implicit")
    return report


def dynamics_table(report):
    totals = report["totals"]
    lines = ["# Static G1 dynamics comparison", "",
        "Measured runtime snapshot versus published deployment data. No controller execution or causal attribution.", "",
        "| Quantity | Source | Runtime |", "| --- | ---: | ---: |",
        f"| Total mass (kg) | {totals['source_mass_kg']:.8g} | {totals['runtime_mass_kg']:.8g} |",
        f"| Rigid bodies | {report['source_body_count']} | {report['runtime_body_count']} |",
        f"| Actuated joints | {report['source_actuated_joint_count']} | {report['runtime_actuated_joint_count']} |", "",
        f"Whole-robot COM difference in the source root frame (m): {totals['com_difference_m']}", "",
        "Same-name body rows may differ due to split/merged bodies. All runtime bodies contribute to totals.", "",
        "| Body | Source mass (kg) | Runtime mass (kg) | COM difference (m) | Source principal inertia (kg m2) | Runtime principal inertia (kg m2) |",
        "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in report["bodies"]:
        lines.append(f"| {row['body']} | {row['source_mass_kg']:.8g} | {row['runtime_mass_kg']:.8g} | "
            f"{row['com_difference_norm_m']:.6g} | {row['source_principal_inertia_kg_m2']} | {row['runtime_principal_inertia_kg_m2']} |")
    lines.extend(["", f"Runtime-only bodies: {report['runtime_only_bodies']}", f"Source-only bodies: {report['source_only_bodies']}", "",
        "| Joint | Source model | Runtime model | Source kp / kd | Runtime kp / kd | Runtime target (rad) |",
        "| --- | --- | --- | --- | --- | ---: |"])
    for row in report["controls"]:
        source_gains = f"{row['source_kp']} / {row['source_kd']}" if row["source_actuated"] else "not actuated"
        lines.append(f"| {row['joint']} | {row['source_actuator']} | {row['runtime_actuator']} | "
                     f"{source_gains} | {row['runtime_kp']:.6g} / {row['runtime_kd']:.6g} | {row['runtime_target_position_rad']:.6g} |")
    return "\n".join([*lines, "", *[f"- {s}" for s in report["limitations"]], ""])


def read_collision_geometry(stage, body_poses):
    """Use measured body poses plus authored relative collision transforms.

    Fabric can leave USD world body transforms stale. Only shape-to-body
    transforms come from USD; current body world poses come from Isaac buffers.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()

    def matrix(transform):
        # Gf uses row vectors; store explicit column-vector matrices for NumPy.
        result = np.eye(4)
        result[:3, 3] = transform.Transform(Gf.Vec3d(0, 0, 0))
        for i in range(3):
            result[:3, i] = transform.TransformDir(Gf.Vec3d(*(float(v) for v in np.eye(3)[i])))
        return result.tolist()

    def shapes(path):
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Missing initial collision scope: {path}")
        result = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if enabled is True:
                    result.append(prim)
                elif enabled is not False:
                    raise RuntimeError(f"Unresolved collision-enabled state: {prim.GetPath()}")
        if not result:
            raise RuntimeError(f"No enabled initial collision shapes: {path}")
        return root, result

    _, ground = shapes("/World/Ground")
    if len(ground) != 1 or ground[0].GetTypeName() != "Plane":
        raise RuntimeError("Initial placement requires exactly one ground collision Plane")
    plane = ground[0]
    report = {"up_axis": str(UsdGeom.GetStageUpAxis(stage)),
              "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
              "ground": {"collision_path": str(plane.GetPath()), "type": "Plane",
                         "axis": str(plane.GetAttribute("axis").Get()),
                         "world_from_plane": matrix(cache.GetLocalToWorldTransform(plane))}, "feet": {}}
    for name, pose in body_poses.items():
        root, spheres = shapes(f"/World/envs/env_0/Robot/{name}")
        body_inverse = cache.GetLocalToWorldTransform(root).GetInverse()
        foot = {"body_pose_xyzw": list(pose), "colliders": []}
        report["feet"][name] = foot
        for prim in spheres:
            if prim.GetTypeName() != "Sphere":
                raise RuntimeError(f"Unsupported initial foot collider: {prim.GetPath()} ({prim.GetTypeName()})")
            foot["colliders"].append({"collision_path": str(prim.GetPath()), "type": "Sphere",
                "radius_m": UsdGeom.Sphere(prim).GetRadiusAttr().Get(),
                "body_from_shape": matrix(cache.GetLocalToWorldTransform(prim)*body_inverse)})
    return report


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
    if (directory / "dynamics-runtime.json").exists():
        return analyze_dynamics(directory)
    profile = json.loads((directory / "profile.json").read_text())
    initial = json.loads((directory / "initial-state.json").read_text())
    records = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    report = {"status": "motion_only", "motion": motion_summary(initial["root_pose_xyzw"], records),
              "stationary_success_assessed": False, "contacts": None, "policy_inputs_verified": False}
    worker_path = directory / "balance-worker.json"
    report["run_status"] = json.loads(worker_path.read_text()).get("status") if worker_path.exists() else "unknown"
    if profile.get("initialization"):
        from g1_wbc import placement_summary
        report["initial_placement"] = placement_summary(
            json.loads((directory / "initial-placement.json").read_text()), profile["initialization"],
            json.loads((directory / "controller-contract.json").read_text())["height_command"], initial["root_pose_xyzw"])
    if profile.get("inertial_override"):
        report["inertial_override"] = analyze_palm_override(directory, profile,
            json.loads((directory / "controller-contract.json").read_text()))
    if profile.get("contact_override"):
        from g1_wbc import contact_override_summary
        evidence = json.loads((directory / "contact-override.json").read_text())
        if evidence["status"] != "passed_source_contact":
            raise ValueError("Incomplete source contact evidence")
        report["contact_override"] = contact_override_summary(evidence,
            json.loads((directory / "initial-placement.json").read_text()),
            json.loads((directory / "deployment-runtime.json").read_text()),
            "multiply" if profile.get("push_object") else None)
    if not profile.get("diagnostics"):
        report["limitation"] = "This historical run did not record contact forces or policy inputs; foot slip cannot be determined."
        return report
    diagnostics = profile["diagnostics"]
    asset = json.loads((directory / "asset-profile.json").read_text())
    dofs = len(json.loads((directory / "asset-inventory.json").read_text())["joint_names"])
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
        q = finite(state["joint_positions"], (dofs,))
        dq = finite(state["joint_velocities"], (dofs,))
        pose = finite(state["root_pose_xyzw"], (7,))
        velocity = finite(state["root_velocity_world"], (6,))
        observation = history.observe(q[indices], dq[indices], pose[3:], velocity[3:], rotation, profile.get("command", (0, 0, 0)),
                                      profile.get("rpy_command", (0, 0, 0)))
        if not np.allclose(observation, finite(row["policy_observation"], (516,)), atol=1e-6, rtol=0):
            raise ValueError("Logged policy observation disagrees with its saved input state/history")
        history.targets(row["action"], limits)
        if profile.get("deployment"):
            expected_applied = np.zeros(15) if row["step"] == 1 else records[row["step"]-2]["action"]
            if (row["policy_phase"] != "after_physics" or
                    not np.isclose(row["policy_input_time_seconds"], row["sim_time_seconds"], atol=1e-9, rtol=0) or
                    not np.allclose(row["applied_action"], expected_applied, atol=1e-7, rtol=0) or
                    not np.allclose(state["root_pose_xyzw"], row["root_pose_xyzw"], atol=1e-7, rtol=0)):
                raise ValueError("Deployment control-phase evidence differs from the source schedule")
    report["policy_inputs_verified"] = bool(records)
    report["status"] = "diagnostics_complete" if len(samples) == expected and len(records) == profile["control_steps"] else "diagnostics_partial"
    report["limitation"] = "Input-state consistency is checked with the adapter; it does not independently establish equivalence to upstream training observations."
    if profile.get("deployment"):
        from g1_wbc import functional_summary
        effective, _ = trial_contract(profile)
        if effective != contract:
            raise ValueError("Saved deployment controller differs from the pinned effective contract")
        report["deployment"] = validate_deployment_evidence(directory, asset, contract, profile.get("arm_hold") == "implicit")
        placement = json.loads((directory / "initial-placement.json").read_text())
        foot_start = {n: placement["after_geometry"]["feet"][n]["body_pose_xyzw"][:2] for n in diagnostics["foot_body_names"]}
        object_start = None
        if profile.get("push_object"):
            from g1_wbc import push_object_summary
            pushed = json.loads((directory / "push-object.json").read_text())
            if pushed["status"] != "passed_push_object":
                raise ValueError("Incomplete push object evidence")
            report["push_object"] = push_object_summary(pushed, profile["push_object"])
            object_start = pushed["initial_pose_xyzw"]
            if (directory / "hand-contact-metadata.json").exists():
                from g1_wbc import push_force_summary
                report["push_force"] = push_force_summary(samples)
        report["functional"] = functional_summary(profile["functional_case"], initial, records, foot_start, samples, object_start)
        report["stationary_success_assessed"] = profile["functional_case"] == "stand" and report["status"] == "diagnostics_complete"
    return report


@contextmanager
def record_worker_failure(directory, result, stage="balance"):
    """Persist the original cause before the enclosing Kit launcher can exit."""
    try:
        yield
    except BaseException as error:
        message = result.get("error") or str(error) or repr(error)
        result.update(status="failed", error=message, error_type=type(error).__name__)
        write_json(directory / f"{stage}-worker.json", result)
        (directory / f"{stage}-error.log").write_text(traceback.format_exc())
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
        if base != DEFAULT_PROFILE or set(profile) - {"base_profile", "diagnostics", "initialization", "inertial_override", "contact_override", "arm_hold", "push_object", "rpy_command", "deployment", "command", "functional_case", "scope"}:
            raise ValueError("Trial extensions may only add diagnostics, placement and the reviewed palm/contact overrides")
        profile = {**json.loads(base.read_text()), **profile}
    if (diagnostics := profile.get("diagnostics")) is not None:
        if (not isinstance(diagnostics, dict) or
                diagnostics.get("foot_body_names") != ["left_ankle_roll_link", "right_ankle_roll_link"] or
                type(diagnostics.get("minimum_normal_force_n")) not in (float, int) or
                not 0.1 <= diagnostics["minimum_normal_force_n"] <= 20):
            raise ValueError("Invalid foot contact diagnostic configuration")
    deployment = profile.get("deployment")
    if not deployment and any(k in profile for k in ("command", "functional_case")):
        raise ValueError("Motion commands require the reviewed deployment protocol")
    if deployment is not None:
        from g1_wbc import load_deployment_reference
        source = load_deployment_reference()
        commands = {"stand": [0, 0, 0], "forward": [0.2, 0, 0], "turn": [0, 0, 0.2], "push": [0.2, 0, 0]}
        if (deployment != "assets/references/g1_wbc_deployment.json" or
                profile.get("functional_case") not in commands or
                profile.get("command") != commands.get(profile.get("functional_case")) or
                profile["control_steps"] != 250 or not profile.get("inertial_override")):
            raise ValueError("Invalid coherent deployment/functional protocol")
    if (initialization := profile.get("initialization")) is not None:
        # The fixed source pose requires nonpenetration; automatic placement adds a margin.
        valid_source_pose = bool(deployment) and initialization == {
            "mode": "source_pose", "minimum_clearance_m": 0.0,
            "maximum_vertical_shift_m": 0.1, "root_height_m": source["initial_root_height_m"]}
        if (not profile.get("diagnostics") or not isinstance(initialization, dict) or
                (not valid_source_pose and (set(initialization) != {"mode", "minimum_clearance_m", "maximum_vertical_shift_m"} or
                initialization["mode"] != "collision_clearance")) or
                type(initialization["minimum_clearance_m"]) not in (int, float) or
                (not valid_source_pose and not 0.0005 <= initialization["minimum_clearance_m"] <= 0.005) or
                type(initialization["maximum_vertical_shift_m"]) not in (int, float) or
                not 0 < initialization["maximum_vertical_shift_m"] <= 0.1):
            raise ValueError("Invalid initial placement experiment configuration")
        # A default-crouch start with clearance placement is reviewed only with the implicit arm hold.
        if deployment and not valid_source_pose and profile.get("arm_hold") != "implicit":
            raise ValueError("Deployment requires the source root pose without height adjustment")
    if "inertial_override" in profile:
        if (not profile.get("initialization") or
                profile["inertial_override"] != {"body": "left_hand_palm_link", "source": "pinned_urdf"}):
            raise ValueError("Only the pinned left-palm inertial override with initial placement is reviewed")
    if "contact_override" in profile and (
            not deployment or profile["contact_override"] != {"source": "assets/references/g1_wbc_contact.json"}):
        raise ValueError("Only the pinned source foot contact within the deployment protocol is reviewed")
    if "arm_hold" in profile and (not profile.get("contact_override") or profile["arm_hold"] != "implicit"):
        raise ValueError("Only an implicit arm hold on the source-contact deployment is reviewed")
    push = profile.get("push_object")
    if (push is not None or profile.get("functional_case") == "push") and (
            profile.get("functional_case") != "push" or profile.get("arm_hold") != "implicit" or not isinstance(push, dict) or
            set(push) != {"size_m", "mass_kg", "friction", "gap_m"} or
            not isinstance(push["size_m"], list) or len(push["size_m"]) != 3 or
            any(type(v) not in (int, float) or not 0.1 <= v <= 2.0 for v in push["size_m"]) or
            type(push["mass_kg"]) not in (int, float) or not 0.5 <= push["mass_kg"] <= 100 or
            type(push["friction"]) not in (int, float) or not 0.05 <= push["friction"] <= 1.0 or
            type(push["gap_m"]) not in (int, float) or not 0 <= push["gap_m"] <= 0.5):
        raise ValueError("Invalid push object experiment configuration")
    # Only a bounded forward torso-pitch lean is reviewed; the policy observes it as its rpy command.
    rpy = profile.get("rpy_command")
    if rpy is not None and (push is None or not isinstance(rpy, list) or len(rpy) != 3 or
                            any(type(v) not in (int, float) for v in rpy) or rpy[0] != 0 or rpy[2] != 0 or
                            not 0 <= rpy[1] <= 0.3):
        raise ValueError("Invalid lean command configuration")
    asset, runtime = load_asset_profile(ROOT / profile["asset_profile"])
    if deployment:
        asset["fixed_hand_joints"] = asset["joint_groups"]["hands"]
        asset["joint_groups"]["hands"] = []
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


def balance_check(profile, asset, runtime, contract, reference, model_path, directory, *, audit_only=False):
    import numpy as np
    import torch
    import warp as wp
    from g1_wbc import (BalancePolicy, ObservationHistory, FrameMismatchError, actuator_limits, check_frames,
                        finite, joint_anchor_poses, matrix_quat_xyzw, quat_xyzw_matrix)

    deployment = bool(profile.get("deployment"))
    dofs = len(reference.limits)
    encoder = None if audit_only else require_ffmpeg(directory)
    torch.manual_seed(runtime["seed"])
    policy = None if audit_only else BalancePolicy(model_path, contract)
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
    implicit_arms = profile.get("arm_hold") == "implicit"
    explicit_upper = deployment and not implicit_arms
    candidate.actuators = {"reference_pd": IdealPDActuatorCfg(
        joint_names_expr=lower, stiffness={n: kp[n] for n in lower}, damping={n: kd[n] for n in lower},
        actuator_effort_limit={n: motor_limits[n]["effort"] for n in lower},
        joint_effort_limit={n: motor_limits[n]["effort"] for n in lower},
        joint_velocity_limit={n: motor_limits[n]["velocity"] for n in lower},
        armature=contract["lower_body_armature"], friction=0.1 if deployment else 0.0,
        dynamic_friction=0.1 if deployment else 0.0, viscous_friction=0.001 if deployment else 0.0,
    ), "upper_hold": (IdealPDActuatorCfg if explicit_upper else ImplicitActuatorCfg)(
        joint_names_expr=upper, stiffness={n: kp[n] for n in upper}, damping={n: kd[n] for n in upper},
        **({"actuator_effort_limit": {n: motor_limits[n]["effort"] for n in upper}} if explicit_upper else {}),
        joint_effort_limit={n: motor_limits[n]["effort"] for n in upper},
        joint_velocity_limit={n: motor_limits[n]["velocity"] for n in upper},
        armature=contract["upper_body_armature"], friction=0.1 if deployment else 0.0,
        dynamic_friction=0.1 if deployment else 0.0, viscous_friction=0.001 if deployment else 0.0,
    )}
    material = {}
    push = profile.get("push_object")
    combine_mode = "multiply" if push else None
    if profile.get("contact_override"):
        from g1_wbc import load_contact_reference
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
        # Bind the source coefficient to the ground and to unbound foot colliders. Multiply keeps
        # foot-ground friction at 1.0 x 1.0 while a pushed object keeps its own coefficient.
        surface = load_contact_reference()["isaac_material"]
        material = {"physics_material": PhysxRigidBodyMaterialCfg(**{**surface, **({"friction_combine_mode": combine_mode} if push else {})})}
    if push:
        from isaaclab.assets import RigidObjectCfg
        # Hand reach: the farthest default-pose hand link origin plus an allowance for the distal finger
        # segment (~0.046 m); run 143003 stalled with fingertips touching at ~0.40 m.
        default_pose = reference.forward({**dict.fromkeys(reference.limits, 0.0), **dict(zip(lower, contract["default_lower_angles"]))})
        hand_reach = max(float(pose[0, 3]) for name, pose in default_pose.items() if "_hand_" in name) + FINGERTIP_ALLOWANCE_M
        furniture_cfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Furniture", spawn=sim_utils.CuboidCfg(
            size=tuple(push["size_m"]), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.22)),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(), mass_props=sim_utils.MassCfg(mass=push["mass_kg"]),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(), physics_material_path="physicsMaterial",
            physics_material=PhysxRigidBodyMaterialCfg(static_friction=push["friction"], dynamic_friction=push["friction"],
                                                       friction_combine_mode="multiply")),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(hand_reach+push["gap_m"]+push["size_m"][0]/2, 0.0, push["size_m"][2]/2)))

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(**material))
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500))
        robot = candidate
        if push:
            furniture = furniture_cfg
        if not audit_only:
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
        if push:
            hand_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_hand/.*_hand_.*_link", update_period=0.0, history_length=1,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Furniture"], debug_vis=False)

    @configclass
    class LaunchCfg:
        sim = sim_utils.SimulationCfg(device=runtime["device"], dt=asset["physics_dt"], physics=PhysxCfg(), **material)
        scene = SceneCfg(num_envs=1, env_spacing=3.0)

    cfg = LaunchCfg()
    stage = "dynamics" if audit_only else "balance"
    result = {"status": "running", "steps_completed": 0, "resets": 0,
              "controller": policy.metadata if policy else {},
              "mode": "static_dynamics_audit_no_control" if audit_only else ("deployment_"+profile["functional_case"] if deployment else "stand_only_zero_navigation"),
              "controller_inference_verified": False, "walking_verified": False}
    write_json(directory / f"{stage}-worker.json", result)
    actuator_evidence = {"kp": kp, "kd": kd,
                         "configuration": {n: cfg.to_dict() for n, cfg in candidate.actuators.items()},
                         "limit_origin": contract["actuator_limit_origin"],
                         "actuation_origin": contract["actuation_origin"],
                         "upper_hold_gain_origin": contract["hold_gain_origin"]}
    write_json(directory / "actuators.json", actuator_evidence)
    with launch_simulation(cfg, {"headless": True, "device": runtime["device"],
                                 "enable_cameras": not audit_only, "kit_args": "--allow-root"}), record_worker_failure(directory, result, stage):
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        if deployment:
            apply_fixed_hands(sim.stage, scene["robot"], directory)
        palm_override = apply_palm_override(sim.stage, scene["robot"], contract, directory) if profile.get("inertial_override") else None
        contact_override = apply_source_contact(sim.stage, scene["robot"], directory) if profile.get("contact_override") else None
        sim.reset()
        robot, camera = scene["robot"], None if audit_only else scene["camera"]
        if palm_override is not None:
            from g1_wbc import palm_override_summary, palm_inertial_reference
            palm_override["runtime"] = read_palm_runtime(robot)
            write_json(directory / "palm-inertial-override.json", palm_override)
            palm_override["verification"] = palm_override_summary(palm_override, palm_inertial_reference(contract))
            palm_override["status"] = "passed_palm_inertial_override"
            write_json(directory / "palm-inertial-override.json", palm_override)

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
                "sampling": "Post-physics-step poses, world twists and forces; efforts are recorded collection staging buffers",
                "effort_telemetry_limitation": "Native execution bypasses these processed-command buffers; actual submitted effort and clipping are not established.",
                "point_velocity_limitation": "Velocity is measured at the rigid-body origin, not a contact point"})
        if push:
            hand_sensor = scene["hand_contact"]
            hand_names = list(hand_sensor.body_names)
            hand_sides = {side: [i for i, n in enumerate(hand_names) if n.startswith(f"{side}_hand_")] for side in ("left", "right")}
            if not all(hand_sides.values()) or sum(map(len, hand_sides.values())) != len(hand_names):
                raise RuntimeError(f"Hand contact sensor resolved unexpected bodies: {hand_names}")
            write_json(directory / "hand-contact-metadata.json", {
                "sensor_body_names": hand_names, "filter_prim_path": "/World/envs/env_0/Furniture",
                "force_scope": "Filtered hand-link forces from the box only, summed per side",
                "limitation": "Wrist and forearm contacts with the box are not sensed"})

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
        if deployment:
            snapshot = read_runtime_dynamics(robot, np.eye(3), candidate.spawn.usd_path,
                                             directory=directory, filename="deployment-runtime.json")
            write_json(directory / "deployment-runtime.json", snapshot)
            write_json(directory / "deployment-audit.json", validate_deployment_evidence(directory, asset, contract, implicit_arms))
        applied_armature = finite(array(robot.data.joint_armature), (dofs,))
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
            q = finite(array(robot.data.joint_pos), (dofs,))
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
        defaults = np.zeros(dofs)
        defaults[lower_indices] = contract["default_lower_angles"]

        def write_joints(q):
            robot.write_joint_position_to_sim_index(position=device(q))
            robot.write_joint_velocity_to_sim_index(velocity=device(np.zeros(dofs)))
            sim.forward()
            scene.update(0.0)

        for probe_index, delta in enumerate((-0.04, 0.04)):
            write_joints(np.clip(defaults+delta, limits[:, 0], limits[:, 1]))
            probe = verify_frames(rotation, label=f"probe-{probe_index}")
            frame_evidence["probes"].append(probe)
        write_json(directory / "frames.json", {"status": "passed_kinematic_probes", **frame_evidence})

        source_pose = (profile.get("initialization") or {}).get("mode") == "source_pose"
        initial_joints = np.zeros(dofs) if source_pose else defaults
        root_height = profile["initialization"]["root_height_m"] if source_pose else contract["height_command"]
        root_pose = np.r_[0.0, 0.0, root_height, matrix_quat_xyzw(rotation.T)]
        robot.write_root_pose_to_sim_index(root_pose=device(root_pose))
        robot.write_root_velocity_to_sim_index(root_velocity=device(np.zeros(6)))
        write_joints(initial_joints)
        scene.reset()
        robot.set_joint_position_target_index(target=device(defaults))
        robot.set_joint_velocity_target_index(target=device(np.zeros(dofs)))
        sim.forward()
        scene.update(0.0)
        if profile.get("initialization"):
            from g1_wbc import collision_clearance, placement_shift, placement_summary
            settings = profile["initialization"]
            placement = {"status": "measuring_initial_placement", "mode": settings["mode"],
                         "policy_height_command_m": contract["height_command"],
                         "before_root_pose_xyzw": array(robot.data.root_link_pose_w).tolist()}
            write_json(directory / "initial-placement.json", placement)

            def geometry():
                poses = frames()
                return read_collision_geometry(sim.stage, {n: poses[n].tolist() for n in diagnostics["foot_body_names"]})

            placement["before_geometry"] = geometry()
            write_json(directory / "initial-placement.json", placement)
            placement["before_clearance"] = collision_clearance(placement["before_geometry"])
            write_json(directory / "initial-placement.json", placement)
            lift = 0.0 if source_pose else placement_shift(placement["before_clearance"]["minimum_clearance_m"], settings)
            root_pose[2] += lift
            placement["vertical_shift_m"] = lift
            robot.write_root_pose_to_sim_index(root_pose=device(root_pose))
            sim.forward()
            scene.update(0.0)
            placement["after_root_pose_xyzw"] = array(robot.data.root_link_pose_w).tolist()
            placement["after_geometry"] = geometry()
            write_json(directory / "initial-placement.json", placement)
            placement["after_clearance"] = collision_clearance(placement["after_geometry"])
            placement["status"] = "measured_initial_placement"
            write_json(directory / "initial-placement.json", placement)
            placement["status"] = "passed_initial_placement"
            placement_summary(placement, settings, contract["height_command"], placement["after_root_pose_xyzw"])
            write_json(directory / "initial-placement.json", placement)
        if contact_override is not None:
            from g1_wbc import contact_override_summary
            contact_override["materials"] = {"default": read_material(sim.stage, "/physicsScene/defaultMaterial"),
                                             "ground": read_material(sim.stage, "/World/Ground/physicsMaterial")}
            write_json(directory / "contact-override.json", contact_override)
            contact_override["verification"] = contact_override_summary(contact_override, placement,
                json.loads((directory / "deployment-runtime.json").read_text()), combine_mode)
            contact_override["status"] = "passed_source_contact"
            write_json(directory / "contact-override.json", contact_override)
        furniture = scene["furniture"] if push else None
        if push:
            from g1_wbc import push_object_summary
            pushed = {"status": "measuring_push_object", "size_m": list(push["size_m"]), "hand_reach_m": hand_reach,
                      "fingertip_allowance_m": FINGERTIP_ALLOWANCE_M,
                      "material": read_material(sim.stage, "/World/envs/env_0/Furniture/geometry/physicsMaterial"),
                      "runtime_mass_kg": float(array(furniture.data.body_mass)[0]),
                      "initial_pose_xyzw": array(furniture.data.root_link_pose_w).tolist()}
            write_json(directory / "push-object.json", pushed)
            pushed["verification"] = push_object_summary(pushed, push)
            pushed["status"] = "passed_push_object"
            write_json(directory / "push-object.json", pushed)
        pose = finite(array(robot.data.root_link_pose_w), (7,))
        q = finite(array(robot.data.joint_pos), (dofs,))
        velocity = finite(array(robot.data.root_link_vel_w), (6,))
        dq = finite(array(robot.data.joint_vel), (dofs,))
        source_rotation = quat_xyzw_matrix(pose[3:]) @ rotation
        initial = {"status": "initialization_mismatch", "wanted_root_pose_xyzw": root_pose.tolist(),
                   "wanted_joint_positions": initial_joints.tolist(), "root_pose_xyzw": pose.tolist(),
                   "source_rotation_world": source_rotation.tolist(), "joint_positions": q.tolist(),
                   "root_vel_world": velocity.tolist(), "joint_velocities": dq.tolist()}
        write_json(directory / "initial-state.json", initial)
        if (np.max(np.abs(q-initial_joints)) > profile["initial_joint_tolerance_rad"] or
                np.max(np.abs(pose[:3]-root_pose[:3])) > profile["initial_pose_tolerance"] or
                not np.allclose(source_rotation, np.eye(3), atol=profile["initial_pose_tolerance"]) or
                np.max(np.abs(velocity)) > 1e-4 or np.max(np.abs(dq)) > 1e-4):
            raise RuntimeError("Explicit initial joint/root state was not established")
        initial["status"] = "passed_explicit_initialization"
        write_json(directory / "initial-state.json", initial)
        if audit_only:
            from g1_wbc import load_dynamics_reference
            write_json(directory / "dynamics-reference.json", load_dynamics_reference())
            snapshot = read_runtime_dynamics(robot, rotation, candidate.spawn.usd_path, directory=directory)
            write_json(directory / "dynamics-runtime.json", snapshot)
            comparison = analyze_dynamics(directory)
            write_json(directory / "dynamics-comparison.json", comparison)
            (directory / "dynamics-comparison.md").write_text(dynamics_table(comparison))
            result["status"] = "passed_dynamics_audit"
            write_json(directory / "dynamics-worker.json", result)
            return
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

        def infer(step, input_time):
            q, dq = array(robot.data.joint_pos), array(robot.data.joint_vel)
            pose, vel = array(robot.data.root_link_pose_w), array(robot.data.root_link_vel_w)
            observation = history.observe(q[indices], dq[indices], pose[3:], vel[3:], rotation, profile.get("command", (0, 0, 0)),
                                          profile.get("rpy_command", (0, 0, 0)))
            if diagnostics:
                input_state = {"joint_positions": q.tolist(), "joint_velocities": dq.tolist(),
                               "root_pose_xyzw": pose.tolist(), "root_velocity_world": vel.tolist()}
            action = policy(observation)
            try:
                target, clipped = history.targets(action, limits[lower_indices])
            except (RuntimeError, ValueError):
                write_json(directory / "rejected-action.json", {
                    "step": step+1, "sim_time_seconds": input_time, "applied": False,
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
            return action, clipped, observation, input_state if diagnostics else None

        capture(directory / "initial.png")
        error = None
        try:
            with (directory / "trajectory.jsonl").open("w") as trajectory, \
                    ((directory / "contacts.jsonl").open("w") if diagnostics else nullcontext(None)) as contact_log:
                for step in range(profile["control_steps"]):
                    applied_action = history.last_action.copy()
                    if not deployment:
                        action, clipped, observation, input_state = infer(step, step*dt)
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
                            computed = finite(array(robot.actuators.computed_effort), (dofs,))[lower_indices]
                            applied = finite(array(robot.actuators.applied_effort), (dofs,))[lower_indices]
                            physics_step = step*asset["control_decimation"]+substep+1
                            sample = {
                                "physics_step": physics_step, "control_step": step+1,
                                "sim_time_seconds": physics_step*asset["physics_dt"],
                                "lower_body_computed_effort": computed.tolist(),
                                "lower_body_applied_effort": applied.tolist(),
                                "feet": {name: {"pose_xyzw": foot_poses[i].tolist(),
                                    "velocity_world": foot_velocities[i].tolist(),
                                    "normal_force_world_n": normal[i].tolist(),
                                    "friction_force_world_n": friction[i].tolist(),
                                    "total_force_world_n": total[i].tolist()} for i, name in enumerate(feet)}}
                            if push:
                                matrix = finite(array(hand_sensor.data.force_matrix_w), (len(hand_names), 1, 3))[:, 0]
                                sample["hand_box_force_world_n"] = {side: matrix[rows].sum(axis=0).tolist() for side, rows in hand_sides.items()}
                            contact_log.write(json.dumps(sample, allow_nan=False)+"\n")
                    if deployment:
                        action, clipped, observation, input_state = infer(step, (step+1)*dt)
                    pose = finite(array(robot.data.root_link_pose_w), (7,))
                    q = finite(array(robot.data.joint_pos), (dofs,))
                    dq = finite(array(robot.data.joint_vel), (dofs,))
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
                    if push:
                        record.update(object_pose_xyzw=finite(array(furniture.data.root_link_pose_w), (7,)).tolist(),
                                      object_velocity_world=finite(array(furniture.data.root_link_vel_w), (6,)).tolist())
                    if deployment:
                        record.update(policy_phase="after_physics", policy_input_time_seconds=(step+1)*dt,
                                      applied_action=applied_action.tolist(), command=profile["command"])
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
    wanted = {"policy": "passed_policy_contract", "gpu": "checks_completed", "balance": "passed_balance_rollout",
              "dynamics": "passed_dynamics_audit"}[stage]
    if evidence.get("status") != wanted:
        raise RuntimeError(f"Incomplete {stage} evidence")
    if stage == "dynamics":
        if (evidence["steps_completed"] != 0 or evidence["controller_inference_verified"] is not False or
                evidence["mode"] != "static_dynamics_audit_no_control"):
            raise RuntimeError("Static dynamics audit unexpectedly executed a controller")
        report = analyze_dynamics(directory)
        if report["status"] != "compared_runtime_dynamics":
            raise RuntimeError("Incomplete dynamics comparison")
    if stage == "balance":
        import numpy as np
        from g1_wbc import check_frames, finite, joint_anchor_poses, quat_xyzw_matrix
        contract, reference = trial_contract(profile)
        dofs = len(reference.limits)
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
            finite(item["joint_positions"], (dofs,))
            finite(item["joint_velocities"], (dofs,))
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
        armature = json.loads((directory / "actuators.json").read_text())["joint_armature"]
        asset = (json.loads((directory / "asset-profile.json").read_text()) if profile.get("deployment")
                 else load_asset_profile(ROOT / profile["asset_profile"])[0])
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
        if profile.get("initialization"):
            from g1_wbc import placement_summary
            placement_summary(json.loads((directory / "initial-placement.json").read_text()),
                              profile["initialization"], contract["height_command"], initial["root_pose_xyzw"])
        pose = finite(initial["root_pose_xyzw"], (7,))
        wanted_pose = finite(initial["wanted_root_pose_xyzw"], (7,))
        if (np.max(np.abs(pose[:3]-wanted_pose[:3])) > profile["initial_pose_tolerance"] or
                np.max(np.abs(finite(initial["joint_positions"], (dofs,))-finite(initial["wanted_joint_positions"], (dofs,)))) > profile["initial_joint_tolerance_rad"] or
                not np.allclose(finite(initial["source_rotation_world"], (3, 3)), np.eye(3), atol=profile["initial_pose_tolerance"]) or
                np.max(np.abs(finite(initial["root_vel_world"], (6,)))) > 1e-4 or
                np.max(np.abs(finite(initial["joint_velocities"], (dofs,)))) > 1e-4):
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
            if profile.get("deployment") and analysis["functional"]["status"] != "passed_functional_evaluation":
                failed = [k for k, passed in analysis["functional"]["checks"].items() if not passed]
                raise RuntimeError(f"Functional gate failed: {failed}; inspect diagnostics-summary.json. No automatic retry.")
            materials = json.loads((directory / "contact-materials.json").read_text())
            if (set(materials["scopes"]) != {"ground", *profile["diagnostics"]["foot_body_names"]} or
                    any(not shapes for shapes in materials["scopes"].values())):
                raise RuntimeError("Incomplete diagnostic material evidence")


def worker(stage, profile, asset, runtime, contract_path, directory):
    try:
        write_json(directory / "versions.json", check_versions(runtime))
        from g1_wbc import load_contract, prepare_model
        contract, reference = trial_contract(profile)
        if stage == "gpu":
            write_json(directory / "gpu-worker.json", {"status": "checks_completed", **gpu_check(runtime, directory)})
        elif stage == "dynamics":
            balance_check(profile, asset, runtime, contract, reference, None, directory, audit_only=True)
        else:
            path = prepare_model(contract)
            if stage == "policy":
                write_json(directory / "policy-worker.json", policy_check(contract, path))
            else:
                balance_check(profile, asset, runtime, contract, reference, path, directory)
        return 0
    except BaseException as error:
        if stage in ("balance", "dynamics") and isinstance(error, SystemExit) and error.code in (None, 0):
            try:
                validate_stage(stage, directory, profile)
            except (OSError, ValueError, RuntimeError, KeyError) as validation_error:
                if profile.get("deployment"):
                    error = validation_error
            else:
                return 0
        path = directory / f"{stage}-worker.json"
        evidence = json.loads(path.read_text()) if path.exists() else {}
        message = (evidence.get("error") if isinstance(error, SystemExit) else None) or str(error) or repr(error)
        write_json(path, {**evidence, "status": "failed", "error": message,
                          "error_type": type(error).__name__})
        (directory / f"{stage}-error.log").write_text(traceback.format_exc())
        if profile.get("deployment"):
            raise error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--stage", choices=("policy", "balance", "dynamics"), default="policy")
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-run", type=Path, help="Analyze saved artifacts on CPU without Isaac, ONNX or Modal")
    parser.add_argument("--stand-run", type=Path, help="Verified deployment standing run required before moving trials")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", choices=("policy", "gpu", "balance", "dynamics"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.analyze_run:
        print(json.dumps(analyze_run(args.analyze_run), indent=2))
        return 0
    profile, asset, runtime = load_plan(args.profile)
    if (args.stage == "dynamics" or args.worker == "dynamics") and (not profile.get("initialization") or args.prepare_model):
        parser.error("Dynamics audit requires the placement profile and does not use --prepare-model")
    stages = ["gpu", "dynamics"] if args.stage == "dynamics" else (["policy"] if args.stage == "policy" else ["policy", "gpu", "balance"])
    if args.dry_run:
        dry_contract, _ = trial_contract(profile)
        print(json.dumps({"status": "local_plan_only", "stages": stages, "asset": asset["asset_config"],
                          "control_steps": 0 if args.stage == "dynamics" else profile["control_steps"],
                          "simulated_seconds": 0 if args.stage == "dynamics" else profile["control_steps"]*0.02,
                          "model_download_bytes": dry_contract["model"]["bytes"] if args.prepare_model else 0,
                          "policy_role": dry_contract.get("policy_role", "balance"),
                          "active_joints": sum(len(v) for v in asset["joint_groups"].values()),
                          "functional_case": profile.get("functional_case"),
                          "policy_phase": "after_physics" if profile.get("deployment") else "before_physics",
                          "contact_diagnostics": bool(profile.get("diagnostics")),
                          "initial_placement": profile.get("initialization", {"mode": "legacy_policy_height"}),
                          "inertial_override": profile.get("inertial_override"),
                          "contact_override": profile.get("contact_override"),
                          "arm_hold": profile.get("arm_hold", "explicit" if profile.get("deployment") else None),
                          "push_object": profile.get("push_object"),
                          "physics_samples": profile["control_steps"]*asset["control_decimation"] if args.stage == "balance" and profile.get("diagnostics") else 0,
                          "runtime_image_change": False, "training": False}, indent=2))
        return 0
    prerequisite = None
    if (profile.get("deployment") and profile["functional_case"] != "stand" and args.stage == "balance"
            and not args.worker):
        if args.stand_run is None:
            parser.error("Moving deployment trials require --stand-run from a successful standing evaluation")
        stand_dir = args.stand_run.expanduser().resolve()
        stand_profile = json.loads((stand_dir / "profile.json").read_text())
        if (stand_profile.get("deployment") != profile["deployment"] or stand_profile.get("functional_case") != "stand" or
                any(stand_profile.get(k) != profile.get(k) for k in ("contact_override", "arm_hold", "initialization"))):
            parser.error("--stand-run must use the same reviewed deployment, contact, arm hold and startup with a standing case")
        if json.loads((stand_dir / "runtime-config.json").read_text()) != runtime:
            parser.error("--stand-run must use the same runtime configuration")
        validate_stage("balance", stand_dir, stand_profile)
        prerequisite = {"run_id": stand_dir.name, "functional": analyze_run(stand_dir)["functional"],
                        "controller_contract_sha256": file_sha256(stand_dir / "controller-contract.json")}
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    contract_path = ROOT / profile["controller_contract"]
    if args.worker:
        if args.run_dir is None:
            parser.error("Internal worker requires --run-dir")
        return worker(args.worker, profile, asset, runtime, contract_path, args.run_dir)
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-wbc")
    directory.mkdir(parents=True)
    if prerequisite is not None:
        write_json(directory / "prerequisite-stand.json", prerequisite)
    write_json(directory / "profile.json", profile)
    write_json(directory / "asset-profile.json", asset)
    write_json(directory / "runtime-config.json", runtime)
    effective_contract, _ = trial_contract(profile)
    write_json(directory / "controller-contract.json", effective_contract)
    summary = {"status": "running", "stages": {}, "run_dir": str(directory)}
    write_json(directory / "summary.json", summary)
    try:
        from g1_wbc import load_contract, prepare_model
        contract, _ = trial_contract(profile)
        if args.prepare_model:
            print(f"Preparing pinned {contract['model']['bytes']:,}-byte {contract.get('policy_role', 'balance')} checkpoint and verifying SHA-256...", flush=True)
        if args.stage != "dynamics":
            prepare_model(contract, args.prepare_model)
        for stage in stages:
            print(f"Running {stage}; log: {directory / (stage+'.log')}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--profile", str(args.profile.resolve()),
                       "--worker", stage, "--run-dir", str(directory)]
            timeout = profile["stage_timeout_seconds"] if stage in ("balance", "dynamics") else (120 if stage == "gpu" else 60)
            code = run_stage(command, directory / f"{stage}.log", timeout)
            if code:
                raise stage_failure(stage, code, directory)
            validate_stage(stage, directory, profile)
            summary["stages"][stage] = "passed"
            write_json(directory / "summary.json", summary)
        summary["status"] = {"policy": "passed_policy_contract", "balance": "passed_balance_rollout", "dynamics": "passed_dynamics_audit"}[args.stage]
        if profile.get("deployment") and args.stage == "balance":
            summary["functional"] = analyze_run(directory)["functional"]
            summary["status"] = "passed_deployment_" + profile["functional_case"]
        return 0
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        (directory / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        if profile.get("diagnostics") and (directory / "trajectory.jsonl").is_file():
            try:
                analysis = analyze_run(directory)
                write_json(directory / "diagnostics-summary.json", analysis)
                if "functional" in analysis:
                    summary["functional"] = analysis["functional"]
            except (OSError, ValueError, RuntimeError, KeyError):
                (directory / "diagnostics-error.log").write_text(traceback.format_exc())
        write_json(directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
