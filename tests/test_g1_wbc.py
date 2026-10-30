"""Numerical convention and controller-transfer failure checks, without a GPU."""

from contextlib import ExitStack, contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g1_wbc as wbc
import isaac_g1_wbc as runner


def joint_frame_fixture(reference):
    return {name: {"parent": parent, "child": child, "position_in_child": [0, 0, 0],
                   "axis_in_child": list(axis)}
            for name, parent, child, kind, _, axis in reference.joints if kind == "revolute"}


def collision_fixture(poses):
    return {"up_axis": "Z", "meters_per_unit": 1.0,
            "ground": {"collision_path": "/World/Ground/CollisionPlane", "type": "Plane", "axis": "Z",
                       "world_from_plane": np.eye(4).tolist()},
            "feet": {name: {"body_pose_xyzw": list(pose), "colliders": [{"type": "Sphere",
                "collision_path": f"/World/envs/env_0/Robot/{name}/collisions/Sphere",
                "radius_m": 0.005, "body_from_shape": [[1,0,0,0.1],[0,1,0,0],[0,0,1,-0.03],[0,0,0,1]]}]}
                for name,pose in poses.items()}}


def dynamics_fixture(contract, reference, asset, *, body_names=None, joint_names=None, q=None, root=None, poses=None, implicit_arms=False):
    source = wbc.load_dynamics_reference()
    joints = joint_names or asset["controller_reference"]["joint_order"]+asset["joint_groups"]["hands"]
    positions = {n:0.0 for n in joints}
    positions.update(zip(asset["joint_groups"]["lower_body"], contract["default_lower_angles"]))
    q = np.asarray(q).tolist() if q is not None else [positions[n] for n in joints]
    root = np.asarray(root).tolist() if root is not None else [0,0,0.786,0,0,0,1]
    fk = reference.forward(dict(zip(joints,q,strict=True)))
    names = body_names or list(reversed(source["bodies"]))
    models = {};solver_kp={};solver_kd={}
    for key,names_group in asset["joint_groups"].items():
        implicit = key != "lower_body" and (implicit_arms or not contract.get("deployment_sha256"))
        kp = contract["kp"] if key == "lower_body" else [contract["upper_hold_kp"] if key == "arms" else contract["hand_hold_kp"]]*len(names_group)
        kd = contract["kd"] if key == "lower_body" else [contract["upper_hold_kd"] if key == "arms" else contract["hand_hold_kd"]]*len(names_group)
        models[key] = {"joint_names":names_group,"class":"fixture."+key,"implicit":implicit,"kp":kp,"kd":kd}
        solver_kp.update(zip(names_group,kp if implicit else [0]*len(names_group)))
        solver_kd.update(zip(names_group,kd if implicit else [0]*len(names_group)))
    fields = {"status":"captured_runtime_dynamics","body_names":names,"joint_names":joints,
        "root_pose_xyzw":root,"source_to_asset_rotation":np.eye(3).tolist(),"joint_positions":q,
        "joint_position_targets":q,"joint_velocity_targets":[0.0]*len(joints),
        "solver_joint_stiffness":[solver_kp[n]for n in joints],"solver_joint_damping":[solver_kd[n]for n in joints],
        "joint_armature":[contract["lower_body_armature"] if n in asset["joint_groups"]["lower_body"]else contract["upper_body_armature"] for n in joints],
        "joint_effort_limits":[wbc.actuator_limits(contract,reference)[n]["effort"] for n in joints],"actuators":models,
        "body_mass_kg":[],"body_com_pose_b_xyzw":[],"body_inertia_backend_kg_m2":[],"body_link_pose_w_xyzw":[]}
    for name in names:
        s=source["bodies"].get(name,{"mass_kg":0.01,"com_position_b_m":[0,0,0],"principal_inertia_kg_m2":[1e-5,2e-5,3e-5]})
        fields["body_mass_kg"].append(s["mass_kg"])
        fields["body_com_pose_b_xyzw"].append([*s["com_position_b_m"],0,0,0,1])
        rotation=wbc.axis_rotation([0,1,0],0.5)
        fields["body_inertia_backend_kg_m2"].append((rotation@np.diag(s["principal_inertia_kg_m2"])@rotation.T).reshape(9).tolist())
        transform=wbc.pose_matrix(root)@fk[name]
        fields["body_link_pose_w_xyzw"].append([*transform[:3,3],*wbc.matrix_quat_xyzw(transform[:3,:3])])
    if poses is not None: fields["body_link_pose_w_xyzw"]=poses
    if contract.get("deployment_sha256"):
        fields.update(joint_static_friction_nm=[0.1]*29, joint_dynamic_friction_nm=[0.1]*29,
                      joint_viscous_friction_nm_s_rad=[0.001]*29)
    return fields


def palm_fixture(contract):
    reference = wbc.palm_inertial_reference(contract)
    left = {k: copy.deepcopy(reference[k]) for k in ("mass_kg", "com_b_m", "principal_inertia_kg_m2", "principal_axes_xyzw")}
    left["has_mass_api"] = True
    mirror = np.diag([1, -1, 1])
    right_tensor = mirror @ np.array(reference["inertia_b_kg_m2"]) @ mirror
    moments, axes = np.linalg.eigh(right_tensor)
    if np.linalg.det(axes) < 0: axes[:, -1] *= -1
    right = {"has_mass_api": True, "mass_kg": reference["mass_kg"],
             "com_b_m": (mirror @ reference["com_b_m"]).tolist(),
             "principal_inertia_kg_m2": moments.tolist(), "principal_axes_xyzw": wbc.matrix_quat_xyzw(axes).tolist()}
    return {"status": "passed_palm_inertial_override", "reference": reference,
        "applied_before_reset": True, "edit_layer": "session",
        "prim_path": "/World/envs/env_0/Robot/left_hand/left_hand_palm_link",
        "before_usd": {"left": {"has_mass_api": False}, "right": copy.deepcopy(right)},
        "after_usd": {"left": left, "right": right},
        "runtime": {"left": {"mass_kg": reference["mass_kg"], "com_b_m": reference["com_b_m"],
                             "inertia_b_kg_m2": reference["inertia_b_kg_m2"]},
                    "right": {"mass_kg": right["mass_kg"], "com_b_m": right["com_b_m"],
                              "inertia_b_kg_m2": right_tensor.tolist()}}}


ACTUAL_FOOT_SPHERES = (("05", 0.12, 0.026), ("04", 0.12, -0.026), ("07", -0.02, -0.026), ("06", -0.02, 0.026))
SPHERE_SCALE = 0.009999999776482582


def contact_geometry(poses):
    """Initialized geometry with four scaled unit spheres at the source centers."""
    geometry = collision_fixture(poses)
    for name, foot in geometry["feet"].items():
        foot["colliders"] = []
        for index, center in zip(("06", "07", "05", "04"), wbc.load_contact_reference()["foot_sphere_centers_m"]):
            transform = np.eye(4)*SPHERE_SCALE; transform[:3, 3] = center; transform[3, 3] = 1
            foot["colliders"].append({"type": "Sphere", "radius_m": 0.5, "body_from_shape": transform.tolist(),
                "collision_path": f"/World/envs/env_0/Robot/{name}/collisions/spheres/Sphere_{index}"})
    return geometry


def contact_fixture():
    reference, source = wbc.load_contact_reference(), wbc.load_dynamics_reference()
    linear = (np.eye(3)*SPHERE_SCALE).tolist()
    evidence = {"status": "passed_source_contact", "reference_sha256": wbc.CONTACT_REFERENCE_SHA256,
                "edit_layer": "session", "applied_before_reset": True, "feet": {},
                "materials": {scope: {"path": path, "static_friction": 1.0, "dynamic_friction": 1.0, "friction_combine_mode": "max"}
                              for scope, path in (("default", "/physicsScene/defaultMaterial"), ("ground", "/World/Ground/physicsMaterial"))}}
    runtime = {"body_names": [], "body_mass_kg": [], "body_com_pose_b_xyzw": [], "body_inertia_backend_kg_m2": []}
    rotation = wbc.axis_rotation([0, 1, 0], 0.5)
    for name in reference["foot_bodies"]:
        body = source["bodies"][name]
        evidence["feet"][name] = {"authored_mass_kg": body["mass_kg"], "authored_com_b_m": body["com_position_b_m"],
            "authored_principal_inertia_kg_m2": body["principal_inertia_kg_m2"], "spheres": [
                {"path": f"/World/envs/env_0/Robot/{name}/collisions/spheres/Sphere_{index}", "radius": 0.5,
                 "center_body_m": [x, y, -0.031], "linear_body": linear, "after_linear_body": linear,
                 "after_center_body_m": wbc.source_sphere_center([x, y, -0.031], reference).tolist()}
                for index, x, y in ACTUAL_FOOT_SPHERES]}
        runtime["body_names"].append(name)
        runtime["body_mass_kg"].append(body["mass_kg"])
        runtime["body_com_pose_b_xyzw"].append([*body["com_position_b_m"], 0, 0, 0, 1])
        runtime["body_inertia_backend_kg_m2"].append((rotation @ np.diag(body["principal_inertia_kg_m2"]) @ rotation.T).reshape(9).tolist())
    placement = {"before_geometry": contact_geometry({n: [0, 0, 0, 0, 0, 0, 1] for n in reference["foot_bodies"]})}
    return evidence, placement, runtime


class ControllerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.contract, self.robot = wbc.load_contract(ROOT / "configs/g1_wbc_reference.json")
        self.asset = json.loads((ROOT / "configs/isaac_g1_asset.json").read_text())

    def test_deployment_contract_matches_body_order_caps_and_selects_policy(self):
        original = copy.deepcopy(self.contract)
        for case, file, role in (("stand", "isaac_g1_deployment.json", "balance"),
                                 ("forward", "isaac_g1_deployment_forward.json", "walk"),
                                 ("turn", "isaac_g1_deployment_turn.json", "walk")):
            profile, asset, _ = runner.load_plan(ROOT / "configs" / file)
            self.assertEqual(profile["initialization"]["minimum_clearance_m"], 0.0)
            contract, robot = runner.trial_contract(profile)
            self.assertEqual(contract["policy_role"], role)
            self.assertEqual(set(robot.limits), set(asset["controller_reference"]["joint_order"]))
            self.assertEqual(len(robot.limits), 29)
            self.assertFalse(asset["joint_groups"]["hands"])
            self.assertEqual(len(asset["fixed_hand_joints"]), 14)
            self.assertEqual((contract["upper_hold_kp"], contract["upper_hold_kd"], contract["upper_body_armature"]), (100, 0.5, 0.01))
            self.assertEqual(wbc.actuator_limits(contract, robot)["left_hip_roll_joint"]["effort"], 139)
            self.assertEqual(wbc.actuator_limits(contract, robot)["left_ankle_pitch_joint"]["velocity"], 37)
            before = self.robot.forward({n:0 for n in self.robot.limits})
            after = robot.forward({n:0 for n in robot.limits})
            for name in before: np.testing.assert_allclose(before[name], after[name], atol=1e-12, rtol=0)
            with mock.patch.object(wbc,"prepare_model",side_effect=AssertionError("model access")), \
                    mock.patch.object(runner,"run_stage",side_effect=AssertionError("GPU launch")), mock.patch("builtins.print"):
                self.assertEqual(runner.main(["--profile",str(ROOT/"configs"/file),"--stage","balance","--dry-run"]),0)
        self.assertEqual(self.contract, original)
        self.assertEqual(len(self.robot.limits), 43)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"invalid.json"
            raw=json.loads((ROOT/"configs/isaac_g1_deployment.json").read_text())
            for changes in ({"command":[1,0,0]}, {"functional_case":"turn"}, {"control_steps":4},
                            {"initialization":{**raw["initialization"],"minimum_clearance_m":0.001}}):
                path.write_text(json.dumps({**raw,**changes}))
                with self.assertRaises(ValueError):runner.load_plan(path)

    def test_functional_gates_reject_backward_motion_and_wrong_turn_direction(self):
        initial={"root_pose_xyzw":[0,0,0.793,0,0,0,1]}
        def trace(vx=0, yaw_rate=0):
            return [{"step":i,"sim_time_seconds":i*0.02,
                "root_pose_xyzw":[vx*i*0.02,0,0.793,0,0,np.sin(yaw_rate*i*0.01),np.cos(yaw_rate*i*0.01)],
                "root_velocity_world":[vx,0,0,0,0,yaw_rate],"tilt_rad":0,
                "action":[0]*15,"clipped_joint_targets":0} for i in range(1,251)]
        start={"left_ankle_roll_link":[0,0.12],"right_ankle_roll_link":[0,-0.12]}
        def feet(slide=0.0):
            return [{"feet":{n:{"pose_xyzw":[xy[0]-slide*min(k,9)/9,xy[1],0.035,0,0,0,1]} for n,xy in start.items()}} for k in range(1000)]
        def squat():
            # Pelvis moves back 0.10 m and settles while the feet stay planted.
            rows=trace()
            for r in rows: r["root_pose_xyzw"][0]=-0.10*min(r["step"],40)/40
            return rows
        for case, rows in (("stand",trace()),("stand",squat()),("forward",trace(0.2)),("turn",trace(yaw_rate=0.2))):
            self.assertEqual(wbc.functional_summary(case,initial,rows,start,feet())["status"],"passed_functional_evaluation")
        self.assertAlmostEqual(wbc.functional_summary("stand",initial,squat(),start,feet())["maximum_pelvis_displacement_m"],0.10)
        for case, rows, samples in (("stand",trace(-0.3),feet()),("forward",trace(-0.2),feet()),
                                    ("turn",trace(yaw_rate=-0.2),feet()),("stand",trace()[:100],feet()),
                                    ("stand",squat(),feet(0.03)),("stand",trace(),feet()[:999]),("stand",trace(),())):
            self.assertEqual(wbc.functional_summary(case,initial,rows,start,samples)["status"],"failed_functional_evaluation")
        result=wbc.functional_summary("stand",initial,trace(),start,feet(0.03))
        self.assertFalse(result["checks"]["foot_displacement_within_0_02m"])
        self.assertAlmostEqual(max(result["maximum_foot_displacement_m"].values()),0.03)

    def test_source_pose_accepts_actual_positive_submillimeter_gap_and_rejects_penetration(self):
        # Numeric fields supplied from run 20261030T082959.221472Z-wbc.
        root_pose=[0,0,0.7929999828338623,-1.0107841319495492e-7,2.849920832659336e-8,2.1509819703169342e-8,1]
        poses={
            "left_ankle_roll_link":[-2.386048436164856e-6,0.11850626021623611,0.03613623231649399,
                -9.543479251306053e-8,-1.6414904990824652e-8,1.4057685859825142e-7,0.9999999403953552],
            "right_ankle_roll_link":[-2.3953616619110107e-6,-0.1185065433382988,0.036136433482170105,
                -1.0301163655412893e-7,-1.440671471009125e-9,-7.222333842094031e-9,0.9999998807907104]}
        geometry=collision_fixture(poses)
        basis=np.array([[0.009999999776481742,-4.097141855155327e-9,2.8990830818410147e-12],
            [4.097141855328148e-9,0.009999999776481725,-5.960393839798507e-10],
            [-2.8988388760445647e-12,5.96039385167596e-10,0.009999999776482565]])
        for name,foot in geometry["feet"].items():
            foot["colliders"]=[]
            for index,x,y in (("05",0.12,0.026),("04",0.12,-0.026),("07",-0.02,-0.026),("06",-0.02,0.026)):
                transform=np.eye(4);transform[:3,:3]=basis;transform[:3,3]=[x,y,-0.031]
                foot["colliders"].append({"type":"Sphere","radius_m":0.5,
                    "collision_path":f"/World/envs/env_0/Robot/{name}/collisions/spheres/Sphere_{index}",
                    "body_from_shape":transform.tolist()})
        clearance=wbc.collision_clearance(geometry)
        self.assertAlmostEqual(clearance["minimum_clearance_m"],0.00013622680904792211,places=14)
        evidence={"status":"passed_initial_placement","mode":"source_pose","policy_height_command_m":0.74,
            "before_root_pose_xyzw":root_pose,"after_root_pose_xyzw":root_pose,"vertical_shift_m":0.0,
            "before_geometry":geometry,"after_geometry":copy.deepcopy(geometry),
            "before_clearance":clearance,"after_clearance":copy.deepcopy(clearance)}
        settings={"mode":"source_pose","minimum_clearance_m":0.001,"maximum_vertical_shift_m":0.1,"root_height_m":0.793}
        with self.assertRaisesRegex(ValueError,r"after=0\.000136.*required=0\.001"):
            wbc.placement_summary(evidence,settings,0.74,root_pose)
        settings["minimum_clearance_m"]=0.0
        result=wbc.placement_summary(evidence,settings,0.74,root_pose)
        self.assertEqual(result["vertical_shift_m"],0)
        self.assertEqual(result["spawn_height_m"],root_pose[2])
        self.assertEqual(result["policy_height_command_m"],0.74)
        invalid=copy.deepcopy(evidence)
        for scope in ("before_geometry","after_geometry"):
            for foot in invalid[scope]["feet"].values():
                for shape in foot["colliders"]:shape["body_from_shape"][2][3]-=0.001
        invalid["before_clearance"]=wbc.collision_clearance(invalid["before_geometry"])
        invalid["after_clearance"]=wbc.collision_clearance(invalid["after_geometry"])
        with self.assertRaisesRegex(ValueError,"Initial foot clearance"):
            wbc.placement_summary(invalid,settings,0.74,root_pose)

    def test_movement_prerequisite_and_clean_exit_cannot_hide_functional_failure(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as exit_error:
            runner.main(["--profile",str(ROOT/"configs/isaac_g1_deployment_forward.json"),"--stage","balance"])
        self.assertEqual(exit_error.exception.code,2)
        profile,asset,runtime=runner.load_plan(ROOT/"configs/isaac_g1_deployment.json")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner,"check_versions",return_value={}), \
                mock.patch.object(wbc,"prepare_model",return_value=Path("unused.onnx")), \
                mock.patch.object(runner,"balance_check",side_effect=SystemExit(0)), \
                mock.patch.object(runner,"validate_stage",side_effect=RuntimeError("Functional gate failed")):
            with self.assertRaisesRegex(RuntimeError,"Functional gate failed"):
                runner.worker("balance",profile,asset,runtime,ROOT/profile["controller_contract"],Path(tmp))
            evidence=json.loads((Path(tmp)/"balance-worker.json").read_text())
            self.assertEqual(evidence["error_type"],"RuntimeError")
            self.assertEqual(evidence["error"],"Functional gate failed")

    def test_fixed_hand_authoring_changes_only_joint_types_in_session_layer(self):
        names=wbc.load_deployment_reference()["fixed_hand_joints"]
        class Prim:
            def __init__(self,name):self.name=name;self.kind="Revolute";self.frame=[0,0,0,0,0,0,1]
            def GetName(self):return self.name
            def GetPath(self):return "/World/envs/env_0/Robot/"+self.name.split('_')[0]+"_hand/joints/"+self.name
            def GetTypeName(self):return "PhysicsFixedJoint" if self.kind=="Fixed" else "PhysicsRevoluteJoint"
            def IsA(self,kind):return self.kind==kind
            def IsInstanceProxy(self):return False
            def GetBody0Rel(self):return SimpleNamespace(GetTargets=lambda:["parent/"+self.name])
            def GetBody1Rel(self):return SimpleNamespace(GetTargets=lambda:["child/"+self.name])
        prims=[Prim(n) for n in names]
        stage=SimpleNamespace(target="root",Traverse=lambda:iter(prims),GetSessionLayer=lambda:"session")
        @contextmanager
        def edit(stage,layer):
            stage.target=layer
            try:yield
            finally:stage.target="root"
        def define(stage,path):
            self.assertEqual(stage.target,"session")
            prim=next(p for p in prims if p.GetPath()==path);prim.kind="Fixed"
            return SimpleNamespace(GetPrim=lambda:prim)
        pxr=SimpleNamespace(Usd=SimpleNamespace(EditContext=edit),UsdPhysics=SimpleNamespace(
            RevoluteJoint="Revolute",Joint=lambda prim:prim,FixedJoint=SimpleNamespace(Define=define)))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules,{"pxr":pxr}):
            runner.apply_fixed_hands(stage,SimpleNamespace(is_initialized=False),Path(tmp))
            record=json.loads((Path(tmp)/"deployment-joints.json").read_text())
            self.assertEqual(len(record["joints"]),14)
            self.assertEqual(stage.target,"root")
            self.assertTrue(all(p.kind=="Fixed" and p.frame==[0,0,0,0,0,0,1] for p in prims))
            with self.assertRaisesRegex(RuntimeError,"before reset"):
                runner.apply_fixed_hands(stage,SimpleNamespace(is_initialized=True),Path(tmp))

    def test_contact_reference_maps_quadrants_and_rejects_unapplied_contact(self):
        reference = wbc.load_contact_reference()
        self.assertEqual(reference["source"]["sha256"], wbc.load_deployment_reference()["sources"][2]["sha256"])
        targets = {tuple(wbc.source_sphere_center([x, y, -0.031], reference)) for _, x, y in ACTUAL_FOOT_SPHERES}
        self.assertEqual(targets, set(map(tuple, reference["foot_sphere_centers_m"])))
        with self.assertRaisesRegex(ValueError, "quadrant"):
            wbc.source_sphere_center([0.12, 0.0, -0.031], reference)
        evidence, placement, runtime = contact_fixture()
        result = wbc.contact_override_summary(evidence, placement, runtime)
        self.assertEqual(result["status"], "verified_source_contact")
        self.assertEqual(result["effective_foot_ground_friction"], 1.0)
        variants = []
        invalid = copy.deepcopy(evidence); invalid["feet"]["left_ankle_roll_link"]["spheres"][2]["after_center_body_m"] = [-0.02, -0.026, -0.031]; variants.append((invalid, placement, runtime))
        invalid = copy.deepcopy(evidence); invalid["feet"]["right_ankle_roll_link"]["spheres"][0]["radius"] = 1.0; variants.append((invalid, placement, runtime))
        invalid = copy.deepcopy(evidence); invalid["feet"]["right_ankle_roll_link"]["spheres"].pop(); variants.append((invalid, placement, runtime))
        invalid = copy.deepcopy(evidence); invalid["materials"]["ground"]["static_friction"] = 0.5; variants.append((invalid, placement, runtime))
        invalid = copy.deepcopy(evidence); invalid["materials"]["default"]["friction_combine_mode"] = None; variants.append((invalid, placement, runtime))
        invalid = copy.deepcopy(evidence); invalid["applied_before_reset"] = False; variants.append((invalid, placement, runtime))
        stale = copy.deepcopy(placement); stale["before_geometry"]["feet"]["left_ankle_roll_link"]["colliders"][0]["body_from_shape"][0][3] = -0.02; variants.append((evidence, stale, runtime))
        heavier = copy.deepcopy(runtime); heavier["body_mass_kg"][1] = 0.7; variants.append((evidence, placement, heavier))
        for args in variants:
            with self.subTest(args=args), self.assertRaises(ValueError): wbc.contact_override_summary(*args)
        with mock.patch.object(wbc.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: "wrong")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"): wbc.load_contact_reference()

    def test_contact_profiles_preserve_deployment_and_reject_unreviewed_contact(self):
        for base, file in (("isaac_g1_deployment.json", "isaac_g1_contact.json"),
                           ("isaac_g1_deployment_forward.json", "isaac_g1_contact_forward.json"),
                           ("isaac_g1_deployment_turn.json", "isaac_g1_contact_turn.json")):
            before, asset, runtime = runner.load_plan(ROOT / "configs" / base)
            after, contact_asset, contact_runtime = runner.load_plan(ROOT / "configs" / file)
            self.assertEqual((asset, runtime), (contact_asset, contact_runtime))
            self.assertEqual({k: v for k, v in before.items() if k != "scope"},
                             {k: v for k, v in after.items() if k not in ("scope", "contact_override")})
            self.assertEqual(runner.trial_contract(before)[0], runner.trial_contract(after)[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            raw = json.loads((ROOT / "configs/isaac_g1_contact.json").read_text())
            palm = json.loads((ROOT / "configs/isaac_g1_palm.json").read_text())
            for data in ({**raw, "contact_override": {"source": "guessed.json"}},
                         {**raw, "contact_override": {**raw["contact_override"], "friction": 2.0}},
                         {**palm, "contact_override": raw["contact_override"]}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError): runner.load_plan(path)
            stand = Path(tmp) / "stand"; stand.mkdir()
            (stand / "profile.json").write_text((ROOT / "configs/isaac_g1_deployment.json").read_text())
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as exit_error:
                runner.main(["--profile", str(ROOT / "configs/isaac_g1_contact_forward.json"), "--stage", "balance",
                             "--stand-run", str(stand)])
            self.assertEqual(exit_error.exception.code, 2)

    def test_arm_hold_profiles_change_only_the_arm_actuator_model(self):
        for base, file in (("isaac_g1_contact.json", "isaac_g1_arm_hold.json"),
                           ("isaac_g1_contact_forward.json", "isaac_g1_arm_hold_forward.json"),
                           ("isaac_g1_contact_turn.json", "isaac_g1_arm_hold_turn.json")):
            before, asset, runtime = runner.load_plan(ROOT / "configs" / base)
            after, arm_asset, arm_runtime = runner.load_plan(ROOT / "configs" / file)
            self.assertEqual((asset, runtime), (arm_asset, arm_runtime))
            self.assertEqual({k: v for k, v in before.items() if k != "scope"},
                             {k: v for k, v in after.items() if k not in ("scope", "arm_hold")})
            explicit, implicit = runner.trial_contract(before)[0], runner.trial_contract(after)[0]
            self.assertEqual({k: v for k, v in explicit.items() if k != "actuation_origin"},
                             {k: v for k, v in implicit.items() if k != "actuation_origin"})
            self.assertIn("implicit", implicit["actuation_origin"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            raw = json.loads((ROOT / "configs/isaac_g1_arm_hold.json").read_text())
            deployment = json.loads((ROOT / "configs/isaac_g1_deployment.json").read_text())
            for data in ({**raw, "arm_hold": "explicit"}, {**raw, "arm_hold": {"kp": 40}}, {**deployment, "arm_hold": "implicit"}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError): runner.load_plan(path)
            stand = Path(tmp) / "stand"; stand.mkdir()
            (stand / "profile.json").write_text((ROOT / "configs/isaac_g1_contact.json").read_text())
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as exit_error:
                runner.main(["--profile", str(ROOT / "configs/isaac_g1_arm_hold_forward.json"), "--stage", "balance",
                             "--stand-run", str(stand)])
            self.assertEqual(exit_error.exception.code, 2)

    def test_crouch_start_profiles_change_only_the_startup_placement(self):
        placement = json.loads((ROOT / "configs/isaac_g1_placement.json").read_text())["initialization"]
        for base, file in (("isaac_g1_arm_hold.json", "isaac_g1_crouch_start.json"),
                           ("isaac_g1_arm_hold_forward.json", "isaac_g1_crouch_start_forward.json"),
                           ("isaac_g1_arm_hold_turn.json", "isaac_g1_crouch_start_turn.json")):
            before, asset, runtime = runner.load_plan(ROOT / "configs" / base)
            after, crouch_asset, crouch_runtime = runner.load_plan(ROOT / "configs" / file)
            self.assertEqual((asset, runtime), (crouch_asset, crouch_runtime))
            self.assertEqual(after["initialization"], placement)
            self.assertEqual({k: v for k, v in before.items() if k not in ("scope", "initialization")},
                             {k: v for k, v in after.items() if k not in ("scope", "initialization")})
            self.assertEqual(runner.trial_contract(before)[0], runner.trial_contract(after)[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            raw = json.loads((ROOT / "configs/isaac_g1_crouch_start.json").read_text())
            contact = json.loads((ROOT / "configs/isaac_g1_contact.json").read_text())
            for data in ({**contact, "initialization": raw["initialization"]},
                         {**raw, "initialization": {**raw["initialization"], "minimum_clearance_m": 0}},
                         {**raw, "initialization": {**raw["initialization"], "maximum_vertical_shift_m": 0.2}}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError): runner.load_plan(path)
            stand = Path(tmp) / "stand"; stand.mkdir()
            (stand / "profile.json").write_text((ROOT / "configs/isaac_g1_arm_hold.json").read_text())
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as exit_error:
                runner.main(["--profile", str(ROOT / "configs/isaac_g1_crouch_start_forward.json"), "--stage", "balance",
                             "--stand-run", str(stand)])
            self.assertEqual(exit_error.exception.code, 2)

    def test_push_profile_object_checks_and_gate(self):
        base, asset, runtime = runner.load_plan(ROOT / "configs/isaac_g1_crouch_start_forward.json")
        push, push_asset, push_runtime = runner.load_plan(ROOT / "configs/isaac_g1_push.json")
        self.assertEqual((asset, runtime), (push_asset, push_runtime))
        self.assertEqual({k: v for k, v in base.items() if k not in ("scope", "functional_case")},
                         {k: v for k, v in push.items() if k not in ("scope", "functional_case", "push_object", "rpy_command")})
        self.assertEqual(runner.trial_contract(push)[0]["policy_role"], "walk")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            raw = json.loads((ROOT / "configs/isaac_g1_push.json").read_text())
            forward = json.loads((ROOT / "configs/isaac_g1_crouch_start_forward.json").read_text())
            for data in ({**raw, "push_object": {**raw["push_object"], "mass_kg": 500}},
                         {**raw, "push_object": {**raw["push_object"], "size_m": [0.7, 0.8]}},
                         {**raw, "push_object": {**raw["push_object"], "friction": 0}},
                         {**raw, "push_object": {**raw["push_object"], "anchor": "weld"}},
                         {**raw, "rpy_command": [0, 0.5, 0]}, {**raw, "rpy_command": [0.1, 0.2, 0]},
                         {**forward, "rpy_command": [0, 0.2, 0]},
                         {k: v for k, v in raw.items() if k != "push_object"},
                         {**forward, "push_object": raw["push_object"]}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError): runner.load_plan(path)
        settings = push["push_object"]
        evidence = {"size_m": [0.7, 0.8, 1.0], "hand_reach_m": 0.291, "runtime_mass_kg": 10.0,
                    "material": {"static_friction": 0.3, "dynamic_friction": 0.3, "friction_combine_mode": "multiply"},
                    "initial_pose_xyzw": [0.241+0.1+0.35, 0, 0.5, 0, 0, 0, 1]}
        self.assertAlmostEqual(wbc.push_object_summary(evidence, settings)["near_face_x_m"], 0.341)
        for key, value in (("runtime_mass_kg", 9.0), ("material", {**evidence["material"], "friction_combine_mode": "max"}),
                           ("initial_pose_xyzw", [1.0, 0, 0.5, 0, 0, 0, 1]),
                           ("initial_pose_xyzw", [0.691, 0, 0.5, *wbc.matrix_quat_xyzw(wbc.axis_rotation([0, 1, 0], 0.1))])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                wbc.push_object_summary({**evidence, key: value}, settings)
        initial = {"root_pose_xyzw": [0, 0, 0.785, 0, 0, 0, 1]}
        def trace(dx, tip=0.0, kick=0.0):
            quat = wbc.matrix_quat_xyzw(wbc.axis_rotation([0, 1, 0], tip)).tolist()
            return [{"step": i, "sim_time_seconds": i*0.02, "root_pose_xyzw": [0.0006*i, 0, 0.75, 0, 0, 0, 1],
                     "root_velocity_world": [0.03, 0, 0, 0, 0, 0], "tilt_rad": 0, "action": [0]*15, "clipped_joint_targets": 0,
                     "object_pose_xyzw": [0.691+dx*max(0, i-25)/225+kick, 0, 0.5, *quat]} for i in range(1, 251)]
        start = [0.691, 0, 0.5, 0, 0, 0, 1]
        self.assertEqual(wbc.functional_summary("push", initial, trace(0.3), object_start=start)["status"], "passed_functional_evaluation")
        for rows, failed in ((trace(0.05), "object_progress_at_least_0_20m"), (trace(0.3, tip=0.3), "object_tilt_within_10_degrees"),
                             (trace(0.0, kick=0.068), "object_still_through_0_20s")):
            result = wbc.functional_summary("push", initial, rows, object_start=start)
            self.assertFalse(result["checks"][failed])
        samples = [{"sim_time_seconds": k*0.005, "hand_box_force_world_n": {side: [-push, 0, 1.0] if k >= 300 else [0.0, 0, 0]
                    for side, push in (("left", 10.0), ("right", 8.0))}} for k in range(1, 1001)]
        force = wbc.push_force_summary(samples)
        self.assertEqual((force["maximum_push_n"], force["mean_push_during_contact_n"], force["first_contact_s"]), (18.0, 18.0, 1.5))
        with self.assertRaisesRegex(ValueError, "Missing"): wbc.push_force_summary([{"sim_time_seconds": 0.005}])
        history = wbc.ObservationHistory(self.contract)
        frame = history.observe(np.zeros(29), np.zeros(29), [0, 0, 0, 1], np.zeros(3), np.eye(3), (0.2, 0, 0), (0, 0.2, 0))[-86:]
        np.testing.assert_allclose(frame[:7], [0.2*self.contract["command_scale"][0], 0, 0, self.contract["height_command"], 0, 0.2, 0], atol=1e-6)

    def test_contact_authoring_deinstances_and_moves_only_sphere_centers(self):
        stage = SimpleNamespace(target="root", writes=[], GetSessionLayer=lambda: "session")
        self_test = self

        class Matrix:
            def __init__(self, value): self.value = np.asarray(value, dtype=float)
            def __mul__(self, other): return Matrix(self.value @ other.value)
            def __getitem__(self, index): return self.value[index]
            def GetInverse(self): return Matrix(np.linalg.inv(self.value))
            def ExtractTranslation(self): return self.value[3, :3]

        class Attribute:
            def __init__(self, prim): self.prim = prim
            def Set(self, value):
                self_test.assertEqual(stage.target, "session")
                self_test.assertFalse(self.prim.IsInstanceProxy())
                self.prim.translate = list(value); stage.writes.append(self.prim.path)
                return True

        class Prim:
            def __init__(self, path, parent=None, translate=None):
                self.path, self.parent, self.translate, self.instanceable = path, parent, translate, path.endswith("/collisions")
            def __bool__(self): return True
            def GetPath(self): return self.path
            def GetParent(self): return self.parent
            def GetTypeName(self): return "Sphere" if self.translate else "Xform"
            def HasAPI(self, api): return api == "Collision" if self.translate else api != "Collision" and self.parent.parent is None
            def IsInstance(self): return self.instanceable
            def IsInstanceProxy(self): return any(p.instanceable for p in self.ancestors())
            def ancestors(self):
                prim = self.parent
                while prim is not None: yield prim; prim = prim.parent
            def SetInstanceable(self, value):
                self_test.assertEqual(stage.target, "session"); self.instanceable = value; stage.writes.append(self.path)
            def GetAttribute(self, name): return Attribute(self) if name == "xformOp:translate" and self.translate else None
            def world(self):
                local = np.eye(4)
                if self.translate: local[:3, :3] *= SPHERE_SCALE; local[3, :3] = self.translate
                elif self.parent.parent is None: local[3, :3] = [0.0, 0.12, 0.036]
                return local @ (self.parent.world() if self.parent.parent is not None else np.eye(4))

        robot = Prim("/World/envs/env_0/Robot")
        prims = {}
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            body = Prim(f"{robot.path}/{name}", robot)
            collisions = Prim(body.path + "/collisions", body)
            group = Prim(collisions.path + "/spheres", collisions)
            prims.update({p.path: p for p in (body, collisions, group)})
            for index, x, y in ACTUAL_FOOT_SPHERES:
                sphere = Prim(f"{group.path}/Sphere_{index}", group, [x, y, -0.031])
                prims[sphere.path] = sphere
        stage.GetPrimAtPath = lambda path: prims[path]

        @contextmanager
        def edit(target, layer):
            target.target = layer
            try: yield
            finally: target.target = "root"

        foot = wbc.load_dynamics_reference()["bodies"]["left_ankle_roll_link"]
        mass = SimpleNamespace(GetMassAttr=lambda: SimpleNamespace(Get=lambda: foot["mass_kg"]),
                               GetCenterOfMassAttr=lambda: SimpleNamespace(Get=lambda: tuple(foot["com_position_b_m"])),
                               GetDiagonalInertiaAttr=lambda: SimpleNamespace(Get=lambda: tuple(foot["principal_inertia_kg_m2"])))
        pxr = SimpleNamespace(Gf=SimpleNamespace(Vec3d=lambda *v: tuple(v)),
            Usd=SimpleNamespace(EditContext=edit, TraverseInstanceProxies=lambda: None,
                                PrimRange=lambda body, predicate: [p for p in prims.values() if p.path.startswith(body.path)]),
            UsdGeom=SimpleNamespace(XformCache=lambda: SimpleNamespace(GetLocalToWorldTransform=lambda p: Matrix(p.world())),
                                    Sphere=lambda p: SimpleNamespace(GetRadiusAttr=lambda: SimpleNamespace(Get=lambda: 0.5))),
            UsdPhysics=SimpleNamespace(RigidBodyAPI="Rigid", MassAPI=lambda prim: mass, CollisionAPI="Collision"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"pxr": pxr}):
            report = runner.apply_source_contact(stage, SimpleNamespace(is_initialized=False), Path(tmp))
            self.assertEqual(stage.target, "root")
            self.assertEqual(json.loads((Path(tmp) / "contact-override.json").read_text())["status"], "authored_before_physics_reset")
            self.assertEqual(len(stage.writes), 10)  # Two de-instanced collider scopes and eight sphere centers.
            self.assertFalse(any(p.instanceable for p in prims.values()))
            evidence, placement, runtime = contact_fixture()
            report["materials"] = evidence["materials"]
            self.assertEqual(wbc.contact_override_summary(report, placement, runtime)["status"], "verified_source_contact")
            for sphere in report["feet"]["left_ankle_roll_link"]["spheres"]:
                np.testing.assert_allclose(sphere["linear_body"], np.eye(3)*SPHERE_SCALE, atol=1e-15, rtol=0)
            with self.assertRaisesRegex(RuntimeError, "precede physics"):
                runner.apply_source_contact(stage, SimpleNamespace(is_initialized=True), Path(tmp))
            self.assertEqual(len(stage.writes), 10)

    def test_xyzw_quarter_turns_and_half_turns(self):
        q = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        np.testing.assert_allclose(wbc.quat_xyzw_matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        for axis in np.eye(3):
            for angle in (0, np.pi/2, np.pi):
                matrix = wbc.axis_rotation(axis, angle)
                np.testing.assert_allclose(wbc.quat_xyzw_matrix(wbc.matrix_quat_xyzw(matrix)), matrix, atol=1e-12)
        with self.assertRaises(ValueError):
            wbc.quat_xyzw_matrix([0, 0, 0, 0])

    def test_observation_has_source_order_scales_and_zero_padded_history(self):
        history = wbc.ObservationHistory(self.contract)
        defaults = np.r_[self.contract["default_lower_angles"], np.zeros(14)]
        observation = history.observe(defaults, np.zeros(29), [0, 0, 0, 1], [0, 0, 0], np.eye(3), [1, 2, 3])
        golden = np.zeros(516, dtype=np.float32)
        golden[-86:-83] = [2, 4, 1.5]
        golden[-83] = 0.74
        golden[-74] = -1  # frame[12]: downward gravity in the upright base frame.
        np.testing.assert_allclose(observation, golden, atol=1e-7)
        action = np.linspace(-1, 1, 15)
        limits = np.tile([-3, 3], (15, 1))
        target, count = history.targets(action, limits)
        np.testing.assert_allclose(target, self.contract["default_lower_angles"] + 0.25*action)
        self.assertEqual(count, 0)
        second = history.observe(defaults, np.zeros(29), [0, 0, 0, 1], [0, 0, 0], np.eye(3))
        np.testing.assert_allclose(second[-15:], action)
        np.testing.assert_allclose(second[-172:-86], observation[-86:])
        history.reset()
        self.assertEqual(len(history.history), 0)
        self.assertTrue(np.all(history.last_action == 0))

    def test_world_gyro_and_gravity_rotate_into_controller_frame(self):
        history = wbc.ObservationHistory(self.contract)
        q = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        out = history.observe(np.r_[self.contract["default_lower_angles"], np.zeros(14)],
                              np.zeros(29), q, [0, 1, 0], np.eye(3))[-86:]
        np.testing.assert_allclose(out[7:10], [0, 0, -0.5], atol=1e-7)
        np.testing.assert_allclose(out[10:13], [0, -1, 0], atol=1e-7)

    def test_bad_action_or_observation_stops_before_targets(self):
        history = wbc.ObservationHistory(self.contract)
        with self.assertRaises(RuntimeError):
            history.targets(np.full(15, 6), np.tile([-3, 3], (15, 1)))
        with self.assertRaises(ValueError):
            history.targets(np.full(15, np.nan), np.tile([-3, 3], (15, 1)))
        narrow = np.tile([-0.01, 0.01], (15, 1))
        result, count = history.targets(np.zeros(15), narrow)
        self.assertGreater(count, 0)
        self.assertTrue(np.all(np.abs(result) <= 0.01))

    def test_frame_calibration_recovers_known_rotation_and_detects_wrong_pose(self):
        positions = {name: 0.0 for name in self.robot.limits}
        source = self.robot.forward(positions)
        rotation = wbc.axis_rotation([1, 0, 0], 0.3)
        # Root actor and source frame have a known nonidentity relationship.
        world = {}
        for name, transform in source.items():
            r = rotation @ transform[:3, :3] if name != "pelvis" else np.eye(3)
            world[name] = np.r_[rotation @ transform[:3, 3], wbc.matrix_quat_xyzw(r)]
        result = wbc.check_frames(self.robot, positions, world, 0.005, 0.02)
        np.testing.assert_allclose(result["source_to_asset_rotation"], rotation, atol=1e-7)
        wrong = copy.deepcopy(world)
        wrong["left_ankle_roll_link"][0] += 0.1
        with self.assertRaises(RuntimeError):
            wbc.check_frames(self.robot, positions, wrong, 0.005, 0.02, rotation)
        changed = {**positions, "left_knee_joint": 0.1}
        with self.assertRaises(RuntimeError):
            wbc.check_frames(self.robot, changed, world, 0.005, 0.02, rotation)

    def test_pinned_urdf_forward_kinematics_and_joint_set(self):
        expected = sum(self.asset["joint_groups"].values(), [])
        self.assertEqual(set(expected), set(self.robot.limits))
        poses = self.robot.forward({n: 0 for n in expected})
        np.testing.assert_allclose(poses["left_hip_pitch_link"][:3, 3], [0, 0.064452, -0.1027], atol=1e-6)

    def test_reference_revision_matches_deployment_geometry_and_preserves_trial_limits(self):
        self.assertEqual(self.contract["reference_urdf_source_path"],
                         "gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf")
        joints = {name: origin for name, _, _, _, origin, _ in self.robot.joints}
        np.testing.assert_allclose(joints["waist_roll_joint"][:3, 3], [-0.0039635, 0, 0.044], atol=1e-10)
        np.testing.assert_allclose(joints["waist_pitch_joint"][:3, 3], [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(joints["left_shoulder_pitch_joint"][:3, 3], [0.0039563, 0.10022, 0.24778], atol=1e-10)
        limits = wbc.actuator_limits(self.contract, self.robot)
        self.assertEqual(limits["left_hip_roll_joint"], {"effort": 88.0, "velocity": 32.0})
        self.assertEqual(limits["left_ankle_pitch_joint"], {"effort": 50.0, "velocity": 37.0})
        self.assertEqual(limits["waist_pitch_joint"], {"effort": 50.0, "velocity": 37.0})
        self.assertEqual(limits["left_hand_thumb_0_joint"]["velocity"], 6.857)
        self.assertEqual(self.robot.limits["left_hip_roll_joint"]["effort"], 139.0)
        for changes in ({"unknown_joint": {"effort": 1}},
                        {"waist_pitch_joint": {"effort": float("nan")}},
                        {"waist_pitch_joint": {"velocity": -1}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                wbc.actuator_limits({**self.contract, "actuator_limit_overrides": changes}, self.robot)
        for field in ("lower_body_armature", "upper_body_armature"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as name:
                path = Path(name) / "contract.json"
                path.write_text(json.dumps({**self.contract, field: 0.0}))
                with self.assertRaisesRegex(ValueError, "armature"):
                    wbc.load_contract(path)

    def test_usd_body_origins_are_converted_using_authored_joint_pivots(self):
        metadata = joint_frame_fixture(self.robot)
        positions = {name: 0.1 for name in self.robot.limits}
        local = self.robot.forward(positions)
        body = {n: np.r_[v[:3, 3], wbc.matrix_quat_xyzw(v[:3, :3])] for n, v in local.items()}
        for name, shift in (("waist_roll_joint", [0, 0, 0.009]),
                            ("waist_pitch_joint", [0, 0, -0.010])):
            frame = metadata[name]
            child = frame["child"]
            shift = np.asarray(shift)
            body[child][:3] += local[child][:3, :3] @ shift
            frame["position_in_child"] = (-shift).tolist()
        with self.assertRaises(wbc.FrameMismatchError) as caught:
            wbc.check_frames(self.robot, positions, body, 0.005, 0.02)
        self.assertEqual(caught.exception.report["worst_position_link"], "torso_link")
        anchors = wbc.joint_anchor_poses(self.robot, body, metadata)
        report = wbc.check_frames(self.robot, positions, anchors, 0.005, 0.02)
        self.assertEqual(report["links_checked"], 44)
        self.assertLess(report["maximum_position_error_m"], 1e-10)
        # A physically wrong pivot must still fail; no fitted offset hides it.
        metadata["waist_pitch_joint"]["position_in_child"][2] += 0.02
        with self.assertRaises(wbc.FrameMismatchError):
            wbc.check_frames(self.robot, positions, wbc.joint_anchor_poses(self.robot, body, metadata), 0.005, 0.02)
        metadata["waist_pitch_joint"]["axis_in_child"] = [1, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "axis differs"):
            wbc.joint_anchor_poses(self.robot, body, metadata)
        metadata.pop("waist_pitch_joint")
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            wbc.joint_anchor_poses(self.robot, body, metadata)

    def test_usd_reader_preserves_authored_pivots_and_rotates_joint_axis(self):
        fields = {}
        class Prim:
            def GetName(self): return "left_hip_pitch_joint"
            def GetPath(self): return "/World/envs/env_0/Robot/joints/left_hip_pitch_joint"
            def IsA(self, schema): return True
            def GetBody0Rel(self): return SimpleNamespace(GetTargets=lambda: ["/World/envs/env_0/Robot/pelvis"])
            def GetBody1Rel(self): return SimpleNamespace(GetTargets=lambda: ["/World/envs/env_0/Robot/left_hip_pitch_link"])
            def GetLocalPos1Attr(self): return SimpleNamespace(Get=lambda: [0, 0, 0.009])
            def GetAxisAttr(self): return SimpleNamespace(Get=lambda: "X")
            def GetLocalRot1Attr(self):
                return SimpleNamespace(Get=lambda: SimpleNamespace(
                    GetReal=lambda: np.sqrt(0.5), GetImaginary=lambda: [0, 0, np.sqrt(0.5)]))
        module = ModuleType("pxr")
        module.UsdPhysics = SimpleNamespace(RevoluteJoint=lambda prim: prim)
        with mock.patch.dict(sys.modules, {"pxr": module}):
            fields = runner.read_joint_frames(SimpleNamespace(Traverse=lambda: [Prim()]), self.robot)
        self.assertEqual(fields["left_hip_pitch_joint"]["position_in_child"], [0, 0, 0.009])
        np.testing.assert_allclose(fields["left_hip_pitch_joint"]["axis_in_child"], [0, 1, 0], atol=1e-12)

    def test_worker_error_details_survive_empty_exceptions_and_kit_shutdown(self):
        profile, asset, runtime = runner.load_plan(runner.DEFAULT_PROFILE)
        with tempfile.TemporaryDirectory() as name, \
                mock.patch.object(runner, "check_versions", return_value={}), \
                mock.patch.object(wbc, "prepare_model", return_value=Path("unused.onnx")):
            directory = Path(name)
            for error in (AssertionError(), SystemExit()):
                with self.subTest(error=type(error).__name__), \
                        mock.patch.object(runner, "balance_check", side_effect=error), \
                        self.assertRaises(type(error)):
                    runner.worker("balance", profile, asset, runtime,
                                  ROOT / profile["controller_contract"], directory)
                evidence = json.loads((directory / "balance-worker.json").read_text())
                self.assertTrue(evidence["error"])
                self.assertEqual(evidence["error_type"], type(error).__name__)
                self.assertIn(type(error).__name__, str(runner.stage_failure("balance", 1, directory)))
            # Preexisting failure cause must survive a clean Kit exit.
            (directory / "balance-worker.json").write_text(json.dumps({"status": "failed", "error": "Frame mismatch cause"}))
            with mock.patch.object(runner, "balance_check", side_effect=SystemExit()), self.assertRaises(SystemExit):
                runner.worker("balance", profile, asset, runtime, ROOT / profile["controller_contract"], directory)
            self.assertEqual(json.loads((directory / "balance-worker.json").read_text())["error"], "Frame mismatch cause")
            (directory / "balance-worker.json").write_text("invalid JSON")
            (directory / "balance-error.log").unlink()
            (directory / "balance.log").write_text("Original runtime failure\n")
            self.assertIn("Original runtime failure", str(runner.stage_failure("balance", 1, directory)))
            @contextmanager
            def kit_shutdown():
                try:
                    yield
                finally:
                    raise SystemExit(1)
            with self.assertRaises(SystemExit):
                with kit_shutdown(), runner.record_worker_failure(directory, {"status": "running"}):
                    raise RuntimeError("Original frame mismatch before Kit shutdown")
            evidence = json.loads((directory / "balance-worker.json").read_text())
            self.assertEqual(evidence["error"], "Original frame mismatch before Kit shutdown")
            self.assertIn("Original frame mismatch", str(runner.stage_failure("balance", 1, directory)))

    def test_weight_cache_rejects_lfs_pointers_and_corruption_without_download(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bad.onnx"
            path.write_bytes(b"version https://git-lfs.github.com/spec/v1")
            with self.assertRaises(ValueError):
                wbc.verify_model(path, self.contract["model"])
            contract = copy.deepcopy(self.contract)
            contract["model"]["cache_path"] = str(Path(name) / "missing.onnx")
            with mock.patch.object(wbc, "urlopen", side_effect=AssertionError("network called")):
                with self.assertRaises(FileNotFoundError):
                    wbc.prepare_model(contract)

    def test_balance_plan_rejects_weakened_guards(self):
        profile = json.loads(runner.DEFAULT_PROFILE.read_text())
        profile["maximum_tilt_rad"] = 2
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "profile.json"
            path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):
                runner.load_plan(path)

    def test_onnx_boundary_rejects_wrong_export_and_nonfinite_outputs(self):
        class Session:
            dimension = 516
            def __init__(self, *args, **kwargs): pass
            def get_inputs(self): return [SimpleNamespace(name="input", type="tensor(float)", shape=["batch", self.dimension])]
            def get_outputs(self): return [SimpleNamespace(name="output", type="tensor(float)", shape=["batch", 15])]
            def get_providers(self): return ["CPUExecutionProvider"]
            def run(self, outputs, inputs): return [np.full((1,15), np.nan, dtype=np.float32)]

        module = ModuleType("onnxruntime")
        module.SessionOptions = SimpleNamespace
        module.InferenceSession = Session
        contract = copy.deepcopy(self.contract)
        with tempfile.TemporaryDirectory() as name, mock.patch.dict(sys.modules, {"onnxruntime": module}), \
                mock.patch.object(wbc.importlib.metadata, "version", return_value="1.26.0"):
            path = Path(name) / "model.onnx"
            path.write_bytes(b"test-session-interface")
            contract["model"].update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            policy = wbc.BalancePolicy(path, contract)
            with self.assertRaises(ValueError):
                policy(np.zeros(516))
            Session.dimension = 123
            with self.assertRaises(RuntimeError):
                wbc.BalancePolicy(path, contract)

    def test_dry_run_is_cloud_and_weight_download_free(self):
        with mock.patch.object(wbc, "prepare_model", side_effect=AssertionError("model accessed")), \
                mock.patch.object(runner, "run_stage", side_effect=AssertionError("worker called")), \
                mock.patch("builtins.print"):
            self.assertEqual(runner.main(["--stage", "balance", "--prepare-model", "--dry-run"]), 0)
            self.assertEqual(runner.main(["--stage","dynamics","--profile",str(ROOT/"configs/isaac_g1_placement.json"),"--dry-run"]),0)

    def test_dynamics_comparison_maps_bodies_and_preserves_inertia_eigenvalues(self):
        source = wbc.load_dynamics_reference()
        with mock.patch.object(wbc,"DYNAMICS_REFERENCE_SHA256","0"*64),self.assertRaisesRegex(ValueError,"reference"):
            wbc.load_dynamics_reference()
        self.assertAlmostEqual(sum(b["mass_kg"]for b in source["bodies"].values()),34.3942352)
        measured = dynamics_fixture(self.contract,self.robot,self.asset,
            root=[1,2,3,*wbc.matrix_quat_xyzw(wbc.axis_rotation([0,0,1],0.7))])
        report=wbc.dynamics_comparison(measured,self.robot,source,self.contract,self.asset)
        self.assertAlmostEqual(report["totals"]["mass_difference_kg"],0.0)
        np.testing.assert_allclose(report["totals"]["com_difference_m"],[0,0,0],atol=1e-12)
        self.assertEqual(len(report["controls"]),43)
        for row in report["bodies"]:
            np.testing.assert_allclose(row["runtime_principal_inertia_kg_m2"],row["source_principal_inertia_kg_m2"],atol=1e-12)
        elbow=next(r for r in report["controls"]if r["joint"]=="left_elbow_joint")
        self.assertEqual((elbow["source_kp"],elbow["source_kd"]),(100.0,0.5))
        self.assertEqual((elbow["runtime_kp"],elbow["runtime_kd"]),(40.0,2.0))
        self.assertEqual(elbow["runtime_actuator"],"implicit_pd")
        hand=next(r for r in report["controls"]if r["joint"]=="left_hand_thumb_0_joint")
        self.assertFalse(hand["source_actuated"])
        for corrupt in (lambda m:m["body_mass_kg"].__setitem__(0,0),
                        lambda m:m["body_inertia_backend_kg_m2"][0].__setitem__(0,-1),
                        lambda m:m["body_com_pose_b_xyzw"][0].__setitem__(0,float("nan")),
                        lambda m:m["actuators"].pop("hands")):
            invalid=copy.deepcopy(measured);corrupt(invalid)
            with self.assertRaises(ValueError):wbc.dynamics_comparison(invalid,self.robot,source,self.contract,self.asset)

    def test_runtime_dynamics_reader_collects_actual_buffers_and_actuator_parameters(self):
        expected=dynamics_fixture(self.contract,self.robot,self.asset)
        class Tensor:
            def __init__(self,v):self.v=np.asarray(v)
            def __getitem__(self,i):return Tensor(self.v[i])
            def detach(self):return self
            def cpu(self):return self
            def numpy(self):return self.v
        mapping={"body_mass":"body_mass_kg","body_com_pose_b":"body_com_pose_b_xyzw",
                 "body_link_pose_w":"body_link_pose_w_xyzw","body_inertia":"body_inertia_backend_kg_m2",
                 "root_link_pose_w":"root_pose_xyzw","joint_pos":"joint_positions",
                 "joint_pos_target":"joint_position_targets","joint_vel_target":"joint_velocity_targets",
                 "joint_stiffness":"solver_joint_stiffness","joint_damping":"solver_joint_damping",
                 "joint_armature":"joint_armature","joint_effort_limits":"joint_effort_limits"}
        data=SimpleNamespace(**{k:SimpleNamespace(torch=Tensor([expected[v]]))for k,v in mapping.items()})
        class LabActuator(SimpleNamespace):pass
        class ImplicitConfig:pass
        class NativeActuator:pass  # Native owner has no Lab model attributes.
        models={k:LabActuator(joint_names=v["joint_names"],is_implicit_model=v["implicit"],
                             stiffness=Tensor([v["kp"]]),damping=Tensor([v["kd"]]))for k,v in expected["actuators"].items()}
        configs={k:SimpleNamespace(joint_names_expr=list(reversed(v["joint_names"])))for k,v in expected["actuators"].items()}
        robot=SimpleNamespace(data=data,body_names=expected["body_names"],joint_names=expected["joint_names"],actuators=models,
            cfg=SimpleNamespace(actuators=configs),find_joints=lambda expr: ([],[n for n in expected["joint_names"]if n in expr]))
        api=mock.Mock(side_effect=AssertionError("Lab model must not use native reads"))
        module=ModuleType("isaaclab.actuators");module.ActuatorBase=LabActuator;module.ImplicitActuatorCfg=ImplicitConfig
        native=ModuleType("isaaclab.actuators.newton");native.read_group_parameter=api
        with mock.patch.dict(sys.modules,{"isaaclab":ModuleType("isaaclab"),"isaaclab.actuators":module,"isaaclab.actuators.newton":native}):
            actual=runner.read_runtime_dynamics(robot,np.eye(3),"fixture.usd")
            for key in mapping.values():self.assertEqual(actual[key],expected[key])
            self.assertEqual(actual["actuators"]["arms"]["kp"],[40.0]*14)
            self.assertEqual(actual["actuators"]["lower_body"]["parameter_source"],"isaaclab_model")
            def native_values(collection,name,component,attr):
                self.assertIs(collection,models);self.assertEqual((name,component),("lower_body","controller"))
                return Tensor([expected["actuators"][name][attr]])
            api.side_effect=native_values
            for owner in (NativeActuator(),(NativeActuator(),NativeActuator())):
                models["lower_body"]=owner
                actual=runner.read_runtime_dynamics(robot,np.eye(3),"fixture.usd")
                lower=actual["actuators"]["lower_body"]
                self.assertEqual(lower["joint_names"],expected["actuators"]["lower_body"]["joint_names"])
                self.assertEqual(lower["kp"],expected["actuators"]["lower_body"]["kp"])
                self.assertEqual(lower["parameter_source"],"native_group_parameter_api")
                self.assertFalse(lower["implicit"])
                self.assertEqual(len(lower["owner_classes"]),len(owner)if isinstance(owner,tuple)else 1)
            self.assertEqual(api.call_count,4)
            with tempfile.TemporaryDirectory() as tmp:
                api.side_effect=RuntimeError("native query failure")
                with self.assertRaisesRegex(RuntimeError,"native query failure"):
                    runner.read_runtime_dynamics(robot,np.eye(3),"fixture.usd",directory=Path(tmp))
                partial=json.loads((Path(tmp)/"dynamics-runtime.json").read_text())
                self.assertEqual(partial["status"],"partial_runtime_dynamics")
                self.assertEqual(partial["body_mass_kg"],expected["body_mass_kg"])
                self.assertEqual(partial["collection_phase"],"actuator:lower_body")
                self.assertEqual(partial["collection_error"],"native query failure")
                with self.assertRaisesRegex(ValueError,"Incomplete runtime dynamics capture"):
                    wbc.dynamics_comparison(partial,self.robot,wbc.load_dynamics_reference(),self.contract,self.asset)

    def test_diagnostic_profile_preserves_the_working_control_configuration(self):
        baseline, asset, runtime = runner.load_plan(runner.DEFAULT_PROFILE)
        profile, diagnostic_asset, diagnostic_runtime = runner.load_plan(ROOT / "configs/isaac_g1_diagnostics.json")
        self.assertEqual(asset, diagnostic_asset)
        self.assertEqual(runtime, diagnostic_runtime)
        self.assertEqual({k:v for k,v in profile.items() if k not in ("base_profile", "diagnostics", "scope")},
                         {k:v for k,v in baseline.items() if k != "scope"})
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "profile.json"
            for change in ({"base_profile": "configs/isaac_smoke.json"}, {"control_steps": 500},
                           {"diagnostics": {"foot_body_names": [], "minimum_normal_force_n": 1}},
                           {"diagnostics": {**profile["diagnostics"], "minimum_normal_force_n": float("nan")}}):
                path.write_text(json.dumps({"base_profile": "configs/isaac_g1_balance.json",
                                            "diagnostics": profile["diagnostics"], **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    runner.load_plan(path)

    def test_contact_summary_checks_raw_force_timing_and_loaded_motion(self):
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        samples = []
        for i in range(1, 9):
            sample = {"physics_step": i, "control_step": (i-1)//4+1, "sim_time_seconds": i*0.005,
                      "lower_body_computed_effort": [2.0]*15, "lower_body_applied_effort": [1.0]*15, "feet": {}}
            for j,name in enumerate(feet):
                normal = 100.0 if ((i <= 4) == (j == 0)) else 0.0
                sample["feet"][name] = {"pose_xyzw": [0,0,0.02,0,0,0,1], "velocity_world": [0.2,0,0,0,0,0],
                    "normal_force_world_n": [0,0,normal], "friction_force_world_n": [-0.1*normal,0,0],
                    "total_force_world_n": [-0.1*normal,0,normal]}
            samples.append(sample)
        report = wbc.contact_summary(samples, feet, 1.0, 0.005, 4)
        self.assertTrue(report["force_evidence_present"])
        self.assertEqual(report["feet"][feet[0]]["loaded_fraction"], 0.5)
        self.assertAlmostEqual(report["feet"][feet[1]]["loaded_planar_speed_mean_m_s"], 0.2)
        self.assertAlmostEqual(report["feet"][feet[1]]["mean_friction_normal_ratio_when_loaded"], 0.1)
        self.assertEqual(report["effort_clipped_samples_by_lower_body_index"], [8]*15)
        for corrupt in (lambda rows: rows[2].update(physics_step=99),
                        lambda rows: rows[2]["feet"][feet[0]].update(total_force_world_n=[0,0,0]),
                        lambda rows: rows[2]["feet"][feet[0]].update(velocity_world=[float("nan")]*6)):
            invalid = copy.deepcopy(samples)
            corrupt(invalid)
            with self.assertRaises(ValueError):
                wbc.contact_summary(invalid, feet, 1.0, 0.005, 4)
        self.assertFalse(wbc.contact_summary([], feet, 1.0, 0.005, 4)["force_evidence_present"])

    def test_initial_placement_profile_preserves_control_and_rejects_unsafe_configuration(self):
        diagnostic, asset, runtime = runner.load_plan(ROOT / "configs/isaac_g1_diagnostics.json")
        trial, trial_asset, trial_runtime = runner.load_plan(ROOT / "configs/isaac_g1_placement.json")
        self.assertEqual(asset, trial_asset)
        self.assertEqual(runtime, trial_runtime)
        self.assertEqual({k:v for k,v in diagnostic.items() if k != "scope"},
                         {k:v for k,v in trial.items() if k not in ("scope", "initialization")})
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "profile.json"
            raw = json.loads((ROOT / "configs/isaac_g1_placement.json").read_text())
            for key,value in (("mode", "fixed_height"), ("minimum_clearance_m", 0),
                              ("minimum_clearance_m", float("nan")), ("maximum_vertical_shift_m", 0.11),
                              ("maximum_vertical_shift_m", True), ("unknown", 1)):
                data = copy.deepcopy(raw)
                data["initialization"][key] = value
                path.write_text(json.dumps(data))
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    runner.load_plan(path)
            del raw["diagnostics"]
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                runner.load_plan(path)

    def test_palm_profile_preserves_placement_and_rejects_other_interventions(self):
        base, asset, runtime = runner.load_plan(ROOT / "configs/isaac_g1_placement.json")
        trial, trial_asset, trial_runtime = runner.load_plan(ROOT / "configs/isaac_g1_palm.json")
        self.assertEqual((asset, runtime), (trial_asset, trial_runtime))
        self.assertEqual({k: v for k, v in base.items() if k != "scope"},
                         {k: v for k, v in trial.items() if k not in ("scope", "inertial_override")})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            for override in (None, {"body": "right_hand_palm_link", "source": "pinned_urdf"},
                             {"body": "left_hand_palm_link", "source": "guessed"},
                             {**trial["inertial_override"], "mass_kg": 1}):
                path.write_text(json.dumps({**trial, "inertial_override": override}))
                with self.subTest(override=override), self.assertRaises(ValueError): runner.load_plan(path)
        with mock.patch.object(wbc, "prepare_model", side_effect=AssertionError("model access")), \
                mock.patch.object(runner, "run_stage", side_effect=AssertionError("GPU launch")), mock.patch("builtins.print"):
            self.assertEqual(runner.main(["--profile", str(ROOT / "configs/isaac_g1_palm.json"), "--stage", "dynamics", "--dry-run"]), 0)

    def test_palm_tensor_preserves_mirrored_offdiagonals_and_proper_axes(self):
        reference = wbc.palm_inertial_reference(self.contract)
        self.assertEqual(reference["mass_kg"], 0.37283854)
        self.assertLess(reference["com_b_m"][1], 0)
        tensor = np.array(reference["inertia_b_kg_m2"])
        self.assertAlmostEqual(tensor[0, 1], -0.00001595519465, places=14)
        self.assertAlmostEqual(tensor[1, 2], -0.00000042279435, places=14)
        axes = wbc.quat_xyzw_matrix(reference["principal_axes_xyzw"])
        self.assertAlmostEqual(np.linalg.det(axes), 1)
        np.testing.assert_allclose(axes @ np.diag(reference["principal_inertia_kg_m2"]) @ axes.T, tensor, atol=1e-12, rtol=0)
        with mock.patch.object(wbc.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: "wrong")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"): wbc.palm_inertial_reference(self.contract)

    def test_palm_verification_rejects_missing_mass_wrong_axes_and_mirrored_com(self):
        evidence = palm_fixture(self.contract)
        expected = evidence["reference"]
        self.assertEqual(wbc.palm_override_summary(evidence, expected)["status"], "verified_palm_inertial_override")
        equivalent = copy.deepcopy(evidence)
        axes = wbc.quat_xyzw_matrix(expected["principal_axes_xyzw"]) @ np.diag([-1, -1, 1])
        equivalent["reference"]["principal_axes_xyzw"] = wbc.matrix_quat_xyzw(axes).tolist()
        equivalent["reference"]["principal_inertia_kg_m2"][0] += 1e-16
        self.assertEqual(wbc.palm_override_summary(equivalent, expected)["status"], "verified_palm_inertial_override")
        variants = []
        invalid = copy.deepcopy(evidence); invalid["runtime"]["left"]["mass_kg"] = 0.214904502; variants.append(invalid)
        invalid = copy.deepcopy(evidence); invalid["runtime"]["left"]["com_b_m"][1] *= -1; variants.append(invalid)
        invalid = copy.deepcopy(evidence); invalid["runtime"]["left"]["inertia_b_kg_m2"] = evidence["runtime"]["right"]["inertia_b_kg_m2"]; variants.append(invalid)
        invalid = copy.deepcopy(evidence); invalid["runtime"]["left"]["mass_kg"] = float("nan"); variants.append(invalid)
        invalid = copy.deepcopy(evidence); invalid["before_usd"]["right"]["mass_kg"] = 1; variants.append(invalid)
        invalid = copy.deepcopy(evidence); invalid["applied_before_reset"] = False; variants.append(invalid)
        for invalid in variants:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): wbc.palm_override_summary(invalid, expected)
        # The wrong mirrored tensor has the same spectrum: eigenvalues alone cannot validate this override.
        np.testing.assert_allclose(np.linalg.eigvalsh(evidence["runtime"]["left"]["inertia_b_kg_m2"]),
                                   np.linalg.eigvalsh(evidence["runtime"]["right"]["inertia_b_kg_m2"]))

    def test_palm_override_authors_session_only_and_rejects_late_or_wrong_unit_edits(self):
        evidence = palm_fixture(self.contract)
        stage = SimpleNamespace(target="root", writes=[], units=1.0)
        values = {"left": {}, "right": copy.deepcopy(evidence["after_usd"]["right"])}

        class Quat:
            def __init__(self, real, imaginary): self.real, self.imaginary = float(np.float32(real)), imaginary
            def GetReal(self): return self.real
            def GetImaginary(self): return self.imaginary

        class Attribute:
            def __init__(self, side, key): self.side, self.key = side, key
            def HasAuthoredValueOpinion(self): return self.key in values[self.side]
            def Get(self):
                value = values[self.side][self.key]
                return Quat(value[3], value[:3]) if self.key == "principal_axes_xyzw" else value
            def Set(self, value):
                self_test.assertEqual(stage.target, "session")
                self_test.assertEqual(self.side, "left")
                if isinstance(value, Quat): value = [*value.GetImaginary(), value.GetReal()]
                values[self.side][self.key] = copy.deepcopy(value)
                stage.writes.append(self.key)
                return True

        class Prim:
            def __init__(self, side): self.side = side
            def HasAPI(self, api): return api == "Rigid" or values[self.side].get("has_mass_api", False)
            def IsInstanceProxy(self): return False

        class MassAPI:
            def __init__(self, prim): self.side = prim.side
            @staticmethod
            def Apply(prim):
                self_test.assertEqual(stage.target, "session")
                values[prim.side]["has_mass_api"] = True
                return MassAPI(prim)
            def GetMassAttr(self): return Attribute(self.side, "mass_kg")
            def GetCenterOfMassAttr(self): return Attribute(self.side, "com_b_m")
            def GetDiagonalInertiaAttr(self): return Attribute(self.side, "principal_inertia_kg_m2")
            def GetPrincipalAxesAttr(self): return Attribute(self.side, "principal_axes_xyzw")
            CreateMassAttr = GetMassAttr
            CreateCenterOfMassAttr = GetCenterOfMassAttr
            CreateDiagonalInertiaAttr = GetDiagonalInertiaAttr
            CreatePrincipalAxesAttr = GetPrincipalAxesAttr

        @contextmanager
        def edit(stage, layer):
            old = stage.target; stage.target = layer
            try: yield
            finally: stage.target = old

        self_test = self
        stage.GetPrimAtPath = lambda path: Prim("left" if "/left_hand/" in path else "right")
        stage.GetSessionLayer = lambda: "session"
        pxr = SimpleNamespace(Gf=SimpleNamespace(Vec3f=lambda *v: np.array(v, dtype=np.float32).tolist(), Quatf=Quat),
            Usd=SimpleNamespace(EditContext=edit), UsdGeom=SimpleNamespace(GetStageMetersPerUnit=lambda s:s.units),
            UsdPhysics=SimpleNamespace(MassAPI=MassAPI, RigidBodyAPI="Rigid", GetStageKilogramsPerUnit=lambda s:1.0))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"pxr": pxr}):
            directory = Path(tmp)
            report = runner.apply_palm_override(stage, SimpleNamespace(is_initialized=False), self.contract, directory)
            report["runtime"] = evidence["runtime"]
            wbc.palm_override_summary(report, evidence["reference"])
            self.assertEqual(stage.target, "root")
            self.assertEqual(len(stage.writes), 4)
            self.assertFalse(report["before_usd"]["left"]["has_mass_api"])
            self.assertEqual(report["before_usd"]["right"], report["after_usd"]["right"])
            self.assertEqual(json.loads((directory / "palm-inertial-override.json").read_text())["status"], "authored_before_physics_reset")
            for initialized, units in ((True, 1.0), (False, 0.01)):
                stage.units = units
                with self.assertRaises(RuntimeError):
                    runner.apply_palm_override(stage, SimpleNamespace(is_initialized=initialized), self.contract, directory)
                self.assertEqual(len(stage.writes), 4)

    def test_palm_runtime_reader_maps_names_and_static_replay_checks_shared_raw_values(self):
        evidence = palm_fixture(self.contract)
        sides = ["right", "left"]

        class Tensor:
            def __init__(self, value): self.value = np.array(value, dtype=np.float32)
            def __getitem__(self, key): return Tensor(self.value[key])
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        masses = [evidence["runtime"][s]["mass_kg"] for s in sides]
        poses = [[*evidence["runtime"][s]["com_b_m"], *evidence["after_usd"][s]["principal_axes_xyzw"]] for s in sides]
        tensors = [np.array(evidence["runtime"][s]["inertia_b_kg_m2"]).reshape(9, order="F").tolist() for s in sides]
        robot = SimpleNamespace(body_names=[f"{s}_hand_palm_link" for s in sides], data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=Tensor([masses])), body_com_pose_b=Tensor([poses]), body_inertia=Tensor([tensors])))
        evidence["runtime"] = runner.read_palm_runtime(robot)
        wbc.palm_override_summary(evidence, evidence["reference"])
        measured = {"body_names": robot.body_names, "body_mass_kg": masses,
                    "body_com_pose_b_xyzw": poses, "body_inertia_backend_kg_m2": tensors}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "palm-inertial-override.json"
            path.write_text(json.dumps(evidence))
            profile = {"inertial_override": {"body": "left_hand_palm_link", "source": "pinned_urdf"}}
            result = runner.analyze_palm_override(directory, profile, self.contract, measured)
            self.assertEqual(result["status"], "verified_palm_inertial_override")
            measured["body_mass_kg"][1] -= 0.01
            with self.assertRaisesRegex(ValueError, "differs from the static dynamics snapshot"):
                runner.analyze_palm_override(directory, profile, self.contract, measured)
            evidence["status"] = "authored_before_physics_reset"
            path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "Incomplete palm"):
                runner.analyze_palm_override(directory, profile, self.contract)

    def test_clearance_uses_full_sphere_surface_and_rejects_invalid_or_changed_geometry(self):
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        geometry = collision_fixture({n: [0,0,0.02,0,0,0,1] for n in feet})
        before = wbc.collision_clearance(geometry)
        self.assertAlmostEqual(before["minimum_clearance_m"], -0.015)
        settings = {"mode": "collision_clearance", "minimum_clearance_m": 0.001, "maximum_vertical_shift_m": 0.1}
        self.assertAlmostEqual(wbc.placement_shift(before["minimum_clearance_m"], settings), 0.016)
        self.assertEqual(wbc.placement_shift(0.01, settings), 0.0)
        with self.assertRaisesRegex(RuntimeError, "bound"):
            wbc.placement_shift(-0.2, settings)
        # A tilted foot changes a sphere's world center even though its radius is unchanged.
        tilted = copy.deepcopy(geometry)
        tilted["feet"][feet[0]]["body_pose_xyzw"][3:] = wbc.matrix_quat_xyzw(wbc.axis_rotation([0,1,0], np.pi/2)).tolist()
        self.assertAlmostEqual(wbc.collision_clearance(tilted)["minimum_clearance_m"], -0.085)
        for corrupt in (lambda g: g.update(meters_per_unit=0.01),
                        lambda g: g["ground"].update(axis="X"),
                        lambda g: g["feet"][feet[0]].update(colliders=[]),
                        lambda g: g["feet"][feet[0]]["colliders"][0].update(radius_m=float("nan")),
                        lambda g: g["feet"][feet[0]]["colliders"][0]["body_from_shape"][0].__setitem__(0, 2)):
            bad = copy.deepcopy(geometry)
            corrupt(bad)
            with self.assertRaises(ValueError):
                wbc.collision_clearance(bad)
        # Revalidation catches a falsely reported clearance and an undeclared horizontal move.
        before_root = [0,0,0.74,0,0,0,1]
        after_root = [0,0,0.756,0,0,0,1]
        after_geometry = copy.deepcopy(geometry)
        for foot in after_geometry["feet"].values(): foot["body_pose_xyzw"][2] += 0.016
        evidence = {"status": "passed_initial_placement", "mode": "collision_clearance", "policy_height_command_m": 0.74,
            "vertical_shift_m": 0.016, "before_root_pose_xyzw": before_root, "after_root_pose_xyzw": after_root,
            "before_geometry": geometry, "after_geometry": after_geometry, "before_clearance": before,
            "after_clearance": wbc.collision_clearance(after_geometry)}
        self.assertAlmostEqual(wbc.placement_summary(evidence, settings, 0.74, after_root)["minimum_clearance_after_m"], 0.001)
        rounded = copy.deepcopy(evidence)
        rounded["after_clearance"]["minimum_clearance_m"] += 1e-12
        wbc.placement_summary(rounded, settings, 0.74, after_root)
        for corrupt in (lambda e: e["after_clearance"].update(minimum_clearance_m=0.1),
                        lambda e: e["after_root_pose_xyzw"].__setitem__(0, 0.1),
                        lambda e: e["after_geometry"]["feet"][feet[0]]["colliders"][0].update(radius_m=0.006)):
            bad = copy.deepcopy(evidence)
            corrupt(bad)
            with self.assertRaises(ValueError): wbc.placement_summary(bad, settings, 0.74, after_root)

    def test_scaled_unit_spheres_match_direct_radii_and_world_size_guard_remains_enforced(self):
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        direct = collision_fixture({n:[0,0,0.05,0,0,0,1] for n in feet})
        expected = wbc.collision_clearance(direct)
        for radius,scale in ((1.0,0.005), (0.5,0.01), (0.0025,2.0)):
            scaled = copy.deepcopy(direct)
            for foot in scaled["feet"].values():
                shape = foot["colliders"][0]
                shape["radius_m"] = radius
                matrix = np.asarray(shape["body_from_shape"])
                matrix[:3,:3] *= scale
                shape["body_from_shape"] = matrix.tolist()
            actual = wbc.collision_clearance(scaled)
            with self.subTest(radius=radius,scale=scale):
                self.assertAlmostEqual(actual["minimum_clearance_m"], expected["minimum_clearance_m"])
                for name in feet:
                    self.assertAlmostEqual(actual["feet"][name]["colliders"][0]["radius_world_m"], 0.005)
        for radius,scale in ((0.005,100.0), (1.0,1.0), (0.0,1.0), (-1.0,0.005)):
            invalid = copy.deepcopy(direct)
            shape = invalid["feet"][feet[0]]["colliders"][0]
            shape["radius_m"] = radius
            matrix = np.asarray(shape["body_from_shape"])
            matrix[:3,:3] *= scale
            shape["body_from_shape"] = matrix.tolist()
            with self.subTest(radius=radius,scale=scale), self.assertRaisesRegex(ValueError, feet[0]):
                wbc.collision_clearance(invalid)

    def test_usd_clearance_reader_uses_body_relative_geometry_with_stale_world_transforms(self):
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        class Matrix:
            def __init__(self, a): self.a = np.array(a)
            def Transform(self, p): return (self.a @ np.r_[p,1])[:3]
            def TransformDir(self, p): return self.a[:3,:3] @ p
            def GetInverse(self): return Matrix(np.linalg.inv(self.a))
            def __mul__(self, other): return Matrix(other.a @ self.a)  # Gf row-vector composition.
        class Prim:
            def __init__(self, path, kind="Xform", enabled=True): self.path,self.kind,self.enabled=path,kind,enabled
            def IsValid(self): return True
            def HasAPI(self, api): return self.kind != "Xform"
            def GetTypeName(self): return self.kind
            def GetPath(self): return self.path
            def GetAttribute(self, name): return SimpleNamespace(Get=lambda: "Z")
        roots = {"/World/Ground": Prim("/World/Ground")}
        shapes = {"/World/Ground": [Prim("/World/Ground/Plane", "Plane")]}
        matrices = {"/World/Ground/Plane": np.eye(4)}
        actual = {n: [0,0,0.05,0,0,0,1] for n in feet}
        for n in feet:
            path = f"/World/envs/env_0/Robot/{n}"
            roots[path] = Prim(path)
            shapes[path] = [Prim(path+"/collisions/Sphere", "Sphere"), Prim(path+"/disabled", "Cube", False)]
            stale = wbc.pose_matrix([100,50,-10,*wbc.matrix_quat_xyzw(wbc.axis_rotation([1,0,0], 0.6))])
            local = np.eye(4); local[2,3] = -0.03
            matrices[path] = stale
            matrices[path+"/collisions/Sphere"] = stale @ local
        module = ModuleType("pxr")
        module.Gf = SimpleNamespace(Vec3d=lambda *v: np.array(v))
        module.Usd = SimpleNamespace(PrimRange=lambda root, flags: [root,*shapes[root.path]], TraverseInstanceProxies=lambda: None)
        module.UsdPhysics = SimpleNamespace(CollisionAPI=lambda p: SimpleNamespace(GetCollisionEnabledAttr=lambda: SimpleNamespace(Get=lambda:p.enabled)))
        radius = [0.005]
        module.UsdGeom = SimpleNamespace(GetStageUpAxis=lambda stage: "Z", GetStageMetersPerUnit=lambda stage: 1.0,
            XformCache=lambda: SimpleNamespace(GetLocalToWorldTransform=lambda p:Matrix(matrices[p.path])),
            Sphere=lambda p:SimpleNamespace(GetRadiusAttr=lambda:SimpleNamespace(Get=lambda:radius[0])))
        with mock.patch.dict(sys.modules, {"pxr":module}):
            geometry = runner.read_collision_geometry(SimpleNamespace(GetPrimAtPath=lambda p:roots[p]), actual)
            self.assertAlmostEqual(wbc.collision_clearance(geometry)["minimum_clearance_m"], 0.015)
            self.assertEqual(len(geometry["feet"][feet[0]]["colliders"]), 1)
            radius[0] = 1.0
            for name in feet:
                path = f"/World/envs/env_0/Robot/{name}"
                local = np.eye(4)*0.005; local[3,3]=1.0; local[2,3]=-0.03
                matrices[path+"/collisions/Sphere"] = matrices[path] @ local
            scaled = runner.read_collision_geometry(SimpleNamespace(GetPrimAtPath=lambda p:roots[p]), actual)
            self.assertAlmostEqual(wbc.collision_clearance(scaled)["minimum_clearance_m"], 0.015)
            shapes[f"/World/envs/env_0/Robot/{feet[0]}"][0].kind = "Mesh"
            with self.assertRaisesRegex(RuntimeError, "Unsupported initial foot collider"):
                runner.read_collision_geometry(SimpleNamespace(GetPrimAtPath=lambda p:roots[p]), actual)

    def test_material_reader_records_unresolved_bindings_without_inventing_friction(self):
        class Prim:
            def __init__(self, path): self.path = path
            def IsValid(self): return True
            def HasAPI(self, schema): return schema == "collision" or self.path.endswith("material")
            def GetPath(self): return self.path
            def GetAttribute(self, name):
                return SimpleNamespace(Get=lambda: {"physics:staticFriction": 0.5, "physics:dynamicFriction": 0.4}.get(name))
        class Binding:
            def __init__(self, prim): self.prim = prim
            def ComputeBoundMaterial(self, **kwargs):
                if self.prim.path == "/World/Ground":
                    return SimpleNamespace(GetPrim=lambda: Prim("/World/Ground/material")), None
                return None, None
        module = ModuleType("pxr")
        module.Usd = SimpleNamespace(PrimRange=lambda root, flags: [root], TraverseInstanceProxies=lambda: None)
        module.UsdPhysics = SimpleNamespace(CollisionAPI="collision", MaterialAPI="material")
        module.UsdShade = SimpleNamespace(MaterialBindingAPI=Binding)
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        with mock.patch.dict(sys.modules, {"pxr": module}):
            result = runner.read_contact_materials(SimpleNamespace(GetPrimAtPath=lambda path: Prim(path)), feet)
        self.assertEqual(result["scopes"]["ground"][0]["physics:dynamicFriction"], 0.4)
        self.assertFalse(result["scopes"][feet[0]][0]["physics_material_found"])
        self.assertIsNone(result["scopes"][feet[0]][0]["material_path"])


class ControllerLaunchTests(unittest.TestCase):
    def test_actual_balance_flow_initializes_frames_gains_and_contiguous_steps(self):
        profile, asset, runtime = runner.load_plan(runner.DEFAULT_PROFILE)
        profile = {**profile, "control_steps": 4,
                   "camera": {"width": 64, "height": 64, "capture_every_control_steps": 2}}
        contract, reference = wbc.load_contract(ROOT / profile["controller_contract"])
        names = list(reversed(sum(asset["joint_groups"].values(), [])))

        class Tensor:
            is_cuda = True
            def __init__(self, value): self.value = np.asarray(value)
            def __getitem__(self, index): return Tensor(self.value[index])
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        class Data:
            @property
            def joint_pos(self): return Tensor([robot.q])
            @property
            def joint_vel(self): return Tensor([robot.dq])
            @property
            def default_joint_pos(self): return Tensor([np.zeros(len(names))])
            @property
            def joint_pos_limits(self): return Tensor([limits])
            @property
            def joint_armature(self):
                return Tensor([[contract["lower_body_armature"] if n in asset["joint_groups"]["lower_body"]
                                else contract["upper_body_armature"] for n in names]])
            @property
            def root_link_pose_w(self): return Tensor([robot.pose])
            @property
            def root_link_vel_w(self): return Tensor([robot.velocity])
            @property
            def body_link_pose_w(self):
                root = wbc.pose_matrix(robot.pose)
                local = reference.forward(dict(zip(names, robot.q, strict=True)))
                poses = [root @ local[n] for n in robot.body_names]
                return Tensor([[np.r_[p[:3, 3], wbc.matrix_quat_xyzw(p[:3, :3])] for p in poses]])
            @property
            def body_link_vel_w(self):
                return Tensor([[[0.2,0,0,0,0,0] for n in robot.body_names]])

        class Robot:
            joint_names = names
            body_names = list(reference.forward({n: 0 for n in names}))
            is_fixed_base = False
            is_initialized = False
            q = np.zeros(43)
            dq = np.zeros(43)
            pose = np.r_[0, 0, 0.75, wbc.matrix_quat_xyzw(wbc.axis_rotation([1,0,0], 0.7))]
            velocity = np.zeros(6)
            data = Data()
            def write_joint_position_to_sim_index(self, *, position): self.q = position.value[0].copy()
            def write_joint_velocity_to_sim_index(self, *, velocity): self.dq = velocity.value[0].copy()
            def write_root_pose_to_sim_index(self, *, root_pose): self.pose = root_pose.value[0].copy()
            def write_root_velocity_to_sim_index(self, *, root_velocity): self.velocity = root_velocity.value[0].copy()
            def set_joint_position_target_index(self, *, target): self.targets = target.value[0].copy()
            def set_joint_velocity_target_index(self, *, target): pass

        robot = Robot()
        robot.actuators = SimpleNamespace(computed_effort=Tensor([np.zeros(len(names))]), applied_effort=Tensor([np.zeros(len(names))]))
        limits = [[reference.limits[n]["lower"], reference.limits[n]["upper"]] for n in names]
        rgb = np.repeat(((np.indices((64, 64)).sum(axis=0) % 2)*255)[..., None], 3, axis=-1).astype(np.uint8)

        class Camera:
            data = SimpleNamespace(output={"rgb": Tensor(rgb[None])})
            def set_world_poses_from_view(self, *args): pass
            def update(self, *args, **kwargs): pass

        class Scene(dict):
            def reset(self): pass
            def update(self, dt): pass
            def write_data_to_sim(self): pass

        class Config:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
            def to_dict(self): return self.__dict__

        class ExplicitConfig(Config):
            pass

        class ImplicitConfig(Config):
            pass

        class Simulation:
            physics_manager = type("PhysxManager", (), {"__module__": "isaaclab_physx.physics"})
            stage = object()
            steps = 0
            def __init__(self, config): self.config = config
            def reset(self): robot.is_initialized = True
            def forward(self): pass
            def render(self): pass
            def step(self, *, render): Simulation.steps += 1

        class Policy:
            metadata = {"input": ["batch",516], "output": ["batch",15]}
            output = np.zeros(15)
            observations = []
            simulation_steps_at_calls = []
            def __init__(self, *args): pass
            def __call__(self, observation):
                self.assert_shape = observation.shape == (516,)
                if not self.assert_shape: raise AssertionError("Bad observation passed to policy")
                self.observations.append(observation.copy())
                self.simulation_steps_at_calls.append(Simulation.steps)
                return self.output

        class Image:
            @staticmethod
            def fromarray(array): return Image()
            def save(self, path): Path(path).write_bytes(b"CPU interface image double")

        candidate = Config(spawn=Config(fix_root_link=False, rigid_props=Config(disable_gravity=False),usd_path="fixture.usd"))
        instantiated = []

        def instantiate(config):
            instantiated.append(config)
            robot.is_initialized = False
            scene = Scene(robot=robot, camera=Camera())
            if hasattr(config, "feet_contact"):
                scene["feet_contact"] = SimpleNamespace(
                    body_names=["right_ankle_roll_link", "left_ankle_roll_link"],
                    data=SimpleNamespace(net_normal_forces_w=Tensor([[[0,0,200],[0,0,100]]]),
                        net_friction_forces_w=Tensor([[[-20,0,0],[-10,0,0]]]),
                        net_forces_w=Tensor([[[-20,0,200],[-10,0,100]]])))
            if hasattr(config, "furniture"):
                scene["furniture"] = SimpleNamespace(data=FurnitureData(config.furniture))
            if hasattr(config, "hand_contact"):
                hands = [f"{side}_hand_{part}_link" for side in ("left", "right") for part in
                         ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
                scene["hand_contact"] = SimpleNamespace(body_names=hands,
                    data=SimpleNamespace(force_matrix_w=Tensor([[[[-1.5, 0, 0.2]]]*16])))
            return scene

        class FurnitureData:
            """Pushed object double: rests for 80 physics steps, then slides 0.4 mm along x per step."""
            def __init__(self, config): self.config, self.start = config, Simulation.steps
            @property
            def root_link_pose_w(self):
                x, y, z = self.config.init_state.pos
                return Tensor([[x+0.0004*max(0, Simulation.steps-self.start-80), y, z, 0, 0, 0, 1]])
            @property
            def root_link_vel_w(self): return Tensor([[0.08, 0, 0, 0, 0, 0]])
            @property
            def body_mass(self): return Tensor([[self.config.spawn.mass_props.mass]])

        class RigidObjectConfig(Config):
            InitialStateCfg = Config

        def replace(value, **kwargs):
            value = copy.deepcopy(value)
            value.__dict__.update(kwargs)
            return value

        modules = {}
        for key, fields in {
            "torch": {"float32": np.float32, "manual_seed": lambda seed: None,
                      "as_tensor": lambda value, **kwargs: Tensor(np.asarray(value, dtype=np.float32))},
            "warp": {"config": Config(enable_backward=True)}, "isaaclab": {},
            "isaaclab.app": {"launch_simulation": lambda *args: nullcontext()},
            "isaaclab.actuators": {"IdealPDActuatorCfg": ExplicitConfig, "ImplicitActuatorCfg": ImplicitConfig},
            "isaaclab.assets": {"AssetBaseCfg": Config, "RigidObjectCfg": RigidObjectConfig},
            "isaaclab.scene": {"InteractiveSceneCfg": Config},
            "isaaclab.sensors": {"CameraCfg": Config, "ContactSensorCfg": Config},
            "isaaclab.sim": {"GroundPlaneCfg": Config, "DomeLightCfg": Config, "PinholeCameraCfg": Config,
                             "SimulationCfg": Config, "SimulationContext": Simulation, "CuboidCfg": Config,
                             "PreviewSurfaceCfg": Config, "UsdPhysicsRigidBodyCfg": Config, "MassCfg": Config,
                             "UsdPhysicsCollisionCfg": Config},
            "isaaclab.utils": {"configclass": lambda c: c, "instantiate": instantiate, "replace": replace},
            "isaaclab_assets": {"G1_29DOF_CFG": candidate}, "isaaclab_physx": {},
            "isaaclab_physx.physics": {"PhysxCfg": Config}, "isaaclab_physx.sim": {}, "isaaclab_physx.sim.spawners": {},
            "isaaclab_physx.sim.spawners.materials": {"PhysxRigidBodyMaterialCfg": Config},
            "isaaclab_physx.renderers": {"IsaacRtxRendererCfg": Config}, "PIL": {"Image": Image},
        }.items():
            module = ModuleType(key)
            module.__dict__.update(fields)
            modules[key] = module

        def encode(directory, **kwargs):
            self.assertFalse(kwargs["require_motion"])
            path = directory / "rollout.mp4"
            path.write_bytes(b"CPU codec interface double")
            return {"status": "passed_video_decode", "decoded_frames": kwargs["count"],
                    "width": kwargs["width"], "height": kwargs["height"], "fps": str(kwargs["fps"]),
                    "duration_seconds": float(kwargs["count"]/kwargs["fps"]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        with tempfile.TemporaryDirectory() as name, mock.patch.dict(sys.modules, modules), \
                mock.patch.object(wbc, "BalancePolicy", Policy), \
                mock.patch.object(runner, "read_joint_frames", return_value=joint_frame_fixture(reference)), \
                mock.patch.object(runner, "read_collision_geometry", side_effect=AssertionError("baseline geometry accessed")), \
                mock.patch.object(runner, "read_contact_materials", return_value={"scopes": {n:[{"material_path":None}]
                    for n in ("ground", "left_ankle_roll_link", "right_ankle_roll_link")}}) as materials, \
                mock.patch.object(runner, "require_ffmpeg", return_value="unused-interface-double"), \
                mock.patch.object(runner, "encode_recording", side_effect=encode):
            directory = Path(name)
            runner.balance_check(profile, asset, runtime, contract, reference, Path("unused.onnx"), directory)
            runner.validate_stage("balance", directory, profile)
            materials.assert_not_called()
            self.assertFalse((directory / "contacts.jsonl").exists())
            self.assertEqual(Simulation.steps, 16)
            self.assertEqual(instantiated[0].robot.actuators["reference_pd"].stiffness["waist_yaw_joint"], 250)
            self.assertEqual(instantiated[0].robot.actuators["reference_pd"].actuator_effort_limit["left_hip_roll_joint"], 88)
            self.assertEqual(instantiated[0].robot.actuators["reference_pd"].joint_velocity_limit["waist_pitch_joint"], 37)
            actuators = instantiated[0].robot.actuators
            self.assertIsInstance(actuators["reference_pd"], ExplicitConfig)
            self.assertIsInstance(actuators["upper_hold"], ImplicitConfig)
            self.assertEqual(set(actuators["reference_pd"].joint_names_expr), set(asset["joint_groups"]["lower_body"]))
            self.assertEqual(set(actuators["upper_hold"].joint_names_expr), set(asset["joint_groups"]["arms"] + asset["joint_groups"]["hands"]))
            self.assertEqual(actuators["reference_pd"].armature, 0.01)
            self.assertEqual(actuators["upper_hold"].armature, 0.001)
            self.assertTrue(np.allclose(robot.pose[3:], [0,0,0,1]))
            evidence = json.loads((directory / "balance-worker.json").read_text())
            self.assertTrue(evidence["controller_inference_verified"])
            self.assertFalse(evidence["walking_verified"])
            self.assertEqual(len(list(directory.glob("frame-input-*.json"))), 3)
            motor_path = directory / "actuators.json"
            motor = json.loads(motor_path.read_text())
            original = motor["joint_armature"]["left_ankle_roll_joint"]
            motor["joint_armature"]["left_ankle_roll_joint"] = 0.0
            motor_path.write_text(json.dumps(motor))
            with self.assertRaisesRegex(RuntimeError, "armature"):
                runner.validate_stage("balance", directory, profile)
            motor["joint_armature"]["left_ankle_roll_joint"] = original
            motor_path.write_text(json.dumps(motor))
            records = [json.loads(l) for l in (directory / "trajectory.jsonl").read_text().splitlines()]
            records[1]["root_height_m"] = 0.1
            (directory / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaises(RuntimeError):
                runner.validate_stage("balance", directory, profile)
            failed = directory / "fall"
            failed.mkdir()

            def falling_step(self, *, render):
                robot.pose[2] = 0.1

            with mock.patch.object(Simulation, "step", new=falling_step):
                with self.assertRaisesRegex(RuntimeError, "Balance guard failed"):
                    runner.balance_check(profile, asset, runtime, contract, reference, Path("unused.onnx"), failed)
            failure = json.loads((failed / "balance-worker.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["steps_completed"], 1)
            self.assertIn("Balance guard", failure["error"])
            rejected = directory / "rejected"
            rejected.mkdir()
            steps_before = Simulation.steps
            Policy.output = np.full(15, 6.0)
            with self.assertRaisesRegex(RuntimeError, "transfer bound"):
                runner.balance_check(profile, asset, runtime, contract, reference, Path("unused.onnx"), rejected)
            snapshot = json.loads((rejected / "rejected-action.json").read_text())
            self.assertEqual(snapshot["step"], 1)
            self.assertEqual(snapshot["maximum_absolute_action"], 6.0)
            self.assertFalse(snapshot["applied"])
            self.assertEqual(snapshot["limit"], 5.0)
            self.assertEqual(len(snapshot["observation"]), 516)
            self.assertEqual(Simulation.steps, steps_before)
            Policy.output = np.zeros(15)
            diagnostic_profile, _, _ = runner.load_plan(ROOT / "configs/isaac_g1_diagnostics.json")
            diagnostic_profile.update(control_steps=4, camera=profile["camera"])
            diagnostic = directory / "diagnostic"
            diagnostic.mkdir()
            for file,value in (("profile.json", diagnostic_profile), ("asset-profile.json", asset),
                               ("controller-contract.json", contract)):
                (diagnostic / file).write_text(json.dumps(value))
            runner.balance_check(diagnostic_profile, asset, runtime, contract, reference, Path("unused.onnx"), diagnostic)
            runner.validate_stage("balance", diagnostic, diagnostic_profile)
            analysis = runner.analyze_run(diagnostic)
            self.assertEqual(analysis["status"], "diagnostics_complete")
            self.assertTrue(analysis["policy_inputs_verified"])
            self.assertEqual(analysis["contacts"]["samples"], 16)
            self.assertEqual(analysis["contacts"]["feet"]["left_ankle_roll_link"]["maximum_normal_force_n"], 100)
            self.assertTrue(instantiated[-1].robot.spawn.activate_contact_sensors)
            self.assertTrue(instantiated[-1].feet_contact.track_friction_forces)
            self.assertEqual(instantiated[-1].feet_contact.history_length, 1)
            placement_profile, _, _ = runner.load_plan(ROOT / "configs/isaac_g1_placement.json")
            placement_profile.update(control_steps=4, camera=profile["camera"])
            placed = directory / "placed"
            placed.mkdir()
            for file,value in (("profile.json", placement_profile), ("asset-profile.json", asset), ("controller-contract.json", contract)):
                (placed / file).write_text(json.dumps(value))
            Policy.observations.clear()
            steps_before = Simulation.steps
            with mock.patch.object(runner, "read_collision_geometry", side_effect=lambda stage,poses:collision_fixture(poses)):
                runner.balance_check(placement_profile, asset, runtime, contract, reference, Path("unused.onnx"), placed)
                runner.validate_stage("balance", placed, placement_profile)
            placed_analysis = runner.analyze_run(placed)
            self.assertGreater(placed_analysis["initial_placement"]["spawn_height_m"], 0.74)
            self.assertAlmostEqual(placed_analysis["initial_placement"]["minimum_clearance_after_m"], 0.001, places=6)
            self.assertEqual(Simulation.steps-steps_before, 16)  # No pre-roll physics steps.
            self.assertEqual(len(Policy.observations), 4)
            self.assertAlmostEqual(float(Policy.observations[0][-86+3]), 0.74, places=6)
            self.assertEqual(placed_analysis["status"], "diagnostics_complete")
            geometry_failure = directory / "geometry-failure"
            geometry_failure.mkdir()
            Policy.observations.clear()
            steps_before = Simulation.steps
            def too_low(stage, poses):
                result = collision_fixture(poses)
                for foot in result["feet"].values(): foot["body_pose_xyzw"][2] -= 1.0
                return result
            with mock.patch.object(runner, "read_collision_geometry", side_effect=too_low):
                with self.assertRaisesRegex(RuntimeError, "above the configured bound"):
                    runner.balance_check(placement_profile, asset, runtime, contract, reference, Path("unused.onnx"), geometry_failure)
            self.assertEqual(Simulation.steps, steps_before)
            self.assertEqual(Policy.observations, [])
            audit = directory/"audit";audit.mkdir()
            for file,value in (("profile.json",placement_profile),("asset-profile.json",asset),("controller-contract.json",contract)):
                (audit/file).write_text(json.dumps(value))
            def read_dynamics(robot,rotation,usd_path,*,directory=None,filename="dynamics-runtime.json"):
                snapshot=dynamics_fixture(contract,reference,asset,body_names=robot.body_names,joint_names=names,
                    q=robot.q,root=robot.pose,poses=robot.data.body_link_pose_w.value[0].tolist())
                snapshot["source_to_asset_rotation"]=rotation.tolist()
                return snapshot
            with mock.patch.object(runner,"read_runtime_dynamics",side_effect=read_dynamics), \
                    mock.patch.object(runner,"read_collision_geometry",side_effect=lambda stage,poses:collision_fixture(poses)), \
                    mock.patch.object(runner,"check_versions",return_value={}), \
                    mock.patch.object(wbc,"prepare_model",side_effect=AssertionError("audit model access")), \
                    mock.patch.object(wbc,"BalancePolicy",side_effect=AssertionError("audit ONNX access")), \
                    mock.patch.object(runner,"require_ffmpeg",side_effect=AssertionError("audit video access")):
                self.assertEqual(runner.worker("dynamics",placement_profile,asset,runtime,
                    ROOT/placement_profile["controller_contract"],audit),0)
                runner.validate_stage("dynamics",audit,placement_profile)
                report=runner.analyze_run(audit)
                self.assertEqual(report["status"],"compared_runtime_dynamics")
                extras=sorted(set(robot.body_names)-wbc.load_dynamics_reference()["bodies"].keys())
                self.assertEqual(report["runtime_only_bodies"],extras)
                self.assertAlmostEqual(report["totals"]["mass_difference_kg"],len(extras)*0.01)
                self.assertFalse(hasattr(instantiated[-1],"camera"))
                self.assertFalse((audit/"trajectory.jsonl").exists())
                self.assertFalse((audit/"rollout.mp4").exists())
                self.assertEqual(Simulation.steps,steps_before)
                self.assertEqual(Policy.observations,[])
                with mock.patch("builtins.print"):
                    self.assertEqual(runner.main(["--analyze-run",str(audit)]),0)
                raw_path=audit/"dynamics-runtime.json"
                invalid=json.loads(raw_path.read_text());invalid["body_mass_kg"][0]=-1
                raw_path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError,"mass"):
                    runner.analyze_run(audit)
            palm_profile = {**placement_profile, "inertial_override": {"body": "left_hand_palm_link", "source": "pinned_urdf"}}
            palm = directory / "palm"; palm.mkdir()
            for file, value in (("profile.json", palm_profile), ("asset-profile.json", asset), ("controller-contract.json", contract)):
                (palm / file).write_text(json.dumps(value))
            Policy.observations.clear()
            steps_before = Simulation.steps
            override = palm_fixture(contract)
            def author_palm(stage, target, controller, directory):
                self.assertFalse(target.is_initialized)
                self.assertEqual(Simulation.steps, steps_before)
                self.assertFalse(Policy.observations)
                report = copy.deepcopy(override)
                report["status"] = "authored_before_physics_reset"
                return report
            def read_palms(target):
                self.assertTrue(target.is_initialized)
                self.assertEqual(Simulation.steps, steps_before)
                self.assertFalse(Policy.observations)
                return copy.deepcopy(override["runtime"])
            with mock.patch.object(runner, "apply_palm_override", side_effect=author_palm), \
                    mock.patch.object(runner, "read_palm_runtime", side_effect=read_palms), \
                    mock.patch.object(runner, "read_collision_geometry", side_effect=lambda stage, poses: collision_fixture(poses)):
                runner.balance_check(palm_profile, asset, runtime, contract, reference, Path("unused.onnx"), palm)
                runner.validate_stage("balance", palm, palm_profile)
                self.assertEqual(runner.analyze_run(palm)["inertial_override"]["status"], "verified_palm_inertial_override")
                self.assertEqual(Simulation.steps-steps_before, 16)
                self.assertEqual(len(Policy.observations), 4)
                failed = directory / "palm-not-applied"; failed.mkdir()
                incorrect = copy.deepcopy(override["runtime"]); incorrect["left"]["mass_kg"] = 0.214904502
                Policy.observations.clear()
                steps_before = Simulation.steps
                with mock.patch.object(runner, "read_palm_runtime", return_value=incorrect), self.assertRaisesRegex(ValueError, "override mismatch"):
                    runner.balance_check(palm_profile, asset, runtime, contract, reference, Path("unused.onnx"), failed)
                self.assertEqual(Simulation.steps, steps_before)
                self.assertFalse(Policy.observations)
                partial = json.loads((failed / "palm-inertial-override.json").read_text())
                self.assertEqual(partial["runtime"]["left"]["mass_kg"], 0.214904502)
                self.assertNotEqual(partial["status"], "passed_palm_inertial_override")
                self.assertFalse((failed / "trajectory.jsonl").exists())
            invalid_dimensions = directory / "invalid-dimensions"
            invalid_dimensions.mkdir()
            def oversized_sphere(stage, poses):
                result = collision_fixture(poses)
                for foot in result["feet"].values(): foot["colliders"][0]["radius_m"] = 1.0
                return result
            with mock.patch.object(runner, "read_collision_geometry", side_effect=oversized_sphere):
                with self.assertRaisesRegex(ValueError, "world_radius_m=1"):
                    runner.balance_check(placement_profile, asset, runtime, contract, reference, Path("unused.onnx"), invalid_dimensions)
            partial = json.loads((invalid_dimensions / "initial-placement.json").read_text())
            self.assertEqual(partial["status"], "measuring_initial_placement")
            self.assertEqual(partial["before_geometry"]["feet"]["left_ankle_roll_link"]["colliders"][0]["radius_m"], 1.0)
            self.assertNotIn("before_clearance", partial)
            self.assertEqual(Simulation.steps, steps_before)
            self.assertEqual(Policy.observations, [])
            with mock.patch.object(wbc, "prepare_model", side_effect=AssertionError("model accessed")), \
                    mock.patch.object(runner, "run_stage", side_effect=AssertionError("worker launched")), \
                    mock.patch("builtins.print"):
                self.assertEqual(runner.main(["--analyze-run", str(diagnostic)]), 0)
            trace_path = diagnostic / "trajectory.jsonl"
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            trace[0]["policy_observation"][-1] = 123.0
            trace_path.write_text("\n".join(json.dumps(row) for row in trace))
            with self.assertRaisesRegex(ValueError, "observation"):
                runner.analyze_run(diagnostic)


            # Reuse the same runner doubles for the coherent 29-DOF branch.
            deployment_profile, asset, runtime = runner.load_plan(ROOT / "configs/isaac_g1_deployment.json")
            deployment_profile = {**deployment_profile, "camera": {"width":64,"height":64,"capture_every_control_steps":2}}
            contract, reference = runner.trial_contract(deployment_profile)
            names[:] = [n for n in names if n in reference.limits]
            limits = [[reference.limits[n]["lower"], reference.limits[n]["upper"]] for n in names]
            robot.q = np.zeros(29); robot.dq = np.zeros(29)
            robot.actuators = SimpleNamespace(computed_effort=Tensor([np.zeros(29)]), applied_effort=Tensor([np.zeros(29)]))
            deployment_dir = directory / "deployment"; deployment_dir.mkdir()
            for file, value in (("profile.json",deployment_profile),("asset-profile.json",asset),("controller-contract.json",contract)):
                (deployment_dir/file).write_text(json.dumps(value))
            Policy.observations.clear(); Policy.simulation_steps_at_calls.clear()
            steps_before = Simulation.steps
            override = palm_fixture(contract)
            def fix_hands(stage, target, directory):
                self.assertFalse(target.is_initialized)
                value = {"status":"authored_fixed_hands_before_reset","edit_layer":"session","joints":{
                    n:{"path":"/World/Robot/"+n,"body0":["parent"],"body1":["child"],"authored_type":"PhysicsFixedJoint"}
                    for n in wbc.load_deployment_reference()["fixed_hand_joints"]}}
                (directory/"deployment-joints.json").write_text(json.dumps(value))
            with mock.patch.object(runner,"apply_fixed_hands",side_effect=fix_hands), \
                    mock.patch.object(runner,"apply_palm_override",side_effect=author_palm), \
                    mock.patch.object(runner,"read_palm_runtime",side_effect=read_palms), \
                    mock.patch.object(runner,"read_runtime_dynamics",side_effect=read_dynamics), \
                    mock.patch.object(runner,"read_joint_frames",return_value=joint_frame_fixture(reference)), \
                    mock.patch.object(runner,"read_collision_geometry",side_effect=lambda stage,poses:collision_fixture(poses)):
                runner.balance_check(deployment_profile,asset,runtime,contract,reference,Path("unused.onnx"),deployment_dir)
            runner.validate_stage("balance",deployment_dir,deployment_profile)
            analysis = runner.analyze_run(deployment_dir)
            self.assertEqual(analysis["functional"]["status"],"passed_functional_evaluation")
            self.assertEqual(analysis["deployment"]["active_joints"],29)
            self.assertEqual(Policy.simulation_steps_at_calls,[steps_before+4*(i+1) for i in range(250)])
            self.assertEqual(len(Policy.observations),250)
            first = json.loads((deployment_dir/"trajectory.jsonl").read_text().splitlines()[0])
            self.assertEqual(first["policy_input_time_seconds"],0.02)
            self.assertEqual(first["applied_action"],[0.0]*15)
            self.assertEqual(json.loads((deployment_dir/"initial-state.json").read_text())["wanted_joint_positions"],[0.0]*29)
            self.assertIsInstance(instantiated[-1].robot.actuators["upper_hold"],ExplicitConfig)
            self.assertEqual(instantiated[-1].robot.actuators["upper_hold"].armature,0.01)
            rejected_analysis = copy.deepcopy(analysis)
            rejected_analysis["functional"]["status"] = "failed_functional_evaluation"
            rejected_analysis["functional"]["checks"]["foot_displacement_within_0_02m"] = False
            with mock.patch.object(runner,"analyze_run",return_value=rejected_analysis), self.assertRaisesRegex(RuntimeError,"Functional gate failed"):
                runner.validate_stage("balance",deployment_dir,deployment_profile)

            # Source contact reuses the same doubles; verification must finish before control.
            contact_profile, _, _ = runner.load_plan(ROOT / "configs/isaac_g1_contact.json")
            contact_profile = {**contact_profile, "camera": deployment_profile["camera"]}
            authored, _, _ = contact_fixture()
            del authored["materials"]
            materials = {"/physicsScene/defaultMaterial": 1.0, "/World/Ground/physicsMaterial": 1.0}
            def author_contact(stage, target, directory):
                self.assertFalse(target.is_initialized)
                return {**copy.deepcopy(authored), "status": "authored_before_physics_reset"}
            surface = {"combine": "max", "box_friction": 0.3}
            def read_material(stage, path):
                if path == "/World/envs/env_0/Furniture/geometry/physicsMaterial":
                    return {"path": path, "static_friction": surface["box_friction"], "dynamic_friction": surface["box_friction"],
                            "friction_combine_mode": "multiply"}
                return {"path": path, "static_friction": materials[path], "dynamic_friction": 1.0, "friction_combine_mode": surface["combine"]}
            created = []
            initialize = Simulation.__init__
            def record_simulation(self, config):
                created.append(config); initialize(self, config)
            # The implicit arm hold changes only the arm actuator model and its PhysX drive gains.
            arm_profile, _, _ = runner.load_plan(ROOT / "configs/isaac_g1_arm_hold.json")
            arm_profile = {**arm_profile, "camera": deployment_profile["camera"]}
            arm_contract, _ = runner.trial_contract(arm_profile)
            crouch_profile, _, _ = runner.load_plan(ROOT / "configs/isaac_g1_crouch_start.json")
            crouch_profile = {**crouch_profile, "camera": deployment_profile["camera"]}
            def read_arm_dynamics(robot,rotation,usd_path,*,directory=None,filename="dynamics-runtime.json"):
                snapshot = dynamics_fixture(arm_contract,reference,asset,body_names=robot.body_names,joint_names=names,
                    q=robot.q,root=robot.pose,poses=robot.data.body_link_pose_w.value[0].tolist(),implicit_arms=True)
                snapshot["source_to_asset_rotation"] = rotation.tolist()
                return snapshot
            push_profile, _, _ = runner.load_plan(ROOT / "configs/isaac_g1_push.json")
            push_profile = {**push_profile, "camera": deployment_profile["camera"]}
            push_contract, _ = runner.trial_contract(push_profile)
            for label, case_profile, controller, dynamics, ground, error, box_friction in (
                    ("contact", contact_profile, contract, read_dynamics, 1.0, None, 0.3),
                    ("contact-friction-mismatch", contact_profile, contract, read_dynamics, 0.5, "friction material", 0.3),
                    ("arm-hold", arm_profile, arm_contract, read_arm_dynamics, 1.0, None, 0.3),
                    ("arm-hold-explicit-runtime", arm_profile, arm_contract, read_dynamics, 1.0, "actuator model", 0.3),
                    ("contact-implicit-runtime", contact_profile, contract, read_arm_dynamics, 1.0, "actuator model", 0.3),
                    ("crouch-start", crouch_profile, arm_contract, read_arm_dynamics, 1.0, None, 0.3),
                    ("push", push_profile, push_contract, read_arm_dynamics, 1.0, None, 0.3),
                    ("push-box-friction-mismatch", push_profile, push_contract, read_arm_dynamics, 1.0, "Push object differs", 0.6)):
                contact_dir = directory / label; contact_dir.mkdir()
                for file, value in (("profile.json",case_profile),("asset-profile.json",asset),("controller-contract.json",controller)):
                    (contact_dir/file).write_text(json.dumps(value))
                materials["/World/Ground/physicsMaterial"] = ground
                surface.update(combine="multiply" if case_profile.get("push_object") else "max", box_friction=box_friction)
                Policy.observations.clear()
                steps_before = Simulation.steps
                with ExitStack() as patches:
                    for target, name, kwargs in ((runner,"apply_fixed_hands",{"side_effect":fix_hands}),
                            (runner,"apply_palm_override",{"side_effect":author_palm}),
                            (runner,"read_palm_runtime",{"side_effect":read_palms}),
                            (runner,"apply_source_contact",{"side_effect":author_contact}),
                            (runner,"read_material",{"side_effect":read_material}),
                            (runner,"read_runtime_dynamics",{"side_effect":dynamics}),
                            (runner,"read_joint_frames",{"return_value":joint_frame_fixture(reference)}),
                            (runner,"read_collision_geometry",{"side_effect":lambda stage,poses:contact_geometry(poses)}),
                            (Simulation,"__init__",{"new":record_simulation})):
                        patches.enter_context(mock.patch.object(target,name,**kwargs))
                    if error is None:
                        runner.balance_check(case_profile,asset,runtime,controller,reference,Path("unused.onnx"),contact_dir)
                    else:
                        with self.assertRaisesRegex(ValueError,error):
                            runner.balance_check(case_profile,asset,runtime,controller,reference,Path("unused.onnx"),contact_dir)
                upper_hold = instantiated[-1].robot.actuators["upper_hold"]
                implicit_case = case_profile.get("arm_hold") == "implicit"
                self.assertIsInstance(upper_hold, ImplicitConfig if implicit_case else ExplicitConfig)
                self.assertEqual((upper_hold.stiffness["left_wrist_pitch_joint"], upper_hold.damping["left_wrist_pitch_joint"], upper_hold.armature), (100, 0.5, 0.01))
                self.assertEqual(upper_hold.joint_effort_limit["left_wrist_pitch_joint"], 5)
                if error is None:
                    runner.validate_stage("balance",contact_dir,case_profile)
                    analysis = runner.analyze_run(contact_dir)
                    self.assertEqual(analysis["contact_override"]["status"],"verified_source_contact")
                    self.assertEqual(analysis["deployment"]["arm_hold"],"implicit" if implicit_case else "explicit")
                    initial = json.loads((contact_dir/"initial-state.json").read_text())
                    placed = analysis["initial_placement"]
                    if case_profile["initialization"]["mode"] == "collision_clearance":
                        lower_indices = [names.index(n) for n in asset["joint_groups"]["lower_body"]]
                        self.assertEqual([initial["wanted_joint_positions"][i] for i in lower_indices], arm_contract["default_lower_angles"])
                        self.assertGreater(placed["vertical_shift_m"], 0)
                        self.assertAlmostEqual(placed["minimum_clearance_after_m"], 0.001, places=6)
                        self.assertAlmostEqual(placed["spawn_height_m"], 0.74+placed["vertical_shift_m"], places=6)
                    else:
                        self.assertEqual(initial["wanted_joint_positions"], [0.0]*29)
                        self.assertEqual(placed["vertical_shift_m"], 0)
                    self.assertEqual(analysis["functional"]["status"],"passed_functional_evaluation")
                    self.assertEqual(len(Policy.observations),250)
                    for material in (instantiated[-1].ground.spawn.physics_material, created[-1].physics_material):
                        self.assertEqual((material.static_friction, material.dynamic_friction, material.friction_combine_mode), (1.0, 1.0, surface["combine"]))
                    if case_profile is push_profile:
                        box = instantiated[-1].furniture
                        self.assertEqual((box.spawn.size, box.spawn.mass_props.mass, box.spawn.physics_material.static_friction), ((0.7, 0.8, 1.0), 10.0, 0.3))
                        pushed = json.loads((contact_dir/"push-object.json").read_text())
                        self.assertAlmostEqual(analysis["push_object"]["near_face_x_m"], 0.05+pushed["hand_reach_m"])
                        self.assertAlmostEqual(pushed["hand_reach_m"], 0.365+0.05, places=3)
                        self.assertAlmostEqual(analysis["functional"]["object"]["progress_m"], 0.368, places=6)
                        self.assertAlmostEqual(analysis["push_force"]["maximum_push_n"], 24.0, places=4)
                        np.testing.assert_allclose(Policy.observations[-1][-86+4:-86+7], [0, 0.2, 0], atol=1e-6)  # rpy command slot
                        self.assertTrue(instantiated[-1].hand_contact.filter_prim_paths_expr)
                        self.assertEqual(controller["policy_role"], "walk")
                else:
                    self.assertEqual(Simulation.steps, steps_before)
                    self.assertFalse(Policy.observations)
                    self.assertFalse((contact_dir/"trajectory.jsonl").exists())
            self.assertEqual(json.loads((directory/"contact-friction-mismatch"/"contact-override.json").read_text())["materials"]["ground"]["static_friction"],0.5)

if __name__ == "__main__":
    unittest.main()
