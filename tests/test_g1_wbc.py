"""Numerical convention and controller-transfer failure checks, without a GPU."""

from contextlib import contextmanager, nullcontext
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


class ControllerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.contract, self.robot = wbc.load_contract(ROOT / "configs/g1_wbc_reference.json")
        self.asset = json.loads((ROOT / "configs/isaac_g1_asset.json").read_text())

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
            def default_joint_pos(self): return Tensor([np.zeros(43)])
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
        robot.actuators = SimpleNamespace(computed_effort=Tensor([np.zeros(43)]), applied_effort=Tensor([np.zeros(43)]))
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
            def reset(self): pass
            def forward(self): pass
            def render(self): pass
            def step(self, *, render): Simulation.steps += 1

        class Policy:
            metadata = {"input": ["batch",516], "output": ["batch",15]}
            output = np.zeros(15)
            def __init__(self, *args): pass
            def __call__(self, observation):
                self.assert_shape = observation.shape == (516,)
                if not self.assert_shape: raise AssertionError("Bad observation passed to policy")
                return self.output

        class Image:
            @staticmethod
            def fromarray(array): return Image()
            def save(self, path): Path(path).write_bytes(b"CPU interface image double")

        candidate = Config(spawn=Config(fix_root_link=False, rigid_props=Config(disable_gravity=False)))
        instantiated = []

        def instantiate(config):
            instantiated.append(config)
            scene = Scene(robot=robot, camera=Camera())
            if hasattr(config, "feet_contact"):
                scene["feet_contact"] = SimpleNamespace(
                    body_names=["right_ankle_roll_link", "left_ankle_roll_link"],
                    data=SimpleNamespace(net_normal_forces_w=Tensor([[[0,0,200],[0,0,100]]]),
                        net_friction_forces_w=Tensor([[[-20,0,0],[-10,0,0]]]),
                        net_forces_w=Tensor([[[-20,0,200],[-10,0,100]]])))
            return scene

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
            "isaaclab.assets": {"AssetBaseCfg": Config},
            "isaaclab.scene": {"InteractiveSceneCfg": Config},
            "isaaclab.sensors": {"CameraCfg": Config, "ContactSensorCfg": Config},
            "isaaclab.sim": {"GroundPlaneCfg": Config, "DomeLightCfg": Config, "PinholeCameraCfg": Config,
                             "SimulationCfg": Config, "SimulationContext": Simulation},
            "isaaclab.utils": {"configclass": lambda c: c, "instantiate": instantiate, "replace": replace},
            "isaaclab_assets": {"G1_29DOF_CFG": candidate}, "isaaclab_physx": {},
            "isaaclab_physx.physics": {"PhysxCfg": Config},
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

if __name__ == "__main__":
    unittest.main()
