"""Numerical convention and controller-transfer failure checks, without a GPU."""

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


if __name__ == "__main__":
    unittest.main()
