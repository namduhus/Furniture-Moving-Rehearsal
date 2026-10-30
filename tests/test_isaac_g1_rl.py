import copy
from contextlib import nullcontext
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
sys.path.insert(0, str(ROOT / "tests"))
import g1_wbc as wbc
import isaac_g1_rl as rl
from test_g1_wbc import contact_fixture, contact_geometry, joint_frame_fixture


class BatchAdapterTests(unittest.TestCase):
    def setUp(self):
        profile, _, _ = rl.load_plan(ROOT / "configs/isaac_g1_push_sliders.json")
        self.contract, self.reference = rl.trial_contract(profile)

    def test_batch_history_matches_single_environment_history(self):
        rng = np.random.default_rng(0)
        count = 3
        batch = wbc.BatchHistory(self.contract, count)
        singles = [wbc.ObservationHistory(self.contract) for _ in range(count)]
        rotation = wbc.axis_rotation([0, 0, 1], 0.01)
        limits = np.array([[-1.0, 1.0]]*15)
        for step in range(8):
            q, dq = rng.normal(0, 0.2, (count, 29)), rng.normal(0, 1, (count, 29))
            quats = np.array([wbc.matrix_quat_xyzw(wbc.axis_rotation(rng.normal(size=3), 0.1)) for _ in range(count)])
            gyros, commands, rpys = rng.normal(size=(count, 3)), rng.normal(0, 0.2, (count, 3)), rng.normal(0, 0.1, (count, 3))
            observed = batch.observe(q, dq, quats, gyros, rotation, commands, rpys)
            for i, single in enumerate(singles):
                np.testing.assert_allclose(observed[i], single.observe(q[i], dq[i], quats[i], gyros[i], rotation, commands[i], rpys[i]), atol=1e-6)
            actions = rng.normal(0, 0.5, (count, 15))
            targets, clipped, invalid = batch.targets(actions, limits)
            for i, single in enumerate(singles):
                wanted, count_i = single.targets(actions[i], limits)
                np.testing.assert_allclose(targets[i], wanted, atol=1e-12)
                self.assertEqual(clipped[i], count_i)
            self.assertFalse(invalid.any())
        actions[1, 4] = 9.0
        _, _, invalid = batch.targets(actions, limits)
        self.assertEqual(invalid.tolist(), [False, True, False])
        self.assertEqual(float(np.max(np.abs(batch.last_action[1]))), self.contract["action_limit"])
        quats = np.array([wbc.matrix_quat_xyzw(wbc.axis_rotation([1, 2, 3], 0.4)), [0, 0, 0, 1]])
        np.testing.assert_allclose(wbc.quat_xyzw_matrices(quats), [wbc.quat_xyzw_matrix(q) for q in quats], atol=1e-12)
        with self.assertRaises(ValueError):
            wbc.quat_xyzw_matrices([[0, 0, 0, 2.0]])

    def test_batch_policy_and_configuration_checks(self):
        policy = object.__new__(wbc.BalancePolicy)
        policy.input_name = "obs"
        policy.session = SimpleNamespace(run=lambda _, feed: [np.full((len(feed["obs"]), 15), 0.5, dtype=np.float32)])
        self.assertEqual(policy.batch(np.zeros((4, 516))).shape, (4, 15))
        with self.assertRaises(ValueError):
            policy.batch(np.zeros((4, 515)))
        config, profile, _, _ = rl.load_config(rl.DEFAULT_CONFIG)
        self.assertEqual((config["num_envs"], profile["push_object"]["friction"]), (64, 0.1))
        raw = json.loads(rl.DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rl.json"
            for data in ({**raw, "num_envs": 0}, {**raw, "env_spacing_m": 1.0}, {**raw, "extra": 1},
                         {**raw, "push_profile": "configs/isaac_g1_crouch_start_forward.json"}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    rl.load_config(path)
        with mock.patch("builtins.print"):
            self.assertEqual(rl.main(["--dry-run"]), 0)


class ParallelWorkerTests(unittest.TestCase):
    def test_worker_verifies_every_environment_and_runs_the_batched_loop(self):
        config, profile, asset, runtime = rl.load_config(rl.DEFAULT_CONFIG)
        config = {**config, "num_envs": 3, "control_steps": 30}
        contract, reference = rl.trial_contract(profile)
        count, names = 3, list(reversed(asset["controller_reference"]["joint_order"]))
        lower = asset["joint_groups"]["lower_body"]
        bodies = list(reference.forward(dict.fromkeys(reference.limits, 0.0)))
        source = wbc.load_dynamics_reference()["bodies"]
        palm = wbc.palm_inertial_reference(contract)["mass_kg"]
        masses = [palm if b == "left_hand_palm_link" else source.get(b, {"mass_kg": 1.0})["mass_kg"] for b in bodies]
        origins = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        steps = {"count": 0}

        class Tensor:
            def __init__(self, value): self.value = np.asarray(value, dtype=float)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        class Data:
            def __init__(self, robot): self.robot = robot
            joint_stiffness = property(lambda self: Tensor([[0.0 if n in lower else 100.0 for n in names]]*count))
            joint_damping = property(lambda self: Tensor([[0.0 if n in lower else 0.5 for n in names]]*count))
            joint_armature = property(lambda self: Tensor(np.full((count, 29), 0.01)))
            joint_friction_coeff = property(lambda self: Tensor(np.full((count, 29), 0.1)))
            joint_dynamic_friction_coeff = property(lambda self: Tensor(np.full((count, 29), 0.1)))
            joint_viscous_friction_coeff = property(lambda self: Tensor(np.full((count, 29), 0.001)))
            body_mass = property(lambda self: Tensor([masses]*count))
            joint_pos_limits = property(lambda self: Tensor([[[reference.limits[n]["lower"], reference.limits[n]["upper"]] for n in names]]*count))
            joint_pos = property(lambda self: Tensor(self.robot.q))
            joint_vel = property(lambda self: Tensor(np.zeros((count, 29))))
            root_link_pose_w = property(lambda self: Tensor(self.robot.pose))
            root_link_vel_w = property(lambda self: Tensor(np.zeros((count, 6))))

            @property
            def body_link_pose_w(self):
                out = []
                for q, pose in zip(self.robot.q, self.robot.pose):
                    fk = reference.forward(dict(zip(names, q, strict=True)))
                    root = wbc.pose_matrix(pose)
                    out.append([np.r_[(root @ fk[b])[:3, 3], wbc.matrix_quat_xyzw((root @ fk[b])[:3, :3])] for b in bodies])
                return Tensor(out)

        class Robot:
            joint_names, body_names = names, bodies
            def __init__(self):
                self.q = np.zeros((count, 29))
                self.pose = np.c_[origins + [0, 0, 0.8], np.tile([0, 0, 0, 1.0], (count, 1))]
                self.data = Data(self)
            def write_root_pose_to_sim_index(self, *, root_pose): self.pose = root_pose.value.copy()
            def write_root_velocity_to_sim_index(self, *, root_velocity): pass
            def write_joint_position_to_sim_index(self, *, position): self.q = position.value.copy()
            def write_joint_velocity_to_sim_index(self, *, velocity): pass
            def set_joint_position_target_index(self, *, target): self.targets = target.value.copy()
            def set_joint_velocity_target_index(self, *, target): pass

        class Box:
            """Rests for 80 physics steps, then slides 0.4 mm along x per step."""
            def __init__(self, x): self.start, self.x = steps["count"], x
            @property
            def data(self):
                slide = 0.0004*max(0, steps["count"]-self.start-80)
                return SimpleNamespace(body_mass=Tensor(np.full((count, 1), 10.0)),
                                       root_link_pose_w=Tensor(np.c_[origins + [self.x+slide, 0, 0.5], np.tile([0, 0, 0, 1.0], (count, 1))]))

        robot = Robot()

        class Scene(dict):
            env_origins = Tensor(origins)
            def reset(self): pass
            def update(self, dt): pass
            def write_data_to_sim(self): pass

        class Simulation:
            stage = object()
            def __init__(self, config): pass
            def reset(self): pass
            def forward(self): pass
            def step(self, *, render): steps["count"] += 1

        class Policy:
            def __init__(self, *args): pass
            def batch(self, observations):
                self.last = observations
                return np.zeros((len(observations), 15), dtype=np.float32)

        hand_reach = 0.415
        box = Box(hand_reach + 0.05 + 0.35)
        built = SimpleNamespace(cfg=SimpleNamespace(sim=None, scene=None), hand_reach=hand_reach)
        hands = {"status": "authored_fixed_hands_before_reset", "joints": dict.fromkeys(range(14), {})}
        report, _, _ = contact_fixture()
        modules = {"torch": ModuleType("torch"), "warp": ModuleType("warp"), "isaaclab": ModuleType("isaaclab"),
                   "isaaclab.app": ModuleType("isaaclab.app"), "isaaclab.sim": ModuleType("isaaclab.sim"),
                   "isaaclab.utils": ModuleType("isaaclab.utils")}
        modules["torch"].__dict__.update(float32=np.float32, manual_seed=lambda seed: None,
                                         as_tensor=lambda value, **kwargs: Tensor(value))
        modules["warp"].config = SimpleNamespace(enable_backward=True)
        modules["isaaclab.app"].launch_simulation = lambda *args: nullcontext()
        modules["isaaclab.sim"].SimulationContext = Simulation
        modules["isaaclab.utils"].instantiate = lambda cfg: Scene(robot=robot, furniture=box)
        authored = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, modules), \
                mock.patch.object(wbc, "BalancePolicy", Policy), \
                mock.patch.object(rl, "scene_config", return_value=built) as scene_config, \
                mock.patch.object(rl, "apply_fixed_hands", side_effect=lambda stage, r, d, env: authored.append(env) or hands), \
                mock.patch.object(rl, "apply_palm_override", return_value={"status": "authored_before_physics_reset"}), \
                mock.patch.object(rl, "apply_source_contact", return_value=copy.deepcopy(report)), \
                mock.patch.object(rl, "read_material", side_effect=lambda stage, path: {"path": path, "static_friction": 0.1 if "Furniture" in path else 1.0,
                                                                                         "dynamic_friction": 0.1, "friction_combine_mode": "multiply"}), \
                mock.patch.object(rl, "read_joint_frames", return_value=joint_frame_fixture(reference)), \
                mock.patch.object(rl, "read_collision_geometry", side_effect=lambda stage, poses: contact_geometry(poses)):
            directory = Path(tmp)
            rl.run_worker(config, profile, asset, runtime, Path("unused.onnx"), directory)
            result = json.loads((directory / "rl-worker.json").read_text())
        self.assertEqual(result["status"], "passed_rl_throughput")
        self.assertEqual(authored, [f"/World/envs/env_{i}" for i in range(count)])
        self.assertEqual(scene_config.call_args.kwargs["replicate_physics"], False)
        self.assertEqual(result["setup"]["environments"], count)
        self.assertAlmostEqual(result["initialization"]["clearance_after_m"], 0.001, places=6)
        self.assertEqual(steps["count"], 30*4)
        self.assertEqual((result["fallen_envs"], result["invalid_action_envs"]), (0, 0))
        self.assertAlmostEqual(result["box_progress_m"]["median"], 0.0004*(120-80), places=6)
        self.assertAlmostEqual(result["simulated_seconds_per_env"], 0.6)


if __name__ == "__main__":
    unittest.main()
