from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import g1_wbc as wbc
import isaac_g1_ppo as ppo
from test_isaac_g1_rl import Tensor, parallel_isaac


def yaw_quats(yaws):
    yaws = np.asarray(yaws, dtype=float)
    return np.c_[np.zeros(len(yaws)), np.zeros(len(yaws)), np.sin(yaws/2), np.cos(yaws/2)]


class TaskFunctionTests(unittest.TestCase):
    def setUp(self):
        self.config = ppo.load_config(ppo.DEFAULT_CONFIG)[0]

    def test_actions_map_to_walk_commands_inside_reviewed_bounds(self):
        bounds = self.config["action_bounds"]
        commands, rpys = ppo.action_commands([[-1, -1, -1], [1, 1, 1], [3, 0, -3]], bounds)
        np.testing.assert_allclose(commands, [[0.1, 0, -0.2], [0.4, 0, 0.2], [0.4, 0, 0.0]])
        np.testing.assert_allclose(rpys, [[0, 0.0, 0], [0, 0.3, 0], [0, 0.0, 0]])
        self.assertTrue((np.linalg.norm(commands, axis=1) > 0.05).all())
        scripted = ppo.normalized_action([0.2, 0.0, 0.2], bounds)
        np.testing.assert_allclose(ppo.action_commands([scripted], bounds)[0], [[0.2, 0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(ppo.action_commands([scripted], bounds)[1], [[0, 0.2, 0]], atol=1e-12)

    def test_observation_uses_the_robot_heading_frame(self):
        origins = np.zeros((2, 3))
        robot = np.c_[[[0.0, 0.0, 0.8], [0.0, 0.1, 0.8]], yaw_quats([0.0, np.pi/2])]
        box = np.c_[[[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]], yaw_quats([0.1, 0.1])]
        velocity = np.zeros((2, 6))
        velocity[:, 0] = 0.2
        obs = ppo.task_observation(robot, velocity, box, velocity, origins, np.array([2.0, 2.0]), np.zeros((2, 3)),
                                   np.zeros((2, 3)), np.zeros((2, 1)))
        self.assertEqual(obs.shape, (2, 21))
        np.testing.assert_allclose(obs[0, :4], [1.0, 0.0, 0.1, 0.0], atol=1e-6)
        # Facing +y, a box ahead along world +x lies on the robot's right (negative local y).
        np.testing.assert_allclose(obs[1, :4], [-0.1, -1.0, 0.1 - np.pi/2, np.pi/2], atol=1e-6)
        np.testing.assert_allclose(obs[:, 7], [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(obs[1, 11:13], [0.0, -0.2], atol=1e-6)

    def test_step_rewards_progress_and_reports_the_first_failure(self):
        origins = np.zeros((6, 3))
        start = np.full(6, 0.9)
        box = np.c_[np.tile([1.0, 0.0, 0.5], (6, 1)), yaw_quats(np.zeros(6))]
        box[1, 0] = 1.95                                  # success
        box[2, 1] = 0.4                                   # off course
        box[3, 3:] = wbc.matrix_quat_xyzw(wbc.axis_rotation([1, 0, 0], 0.3))  # tipped
        robot = np.c_[np.zeros((6, 3)), yaw_quats(np.zeros(6))]
        robot[1, 0] = 1.1                                 # following the successful box
        robot[4, 0] = -1.0                                # lost the box
        low = {"invalid_action": np.array([0, 0, 0, 0, 0, 1], bool), "fallen": np.array([0, 0, 0, 0, 0, 1], bool)}
        actions, last = np.zeros((6, 3)), np.zeros((6, 3))
        actions[0] = [0.5, 0, 0]
        reward, done, outcome, progress = ppo.task_step(box, robot, origins, start, np.full(6, 0.995), actions, last,
                                                        low, self.config, 0.1)
        names = [ppo.OUTCOMES[i] if i >= 0 else None for i in outcome]
        self.assertEqual(names, [None, "success", "off_course", "box_tipped", "lost_box", "invalid_action"])
        self.assertEqual(done.tolist(), [False, True, True, True, True, True])
        w = self.config["reward"]
        self.assertAlmostEqual(reward[0], w["progress"]*0.005/0.1 - w["action_rate"]*0.25, places=9)
        self.assertAlmostEqual(reward[1], w["progress"]*0.955/0.1 + w["success"], places=9)
        self.assertAlmostEqual(progress[1], 1.05)

    def test_configuration_rejects_unreviewed_commands_and_spawn_contact(self):
        raw = json.loads(ppo.DEFAULT_CONFIG.read_text())
        cases = [{**raw, "extra": 1},
                 {**raw, "action_bounds": {**raw["action_bounds"], "lean_pitch_rad": [0.0, 0.4]}},
                 {**raw, "action_bounds": {**raw["action_bounds"], "forward_mps": [0.0, 0.4]}},
                 {**raw, "randomization": {**raw["randomization"], "gap_m": [0.02, 0.12]}},
                 {**raw, "scripted_action": {**raw["scripted_action"], "forward_mps": 0.5}},
                 {**raw, "push_profile": "configs/isaac_g1_crouch_start_forward.json"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ppo.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    ppo.load_config(path)
        self.assertEqual(ppo.train_config(self.config)["obs_groups"], {"actor": ["policy"], "critic": ["critic"]})


class FakeRunner:
    def __init__(self, env, cfg, log_dir, device):
        self.env, self.cfg = env, cfg
        observations = env.get_observations()
        assert observations["critic"].value.shape == (env.num_envs, 23)

    def learn(self, num_learning_iterations, init_at_random_ep_len):
        # Staggered lengths near the limit, so single environments time out and reset during training.
        self.env.episode_length_buf = Tensor(np.arange(self.env.num_envs) % 3 + self.env.max_episode_length - 3)
        for _ in range(num_learning_iterations*self.cfg["num_steps_per_env"]):
            _, reward, dones, extras = self.env.step(Tensor(np.zeros((self.env.num_envs, 3))))
            assert reward.value.shape == dones.value.shape == extras["time_outs"].value.shape == (self.env.num_envs,)

    def get_inference_policy(self, device):
        return lambda obs: Tensor(np.ones((len(obs["policy"].value), 3)))


class PpoWorkerTests(unittest.TestCase):
    def test_worker_randomizes_boxes_resets_environments_and_evaluates_both_policies(self):
        config, profile, asset, runtime = ppo.load_config(ppo.DEFAULT_CONFIG)
        count = 4
        config = {**config, "num_envs": count, "episode_seconds": 1.0,
                  "ppo": {**config["ppo"], "max_iterations": 2, "num_steps_per_env": 3}}
        modules = {"tensordict": ModuleType("tensordict"), "rsl_rl": ModuleType("rsl_rl"),
                   "rsl_rl.runners": ModuleType("rsl_rl.runners")}
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        modules["rsl_rl.runners"].OnPolicyRunner = FakeRunner
        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(count, profile, asset) as fake, \
                mock.patch.dict(sys.modules, modules):
            sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
            directory = Path(tmp)
            ppo.run_worker(config, profile, asset, runtime, Path("unused.onnx"), directory)
            result = json.loads((directory / "ppo-worker.json").read_text())
            progress = [json.loads(line) for line in (directory / "ppo-progress.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "passed_ppo_training")
        masses, frictions = (np.array(result["box_properties"][k]) for k in ("masses_kg", "frictions"))
        np.testing.assert_allclose(fake.box_mass, masses)
        np.testing.assert_allclose(fake.box_friction, frictions)
        self.assertTrue(((5.0 <= masses) & (masses <= 20.0) & (0.05 <= frictions) & (frictions <= 0.15)).all())
        for name in ("scripted_eval", "learned_eval"):
            self.assertEqual(result[name]["episodes"], count)
            self.assertEqual(result[name]["outcome_rates"]["time_out"], 1.0)
            self.assertEqual(sum(g["episodes"] for g in result[name]["by_slide_force"].values()), count)
        self.assertEqual([p["iteration"] for p in progress], [0, 1])
        self.assertEqual(result["training"]["decisions"], count*2*3)
        # Staggered episode lengths reset single environments, and each reset re-places the box inside the gap range.
        self.assertTrue(any(0 < len(ids) < count for ids in fake.robot.root_writes))
        near = fake.box.pose[:, 0] - fake.origins[:, 0] - 0.415 - 0.35
        self.assertTrue(((near >= 0.06 - 1e-9) & (near <= 0.12 + 1e-9)).all())


if __name__ == "__main__":
    unittest.main()
