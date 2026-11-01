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
import isaac_g1_carry as carry
import isaac_g1_lift as lift
import isaac_g1_room as room
import room_geometry as rg
from test_isaac_g1_rl import Tensor, parallel_isaac

APPROACH = ROOT / "configs/isaac_g1_room_approach.json"
CARRY = ROOT / "configs/isaac_g1_room_carry.json"


def yaw_poses(xyz, yaws):
    yaws = np.asarray(yaws, dtype=float)
    return np.c_[np.asarray(xyz, dtype=float), np.zeros(len(yaws)), np.zeros(len(yaws)), np.sin(yaws/2), np.cos(yaws/2)]


class RoomTaskTests(unittest.TestCase):
    def setUp(self):
        self.config = room.load_config(APPROACH)[0]
        self.carry_config = room.load_config(CARRY)[0]
        self.box_room = {"width_m": 4.0, "depth_m": 4.0, "boxes": np.array([[1.0, 1.0, 0.6, 0.6, 0.8]])}

    def test_episodes_reach_a_pregrasp_or_target_pose_along_a_clear_path(self):
        size = self.config["object"]["size_m"]
        for config in (self.config, self.carry_config):
            for seed in range(5):
                episode = room.sample_episode(config["task"], self.box_room, config["room"], config["goal"], 0.41, size,
                                              np.random.default_rng(seed))
                self.assertIsNotNone(episode)
                np.testing.assert_allclose(episode["path"][0], episode["start"][:2])
                np.testing.assert_allclose(episode["path"][-1], episode["goal"][:2])
                self.assertTrue(config["goal"]["min_distance_m"] <= np.linalg.norm(episode["goal"][:2] - episode["start"][:2])
                                <= config["goal"]["max_distance_m"])
                target = episode["cabinet"] if config["task"] == "approach" else episode["placed"]
                np.testing.assert_allclose(episode["goal"], rg.pregrasp_pose(target[:2], target[2], 0.41), atol=1e-12)
                if config["task"] == "carry":
                    # The held cabinet sits `reach` ahead of the start root.
                    ahead = episode["start"][:2] + 0.41*np.array([np.cos(episode["start"][2]), np.sin(episode["start"][2])])
                    np.testing.assert_allclose(episode["cabinet"][:2], ahead, atol=1e-12)

    def test_state_and_observation_use_the_robot_heading_frame(self):
        origins = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        robot = yaw_poses(origins + [-1.0, 0.0, 0.75], [0.0, np.pi/2])
        cabinet = yaw_poses(origins + [-0.6, 0.0, 0.5], [0.0, 0.0])
        bodies = np.stack([origins + [-1.0, 0.0, 0.8]], axis=1)
        goals = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        paths = [np.array([[-1.0, 0.0], [0.0, 0.0]])]*2
        velocity = np.zeros((2, 6))
        velocity[:, 0] = 0.2
        padded = rg.pad_rooms([self.box_room]*2, 3)
        state = room.room_state(robot, velocity, cabinet, bodies, origins, padded, goals, paths, [0.3, 0.28, 0.9], 0.08)
        np.testing.assert_allclose(state["goal_local"], [[1.0, 0.0], [0.0, -1.0]], atol=1e-12)
        np.testing.assert_allclose(state["yaw_error"], [0.0, -np.pi/2], atol=1e-12)
        np.testing.assert_allclose(state["remaining_m"], [1.0, 1.0])
        np.testing.assert_allclose(state["lookahead"][0], [0.3, 0, 0.6, 0, 1.0, 0], atol=1e-12)
        np.testing.assert_allclose(state["velocity"][1], [0.0, -0.2, 0.0], atol=1e-12)
        # Ray 0 points along the heading: (-0.55, 0) is 1.45 m from the left wall and closer to the box corner (0.7, 0.7).
        self.assertAlmostEqual(state["rays"][0, 0], np.hypot(1.25, 0.7))
        self.assertAlmostEqual(state["robot_clearance_m"][0], 1.0 - 0.08)
        obs = room.task_observation(state, np.array([False, True]), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 1)),
                                    np.zeros((2, 1)))
        self.assertEqual(obs.shape, (2, room.OBSERVATIONS))
        self.assertEqual((obs[0, 19], obs[0, 26]), (0.0, 0.0))         # nothing carried: cabinet terms are zero
        self.assertAlmostEqual(obs[1, 26], 0.05, places=6)

    def test_outcomes_require_stopping_at_the_pose_and_penalize_pushing(self):
        config = self.config
        n = 5
        state = {"position_error": np.array([0.02, 0.02, 0.5, 0.02, 0.02]), "yaw_error": np.zeros(n),
                 "remaining_m": np.full(n, 0.5), "robot_clearance_m": np.full(n, 0.3), "object_clearance_m": np.full(n, 0.3),
                 "bottom_m": np.full(n, 0.05), "tilt_rad": np.zeros(n)}
        speed = np.array([0.01, 0.2, 0.01, 0.01, 0.01])
        pushed = np.array([0.0, 0.0, 0.0, 0.05, 0.0])
        prefix = np.array([False, False, False, False, True])
        low = {"invalid_action": np.zeros(n, bool), "fallen": np.zeros(n, bool)}
        actions = np.zeros((n, 3))
        reward, done, outcome = room.task_step(state, np.full(n, 0.6), prefix, np.zeros(n, bool), pushed, speed, actions,
                                               actions, low, config, 0.1)
        names = [room.OUTCOMES[i] if i >= 0 else None for i in outcome]
        self.assertEqual(names, ["success", None, None, "object_pushed", None])
        w = config["reward"]
        self.assertAlmostEqual(reward[2], w["progress"]*0.1/0.1 - w["pose"]*0.0 - w["time"], places=9)   # 0.5 m is not "near"
        self.assertEqual(reward[4], 0.0)
        # Carrying: a cabinet back on the floor after the prefix fails.
        grounded = {**state, "bottom_m": np.zeros(n)}
        outcome = room.task_step(grounded, np.full(n, 0.6), np.zeros(n, bool), np.ones(n, bool), pushed, speed, actions, actions,
                                 low, self.carry_config, 0.1)[2]
        self.assertTrue(all(room.OUTCOMES[i] == "object_grounded" for i in outcome))

    def test_docking_follows_the_path_then_corrects_and_stops(self):
        obs = np.zeros((3, room.OBSERVATIONS))
        obs[:, 3] = 1.0                                       # yaw error 0
        obs[0, :2], obs[0, 6:8] = [2.0, 0.0], [0.6, 0.0]      # far: walk toward the look-ahead point
        obs[1, :2], obs[1, 2:4] = [0.1, -0.05], [np.sin(0.2), np.cos(0.2)]
        obs[2, :2] = [0.01, 0.0]                              # inside 80% of the 4 cm tolerance: stop
        commands = room.action_commands(room.docking_actions(obs, self.config), self.config["action_bounds"])
        np.testing.assert_allclose(commands, [[0.2, 0, 0], [0.08, -0.04, 0.24], [0, 0, 0]], atol=1e-12)

    def test_configuration_rejects_unsafe_or_overlapping_settings(self):
        raw = json.loads(APPROACH.read_text())
        cases = [{**raw, "extra": 1}, {**raw, "task": "dance"}, {**raw, "mass_kg": [2.0, 6.0]}, {**raw, "env_spacing_m": 5.0},
                 {**raw, "action_bounds": {**raw["action_bounds"], "forward_mps": [0.05, 0.3]}},
                 {**raw, "goal": {**raw["goal"], "position_tolerance_m": 0.5}},
                 {**raw, "room": {**raw["room"], "obstacles": [3, 1]}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "room.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    room.load_config(path)
        carry_raw = json.loads(CARRY.read_text())
        self.assertEqual({k: v for k, v in carry_raw.items() if k not in ("task", "room", "goal", "episode_seconds", "scope")},
                         {k: v for k, v in raw.items() if k not in ("task", "room", "goal", "episode_seconds", "scope")})


class FakeRunner:
    def __init__(self, env, cfg, log_dir, device):
        self.env, self.cfg = env, cfg
        assert env.get_observations()["critic"].value.shape == (env.num_envs, room.OBSERVATIONS + 1)

    def learn(self, num_learning_iterations, init_at_random_ep_len):
        self.env.episode_length_buf = Tensor(np.arange(self.env.num_envs) % 3 + self.env.max_episode_length - 3)
        for _ in range(num_learning_iterations*self.cfg["num_steps_per_env"]):
            self.env.step(Tensor(np.zeros((self.env.num_envs, 3))))

    def get_inference_policy(self, device):
        return lambda obs: Tensor(np.ones((len(obs["policy"].value), 3)))


class RoomWorkerTests(unittest.TestCase):
    def run_task(self, path):
        config, profile, asset, runtime = room.load_config(path)
        count = 2
        config = {**config, "num_envs": count, "episode_seconds": 4.0,
                  "ppo": {**config["ppo"], "max_iterations": 1, "num_steps_per_env": 2}}
        o = config["object"]
        scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": 2.0, "friction": o["friction"],
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        modules = {"tensordict": ModuleType("tensordict"), "rsl_rl": ModuleType("rsl_rl"),
                   "rsl_rl.runners": ModuleType("rsl_rl.runners")}
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        modules["rsl_rl.runners"].OnPolicyRunner = FakeRunner
        attached, walls = [], {}
        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(count, scene_profile, asset) as fake, \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(lift, "apply_hand_attachment", side_effect=lambda stage, r, b, env, f, on: attached.append((env, on)) or {}), \
                mock.patch.object(carry, "apply_corridor", side_effect=lambda stage, env, w: walls.update({env: w}) or []):
            sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
            fake.box.slide = lambda: np.zeros(count)
            room.run_worker(config, profile, asset, runtime, (Path("walk.onnx"), Path("balance.onnx")), Path(tmp))
            result = json.loads((Path(tmp) / "room-worker.json").read_text())
            progress = (Path(tmp) / "room-progress.jsonl").read_text().splitlines()
        return result, attached, walls, fake, progress

    def test_approach_worker_builds_rooms_without_attachments(self):
        result, attached, walls, fake, progress = self.run_task(APPROACH)
        # Hand collisions with the cabinet are filtered in both tasks; the joints stay disabled without load.
        self.assertEqual((result["status"], result["task"]), ("passed_room_training", "approach"))
        self.assertEqual(attached, [("/World/envs/env_0", False), ("/World/envs/env_1", False)])
        for i, env in enumerate(["/World/envs/env_0", "/World/envs/env_1"]):
            boxes = result["rooms"][i]["boxes"]
            self.assertEqual(len(walls[env]), 4 + len(boxes))
        self.assertAlmostEqual(result["object_in_root"][0][3], 0.411, places=2)
        for name in ("scripted_eval", "learned_eval"):
            self.assertEqual(result[name]["episodes"], 2)
            self.assertTrue(all(r["max_pushed_m"] < 1e-6 for r in result[name]["per_env"]))
        # The learned double commands the upper bounds: forward 0.3, lateral 0.15, yaw 0.3.
        commands = result["learned_eval"]["per_env"][0]["commands"]
        self.assertAlmostEqual(commands["forward_mean_mps"], 0.3)
        self.assertAlmostEqual(commands["lateral_abs_mean_mps"], 0.15)
        self.assertEqual(len(progress), 1)

    def test_carry_worker_attaches_and_holds_the_cabinet_at_the_start(self):
        result, attached, walls, fake, _ = self.run_task(CARRY)
        self.assertEqual((result["status"], attached), ("passed_room_training", [("/World/envs/env_0", True), ("/World/envs/env_1", True)]))
        relative = np.array(result["object_in_root"])
        from g1_wbc import pose_matrix
        for i in range(2):
            np.testing.assert_allclose(np.linalg.inv(pose_matrix(fake.robot.pose[i])) @ pose_matrix(fake.box.pose[i]), relative,
                                       atol=1e-9)
        self.assertEqual(result["learned_eval"]["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
