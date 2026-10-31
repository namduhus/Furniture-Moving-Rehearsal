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
import isaac_g1_carry as carry
import isaac_g1_lift as lift
from test_isaac_g1_rl import Tensor, parallel_isaac


def yaw_poses(xyz, yaws):
    yaws = np.asarray(yaws, dtype=float)
    return np.c_[np.asarray(xyz, dtype=float), np.zeros(len(yaws)), np.zeros(len(yaws)), np.sin(yaws/2), np.cos(yaws/2)]


class CorridorGeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = carry.load_config(carry.DEFAULT_CONFIG)[0]
        self.settings = self.config["corridor"]

    def distance(self, points, turn, width=1.2, corner=1.5):
        xy = carry.mirrored(np.asarray(points, dtype=float)[None], [turn])
        return carry.inside_distance(xy, [width], [corner], self.settings["back_m"])[0]

    def test_walls_bound_the_free_region_for_both_turns(self):
        t = self.settings["wall_thickness_m"]
        for turn in (1.0, -1.0):
            walls = carry.corridor_walls(1.2, 1.5, turn, self.settings)
            self.assertEqual(len(walls), 5)
            # Every wall centre lies half a thickness behind the free-space boundary.
            np.testing.assert_allclose(self.distance([c[:2] for c, _ in walls], turn), -t/2, atol=1e-12)
            free = np.array([[0.7, 0.0],            # first leg centre line
                             [2.09, -0.58],         # outer corner, 1 cm from the facing wall
                             [0.85, 0.65],          # inside the inner corner's walls
                             [1.5, 2.5],            # past the open end of the second leg
                             [-0.7, 0.0]])          # behind the start wall
            free[:, 1] *= turn
            np.testing.assert_allclose(self.distance(free, turn), [0.6, 0.01, -0.05, 0.6, -0.1], atol=1e-12)

    def test_state_follows_the_centre_line_in_the_robot_frame(self):
        corridors = {"width_m": np.array([1.2, 1.2]), "corner_m": np.array([1.2, 1.2]), "turn": np.array([1.0, -1.0])}
        origins = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        size, radius = self.config["object"]["size_m"], 0.08
        # At the start, facing +x: the third look-ahead point turns into the second leg.
        robot = yaw_poses(origins + [0, 0, 0.75], [0.0, 0.0])
        cabinet = yaw_poses(origins + [0.7, 0, 0.5], [0.0, 0.0])
        bodies = np.stack([origins + [0.0, 0.5, 0.8], origins + [0.1, 0.0, 0.8]], axis=1)
        state = carry.corridor_state(robot, cabinet, bodies, origins, corridors, self.settings, size, radius)
        np.testing.assert_allclose(state["lookahead"], [[0.5, 0, 1.0, 0, 1.2, 0.3], [0.5, 0, 1.0, 0, 1.2, -0.3]], atol=1e-12)
        np.testing.assert_allclose(state["progress_m"], [0.7, 0.7])
        np.testing.assert_allclose(state["goal_m"], [2.2, 2.2])
        np.testing.assert_allclose(state["object_clearance_m"], 0.6 - size[1]/2)
        # A link 0.5 m to the left is 0.1 m from the wall less the link radius (left wall of a right turn is the outer one).
        np.testing.assert_allclose(state["robot_clearance_m"], [0.1 - radius, 0.1 - radius])
        np.testing.assert_allclose(state["bottom_m"], 0.5 - size[2]/2)
        np.testing.assert_allclose(state["tilt_rad"], 0.0, atol=1e-7)
        # In the second leg, facing along it: points straight ahead; the cabinet is 0.2 m short of the goal line.
        robot = yaw_poses(origins + [[1.2, 0.5, 0.75], [1.2, -0.5, 0.75]], [np.pi/2, -np.pi/2])
        cabinet = yaw_poses(origins + [[1.2, 0.8, 0.5], [1.2, -0.8, 0.5]], [np.pi/2, -np.pi/2])
        state = carry.corridor_state(robot, cabinet, bodies, origins, corridors, self.settings, size, radius)
        np.testing.assert_allclose(state["lookahead"], [[0.5, 0, 1.0, 0, 1.5, 0]]*2, atol=1e-12)
        np.testing.assert_allclose(state["goal_m"] - state["progress_m"], [0.2, 0.2])

    def test_pursuit_steers_toward_the_look_ahead_point(self):
        obs = np.zeros((3, carry.OBSERVATIONS))
        obs[:, 2:4] = [[1.0, 0.0], [1.0, 1.0], [0.0, -1.0]]
        commands = carry.action_commands(carry.pursuit_actions(obs, self.config), self.config["action_bounds"])
        np.testing.assert_allclose(commands, [[0.2, 0, 0.0], [0.2, 0, 0.3], [0.2, 0, -0.3]], atol=1e-12)


class TaskStepTests(unittest.TestCase):
    def test_outcomes_follow_the_first_failure_and_the_prefix(self):
        config = carry.load_config(carry.DEFAULT_CONFIG)[0]
        n = 7
        state = {"progress_m": np.full(n, 1.0), "goal_m": np.full(n, 2.2), "object_clearance_m": np.full(n, 0.3),
                 "robot_clearance_m": np.full(n, 0.3), "bottom_m": np.full(n, 0.05), "tilt_rad": np.full(n, 0.1)}
        state["progress_m"][1] = 2.25                     # success
        state["object_clearance_m"][2] = 0.005            # cabinet within the 1 cm contact distance
        state["tilt_rad"][3] = 0.4                        # tipped
        state["bottom_m"][4] = 0.0                        # back on the floor after the prefix
        state["bottom_m"][5] = 0.0                        # on the floor during the prefix: allowed
        low = {"invalid_action": np.array([0, 0, 0, 0, 0, 0, 1], bool), "fallen": np.zeros(n, bool)}
        prefix = np.array([0, 0, 0, 0, 0, 1, 0], bool)
        actions, last = np.zeros((n, 2)), np.zeros((n, 2))
        actions[0] = [0.5, 0.0]
        reward, done, outcome = carry.task_step(state, np.full(n, 0.99), prefix, actions, last, low, config, 0.1)
        names = [carry.OUTCOMES[i] if i >= 0 else None for i in outcome]
        self.assertEqual(names, [None, "success", "wall_contact", "object_tipped", "object_grounded", None, "invalid_action"])
        self.assertEqual(done.tolist(), [False, True, True, True, True, False, True])
        w = config["reward"]
        self.assertAlmostEqual(reward[0], w["progress"]*0.01/0.1 - w["action_rate"]*0.25, places=9)
        self.assertAlmostEqual(reward[1], w["progress"]*1.26/0.1 + w["success"], places=9)
        # Inside the safe clearance the wall term grows linearly; contact also costs the failure penalty.
        self.assertAlmostEqual(reward[2], w["progress"]*0.01/0.1 - w["wall"]*0.95 - w["failure"], places=9)
        self.assertEqual(reward[5], 0.0)

    def test_configuration_rejects_unreviewed_masses_commands_and_overlap(self):
        raw = json.loads(carry.DEFAULT_CONFIG.read_text())
        cases = [{**raw, "extra": 1},
                 {**raw, "mass_kg": [2.0, 5.0]},
                 {**raw, "env_spacing_m": 4.0},
                 {**raw, "action_bounds": {**raw["action_bounds"], "forward_mps": [-0.1, 0.3]}},
                 {**raw, "scripted": {**raw["scripted"], "forward_mps": 0.5}},
                 {**raw, "corridor": {**raw["corridor"], "leg_two_m": 1.2}},
                 {**raw, "ppo": {**raw["ppo"], "hidden_dims": []}},
                 {**raw, "push_profile": "configs/isaac_g1_crouch_start_forward.json"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "carry.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    carry.load_config(path)
        # The smoke check runs the training setup unchanged except for the iteration count and its timeout.
        smoke = carry.load_config(ROOT / "configs/isaac_g1_carry_smoke.json")[0]
        changed = ("ppo", "stage_timeout_seconds", "scope")
        self.assertEqual({k: v for k, v in smoke.items() if k not in changed}, {k: v for k, v in raw.items() if k not in changed})
        self.assertEqual({k: v for k, v in smoke["ppo"].items() if k not in ("max_iterations", "save_interval")},
                         {k: v for k, v in raw["ppo"].items() if k not in ("max_iterations", "save_interval")})
        self.assertEqual(smoke["ppo"]["max_iterations"], 3)


class FakeRunner:
    def __init__(self, env, cfg, log_dir, device):
        self.env, self.cfg = env, cfg
        assert env.get_observations()["critic"].value.shape == (env.num_envs, carry.OBSERVATIONS + 1)

    def learn(self, num_learning_iterations, init_at_random_ep_len):
        self.env.episode_length_buf = Tensor(np.arange(self.env.num_envs) % 3 + self.env.max_episode_length - 3)
        for _ in range(num_learning_iterations*self.cfg["num_steps_per_env"]):
            _, reward, dones, extras = self.env.step(Tensor(np.zeros((self.env.num_envs, 2))))
            assert reward.value.shape == dones.value.shape == extras["time_outs"].value.shape == (self.env.num_envs,)

    def get_inference_policy(self, device):
        return lambda obs: Tensor(np.ones((len(obs["policy"].value), 2)))


class CarryWorkerTests(unittest.TestCase):
    def test_worker_builds_corridors_lifts_and_evaluates_both_policies(self):
        config, profile, asset, runtime = carry.load_config(carry.DEFAULT_CONFIG)
        count = 4
        config = {**config, "num_envs": count, "episode_seconds": 5.0,
                  "ppo": {**config["ppo"], "max_iterations": 2, "num_steps_per_env": 3}}
        o = config["object"]
        scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": 2.0, "friction": o["friction"],
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        modules = {"tensordict": ModuleType("tensordict"), "rsl_rl": ModuleType("rsl_rl"),
                   "rsl_rl.runners": ModuleType("rsl_rl.runners")}
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        modules["rsl_rl.runners"].OnPolicyRunner = FakeRunner
        attached, walls, targets = [], {}, []
        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(count, scene_profile, asset) as fake, \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(lift, "apply_hand_attachment", side_effect=lambda stage, r, b, env, f: attached.append(env) or {}), \
                mock.patch.object(carry, "apply_corridor", side_effect=lambda stage, env, w: walls.update({env: w}) or ["wall"]*len(w)):
            sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
            setter = fake.robot.set_joint_position_target_index
            fake.robot.set_joint_position_target_index = lambda *, target: targets.append(target.value.copy()) or setter(target=target)
            directory = Path(tmp)
            carry.run_worker(config, profile, asset, runtime, (Path("walk.onnx"), Path("balance.onnx")), directory)
            result = json.loads((directory / "carry-worker.json").read_text())
            progress = [json.loads(line) for line in (directory / "carry-progress.jsonl").read_text().splitlines()]
            reference = lift.trial_contract(profile)[1]
            arm_ids = [fake.robot.joint_names.index(n) for side in lift.SIDES for n in lift.arm_joints(reference, side)]
        self.assertEqual(result["status"], "passed_carry_training")
        envs = [f"/World/envs/env_{i}" for i in range(count)]
        self.assertEqual((attached, result["authored"]["environments"]), (envs, envs))
        corridors = result["corridors"]
        self.assertEqual(corridors["turn"], [1.0, -1.0, 1.0, -1.0])
        for i, env in enumerate(envs):
            wanted = carry.corridor_walls(corridors["width_m"][i], corridors["corner_m"][i], corridors["turn"][i], config["corridor"])
            self.assertEqual(json.loads(json.dumps(walls[env])), json.loads(json.dumps(wanted)))
        np.testing.assert_allclose(fake.box_mass, result["masses_kg"])
        self.assertTrue(all(2.0 <= m <= 4.0 for m in result["masses_kg"]))
        start = result["start"]
        self.assertLess(start["palm_frame_mismatch_max_m"], 1e-6)
        self.assertLess(start["relative_pose_spread"], 1e-9)
        self.assertGreater(start["wall_clearance_min_m"], 0.0)
        # After the last reset (randomized heading and offset) every cabinet still sits where the palm joints hold it.
        for i in range(count):
            np.testing.assert_allclose(np.linalg.inv(wbc.pose_matrix(fake.robot.pose[i])) @ wbc.pose_matrix(fake.box.pose[i]),
                                       start["object_in_root"], atol=1e-9)
        # 5 s episodes: 25 prefix decisions under Balance, then 25 policy decisions until the time-out.
        for name, forward in (("scripted_eval", 0.2), ("learned_eval", 0.3)):
            e = result[name]
            self.assertEqual((e["episodes"], e["outcome_rates"]["time_out"]), (count, 1.0))
            self.assertEqual(sum(g["episodes"] for g in (e["groups"]["left"], e["groups"]["right"])), count)
            for record in e["per_env"]:
                self.assertEqual(record["commands"]["decisions"], 25)
                self.assertAlmostEqual(record["commands"]["forward_max_mps"], forward)
                self.assertGreater(record["progress_m"], 0.1)       # the box double slides along x
        self.assertEqual([p["iteration"] for p in progress], [0, 1])
        self.assertEqual(set(progress[0]["outcome_rates"]), set(carry.OUTCOMES))
        self.assertEqual(result["training"]["decisions"], count*2*3)
        self.assertEqual({role for role, _ in fake.policy_calls}, {"walk", "balance"})
        # The final control step holds the lift pose; the time-out reset that follows restores the start targets.
        lifted = [result["arm_lift_rad"][fake.robot.joint_names[k]] for k in arm_ids]
        np.testing.assert_allclose(targets[-2][:, arm_ids], np.tile(lifted, (count, 1)))
        np.testing.assert_allclose(targets[-1][:, arm_ids], 0.0)


if __name__ == "__main__":
    unittest.main()
