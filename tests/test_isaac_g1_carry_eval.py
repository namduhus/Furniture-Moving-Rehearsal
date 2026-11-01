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
import isaac_g1_carry_eval as ev
import isaac_g1_lift as lift
from test_isaac_g1_ppo_eval import Image, fake_encode
from test_isaac_g1_rl import Tensor, parallel_isaac


class LoadedRunner:
    loaded = []

    def __init__(self, env, cfg, log_dir, device):
        assert log_dir is None and env.get_observations()["policy"].value.shape == (env.num_envs, 23 + env.num_actions)

    def load(self, path, map_location):
        LoadedRunner.loaded.append(path)
        self.current_learning_iteration = 299

    def get_inference_policy(self, device):
        return lambda obs: Tensor(np.ones((len(obs["policy"].value), 2)))


class CarryEvaluationConfigTests(unittest.TestCase):
    def test_configuration_requires_held_out_corridors_and_showcases_inside_training_ranges(self):
        config, training, *_ = ev.load_config(ev.DEFAULT_CONFIG)
        self.assertNotEqual(config["held_out"]["corridor_seed"], training["seed"])
        raw = json.loads(ev.DEFAULT_CONFIG.read_text())
        case = raw["showcase"]["cases"][0]
        cases = [{**raw, "extra": 1},
                 {**raw, "checkpoint": "/tmp/model.pt"},
                 {**raw, "held_out": {**raw["held_out"], "corridor_seed": training["seed"]}},
                 {**raw, "showcase": {**raw["showcase"], "cases": [{**case, "width_m": 1.0}]}},
                 {**raw, "showcase": {**raw["showcase"], "cases": [{**case, "turn": "up"}]}},
                 {**raw, "showcase": {**raw["showcase"], "cases": [{**case, "mass_kg": 5.0}]}},
                 {**raw, "showcase": {**raw["showcase"], "cases": [case]*5}},
                 {**raw, "showcase": {**raw["showcase"], "env_spacing_m": 4.0}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    ev.load_config(path)

    def test_speed_limited_evaluation_reuses_the_unlimited_corridors(self):
        speed, training, *_ = ev.load_config(ROOT / "configs/isaac_g1_carry_speed_eval.json")
        base = ev.load_config(ev.DEFAULT_CONFIG)[0]
        self.assertEqual(training["reward"]["overspeed"], 5.0)
        self.assertNotEqual(speed["checkpoint"], base["checkpoint"])
        self.assertEqual((speed["held_out"], speed["showcase"]), (base["held_out"], base["showcase"]))
        heavy, heavy_training, *_ = ev.load_config(ROOT / "configs/isaac_g1_carry_speed_6kg_eval.json")
        self.assertEqual((heavy_training["mass_kg"], heavy["held_out"]), ([2.0, 6.0], base["held_out"]))
        self.assertEqual(max(c["mass_kg"] for c in heavy["showcase"]["cases"]), 5.8)

    def test_showcase_pairs_each_corridor_and_frames_it_from_its_corner(self):
        cases = [{"width_m": 1.2, "corner_m": 1.5, "turn": "left", "mass_kg": 3.0},
                 {"width_m": 1.4, "corner_m": 1.8, "turn": "right", "mass_kg": 2.5}]
        corridors, masses, learned = ev.showcase_layout(cases)
        np.testing.assert_allclose(corridors["width_m"], [1.2, 1.2, 1.4, 1.4])
        np.testing.assert_allclose(corridors["turn"], [1, 1, -1, -1])
        np.testing.assert_allclose(masses, [3.0, 3.0, 2.5, 2.5])
        self.assertEqual(learned.tolist(), [False, True, False, True])
        origins = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [12.0, 12.0, 0.0]])
        eyes, targets = ev.camera_views(origins, corridors, {"eye_from_corner_m": [-2.0, -4.0, 5.5],
                                                             "target_from_corner_m": [-0.8, 0.6, 0.3]})
        np.testing.assert_allclose(eyes[1], [12.0 + 1.5 - 2.0, -4.0, 5.5])
        np.testing.assert_allclose(eyes[2], [1.8 - 2.0, 12.0 + 4.0, 5.5])     # mirrored for the right turn
        np.testing.assert_allclose(targets[3], [12.0 + 1.8 - 0.8, 12.0 - 0.6, 0.3])


class CarryEvaluationWorkerTests(unittest.TestCase):
    def run_stage(self, stage, count):
        config, training, profile, asset, runtime = ev.load_config(ev.DEFAULT_CONFIG)
        config = {**config, "held_out": {**config["held_out"], "num_envs": count}}
        training = {**training, "episode_seconds": 5.0}
        o = training["object"]
        scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": 2.0, "friction": o["friction"],
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        modules = {"tensordict": ModuleType("tensordict"), "rsl_rl": ModuleType("rsl_rl"),
                   "rsl_rl.runners": ModuleType("rsl_rl.runners"), "PIL": ModuleType("PIL")}
        modules["PIL"].Image = Image
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        modules["rsl_rl.runners"].OnPolicyRunner = LoadedRunner
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        with parallel_isaac(count, scene_profile, asset) as fake, mock.patch.dict(sys.modules, modules), \
                mock.patch.object(lift, "apply_hand_attachment", return_value={}), \
                mock.patch.object(carry, "apply_corridor", side_effect=lambda stage, env, walls: ["wall"]*len(walls)), \
                mock.patch.object(ev, "file_sha256", return_value="f"*64), \
                mock.patch.object(ev, "require_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(ev, "encode_recording", side_effect=fake_encode):
            sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
            ev.run_worker(stage, config, training, profile, asset, runtime, (Path("walk.onnx"), Path("balance.onnx")),
                          directory)
        return json.loads((directory / "eval-worker.json").read_text()), directory, fake, config, training

    def test_held_out_stage_draws_new_corridors_and_evaluates_both_policies(self):
        result, _, fake, config, training = self.run_stage("held_out", 3)
        self.assertEqual((result["status"], result["checkpoint_iteration"]), ("passed_held_out", 299))
        self.assertEqual(LoadedRunner.loaded[-1], str(ROOT / config["checkpoint"]))
        rng = np.random.default_rng(config["held_out"]["corridor_seed"])
        corridors = carry.sample_corridors(training["corridor"], 3, rng)
        np.testing.assert_allclose(result["corridors"]["width_m"], corridors["width_m"])
        np.testing.assert_allclose(result["masses_kg"], rng.uniform(*training["mass_kg"], 3))
        np.testing.assert_allclose(fake.box_mass, result["masses_kg"])
        for name, forward in (("scripted_eval", 0.2), ("learned_eval", 0.3)):
            e = result[name]
            self.assertEqual((e["episodes"], e["seed"]), (3, config["held_out"]["start_seed"]))
            # The robot double never moves, so the logged walking speed is zero.
            self.assertEqual(e["walking_speed_mps"], {"mean": 0.0, "p95": 0.0, "max": 0.0, "fraction_above_limit": 0.0})
            self.assertTrue(all(r["commands"]["forward_max_mps"] == forward for r in e["per_env"]))

    def test_showcase_stage_records_pursuit_and_learned_runs_of_each_corridor(self):
        result, directory, fake, config, _ = self.run_stage("showcase", 8)
        self.assertEqual(result["status"], "passed_showcase")
        runs = result["showcase"]
        self.assertEqual([r["case"] for r in runs[:2]], ["case_0_pursuit", "case_0_learned"])
        self.assertEqual([r["policy"] for r in runs], ["pursuit", "learned"]*4)
        self.assertEqual([r["turn"] for r in runs[::2]], [c["turn"] for c in config["showcase"]["cases"]])
        for run in runs:
            trace = [json.loads(line) for line in (directory / run["case"] / "trace.jsonl").read_text().splitlines()]
            # One initial frame plus one per decision until the 5 s episode times out; 25 lift-prefix decisions.
            self.assertEqual((len(trace), run["frames"], run["outcome"]), (50, 51, "time_out"))
            self.assertEqual(sum(t["lift_prefix"] for t in trace), 25)
            self.assertEqual(trace[-1]["command"]["forward_mps"], 0.3 if run["policy"] == "learned" else 0.2)
        np.testing.assert_allclose(fake.camera.eyes[:, 2], 5.5)
        np.testing.assert_allclose(fake.camera.eyes[1, :2] - fake.origins[1, :2], [1.6 - 2.0, -4.0])
        np.testing.assert_allclose(fake.camera.eyes[2, :2] - fake.origins[2, :2], [1.4 - 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
