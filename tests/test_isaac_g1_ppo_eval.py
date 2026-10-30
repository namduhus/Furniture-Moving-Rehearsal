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
import isaac_g1_ppo_eval as ev
from test_isaac_g1_rl import Tensor, parallel_isaac


class LoadedRunner:
    loaded = []

    def __init__(self, env, cfg, log_dir, device):
        assert log_dir is None and env.get_observations()["policy"].value.shape == (env.num_envs, 21)

    def load(self, path, map_location):
        LoadedRunner.loaded.append(path)
        self.current_learning_iteration = 99

    def get_inference_policy(self, device):
        return lambda obs: Tensor(np.ones((len(obs["policy"].value), 3)))


class Image:
    """PIL double: records the frame shape so the test checks what would be written."""
    @staticmethod
    def fromarray(array):
        assert array.shape == (360, 640, 3)
        return Image()
    def save(self, path): Path(path).write_bytes(b"frame")


def fake_encode(directory, *, binary, count, fps, width, height, require_motion):
    assert len(list((directory / "frames").glob("*.png"))) == count and fps == 10
    return {"status": "passed_video_decode", "decoded_frames": count, "sha256": "0"*64}


class EvaluationConfigTests(unittest.TestCase):
    def test_configuration_requires_held_out_boxes_inside_training_ranges(self):
        config, training, *_ = ev.load_config(ev.DEFAULT_CONFIG)
        self.assertNotEqual(config["held_out"]["box_seed"], training["seed"])
        raw = json.loads(ev.DEFAULT_CONFIG.read_text())
        cases = [{**raw, "extra": 1},
                 {**raw, "checkpoint": "/tmp/model.pt"},
                 {**raw, "checkpoint": "outputs/isaac/../../model.pt"},
                 {**raw, "held_out": {**raw["held_out"], "box_seed": training["seed"]}},
                 {**raw, "showcase": {**raw["showcase"], "cases": [{"mass_kg": 25.0, "friction": 0.1}]}},
                 {**raw, "showcase": {**raw["showcase"], "env_spacing_m": 2.0}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    ev.load_config(path)


class EvaluationWorkerTests(unittest.TestCase):
    def run_stage(self, stage, count):
        config, training, profile, asset, runtime = ev.load_config(ev.DEFAULT_CONFIG)
        config = {**config, "held_out": {**config["held_out"], "num_envs": count}}
        training = {**training, "episode_seconds": 1.0}
        modules = {"tensordict": ModuleType("tensordict"), "rsl_rl": ModuleType("rsl_rl"),
                   "rsl_rl.runners": ModuleType("rsl_rl.runners"), "PIL": ModuleType("PIL")}
        modules["PIL"].Image = Image
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        modules["rsl_rl.runners"].OnPolicyRunner = LoadedRunner
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        with parallel_isaac(count, profile, asset) as fake, mock.patch.dict(sys.modules, modules), \
                mock.patch.object(ev, "file_sha256", return_value="f"*64), \
                mock.patch.object(ev, "require_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(ev, "encode_recording", side_effect=fake_encode):
            sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
            ev.run_worker(stage, config, training, profile, asset, runtime, Path("unused.onnx"), directory)
        return json.loads((directory / "eval-worker.json").read_text()), directory, fake, config

    def test_held_out_stage_uses_new_box_seed_and_records_commands(self):
        result, _, fake, config = self.run_stage("held_out", 3)
        self.assertEqual(result["status"], "passed_held_out")
        self.assertEqual(result["checkpoint_iteration"], 99)
        self.assertEqual(LoadedRunner.loaded[-1], str(ROOT / config["checkpoint"]))
        expected = np.random.default_rng(config["held_out"]["box_seed"]).uniform(5.0, 20.0, 3)
        np.testing.assert_allclose(result["box_properties"]["masses_kg"], expected)
        np.testing.assert_allclose(fake.box_mass, expected)
        self.assertEqual(result["scripted_eval"]["forward_command"]["fraction_above_reviewed_0_2_mps"], 0.0)
        self.assertEqual(result["learned_eval"]["forward_command"]["max_mps"], 0.4)
        command = result["learned_eval"]["per_env"][0]["commands"]
        self.assertEqual((command["decisions"], command["hand_push_mean_during_contact_n"]), (10, 6.0))

    def test_showcase_stage_records_one_video_and_trace_per_box(self):
        result, directory, fake, config = self.run_stage("showcase", 5)
        self.assertEqual(result["status"], "passed_showcase")
        cases = result["showcase"]
        self.assertEqual([c["mass_kg"] for c in cases], [c["mass_kg"] for c in config["showcase"]["cases"]])
        self.assertAlmostEqual(cases[4]["slide_force_n"], 20.0*0.15*9.81)
        for case in cases:
            trace = (directory / case["case"] / "trace.jsonl").read_text().splitlines()
            # One initial frame plus one per decision until the 1 s episode times out.
            self.assertEqual((len(trace), case["frames"], case["outcome"]), (10, 11, "time_out"))
            self.assertEqual(json.loads(trace[-1])["command"]["forward_mps"], 0.4)
        np.testing.assert_allclose(fake.camera.eyes[:, 1] - fake.origins[:, 1], -3.8)


if __name__ == "__main__":
    unittest.main()
