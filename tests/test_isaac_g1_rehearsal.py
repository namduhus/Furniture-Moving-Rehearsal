import json
from contextlib import nullcontext
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
import isaac_g1_ppo_eval as ppo_eval
import isaac_g1_rehearsal as rehearsal
import isaac_g1_room as room
import room_geometry as rg
from test_isaac_g1_lift import Image
from test_isaac_g1_rl import parallel_isaac


class RehearsalLayoutTests(unittest.TestCase):
    def setUp(self):
        self.config, self.trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        self.request, self.scene = rehearsal.load_request(ROOT / self.config["request"])

    def layout(self, request=None, scene=None):
        return rehearsal.rehearsal_layout(scene or self.scene, request or self.request, self.trainings, 0.6, 0.411)

    def test_scene_spec_becomes_a_centred_room_with_a_reachable_cabinet_and_destination(self):
        # The spec frame (x right, y toward the camera) is left-handed; the simulation flips y, so spec yaw 0 (facing
        # the camera) faces sim -y and the robot stands in front facing back (+y); 90 faces +x.
        yaws = [rehearsal.facing_yaw(v) for v in (0.0, 90.0, -90.0)]
        np.testing.assert_allclose(np.c_[np.cos(yaws), np.sin(yaws)], [[0, 1], [-1, 0], [1, 0]], atol=1e-12)
        layout = self.layout()
        r = layout["room"]
        self.assertEqual((r["width_m"], r["depth_m"]), (3.3, 4.0))
        # The armchair (yaw 90: 0.85 m along x, 0.95 m along y), the TV bench and the TV; the target is not an obstacle.
        np.testing.assert_allclose(r["boxes"][0], [0.6 - 1.65, 2.0 - 1.5, 0.85, 0.95, 0.75], atol=1e-12)
        self.assertEqual(len(r["boxes"]), 3)
        # The pedestal (spec (3.0, 2.5), facing -x) is grasped from its -x side; the robot starts 0.6 m inside the
        # right-wall door (1.1 m from the back wall) facing into the room.
        np.testing.assert_allclose(layout["cabinet"], [1.35, -0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(layout["approach_goal"], [1.35 - 0.411, -0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(layout["start"], [1.65 - 0.6, 2.0 - 1.1, np.pi], atol=1e-12)
        np.testing.assert_allclose(layout["placed"][:2], [0.5 - 1.65, 2.0 - 3.0], atol=1e-12)
        np.testing.assert_allclose(layout["carry_goal"][:2], [-1.15 + 0.411, -1.0], atol=1e-12)
        self.assertAlmostEqual(abs(layout["placed"][2]), np.pi)
        for path, start, goal in ((layout["approach_path"], layout["start"], layout["approach_goal"]),
                                  (layout["carry_path"], layout["approach_goal"], layout["carry_goal"])):
            np.testing.assert_allclose((path[0], path[-1]), (start[:2], goal[:2]))
        # Back to the spec frame: the pedestal centre and the destination round-trip.
        np.testing.assert_allclose(rehearsal.to_spec(layout["cabinet"][:2], layout["spec_transform"]), [3.0, 2.5], atol=1e-12)
        np.testing.assert_allclose(rehearsal.to_spec(layout["placed"][:2], layout["spec_transform"]), [0.5, 3.0], atol=1e-12)

    def test_requests_outside_the_carry_tier_or_into_furniture_are_rejected(self):
        r, fix = self.request, self.request["corrections"]
        moved = json.loads(json.dumps(self.scene))
        moved["scene"]["furniture"][-1]["center_m"] = [1.65, 0.5]          # the pedestal inside the TV bench
        for request, scene, message in (({**r, "corrections": {**fix, "mass_kg": 6.0}}, None, "push branch"),
                                        ({**r, "corrections": {**fix, "size_m": [0.7, 0.45, 0.95]}}, None, "push branch"),
                                        ({**r, "target": "sofa_9"}, None, "Unknown target"),
                                        ({**r, "destination": {"center_m": [0.6, 1.5], "yaw_deg": 0.0}}, None, "destination"),
                                        (r, moved, "overlaps")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.layout(request, scene)

    def test_placement_error_and_configuration_checks(self):
        yaw = np.radians(170.0)
        error = rehearsal.placement_error([0.03, 1.04, 0.45, 0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)], np.array([0.0, 1.0, -np.pi]))
        self.assertAlmostEqual(error["position_m"], 0.05)
        self.assertAlmostEqual(error["yaw_deg"], 10.0)
        raw = json.loads(rehearsal.DEFAULT_CONFIG.read_text())
        cases = [{**raw, "extra": 1}, {**raw, "carry_training": raw["approach_training"]},
                 {**raw, "approach_checkpoint": "../model.pt"}, {**raw, "timeline_seconds": {"reach": 1.0}},
                 {**raw, "back_away_mps": 0.5}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rehearsal.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    rehearsal.load_config(path)


class RehearsalWorkerTests(unittest.TestCase):
    def test_worker_chains_approach_grasp_carry_and_set_down(self):
        config, trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        request, scene = rehearsal.load_request(ROOT / config["request"])
        approach, profile, asset, runtime = trainings["approach"]
        o = approach["object"]
        scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": 3.0, "friction": o["friction"],
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        real = rehearsal.rehearsal_layout

        def docked(*args):
            # The doubles do not walk: the robot starts at the pre-grasp pose, and the destination is where the cabinet
            # stands, so both walking phases succeed at once and the placement error is zero.
            layout = real(*args)
            here = layout["approach_goal"].copy()
            return {**layout, "start": here, "approach_path": np.array([here[:2]]*2), "carry_goal": here,
                    "carry_path": np.array([here[:2]]*2), "placed": layout["cabinet"].copy()}

        modules = {"PIL": ModuleType("PIL"), "tensordict": ModuleType("tensordict")}
        modules["PIL"].Image = Image
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        toggles, walls = [], []

        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(1, scene_profile, asset) as fake, \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(rehearsal, "rehearsal_layout", side_effect=docked), \
                mock.patch.object(lift, "apply_hand_attachment", return_value={}), \
                mock.patch.object(carry, "apply_corridor", side_effect=lambda stage, env, w: walls.extend(w) or w), \
                mock.patch.object(rehearsal, "require_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(rehearsal, "encode_recording", return_value={"status": "passed_video_decode"}) as encode:
            sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
            fake.box.slide = lambda: np.zeros(1)

            def attach(stage, env, on, held=None):
                # The double lifts the cabinet 5 cm while the joints are enabled and drops it at the release.
                fake.box.pose[0, 2] += 0.05 if on else -0.05
                toggles.append((env, on, held))
                return True

            # Both checkpoints load through the evaluation loader; the double is the zero residual.
            loaded = mock.patch.object(ppo_eval, "load_policy",
                                       side_effect=lambda env, training, path, device: (room.scripted_actor(training), 299))
            with mock.patch.object(lift, "set_attachment", side_effect=attach), loaded as load:
                directory = Path(tmp)
                rehearsal.run_worker(config, trainings, request, scene, (Path("walk.onnx"), Path("balance.onnx")), directory)
            result = json.loads((directory / "rehearsal-worker.json").read_text())
            replay = json.loads((directory / "replay.json").read_text())
        self.assertEqual(result["status"], "completed_rehearsal")
        self.assertEqual([call.args[2] for call in load.call_args_list],
                         [ROOT / config["approach_checkpoint"], ROOT / config["carry_checkpoint"]])
        self.assertEqual((result["phases"]["approach_checkpoint_iteration"], result["phases"]["carry_checkpoint_iteration"]), (299, 299))
        # Four walls plus the three spec furniture boxes; joints enabled with re-anchored frames, then disabled.
        self.assertEqual(result["walls"], 7)
        self.assertEqual([(on, held is None) for _, on, held in toggles], [(True, False), (False, True)])
        self.assertEqual((result["phases"]["approach"]["outcome"], result["phases"]["carry"]["outcome"]), ("success", "success"))
        # The approach hands over after the 1 s settle prefix (10 decisions) and succeeds on the first policy decision.
        self.assertAlmostEqual(result["phases"]["approach"]["seconds"], 1.1)
        self.assertLess(result["placement"]["position_m"], 1e-9)
        self.assertTrue(result["placement"]["within_tolerance"])
        np.testing.assert_allclose(result["placement"]["cabinet_spec_xy_m"], [3.0, 2.5], atol=1e-9)
        # One frame per decision: start, approach 11, reach 10, grasp 5, lift 10, hold 5, carry 1, lower 15, release 5,
        # withdraw 10, back away 15.
        phases = [frame["phase"] for frame in replay["frames"]]
        self.assertEqual(len(phases), 88)
        self.assertEqual(encode.call_args.kwargs["count"], 88)
        self.assertEqual([phases.count(p) for p in ("approach", "reach", "carry", "back_away")], [11, 10, 1, 15])
        self.assertEqual(len(replay["frames"][0]["bodies"]), len(replay["bodies"]))
        self.assertEqual(replay["frames"][-1]["command"], [-0.1, 0.0, 0.0])
        # The reach measured each palm against its grasp point on the floor-standing cabinet.
        self.assertEqual(set(result["reach"][0]["offset_m"]), {"left", "right"})
        self.assertIsNotNone(result["grasp"]["palm_gap_m"])


if __name__ == "__main__":
    unittest.main()
