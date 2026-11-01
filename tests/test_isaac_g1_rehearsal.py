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
from test_isaac_g1_rl import Tensor, parallel_isaac

# A 4 x 4 m room with one 0.8 m wide, 1.0 m tall storage cabinet facing +x; the request slides it 0.5 m along +x.
PUSH_SCENE = {"scene": {"room": {"width_m": 4.0, "depth_m": 4.0, "height_m": 2.5},
                        "door": {"detection_id": "door_0", "wall": "right", "center_m": 0.9},
                        "furniture": [{"id": "cabinet_0", "category": "storage cabinet", "detection_id": "cabinet_0",
                                       "size_m": [0.8, 0.6, 1.0], "center_m": [1.9, 2.0], "yaw_deg": 90.0,
                                       "estimated_mass_kg": 30.0, "mass_basis": "solid wood"}]}}
# A 0.8 m wide, 0.3 m deep, 1.0 m tall bookcase facing -x: grippable across its depth from an end, pushable on its front.
AUTO_SCENE = {"scene": {**PUSH_SCENE["scene"], "furniture": [
    {"id": "bookcase_0", "category": "bookcase", "detection_id": "bookcase_0", "size_m": [0.8, 0.3, 1.0],
     "center_m": [1.6, 2.0], "yaw_deg": -90.0, "estimated_mass_kg": 25.0, "mass_basis": "solid wood"}]}}
AUTO_REQUEST = {"scene": "configs/scenes/auto.json", "target": "bookcase_0",
                "corrections": {"size_m": [0.8, 0.3, 1.0], "mass_kg": 12.0},
                "destination": {"center_m": [2.1, 2.0], "yaw_deg": -90.0}, "strategy": "auto"}
PUSH_REQUEST = {"scene": "configs/scenes/push.json", "target": "cabinet_0",
                "corrections": {"size_m": [0.8, 0.6, 1.0], "mass_kg": 12.0},
                "destination": {"center_m": [2.4, 2.0], "yaw_deg": 90.0}, "strategy": "push"}


class RehearsalLayoutTests(unittest.TestCase):
    def setUp(self):
        self.config, self.trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        self.request, self.scene = rehearsal.load_request(ROOT / self.config["request"])

    def layout(self, request=None, scene=None):
        return rehearsal.rehearsal_layout(scene or self.scene, request or self.request, self.trainings, 0.6, 0.411,
                                          self.config["push"])

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
                                        ({**r, "corrections": {**fix, "size_m": [0.45, 0.7, 0.95]}}, None, "push branch"),
                                        ({**r, "target": "sofa_9"}, None, "Unknown target"),
                                        ({**r, "destination": {"center_m": [0.6, 1.5], "yaw_deg": 0.0}}, None, "destination"),
                                        (r, moved, "overlaps")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.layout(request, scene)

    def test_push_layout_puts_the_robot_behind_the_pushed_face(self):
        layout = self.layout(PUSH_REQUEST, PUSH_SCENE)
        # Spec size [width 0.8, depth 0.6, height 1.0] -> sim [0.6, 0.8, 1.0]; facing +x, so the 0.6 m depth lies along
        # the push and the palms meet the 0.8 m face, as in push training (0.7 x 0.8 x 1.0 m).
        self.assertEqual(layout["size_m"], [0.6, 0.8, 1.0])
        np.testing.assert_allclose(layout["cabinet"][:2], [-0.1, 0.0], atol=1e-12)
        self.assertAlmostEqual(layout["push"]["distance_m"], 0.5)
        self.assertEqual((layout["push"]["along_m"], layout["push"]["face_m"]), (0.6, 0.8))
        np.testing.assert_allclose(layout["approach_goal"], [-0.1 - (0.3 + 0.09 + 0.415), 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(layout["approach_path"][-1], layout["approach_goal"][:2])
        self.assertNotIn("carry_goal", layout)
        bad = (({"destination": {"center_m": [2.4, 2.3], "yaw_deg": 90.0}}, "one side"),       # diagonal
               ({"destination": {"center_m": [2.4, 2.0], "yaw_deg": 60.0}}, "heading"),
               ({"destination": {"center_m": [3.5, 2.0], "yaw_deg": 90.0}}, "moves the item"),   # 1.6 m
               ({"corrections": {"size_m": [0.8, 0.6, 1.0], "mass_kg": 30.0}}, "push tier"),
               ({"corrections": {"size_m": [0.3, 0.6, 1.0], "mass_kg": 12.0}}, "narrower"))
        for change, message in bad:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.layout({**PUSH_REQUEST, **change}, PUSH_SCENE)

    def test_auto_layout_grips_an_end_and_plans_the_push_of_the_front(self):
        layout = self.layout(AUTO_REQUEST, AUTO_SCENE)
        # Gripped across its 0.3 m depth from the end nearer the door (+y): box x along the 0.8 m width, reach + 0.25 m.
        self.assertEqual(layout["size_m"], [0.8, 0.3, 1.0])
        np.testing.assert_allclose(layout["cabinet"], [-0.4, 0.0, -np.pi/2], atol=1e-12)
        self.assertAlmostEqual(layout["grip_reach_m"], 0.411 + 0.25)
        np.testing.assert_allclose(layout["approach_goal"], [-0.4, 0.661, -np.pi/2], atol=1e-12)
        # The push then slides the 0.8 m front face 0.5 m along +x.
        self.assertEqual((layout["push"]["along_m"], layout["push"]["face_m"]), (0.3, 0.8))
        np.testing.assert_allclose(layout["push_goal"], [-0.4 - (0.15 + 0.09 + 0.415), 0.0, 0.0], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "grip"):
            self.layout({**AUTO_REQUEST, "corrections": {"size_m": [0.8, 0.6, 1.0], "mass_kg": 12.0}}, AUTO_SCENE)

    def test_course_frame_puts_the_push_along_x(self):
        yaw = np.radians(93.0)
        # Item at (1, 2) turned 93 deg, pushed along +y (theta 90 deg): in the course frame it sits at the origin, and its
        # yaw folds to +3 deg; a point 0.5 m further along +y is 0.5 m along course x.
        box, velocity = rehearsal.course_frame([1.0, 2.0, 0.5, 0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)],
                                               [0.0, 0.2, 0.0, 0.0, 0.0, 0.1], np.array([1.0, 2.0]), np.pi/2)
        np.testing.assert_allclose(box[:3], [0.0, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(velocity, [0.2, 0.0, 0.0, 0.0, 0.0, 0.1], atol=1e-12)
        folded = rehearsal.square_yaw(rehearsal.course_frame([1.0, 2.5, 0.5, 0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)],
                                                             np.zeros(6), np.array([1.0, 2.0]), np.pi/2)[0])
        np.testing.assert_allclose(folded[:2], [0.5, 0.0], atol=1e-12)
        self.assertAlmostEqual(float(rehearsal.ppo.yaw_of(rehearsal.square_yaw(box)[None, 3:])[0]), np.radians(3.0), places=9)
        # Tilt survives the fold.
        tilted = rehearsal.square_yaw(np.r_[0.0, 0.0, 0.5, np.sin(0.05), 0.0, 0.0, np.cos(0.05)])
        np.testing.assert_allclose(tilted, [0.0, 0.0, 0.5, np.sin(0.05), 0.0, 0.0, np.cos(0.05)], atol=1e-12)

    def test_placement_goal_puts_the_held_cabinet_on_the_requested_pose(self):
        from g1_wbc import pose_matrix
        yaw = np.radians(8.0)
        held = pose_matrix([0.38, -0.02, -0.29, 0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)])      # 3 cm short, 2 cm right, 8 deg
        placed = np.array([-1.15, -1.0, np.pi])
        goal = rehearsal.placement_goal(held, placed)
        root = pose_matrix([goal[0], goal[1], 0.75, 0.0, 0.0, np.sin(goal[2]/2), np.cos(goal[2]/2)])
        cabinet = root @ held
        np.testing.assert_allclose(cabinet[:2, 3], placed[:2], atol=1e-12)
        self.assertAlmostEqual(float(room.wrap(np.arctan2(cabinet[1, 0], cabinet[0, 0]) - placed[2])), 0.0, places=12)

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
        # One frame per decision: start, approach 11, reach 10, grasp 5, lift 10, hold 5, carry 1, align 40, lower 15,
        # release 5, withdraw 10, back away 15.
        phases = [frame["phase"] for frame in replay["frames"]]
        self.assertEqual(len(phases), 128)
        self.assertEqual(encode.call_args.kwargs["count"], 128)
        self.assertEqual([phases.count(p) for p in ("approach", "reach", "carry", "align", "back_away")], [11, 10, 1, 40, 15])
        # The doubles hold the cabinet straight ahead at the reach, so the placement goal is where the robot stands.
        np.testing.assert_allclose(result["placement_goal"], result["layout"]["approach_goal"], atol=1e-6)
        self.assertEqual(len(replay["frames"][0]["bodies"]), len(replay["bodies"]))
        self.assertEqual(replay["frames"][-1]["command"], [-0.1, 0.0, 0.0])
        # Key frames: the end of the approach (frame 11), of the hold (41), of the carry (42) and the final frame.
        self.assertEqual(result["key_frames"], ["000011_approach.png", "000041_hold.png", "000042_carry.png",
                                                "000127_back_away.png"])
        # The reach measured each palm against its grasp point on the floor-standing cabinet.
        self.assertEqual(set(result["reach"][0]["offset_m"]), {"left", "right"})
        self.assertIsNotNone(result["grasp"]["palm_gap_m"])


    def test_worker_pushes_the_cabinet_on_sliders(self):
        config, trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        approach, profile, asset, runtime = trainings["approach"]
        scene_profile = {**profile, "push_object": {"size_m": [0.6, 0.8, 1.0], "mass_kg": 12.0, "friction": 0.1,
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        real = rehearsal.rehearsal_layout

        def docked(*args):
            # The doubles do not walk: the robot starts at the push pose.
            layout = real(*args)
            here = layout["approach_goal"].copy()
            return {**layout, "start": here, "approach_path": np.array([here[:2]]*2)}

        modules = {"PIL": ModuleType("PIL"), "tensordict": ModuleType("tensordict")}
        modules["PIL"].Image = Image
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(1, scene_profile, asset) as fake, \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(rehearsal, "rehearsal_layout", side_effect=docked), \
                mock.patch.object(lift, "apply_hand_attachment") as attach, \
                mock.patch.object(carry, "apply_corridor", side_effect=lambda stage, env, w: w), \
                mock.patch.object(rehearsal, "require_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(rehearsal, "encode_recording", return_value={"status": "passed_video_decode"}):
            sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
            # The cabinet rests while the robot walks up; the push double then lets it slide 0.4 mm per physics step
            # along world +x, the push direction.
            slide = fake.box.slide
            fake.box.slide = lambda: np.zeros(1)
            observations = []

            def push_actor(obs):
                fake.box.slide = slide
                observations.append(obs["policy"].value.copy())
                return Tensor(np.zeros((1, 3)))

            def load(env, training, path, device):
                if path == ROOT / config["push"]["checkpoint"]:
                    self.assertEqual(env.get_observations()["critic"].value.shape[1], env.get_observations()["policy"].value.shape[1] + 2)
                    return push_actor, 99
                return room.scripted_actor(training), 299

            with mock.patch.object(ppo_eval, "load_policy", side_effect=load) as loaded:
                directory = Path(tmp)
                rehearsal.run_worker(config, trainings, PUSH_REQUEST, PUSH_SCENE, (Path("walk.onnx"), Path("balance.onnx")),
                                     directory)
            result = json.loads((directory / "rehearsal-worker.json").read_text())
            replay = json.loads((directory / "replay.json").read_text())
            friction = float(fake.box_friction[0])
        self.assertEqual(result["status"], "completed_rehearsal")
        # Pushing keeps hand collisions (no attachment or filter), slides on sliders and loads approach + push only.
        attach.assert_not_called()
        self.assertAlmostEqual(friction, 0.1, places=6)
        self.assertEqual([call.args[2] for call in loaded.call_args_list],
                         [ROOT / config["approach_checkpoint"], ROOT / config["push"]["checkpoint"]])
        self.assertEqual(result["phases"]["push_checkpoint_iteration"], 99)
        push = result["phases"]["push"]
        self.assertEqual(push["outcome"], "success")
        self.assertGreaterEqual(push["progress_m"], 0.5)
        self.assertLess(push["lateral_m"], 1e-9)
        # The first push observation: the item 0.3 m + gap + hand reach ahead on the course line, 0.5 m to go.
        first = observations[0][0]
        np.testing.assert_allclose(first[:3], [0.3 + 0.09 + 0.415, 0.0, 0.0], atol=0.06)
        self.assertGreater(first[7], 0.43)
        phases = [frame["phase"] for frame in replay["frames"]]
        self.assertEqual((phases.count("approach"), phases.count("back_away")), (11, 15))
        self.assertEqual(phases.count("push"), len(observations))
        self.assertIn("push", result["key_frames"][-2])
        # The camera looks at the middle of the push (item start (-0.1, 0) and destination (0.4, 0)) from the front.
        np.testing.assert_allclose(fake.camera.targets[0], [0.15, 0.0, config["camera"]["target_height_m"]], atol=1e-9)
        np.testing.assert_allclose(fake.camera.eyes[0][:2], [0.15, -2.0 - config["camera"]["eye_from_front_m"][1]], atol=1e-9)


    def test_worker_tries_to_lift_then_pushes(self):
        config, trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        approach, profile, asset, runtime = trainings["approach"]
        scene_profile = {**profile, "push_object": {"size_m": [0.8, 0.3, 1.0], "mass_kg": 12.0, "friction": 0.1,
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        real_layout, real_push = rehearsal.rehearsal_layout, rehearsal.push_geometry

        def docked(*args):
            layout = real_layout(*args)
            here = layout["approach_goal"].copy()
            return {**layout, "start": here, "approach_path": np.array([here[:2]]*2)}

        modules = {"PIL": ModuleType("PIL"), "tensordict": ModuleType("tensordict")}
        modules["PIL"].Image = Image
        modules["tensordict"].TensorDict = lambda groups, batch_size: groups
        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(1, scene_profile, asset) as fake:
            def walk(*args):
                # The doubles do not walk: planning the push pose puts the robot there.
                goal, course = real_push(*args)
                fake.robot.pose[0, :2] = fake.origins[0, :2] + goal[:2]
                fake.robot.pose[0, 3:] = [0.0, 0.0, np.sin(goal[2]/2), np.cos(goal[2]/2)]
                return goal, course

            slide = fake.box.slide
            fake.box.slide = lambda: np.zeros(1)

            def push_actor(obs):
                if fake.box.slide is not slide:
                    # The double's slide grows with the steps since its last pose write; start it at zero here.
                    fake.box.written[:] = fake.steps["count"] - 80
                    fake.box.slide = slide
                return Tensor(np.zeros((1, 3)))

            def load(env, training, path, device):
                return (push_actor, 99) if path == ROOT / config["push"]["checkpoint"] else (room.scripted_actor(training), 299)

            with mock.patch.dict(sys.modules, modules), \
                    mock.patch.object(rehearsal, "rehearsal_layout", side_effect=docked), \
                    mock.patch.object(rehearsal, "push_geometry", side_effect=walk), \
                    mock.patch.object(lift, "apply_hand_attachment", return_value={}) as attach, \
                    mock.patch.object(lift, "set_attachment", return_value=True) as toggle, \
                    mock.patch.object(carry, "apply_corridor", side_effect=lambda stage, env, w: w), \
                    mock.patch.object(rehearsal, "require_ffmpeg", return_value="ffmpeg"), \
                    mock.patch.object(rehearsal, "encode_recording", return_value={"status": "passed_video_decode"}), \
                    mock.patch.object(rehearsal.advisor, "advise", side_effect=RuntimeError("no META_API_KEY")) as advise, \
                    mock.patch.object(ppo_eval, "load_policy", side_effect=load):
                sys.modules["torch"].__dict__.update(long=np.int64, inference_mode=nullcontext)
                directory = Path(tmp)
                rehearsal.run_worker(config, trainings, AUTO_REQUEST, AUTO_SCENE, (Path("walk.onnx"), Path("balance.onnx")),
                                     directory)
            result = json.loads((directory / "rehearsal-worker.json").read_text())
            replay = json.loads((directory / "replay.json").read_text())
        self.assertEqual(result["status"], "completed_rehearsal")
        # The grip keeps hand collisions (the item may be pushed afterwards); the joints engage and let go once.
        self.assertFalse(attach.call_args.kwargs["filter_hands"])
        self.assertEqual([call.args[2] for call in toggle.call_args_list], [True, False])
        # The 12 kg bookcase stays on the floor; Meta is asked with the attempt and its last frame, and without the API
        # the capability rules decide to push it on sliders.
        self.assertFalse(result["lift"]["lifted"])
        worker, frames = advise.call_args.args[4], advise.call_args.args[5]
        self.assertEqual((worker["status"], worker["lift"]["lifted"]), ("lift_attempt", False))
        # Frames: the end of the hold (frame 41) and the moment of the diagnosis, after lower 15, release 5, withdraw 10
        # and the step back, which runs its full 4 s because the doubles do not move (frame 111).
        self.assertEqual([f.name for f in frames], ["000041.png", "000111.png"])
        self.assertLess(result["phases"]["step_back"]["clearance_m"], 0.5)
        self.assertEqual((result["advice"]["mode"], result["advice"]["decision"]["strategy"]), ("rules_only", "push"))
        self.assertIn("no META_API_KEY", result["advice"]["error"])
        self.assertEqual((result["phases"]["reposition"]["outcome"], result["phases"]["push"]["outcome"]), ("success", "success"))
        phases = [frame["phase"] for frame in replay["frames"]]
        order = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
        self.assertEqual(order, ["start", "approach", "reach", "grasp", "lift", "hold", "lower", "release", "withdraw",
                                 "step_back", "reposition", "push", "back_away"])


if __name__ == "__main__":
    unittest.main()
