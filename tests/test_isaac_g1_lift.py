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
import isaac_g1_lift as lift
from test_isaac_g1_rl import parallel_isaac


class Image:
    @staticmethod
    def fromarray(array):
        assert array.shape == (360, 640, 3)
        return Image()
    def save(self, path): Path(path).write_bytes(b"frame")


class LiftGeometryTests(unittest.TestCase):
    def setUp(self):
        self.config, self.profile, self.asset, _ = lift.load_config(lift.DEFAULT_CONFIG)
        self.contract, self.reference = lift.trial_contract(self.profile)
        self.base = lift.base_pose(self.reference, self.asset, self.contract)

    def test_arm_ik_moves_each_palm_without_rotating_it(self):
        start = self.reference.forward(self.base)
        for dx, dz in ((0.0, 0.06), (0.0, -0.02), (0.08, 0.10), (0.08, -0.02)):
            q = dict(self.base)
            for side in ("left", "right"):
                q.update(lift.arm_pose(self.reference, self.base, side, dz, dx))
            moved = self.reference.forward(q)
            for side in ("left", "right"):
                palm = f"{side}_hand_palm_link"
                np.testing.assert_allclose(moved[palm][:3, 3] - start[palm][:3, 3], [dx, 0, dz], atol=1e-7)
                np.testing.assert_allclose(moved[palm][:3, :3], start[palm][:3, :3], atol=1e-7)
            # Only arm joints move.
            self.assertEqual({n for n in q if q[n] != self.base[n]} - set(lift.arm_joints(self.reference, "left")) -
                             set(lift.arm_joints(self.reference, "right")), set())
        # 16 cm forward at the start height cannot keep the palm orientation (the reach checked before choosing 8 cm).
        with self.assertRaises(RuntimeError):
            lift.arm_pose(self.reference, self.base, "left", 0.0, 0.16)

    def test_both_palm_joints_imply_the_same_side_grasp_pose(self):
        o, height = self.config["object"], self.contract["height_command"]
        start = lift.start_pose(self.reference, self.base, self.config["arm_offsets_m"])
        frames = lift.attachment_frames(self.reference, start, o, height)
        fk = self.reference.forward(start)
        for side in ("left", "right"):
            np.testing.assert_allclose(fk[f"{side}_hand_palm_link"] @ frames["palm_object"][side], frames["pelvis_object"], atol=1e-12)
        obj, size = frames["pelvis_object"], o["size_m"]
        palm = fk["left_hand_palm_link"][:3, 3]
        self.assertAlmostEqual(obj[0, 3] - size[0]/2, palm[0] + o["near_face_ahead_of_palm_m"])
        # The palms moved forward along the side faces; the cabinet stays where the front-edge grasp held it.
        default = self.reference.forward(self.base)["left_hand_palm_link"][:3, 3]
        self.assertAlmostEqual(palm[0] - default[0], self.config["arm_offsets_m"]["forward"])
        self.assertAlmostEqual(obj[0, 3] - size[0]/2, default[0] + 0.02)
        # Standing clearance is set at the policy height, not at the raised crouch start (run 013531).
        self.assertAlmostEqual(height + obj[2, 3] - size[2]/2, o["start_clearance_m"])
        self.assertAlmostEqual(frames["palm_inset_m"], abs(palm[1] - obj[1, 3]) - size[1]/2)
        self.assertTrue(0 <= frames["palm_inset_m"] <= 0.03 and frames["grasp_below_top_m"] >= 0.04)
        for size in ([0.3, 0.4, 0.9], [0.3, 0.28, 0.75]):
            with self.subTest(size=size), self.assertRaises(ValueError):
                lift.attachment_frames(self.reference, self.base, {**o, "size_m": size}, height)
        pose = np.r_[[0.0, 0.0, 0.5], wbc.matrix_quat_xyzw(wbc.axis_rotation([1, 0, 0], np.radians(10)))]
        self.assertAlmostEqual(lift.tilt_deg(pose), 10.0)
        self.assertLess(lift.lowest_point_z(pose, [0.4, 0.4, 0.4]), 0.5 - 0.2)

    def test_feedforward_torque_is_the_gradient_of_the_held_weight(self):
        config = {**self.config, "cases": [{"mass_kg": 2.0, "feedforward": False}, {"mass_kg": 4.0, "feedforward": True},
                                           {"mass_kg": 6.0, "feedforward": False}]}
        start = lift.start_pose(self.reference, self.base, config["arm_offsets_m"])
        frames = lift.attachment_frames(self.reference, start, config["object"], self.contract["height_command"])
        joints, angles, plan, loads = lift.arm_plan(self.reference, self.base, config, frames, self.contract)
        q = {**self.base, **angles["lift"]}
        mass, kp = 3.0, self.contract["upper_hold_kp"]
        efforts = {n: self.reference.limits[n]["effort"] for n in joints}
        for side in ("left", "right"):
            load = lift.load_feedforward(self.reference, q, side, frames, mass, kp, efforts)

            def centre_z(angles):
                return (self.reference.forward(angles)[f"{side}_hand_palm_link"] @ frames["palm_object"][side])[2, 3]

            # Virtual work: the torque that holds half the weight is the gradient of its potential energy.
            for name in lift.arm_joints(self.reference, side):
                gradient = mass*lift.GRAVITY/2*(centre_z({**q, name: q[name] + 1e-5}) - centre_z({**q, name: q[name] - 1e-5}))/2e-5
                self.assertAlmostEqual(load["torque_nm"][name], gradient, places=3)
                self.assertAlmostEqual(load["offset_rad"][name], gradient/kp, places=5)
        # Holding 8 cm further back on the side faces moves the first wrist pitch limit from 3.9 kg to between 4 and 6 kg.
        self.assertEqual(["left_wrist_pitch_joint" in l["left"]["over_limit"] for l in loads], [False, False, True])
        front = {**config, "object": {**config["object"], "near_face_ahead_of_palm_m": 0.02},
                 "arm_offsets_m": {**config["arm_offsets_m"], "forward": 0.0}}
        front_frames = lift.attachment_frames(self.reference, self.base, front["object"], self.contract["height_command"])
        front_loads = lift.arm_plan(self.reference, self.base, front, front_frames, self.contract)[3]
        np.testing.assert_allclose(front_frames["pelvis_object"], frames["pelvis_object"], atol=1e-9)
        wrist = [abs(l["left"]["torque_nm"]["left_wrist_pitch_joint"])/c["mass_kg"] for l, c in ((front_loads[0], config["cases"][0]),
                                                                                                (loads[0], config["cases"][0]))]
        self.assertAlmostEqual(wrist[0], 1.28, places=2)
        self.assertAlmostEqual(wrist[1], 0.89, places=2)
        # Feedforward changes only the lift pose of its own cases; every pose keeps the palms forward.
        lift_angles = [angles["lift"][n] for n in joints]
        for i, case in enumerate(config["cases"]):
            np.testing.assert_allclose(plan["start"][i], [angles["start"][n] for n in joints])
            np.testing.assert_allclose(plan["place"][i], [angles["place"][n] for n in joints])
            offsets = [loads[i]["left" if n.startswith("left") else "right"]["offset_rad"][n] for n in joints]
            np.testing.assert_allclose(plan["lift"][i], np.add(lift_angles, offsets) if case["feedforward"] else lift_angles)

    def test_schedule_and_functional_checks(self):
        steps = lift.schedule(self.config["timeline"], 0.02)
        self.assertEqual(len(steps), 600)
        self.assertEqual([s["phase"] for s in steps if s["release"]], ["release"])
        lift_steps = [s for s in steps if s["phase"] == "lift"]
        self.assertEqual((lift_steps[0]["arms"], lift_steps[-1]["fraction"]), (["start", "lift"], 1.0))

        def record(phase, released, bottom, x, palm, error=0.001):
            return {"phase": phase, "released": released, "object_bottom_m": bottom, "object_xyz": [x, 0, 0.5],
                    "attachment_error_m": error, "fallen": False, "object_tilt_deg": 1.0,
                    "palm_z": {"left": palm, "right": palm}}

        records = ([record("hold", False, 0.08, 0.6, 0.94), record("carry", False, 0.07, 0.6, 0.94),
                    record("carry", False, 0.075, 1.2, 0.94), record("lower", False, 0.0, 1.2, 0.86),
                    record("release", True, 0.0, 1.2, 0.86)])
        stayed = lift.lift_summary(records + [record("withdraw", True, 0.001, 1.2, 0.88)], self.config["checks"])
        self.assertEqual(stayed["status"], "passed_lift_probe")
        self.assertTrue(stayed["release"]["object_stayed"])
        attached = lift.lift_summary(records + [record("withdraw", True, 0.02, 1.2, 0.88)], self.config["checks"])
        self.assertFalse(attached["release"]["object_stayed"])
        dropped = lift.lift_summary([record("carry", False, 0.01, 0.6, 0.94)] + records[2:] +
                                    [record("withdraw", True, 0.0, 1.2, 0.88)], self.config["checks"])
        self.assertEqual((dropped["status"], dropped["checks"]["lifted_during_carry"]), ("failed_lift_probe", False))
        settled = [record("settle", False, 0.0, 0.6, 0.84)]
        bounded = lift.lift_summary(settled + records + [{**record("withdraw", True, 0.001, 1.2, 0.88), "invalid_action": True}],
                                    self.config["checks"])
        self.assertAlmostEqual(bounded["palm_rise_m"], 0.10)
        self.assertEqual(bounded["held_tilt_max_deg"], 1.0)
        self.assertEqual((bounded["status"], bounded["checks"]["actions_in_bound"]), ("failed_lift_probe", False))

    def test_configuration_rejects_unreviewed_lifts(self):
        raw = json.loads(lift.DEFAULT_CONFIG.read_text())
        timeline = raw["timeline"]
        cases = [{**raw, "object": {**raw["object"], "mass_kg": 2.0}},
                 {**raw, "cases": [{"mass_kg": 13.0, "feedforward": False}]},
                 {**raw, "cases": [{"mass_kg": 2.0, "feedforward": True}]*2},
                 {**raw, "camera": {**raw["camera"], "env_spacing_m": 4.0}},
                 {**raw, "object": {**raw["object"], "start_clearance_m": 0.05}},
                 {**raw, "timeline": [{**p, "command": [0.4, 0, 0]} if p["name"] == "carry" else p for p in timeline]},
                 {**raw, "timeline": [p for p in timeline if p["name"] != "withdraw"]},
                 {**raw, "arm_offsets_m": {"forward": 0.08, "lift": 0.2, "place": -0.02}},
                 {**raw, "arm_offsets_m": {"forward": 0.3, "lift": 0.1, "place": -0.02}},
                 {**raw, "arm_offsets_m": {"lift": 0.1, "place": -0.02}}, {**raw, "extra": 1}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lift.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    lift.load_config(path)


class LiftWorkerTests(unittest.TestCase):
    def test_worker_runs_each_mass_case_in_its_own_environment(self):
        config, profile, asset, runtime = lift.load_config(lift.DEFAULT_CONFIG)
        config = {**config, "cases": [{"mass_kg": 2.0, "feedforward": False}, {"mass_kg": 5.0, "feedforward": True}]}
        o = config["object"]
        scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": 2.0, "friction": o["friction"],
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        modules = {"PIL": ModuleType("PIL")}
        modules["PIL"].Image = Image
        attached, released, targets = [], [], []

        def attach(stage, robot, box, env, frames):
            # The enabled joints pull the object toward the palms in the PhysX warmup step of sim.reset().
            box.pose[int(env.rsplit("_", 1)[1]), :3] += [-0.1, 0.0, 0.3]
            attached.append(env)
            return {"joints": {"left": {}, "right": {}}}

        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(2, scene_profile, asset) as fake, \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(lift, "apply_hand_attachment", side_effect=attach), \
                mock.patch.object(lift, "release_attachment", side_effect=lambda stage, env: released.append(env) or True), \
                mock.patch.object(lift, "require_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(lift, "encode_recording", return_value={"status": "passed_video_decode"}) as encode:
            fake.box.slide = lambda: np.zeros(2)
            # The second cabinet presses 5 N on the left knee throughout.
            fake.body.data.force_matrix_w.value[1, 0, 0] = [3.0, 0.0, 4.0]
            setter = fake.robot.set_joint_position_target_index
            fake.robot.set_joint_position_target_index = lambda *, target: targets.append(target.value.copy()) or setter(target=target)
            directory = Path(tmp)
            lift.run_worker(config, profile, asset, runtime, (Path("walk.onnx"), Path("balance.onnx")), directory)
            result = json.loads((directory / "lift-worker.json").read_text())
            lines = [(directory / f"case_{i}" / "trajectory.jsonl").read_text().splitlines() for i in range(2)]
            joints = [n for side in ("left", "right") for n in lift.arm_joints(lift.trial_contract(profile)[1], side)]
            arm_ids = [fake.robot.joint_names.index(n) for n in joints]
        envs = ["/World/envs/env_0", "/World/envs/env_1"]
        self.assertEqual((attached, released), (envs, envs))
        np.testing.assert_allclose(fake.box_mass, [2.0, 5.0])
        start = result["start"]
        self.assertLess(max(start["palm_frame_mismatch_m"]), 1e-6)
        self.assertAlmostEqual(start["after_reset_offset_m"][1][2], o["size_m"][2]/2 + 0.3)
        np.testing.assert_allclose(start["object_bottom_m"], o["start_clearance_m"] + result["initialization"]["lift_m"])
        self.assertEqual([len(case) for case in lines], [600, 600])
        self.assertEqual(json.loads(lines[1][0])["root_xyz"][:2], [0.0, 0.0])
        # One frame every two control steps per case, none before the first step.
        self.assertEqual([call.kwargs["count"] for call in encode.call_args_list], [300, 300])
        # After the crouch-start defaults, the grasp start pose (palms forward) is commanded before the first step.
        start_angles = [result["arm_angles_rad"]["start"][n] for n in joints]
        self.assertGreater(np.abs(start_angles).max(), 0.1)
        np.testing.assert_allclose(targets[0][:, arm_ids], 0.0)
        np.testing.assert_allclose(targets[1][:, arm_ids], np.tile(start_angles, (2, 1)))
        # During hold (control steps 100-149) only the feedforward case carries the load offsets.
        hold = targets[120][:, arm_ids]
        np.testing.assert_allclose(hold[0], [result["arm_angles_rad"]["lift"][n] for n in joints])
        np.testing.assert_allclose(hold[1], result["lift_targets_rad"][1])
        self.assertGreater(np.abs(hold[1] - hold[0]).max(), 0.05)
        # The doubles neither move the palms nor the objects, so the attachment holds but nothing is carried.
        for case in result["results"]:
            checks = case["summary"]["checks"]
            self.assertEqual((checks["attachment_held"], checks["carried_forward"], checks["actions_in_bound"]), (True, False, True))
        self.assertEqual(result["heaviest_passed_kg"], {"uncompensated": None, "feedforward": None})
        # The doubles keep the arms at the start pose, so the hold error is the lift-start difference and the logged
        # torque is kp times it; drives beyond their effort limit are reported as saturated.
        record = json.loads(lines[0][120])
        error = np.subtract(hold[0], start_angles)
        np.testing.assert_allclose(record["arm_error_rad"], error, atol=1e-9)
        np.testing.assert_allclose(record["arm_pd_torque_nm"], 100.0*error, atol=1e-7)
        diagnostics = [case["diagnostics"] for case in result["results"]]
        over = [n.removesuffix("_joint") for n, e, limit in zip(joints, error, result["arm_effort_nm"]) if abs(100.0*e) >= limit]
        self.assertEqual(diagnostics[0]["saturated_joints"], over)
        self.assertAlmostEqual(diagnostics[0]["hold_left_arm"]["wrist_pitch"]["error_rad"], error[joints.index("left_wrist_pitch_joint")])
        self.assertEqual(diagnostics[0]["body_contact"], {"held_max_n": 0.0, "held_fraction": 0.0, "most_frequent_body": None})
        self.assertEqual(diagnostics[1]["body_contact"], {"held_max_n": 5.0, "held_fraction": 1.0, "most_frequent_body": "left_knee_link"})
        self.assertEqual(result["status"], "completed_lift_sweep")
        self.assertEqual({role for role, _ in fake.policy_calls}, {"walk", "balance"})


if __name__ == "__main__":
    unittest.main()
