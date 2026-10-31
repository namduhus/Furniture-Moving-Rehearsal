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

    def test_arm_ik_moves_each_palm_vertically_without_rotating_it(self):
        start = self.reference.forward(self.base)
        for dz in (0.06, -0.02):
            q = dict(self.base)
            for side in ("left", "right"):
                q.update(lift.arm_pose(self.reference, self.base, side, dz))
            moved = self.reference.forward(q)
            for side in ("left", "right"):
                palm = f"{side}_hand_palm_link"
                np.testing.assert_allclose(moved[palm][:3, 3] - start[palm][:3, 3], [0, 0, dz], atol=1e-7)
                np.testing.assert_allclose(moved[palm][:3, :3], start[palm][:3, :3], atol=1e-7)
            # Only arm joints move.
            self.assertEqual({n for n in q if q[n] != self.base[n]} - set(lift.arm_joints(self.reference, "left")) -
                             set(lift.arm_joints(self.reference, "right")), set())

    def test_both_palm_joints_imply_the_same_object_pose_at_the_grasp(self):
        frames = lift.attachment_frames(self.reference, self.base, self.config["object"], 0.415)
        fk = self.reference.forward(self.base)
        for side in ("left", "right"):
            np.testing.assert_allclose(fk[f"{side}_hand_palm_link"] @ frames["palm_object"][side], frames["pelvis_object"], atol=1e-12)
        obj, size = frames["pelvis_object"], self.config["object"]["size_m"]
        self.assertAlmostEqual(obj[0, 3] - size[0]/2, 0.415)
        palm_z = fk["left_hand_palm_link"][2, 3]
        self.assertAlmostEqual(obj[2, 3] + size[2]/2, palm_z + self.config["object"]["grasp_below_top_m"])
        pose = np.r_[[0.0, 0.0, 0.5], wbc.matrix_quat_xyzw(wbc.axis_rotation([1, 0, 0], np.radians(10)))]
        self.assertAlmostEqual(lift.tilt_deg(pose), 10.0)
        self.assertLess(lift.lowest_point_z(pose, [0.4, 0.4, 0.4]), 0.5 - 0.2)

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

    def test_configuration_rejects_unreviewed_lifts(self):
        raw = json.loads(lift.DEFAULT_CONFIG.read_text())
        timeline = raw["timeline"]
        cases = [{**raw, "object": {**raw["object"], "mass_kg": 3.5}},
                 {**raw, "timeline": [{**p, "command": [0.4, 0, 0]} if p["name"] == "carry" else p for p in timeline]},
                 {**raw, "timeline": [p for p in timeline if p["name"] != "withdraw"]},
                 {**raw, "arm_offsets_m": {"lift": 0.2, "place": -0.02}}, {**raw, "extra": 1}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lift.json"
            for data in cases:
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    lift.load_config(path)


class LiftWorkerTests(unittest.TestCase):
    def test_worker_attaches_places_and_logs_every_control_step(self):
        config, profile, asset, runtime = lift.load_config(lift.DEFAULT_CONFIG)
        o = config["object"]
        scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": o["mass_kg"], "friction": o["friction"],
                                                    "gap_m": profile["push_object"]["gap_m"]}}
        modules = {"PIL": ModuleType("PIL")}
        modules["PIL"].Image = Image
        attached, released = [], []

        def attach(stage, robot, box, env, frames):
            # The enabled joints pull the object toward the palms in the PhysX warmup step of sim.reset().
            box.pose[0, :3] += [-0.1, 0.0, 0.3]
            attached.append(env)
            return {"joints": {"left": {}, "right": {}}}

        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(1, scene_profile, asset) as fake, \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(lift, "apply_hand_attachment", side_effect=attach), \
                mock.patch.object(lift, "release_attachment", side_effect=lambda stage, env: released.append(env) or True), \
                mock.patch.object(lift, "require_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(lift, "encode_recording", return_value={"status": "passed_video_decode"}) as encode:
            fake.box.slide = lambda: np.zeros(1)
            directory = Path(tmp)
            lift.run_worker(config, profile, asset, runtime, (Path("walk.onnx"), Path("balance.onnx")), directory)
            result = json.loads((directory / "lift-worker.json").read_text())
            lines = (directory / "lift-trajectory.jsonl").read_text().splitlines()
        self.assertEqual((attached, released), (["/World/envs/env_0"], ["/World/envs/env_0"]))
        self.assertLess(result["start"]["palm_frame_mismatch_m"], 1e-6)
        self.assertAlmostEqual(result["start"]["after_reset_offset_m"][2], o["size_m"][2]/2 + 0.3)
        self.assertTrue(0.002 <= result["start"]["object_bottom_m"] <= 0.06)
        self.assertEqual(len(lines), 600)
        self.assertEqual(encode.call_args.kwargs["count"], 301)
        # The doubles neither move the palms nor the object, so the attachment holds but nothing is carried.
        checks = result["summary"]["checks"]
        self.assertEqual((checks["attachment_held"], checks["carried_forward"]), (True, False))
        self.assertEqual(result["status"], "failed_lift_probe")
        roles = {role for role, _ in fake.policy_calls}
        self.assertEqual(roles, {"walk", "balance"})


if __name__ == "__main__":
    unittest.main()
