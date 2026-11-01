import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_replay
import isaac_g1_rehearsal as rehearsal
import rehearsal_sequence as sequence

TRIO = ROOT / "configs/rehearsal_living_room_trio.json"


def fake_run(directory, target, status, spec_xy, sim_yaw, root, bodies=("pelvis", "torso_link")):
    """A finished rehearsal run folder as the runner leaves it (summary, worker evidence, replay)."""
    run = directory / "20261102T000000Z-rehearsal"
    run.mkdir(parents=True)
    placement = {"position_m": 0.01, "yaw_deg": 1.0, "yaw_rad": sim_yaw, "cabinet_spec_xy_m": spec_xy, "within_tolerance": True}
    (run / "summary.json").write_text(json.dumps({"status": status, "placement": placement}))
    (run / "rehearsal-worker.json").write_text(json.dumps({
        "status": status, "request": {"target": target, "strategy": "carry"}, "phases": {"carry": {"outcome": "success"}},
        "placement": placement, "layout": {"cabinet": [1.6, 0.0, 0.0], "placed": [0.0, -0.8, np.pi/2], "size_m": [0.3, 0.28, 0.9],
                                           "room": {}, "spec_transform": {"x": [1.0, 1.9], "y": [-1.0, 2.3]}}}))
    frame = {"t": 0.0, "phase": "start", "command": [0, 0, 0], "bodies": [list(root), [0, 0, 1, 0, 0, 0, 1]],
             "cabinet": [1.6, 0.0, 0.45, 0, 0, 0, 1]}
    (run / "replay.json").write_text(json.dumps({"bodies": list(bodies), "decision_dt": 0.1, "frame": "f",
                                                 "spec_transform": {"x": [1.0, 1.9], "y": [-1.0, 2.3]},
                                                 "cabinet_size_m": [0.3, 0.28, 0.9],
                                                 "frames": [frame, {**frame, "phase": "back_away"}]}))
    return run


class SequenceTests(unittest.TestCase):
    def test_the_trio_sequence_plans_every_move_within_the_trained_distances(self):
        config, trainings = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        seq = sequence.load_sequence(TRIO)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sequence, "ROOT", Path(tmp)):
            layouts = sequence.plan(seq, config, trainings)
        self.assertEqual([m["target"] for m in seq["moves"]], ["pedestal_3", "pedestal_2", "pedestal_1"])
        for layout in layouts:
            # The pedestals' spec fronts face the wall, so they are gripped from the room side.
            self.assertAlmostEqual(float(np.cos(layout["cabinet"][2])), 1.0)
            for a, b in ((layout["start"], layout["approach_goal"]), (layout["approach_goal"], layout["carry_goal"])):
                self.assertTrue(1.0 <= np.linalg.norm(np.asarray(a[:2]) - b[:2]) <= 3.5)
        # Later moves start behind the item set down before, with the moved items as obstacles.
        np.testing.assert_allclose(layouts[1]["start"], [-1.2, -0.8 - 0.15 - 0.5, np.pi/2], atol=1e-9)
        self.assertEqual(len(layouts[2]["room"]["boxes"]), 4)

    def test_requests_carry_the_corrected_sizes_the_moved_poses_and_the_start(self):
        seq = sequence.load_sequence(TRIO)
        first = sequence.move_request(seq, 0, {}, None)
        self.assertEqual(first["start"], seq["start"])
        self.assertEqual({k: v["size_m"] for k, v in first["furniture"].items()},
                         {t: [0.28, 0.30, 0.90] for t in ("pedestal_1", "pedestal_2", "pedestal_3")})
        moved = {"pedestal_3": {"center_m": [0.71, 3.09], "yaw_deg": 178.0}}
        second = sequence.move_request(seq, 1, moved, [-1.2, -1.45, 1.5])
        self.assertEqual((second["target"], second["start"]), ("pedestal_2", [-1.2, -1.45, 1.5]))
        self.assertEqual(second["furniture"]["pedestal_3"], {"size_m": [0.28, 0.30, 0.90], **moved["pedestal_3"]})
        for yaw in (-90.0, 0.0, 45.0, 180.0):
            self.assertAlmostEqual(sequence.spec_yaw(rehearsal.facing_yaw(yaw)), (yaw + 180.0) % 360.0 - 180.0, places=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.json"
            raw = json.loads(TRIO.read_text())
            for bad in ({**raw, "moves": raw["moves"][:1]*2}, {**raw, "extra": 1}, {**raw, "moves": []}):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    sequence.load_sequence(path)

    def test_moves_chain_through_measured_poses_and_stop_at_a_failure(self):
        seq = sequence.load_sequence(TRIO)
        requests = []

        def runner(path, out):
            request = json.loads(path.read_text())
            requests.append(request)
            k = len(requests) - 1
            yaw = np.pi/2 + 0.02
            status = "completed_rehearsal" if k == 0 else "failed_carry"
            fake_run(out, request["target"], status, [0.71, 3.09], yaw, [-1.2, -1.46, 0.75, 0, 0, np.sin(yaw/2), np.cos(yaw/2)])
            return 0 if k == 0 else 1

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sequence, "ROOT", Path(tmp)):
            record = sequence.run(seq, Path(tmp) / "20261102T000000Z-sequence", runner)
            saved = json.loads((Path(tmp) / "20261102T000000Z-sequence/sequence.json").read_text())
        self.assertEqual((record["status"], saved["status"]), ("stopped_at_move_1", "stopped_at_move_1"))
        self.assertEqual([r["status"] for r in record["runs"]], ["completed_rehearsal", "failed_carry"])
        self.assertEqual(len(requests), 2)
        # The second move starts at the last root pose of the first and sees pedestal_3 where it was measured.
        np.testing.assert_allclose(requests[1]["start"], [-1.2, -1.46, np.pi/2 + 0.02], atol=1e-9)
        self.assertEqual(requests[1]["furniture"]["pedestal_3"]["center_m"], [0.71, 3.09])
        self.assertEqual(requests[1]["furniture"]["pedestal_3"]["yaw_deg"], round(sequence.spec_yaw(np.pi/2 + 0.02), 2))
        self.assertEqual(requests[0]["start"], seq["start"])

    def test_export_merges_the_runs_into_one_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "20261102T000000Z-sequence"
            scene = json.loads((ROOT / "configs/scenes/living_room_trio.json").read_text())
            for k, target in enumerate(("pedestal_3", "pedestal_2")):
                run = fake_run(directory / str(k), target, "completed_rehearsal", [0.7, 3.1], np.pi/2, [0, 0, 0.75, 0, 0, 0, 1])
                (run / "scene.json").write_text(json.dumps(scene))
            (directory / "sequence.json").write_text(json.dumps({"status": "stopped_at_move_2", "runs": [
                {"target": "pedestal_3", "run": f"0/{run.name}"}, {"target": "pedestal_2", "run": f"1/{run.name}"},
                {"target": "pedestal_1", "run": None}]}))
            with mock.patch.object(export_replay, "appearance", return_value={"items": {}}):
                target = export_replay.export_sequence(directory, "trio", out=Path(tmp) / "web")
            replay = json.loads((target / "replay.json").read_text())
            demo = json.loads((target / "rehearsal.json").read_text())
        self.assertEqual(replay["items"], ["pedestal_3", "pedestal_2"])
        self.assertEqual([f["item"] for f in replay["frames"]], [0, 0, 1, 1])
        self.assertEqual([f["t"] for f in replay["frames"]], [0.0, 0.1, 0.2, 0.3])
        self.assertEqual([m["frames"] for m in demo["moves"]], [[0, 1], [2, 3]])
        self.assertEqual(demo["status"], "stopped_at_move_2")
        # The display room drops the two moved pedestals but keeps the armchair, the sideboard and the unmoved one.
        self.assertEqual(len(demo["layout"]["room"]["boxes"]), 3)
        self.assertEqual(replay["room"], demo["layout"]["room"])


if __name__ == "__main__":
    unittest.main()
