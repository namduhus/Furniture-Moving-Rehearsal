import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isaac_g1_rehearsal as rehearsal
import rehearsal_server as server

CONFIG, TRAININGS = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)


def move(name):
    request = json.loads((ROOT / f"configs/rehearsal_living_room_{name}.json").read_text())
    return {"scene": Path(request["scene"]).stem, "target": request["target"], "corrections": request["corrections"],
            "destination": request["destination"]}


class FakeModels:
    def chat(self, model, messages, schema=None):
        return {"content": {"cause": "c", "evidence": [], "strategy": "carry", "explanation": "x", "explanation_ko": "y",
                            "confidence": "high"}, "usage": {}, "cached": False}


class ServerCheckTests(unittest.TestCase):
    def test_scene_view_gives_the_room_items_and_start_in_the_simulation_frame(self):
        with mock.patch.object(server.export_replay, "appearance", return_value={"items": {}}):
            view = server.scene_view("living_room_pedestal", CONFIG, TRAININGS)
        self.assertEqual(view["room"]["ids"], ["armchair_1", "tvstand_1", "tv_1", "cabinet_1"])
        pedestal = next(f for f in view["furniture"] if f["id"] == "cabinet_1")
        self.assertEqual([round(v, 6) for v in pedestal["sim"]], [1.35, -0.5, 0.0])
        self.assertEqual([round(v, 6) for v in view["start"]], [1.05, 0.9, round(3.141592653589793, 6)])
        self.assertEqual(view["carry_tier"]["size_m"], [0.28, 0.30, 0.90])
        with self.assertRaises(ValueError):
            server.scene_path("../secrets")

    def test_the_first_valid_strategy_is_chosen(self):
        for name, expected in (("pedestal", "carry"), ("bookcase", "auto"), ("cabinet", "push")):
            with self.subTest(name=name):
                self.assertEqual(server.check(move(name), CONFIG, TRAININGS)["strategy"], expected)
        # The pedestal moved into the armchair is not possible with any strategy; every reason is reported.
        bad = {**move("pedestal"), "destination": {"center_m": [0.6, 1.5], "yaw_deg": 0.0}}
        result = server.check(bad, CONFIG, TRAININGS)
        self.assertEqual((result["ok"], result["request"]), (False, None))
        self.assertEqual(set(result["reasons"]), {"carry", "auto", "push"})

    def test_feasible_destinations_follow_the_strategy(self):
        pedestal = server.feasible(move("pedestal"), CONFIG, TRAININGS)
        self.assertGreater(len(pedestal["points"]), 50)
        self.assertEqual({p["strategy"] for p in pedestal["points"]}, {"carry"})
        # The cabinet is pushed: candidates lie along its four sides at the push distances and keep its heading.
        cabinet = server.feasible(move("cabinet"), CONFIG, TRAININGS)
        self.assertEqual(cabinet["candidates"], 4*13)
        self.assertTrue(cabinet["points"] and {p["strategy"] for p in cabinet["points"]} == {"push"})
        self.assertEqual({p["yaw_deg"] for p in cabinet["points"]}, {0.0})
        # Every listed point passes the same check.
        point = cabinet["points"][0]
        checked = server.check({**move("cabinet"), "destination": {"center_m": point["center_m"], "yaw_deg": point["yaw_deg"]}},
                               CONFIG, TRAININGS)
        self.assertEqual(checked["strategy"], "push")
        # With Meta's estimate (8 kg side table) nothing passes, and the reasons say why.
        estimate = server.feasible({"scene": "living_room_pedestal_three", "target": "side_table_1",
                                    "corrections": {"size_m": [0.32, 0.32, 0.9], "mass_kg": 8.0}}, CONFIG, TRAININGS)
        self.assertEqual(estimate["points"], [])
        self.assertIn("carry tier", estimate["reasons"]["carry"])
        self.assertEqual(pedestal["reasons"], {})

    def test_queued_moves_and_same_size_items_shape_the_next_move(self):
        # Edited pedestal room: the inner side table is boxed in until the outer one has been moved.
        fix = {"size_m": [0.28, 0.30, 0.90], "mass_kg": 3.0}
        sequence = json.loads((ROOT / "configs/rehearsal_living_room_pedestal_three.json").read_text())
        first, second = sequence["moves"][:2]
        body = {"scene": "living_room_pedestal_three", "corrections": fix}
        alone = server.check({**body, "target": "side_table_0", "destination": second["destination"]}, CONFIG, TRAININGS)
        self.assertFalse(alone["ok"])
        outer = server.check({**body, "target": "side_table_1", "destination": first["destination"], "alike": ["side_table_0"]},
                             CONFIG, TRAININGS)
        self.assertEqual(outer["strategy"], "carry")
        self.assertEqual(outer["request"]["furniture"], {"side_table_0": {"size_m": fix["size_m"]}})
        inner = server.check({**body, "target": "side_table_0", "destination": second["destination"],
                              "previous": [outer["request"]]}, CONFIG, TRAININGS)
        self.assertEqual(inner["strategy"], "carry")
        # The moved side table stands on its destination, and the robot starts behind it.
        self.assertEqual(inner["request"]["furniture"]["side_table_1"]["center_m"], first["destination"]["center_m"])
        self.assertEqual(len(inner["request"]["start"]), 3)
        with self.assertRaisesRegex(ValueError, "already"):
            server.check({**body, "target": "side_table_1", "destination": first["destination"], "previous": [outer["request"]]},
                         CONFIG, TRAININGS)

    def test_meta_plan_is_added_on_request(self):
        result = server.check({**move("pedestal"), "ask_meta": True}, CONFIG, TRAININGS, models=FakeModels())
        self.assertEqual((result["strategy"], result["advice"]["decision"]["strategy"]), ("carry", "carry"))
        self.assertEqual(result["request"]["strategy"], "carry")


class FakeRemote:
    def __init__(self, outputs):
        self.calls, self.outputs, self.fetched = [], list(outputs), []

    def __call__(self, command, stdin=b"", timeout=120):
        self.calls.append((command, stdin))
        return self.outputs.pop(0) if self.outputs else ""

    def fetch(self, name, out):
        self.fetched.append(name)


class JobTests(unittest.TestCase):
    def test_a_job_uploads_the_request_starts_with_the_key_on_stdin_and_comes_back(self):
        request = {**json.loads((ROOT / "configs/rehearsal_living_room_pedestal.json").read_text())}
        summary = {"status": "completed_rehearsal", "run_dir": "/remote/outputs/isaac/20261102T000000Z-rehearsal"}
        remote = FakeRemote(["", "", "still running", "alive",
                             "/r/outputs/isaac/20201231T000000Z-rehearsal\n/r/outputs/isaac/20991231T000000Z-rehearsal",
                             json.dumps({"phases": {"approach": {}, "reach": {}}}), json.dumps(summary)])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "ROOT", Path(tmp)), \
                mock.patch.object(server.export_replay, "export") as export:
            jobs = server.Jobs(remote, key=lambda: "secret-key")
            job = jobs.start(request)
            written = json.loads((Path(tmp) / f"configs/requests/{job}.json").read_text())
            with self.assertRaisesRegex(RuntimeError, "already running"):
                jobs.start(request)
            running = jobs.status(job)
            (Path(tmp) / "outputs/isaac/20261102T000000Z-rehearsal").mkdir(parents=True)
            (Path(tmp) / "outputs/isaac/20261102T000000Z-rehearsal/replay.json").write_text("{}")
            done = jobs.status(job)
        self.assertEqual(written["strategy"], "carry")
        upload, start = remote.calls[0], remote.calls[1]
        self.assertIn(f"configs/requests/{job}.json", upload[0])
        self.assertEqual(json.loads(upload[1]), request)
        # The key travels only on standard input.
        self.assertEqual(start[1], b"secret-key\n")
        self.assertTrue(all("secret-key" not in command for command, _ in remote.calls))
        self.assertIn("read -r META_API_KEY", start[0])
        self.assertEqual((running["status"], running["phases"]), ("running", ["approach", "reach"]))
        self.assertEqual((done["status"], done["demo"], remote.fetched), ("done", job, ["20261102T000000Z-rehearsal"]))
        export.assert_called_once()

    def test_queued_moves_run_as_one_sequence(self):
        moves = [{"scene": "configs/scenes/living_room_pedestal_three.json", "target": t, "strategy": "carry",
                  "corrections": {"size_m": [0.28, 0.30, 0.90], "mass_kg": 3.0},
                  "destination": {"center_m": [1.0, 3.0], "yaw_deg": 90.0},
                  "furniture": {"side_table_0": {"size_m": [0.28, 0.30, 0.90]}}} for t in ("side_table_1", "cabinet_0")]
        summary = {"status": "completed_sequence", "sequence_dir": "/remote/outputs/isaac/20261102T000000Z-sequence"}
        # The older sequence folder (a previous job) is ignored; this job's folder holds move 1's run.
        remote = FakeRemote(["", "", "not json", "alive",
                             "/remote/outputs/isaac/20201231T000000Z-sequence\n/remote/outputs/isaac/20991231T000000Z-sequence",
                             "/remote/outputs/isaac/20991231T000000Z-sequence/1/20991231T000100Z-rehearsal",
                             json.dumps({"phases": {"approach": {}}}), json.dumps(summary)])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "ROOT", Path(tmp)), \
                mock.patch.object(server.export_replay, "export_sequence") as export:
            jobs = server.Jobs(remote, key=lambda: "k")
            job = jobs.start({"moves": moves})
            written = json.loads((Path(tmp) / f"configs/requests/{job}.json").read_text())
            running = jobs.status(job)
            local = Path(tmp) / "outputs/isaac/20261102T000000Z-sequence"
            local.mkdir(parents=True)
            (local / "sequence.json").write_text(json.dumps({"runs": [{"run": "0/x-rehearsal"}]}))
            done = jobs.status(job)
        self.assertEqual([m["target"] for m in written["moves"]], ["side_table_1", "cabinet_0"])
        self.assertEqual(written["furniture"], {"side_table_0": {"size_m": [0.28, 0.30, 0.90]}})
        self.assertIn(f"scripts/rehearsal_sequence.py --sequence configs/requests/{job}.json", remote.calls[1][0])
        self.assertEqual((running["move"], running["phases"]), (2, ["approach"]))
        self.assertEqual((done["status"], done["demo"], remote.fetched), ("done", job, ["20261102T000000Z-sequence"]))
        export.assert_called_once()

    def test_a_previous_jobs_run_is_not_shown_as_progress(self):
        remote = FakeRemote(["", "", "not json", "alive", "/r/outputs/isaac/20201231T000000Z-rehearsal"])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "ROOT", Path(tmp)):
            jobs = server.Jobs(remote, key=lambda: "k")
            job = jobs.start(json.loads((ROOT / "configs/rehearsal_living_room_pedestal.json").read_text()))
            running = jobs.status(job)
        self.assertEqual((running["status"], running["phases"]), ("running", []))
        self.assertFalse(any("rehearsal-worker.json" in command for command, _ in remote.calls))

    def test_a_run_that_stopped_without_a_result_is_reported(self):
        # The session had no Isaac runtime: nohup could not start the interpreter.
        log = "nohup: failed to run command '/opt/isaaclab/.venv/bin/python': No such file or directory"
        remote = FakeRemote(["", "", log, "stopped", log])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "ROOT", Path(tmp)):
            jobs = server.Jobs(remote, key=lambda: "k")
            job = jobs.start(json.loads((ROOT / "configs/rehearsal_living_room_pedestal.json").read_text()))
            failed = jobs.status(job)
        self.assertEqual((failed["status"], failed["error"]), ("failed", log))
        self.assertIn(f"requests/{job}.json", remote.calls[3][0])

    def test_unchecked_requests_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Check the move"):
            server.Jobs(FakeRemote([])).start({"strategy": None})


if __name__ == "__main__":
    unittest.main()
