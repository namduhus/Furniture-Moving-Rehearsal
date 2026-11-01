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
        remote = FakeRemote(["", "", "still running", "20261102T000000Z-rehearsal",
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

    def test_unchecked_requests_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Check the move"):
            server.Jobs(FakeRemote([])).start({"strategy": None})


if __name__ == "__main__":
    unittest.main()
