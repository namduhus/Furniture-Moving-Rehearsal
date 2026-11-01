import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_replay
import fetch_g1_meshes


class MeshFetchTests(unittest.TestCase):
    def test_manifest_covers_the_urdf_visual_meshes(self):
        import re
        manifest = json.loads(fetch_g1_meshes.MANIFEST.read_text())
        names = set(re.findall(r'filename="meshes/([^"]+)"', (ROOT / "assets/references/g1_wbc.urdf").read_text()))
        self.assertEqual(set(manifest["files"]), names)
        self.assertIn(manifest["commit"], manifest["download"])

    def test_fetch_verifies_size_and_hash_and_keeps_good_files(self):
        good, bad = b"solid mesh", b"tampered"
        manifest = {"download": "https://example.invalid/{name}",
                    "files": {"a.STL": {"sha256": hashlib.sha256(good).hexdigest(), "bytes": len(good)}}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            urls = []
            fetched, kept = fetch_g1_meshes.fetch(manifest, out, get=lambda url: urls.append(url) or good)
            self.assertEqual((fetched, kept, urls), (["a.STL"], [], ["https://example.invalid/a.STL"]))
            self.assertEqual(fetch_g1_meshes.fetch(manifest, out, get=lambda url: self.fail("re-downloaded")), ([], ["a.STL"]))
            (out / "a.STL").write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "does not match"):
                fetch_g1_meshes.fetch(manifest, out, get=lambda url: bad)
            self.assertEqual((out / "a.STL").read_bytes(), bad)        # a failed download writes nothing
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                fetch_g1_meshes.fetch({**manifest, "files": {"../x.STL": manifest["files"]["a.STL"]}}, out, get=lambda url: good)


class ExportTests(unittest.TestCase):
    def test_export_copies_the_replay_and_condenses_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261101T000000Z-rehearsal"
            run.mkdir()
            (run / "replay.json").write_text('{"frames": []}')
            (run / "scene.json").write_text(json.dumps({"image": "outputs/meta/images/none.png", "scene": {}}))
            worker = {"status": "completed_rehearsal", "request": {"target": "x"}, "phases": {"push": {"outcome": "success"}},
                      "advice": {"decision": {"strategy": "push"}}, "layout": {"placed": [0, 1, 0]}, "env_internal": 1}
            (run / "rehearsal-worker.json").write_text(json.dumps(worker))
            target = export_replay.export(run, "demo", out=Path(tmp) / "web")
            rehearsal = json.loads((target / "rehearsal.json").read_text())
        self.assertEqual(rehearsal["run"], run.name)
        self.assertEqual(rehearsal["advice"]["decision"]["strategy"], "push")
        self.assertNotIn("env_internal", rehearsal)
        self.assertEqual(rehearsal["walls"], {"thickness_m": 0.1, "height_m": 1.2})


class ThreeVendorTests(unittest.TestCase):
    def test_vendored_three_files_match_the_manifest(self):
        meta = json.loads((ROOT / "web/vendor/three/vendor.json").read_text())
        for name, info in meta["files"].items():
            data = (ROOT / "web/vendor/three" / name).read_bytes()
            self.assertEqual((hashlib.sha256(data).hexdigest(), len(data)), (info["sha256"], info["bytes"]), name)


if __name__ == "__main__":
    unittest.main()
