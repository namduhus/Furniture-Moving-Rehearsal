import hashlib
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
            (run / "scene.json").write_text(json.dumps({"image": "outputs/meta/images/none.png", "scene": {"furniture": []}}))
            worker = {"status": "completed_rehearsal", "request": {"target": "x", "strategy": "carry"},
                      "phases": {"push": {"outcome": "success"}},
                      "advice": {"decision": {"strategy": "push"}}, "layout": {"placed": [0, 1, 0]}, "env_internal": 1}
            (run / "rehearsal-worker.json").write_text(json.dumps(worker))
            target = export_replay.export(run, "demo", out=Path(tmp) / "web")
            rehearsal = json.loads((target / "rehearsal.json").read_text())
        self.assertEqual(rehearsal["run"], run.name)
        self.assertEqual(rehearsal["advice"]["decision"]["strategy"], "push")
        self.assertNotIn("env_internal", rehearsal)
        self.assertEqual(rehearsal["walls"], {"thickness_m": 0.1, "height_m": 1.2})

    def test_pre_fix_auto_heading_is_rotated_into_the_grip_frame(self):
        # Run 20261101T145650: bookcase facing the camera (spec yaw 0), gripped from its end (box yaw pi); the
        # destination keeps the heading but was stored in the front frame (pi/2).
        scene = {"scene": {"furniture": [{"id": "cabinet_1", "yaw_deg": 0.0}]}}
        worker = {"request": {"strategy": "auto", "target": "cabinet_1", "destination": {"yaw_deg": 0.0}},
                  "layout": {"placed": [0.0, 2.3, np.pi/2], "cabinet": [0.0, 1.8, np.pi]},
                  "placement": {"position_m": 0.061, "yaw_rad": -3.003, "yaw_deg": 97.9, "within_tolerance": False}}
        fixed = export_replay.grip_frame_heading(worker, scene)
        self.assertAlmostEqual(fixed["layout"]["placed"][2], np.pi)
        self.assertAlmostEqual(fixed["placement"]["yaw_deg"], np.degrees(np.pi - 3.003), places=6)
        self.assertTrue(fixed["placement"]["within_tolerance"])
        # A run already in the grip frame, or a carry run, is left alone.
        self.assertIs(export_replay.grip_frame_heading(fixed, scene), fixed)
        carry = {**worker, "request": {**worker["request"], "strategy": "carry"}}
        self.assertIs(export_replay.grip_frame_heading(carry, scene), carry)

    def test_appearance_reads_sam_masks_from_the_cache_only(self):
        scene = {"image": "configs/scenes/living_room_pedestal.json",            # any existing file stands in for the photo
                 "detections": [{"id": "sofa_0", "phrase": "sofa", "box_px": [10, 20, 110, 220], "image_px": [640, 480]},
                                {"id": "door_0", "phrase": "door", "box_px": [0, 0, 5, 5], "image_px": [640, 480]}],
                 "scene": {"furniture": [{"id": "sofa_1", "detection_id": "sofa_0"}, {"id": "lamp_1", "detection_id": "door_0"},
                                         {"id": "tv_1", "detection_id": "none"}]}}

        class Models:
            def segment(self, image, phrase):
                if phrase == "door":
                    raise LookupError("not cached")
                return {"records": [{"id": 0, "box": {}, "mask": {}}]}

        mask = np.zeros((200, 100), dtype=np.uint8)
        mask[:, :50] = 1                                                       # left half of the box
        with mock.patch.object(export_replay.meta_models, "mask_raster", return_value=mask):
            looks = export_replay.appearance(scene, Models())
        self.assertEqual(set(looks["items"]), {"sofa_1"})
        grid = looks["items"]["sofa_1"]["grid"]
        half = export_replay.GRID//2
        self.assertEqual((len(grid), grid[0]), (export_replay.GRID, "1"*half + "0"*half))
        self.assertEqual(looks["image_px"], [640, 480])


class ThreeVendorTests(unittest.TestCase):
    def test_vendored_three_files_match_the_manifest(self):
        meta = json.loads((ROOT / "web/vendor/three/vendor.json").read_text())
        for name, info in meta["files"].items():
            data = (ROOT / "web/vendor/three" / name).read_bytes()
            self.assertEqual((hashlib.sha256(data).hexdigest(), len(data)), (info["sha256"], info["bytes"]), name)


if __name__ == "__main__":
    unittest.main()
