import base64
from functools import partial
import http.client
import http.server
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import room_upload as upload
import rehearsal_server as server
import isaac_g1_rehearsal as rehearsal


def png():
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress((b"\x00" + b"\x99\xaa\xbb" * 64) * 64)) + chunk(b"IEND", b""))


class Models:
    def __init__(self):
        self.calls = []

    def segment(self, image, phrase):
        self.calls.append(phrase)
        ids = {"door": 3, "cabinet": 1}
        return {"records": [] if phrase not in ids else [
            {"id": ids[phrase], "box": {"x1": 2, "y1": 2, "x2": 40, "y2": 60, "w": 64, "h": 64, "c": .9}}]}

    def chat(self, model, messages, schema=None):
        self.calls.append(model)
        return {"content": {"room": {"width_m": 4., "depth_m": 5., "height_m": 2.6},
                "door": {"detection_id": "door_3", "wall": "right", "center_m": 1.},
                "furniture": [{"id": "cab", "category": "cabinet", "detection_id": "cabinet_1",
                               "size_m": [.5, .4, .6], "center_m": [1., 2.], "yaw_deg": 0.,
                               "estimated_mass_kg": 12., "mass_basis": "wood"}], "notes": "Estimated layout"}, "usage": {}}


def body():
    return {"image_base64": base64.b64encode(png()).decode(), "confirm_meta": True,
            "reference": {"height_m": 2.1, "width_m": .9}}


class UploadTests(unittest.TestCase):
    def test_upload_round_trip_registers_a_scene_that_existing_viewer_and_requests_can_use(self):
        config, training = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        models = Models()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(upload, "ROOT", Path(tmp)), \
                mock.patch.object(server, "ROOT", Path(tmp)), mock.patch.object(upload.export_replay, "ROOT", Path(tmp)):
            result = upload.convert(body(), models)
            path = server.scene_path(result["id"])
            saved = json.loads(path.read_text())
            view = server.scene_view(result["id"], config, training)
            request = server.move_request({"scene": result["id"], "target": "cab",
                                          "corrections": {"size_m": [.3, .28, .9], "mass_kg": 3}})
            self.assertEqual((Path(tmp) / saved["image"]).read_bytes(), png())
            self.assertEqual(request["scene"], f"configs/scenes/{result['id']}.json")
            self.assertTrue(view["uploaded"])
            self.assertEqual(view["furniture"][0]["id"], "cab")
            self.assertEqual(view["reference"]["height_m"], 2.1)
            self.assertEqual(view["appearance"]["image_px"], [64, 64])
            self.assertNotIn(str(Path(tmp)), path.read_text())
        self.assertEqual(models.calls.count("muse-spark-1.3"), 1)

    def test_invalid_uploads_do_not_call_models(self):
        for changes in ({"confirm_meta": False}, {"reference": None}, {"image_base64": "not base64!"},
                        {"image_base64": base64.b64encode(b"not PNG").decode()},
                        {"reference": {"height_m": float("nan"), "width_m": .8}},
                        {"image_base64": base64.b64encode(png()[:-2]).decode()}):
            models = Models()
            with self.subTest(changes=changes.keys()), self.assertRaises(ValueError):
                upload.convert({**body(), **changes}, models)
            self.assertEqual(models.calls, [])

    def test_invalid_estimate_is_retained_locally_but_not_registered(self):
        models = Models()
        original = models.chat
        def invalid(*args, **kwargs):
            reply = original(*args, **kwargs)
            reply["content"]["furniture"][0]["size_m"] = [float("nan"), .3, .9]
            return reply
        models.chat = invalid
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "size outside"):
            upload.convert(body(), models, root=Path(tmp))
            self.fail("Invalid room registered")

    def test_scene_ids_cannot_traverse_paths(self):
        for scene_id in ("../.env", "foo/bar", "<img>", "bad name", "x;cmd", None):
            with self.subTest(scene_id=scene_id), self.assertRaises(ValueError):
                server.scene_path(scene_id)

    def test_maskless_item_is_a_visible_warning_and_invalid_door_still_blocks(self):
        models = Models()
        reply = models.chat(None, [], None)["content"]
        result = {"scene": reply, "detections": [{"id": "door_3"}], "checks_failed": []}
        upload.validate_scene(result)
        self.assertEqual(result["checks_failed"], ["cab: no SAM detection"])
        reply["door"]["center_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "door center"):
            upload.validate_scene(result)

    def test_parallel_conversion_is_rejected_without_a_paid_call(self):
        models = Models()
        upload.CONVERSION_LOCK.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                upload.convert(body(), models)
        finally:
            upload.CONVERSION_LOCK.release()
        self.assertEqual(models.calls, [])

    def test_http_upload_then_list_and_load_with_no_external_services(self):
        config, training = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(upload, "ROOT", Path(tmp)), \
                mock.patch.object(server, "ROOT", Path(tmp)), mock.patch.object(upload.export_replay, "ROOT", Path(tmp)), \
                mock.patch.object(upload.meta_models, "MetaModels", return_value=Models()):
            handler = partial(server.handler(config, training, None), directory=tmp)
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            worker = threading.Thread(target=httpd.serve_forever, daemon=True); worker.start()
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port)
            try:
                conn.request("POST", "/api/convert-room", json.dumps(body()), {"Content-Type": "application/json"})
                response = conn.getresponse(); converted = json.loads(response.read())
                self.assertEqual(response.status, 200, converted)
                conn.request("GET", "/api/scenes")
                self.assertIn(converted["id"], json.loads(conn.getresponse().read()))
                conn.request("GET", "/api/scene?id=" + converted["id"])
                loaded = json.loads(conn.getresponse().read())
                self.assertEqual(loaded["furniture"][0]["category"], "cabinet")
                conn.request("GET", loaded["photo"])
                self.assertEqual(conn.getresponse().read(), png())
                conn.request("POST", "/api/convert-room", "{}", {"Origin": "https://unrelated.example"})
                response = conn.getresponse(); response.read(); self.assertEqual(response.status, 403)
            finally:
                conn.close(); httpd.shutdown(); httpd.server_close(); worker.join()

    def test_uploaded_scene_and_image_are_sent_before_the_rehearsal_request(self):
        scene_id = "upload-" + "a" * 32
        calls = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, "ROOT", Path(tmp)):
            root = Path(tmp); image = f"outputs/uploads/{scene_id}/photo.png"
            (root / image).parent.mkdir(parents=True); (root / image).write_bytes(png())
            spec = root / f"configs/scenes/{scene_id}.json"; spec.parent.mkdir(parents=True)
            spec.write_text(json.dumps({"image": image}))
            jobs = server.Jobs(lambda command, stdin=b"": calls.append((command, stdin)))
            jobs.start({"scene": f"configs/scenes/{scene_id}.json", "strategy": "carry"})
        self.assertIn(f"configs/scenes/{scene_id}.json", calls[0][0])
        self.assertEqual(calls[1][1], png())
        self.assertIn("configs/requests/", calls[2][0])


if __name__ == "__main__":
    unittest.main()
