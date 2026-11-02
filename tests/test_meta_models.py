import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import meta_models as mm

# Shape of a real SAM 3.1 reply (run 2026-11-01, phrase "box"): the mask raster is cropped to the box (227 x 93).
SAM_LINE = ("<0f>0<|box;x1=202;y1=74;x2=294;y2=300;w=640;h=360;c=0.83|><|mask;x=0;y=0;c=0.8;data=227,93,!!!!tE|>w|>,"
            "4<|box;x1=1;y1=2;x2=3;y2=4;w=640;h=360|>")


class FakeTransport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, headers, body, stream):
        self.calls.append({"url": url, "headers": headers, "body": body, "stream": stream})
        return self.replies.pop(0)


class MetaModelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.image = self.dir / "frame.png"
        self.image.write_bytes(b"\x89PNG fake")
        patcher = mock.patch.dict(os.environ, {"META_API_KEY": "secret-test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def client(self, replies):
        transport = FakeTransport(replies)
        return mm.MetaModels("meta", cache=self.dir / "cache", log=self.dir / "calls.jsonl", transport=transport), transport

    def test_chat_sends_the_image_and_schema_and_caches_the_reply(self):
        reply = {"choices": [{"message": {"content": json.dumps({"objects": []})}}], "usage": {"total_tokens": 9}}
        client, transport = self.client([reply])
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": mm.image_data_url(self.image)}}]}]
        first = client.chat("muse-spark-1.3", messages, schema=mm.SMOKE_SCHEMA)
        second = client.chat("muse-spark-1.3", messages, schema=mm.SMOKE_SCHEMA)
        self.assertEqual((first["content"], first["cached"], second["cached"]), ({"objects": []}, False, True))
        self.assertEqual(len(transport.calls), 1)
        call = transport.calls[0]
        self.assertEqual(call["url"], "https://api.meta.ai/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret-test-key")
        self.assertEqual(call["body"]["response_format"]["json_schema"]["name"], "VisibleObjects")
        self.assertTrue(call["body"]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
        # The call log records both calls and never the key.
        log = (self.dir / "calls.jsonl").read_text()
        self.assertEqual([json.loads(line)["cache_hit"] for line in log.splitlines()], [False, True])
        self.assertNotIn("secret-test-key", log)
        self.assertNotIn("secret-test-key", "".join(p.read_text() for p in (self.dir / "cache").iterdir()))

    def test_segment_streams_one_phrase_and_parses_boxes_and_masks(self):
        events = [{"type": "response.created"}, {"type": "response.output_text.delta", "delta": SAM_LINE[:60]},
                  {"type": "response.output_text.delta", "delta": SAM_LINE[60:]}, {"type": "response.completed"}]
        client, transport = self.client([events])
        result = client.segment(self.image, "box")
        body = transport.calls[0]["body"]
        self.assertEqual((transport.calls[0]["url"], transport.calls[0]["stream"]), ("https://api.meta.ai/v1/responses", True))
        self.assertEqual([part["type"] for part in body["input"][0]["content"]], ["input_text", "input_image"])
        self.assertEqual(body["metadata"], {"mask_encoding": "one_bit", "include_confidence": "true"})
        first, second = result["records"]
        self.assertEqual((first["id"], first["box"]["x2"], first["box"]["c"]), (0, 294, 0.83))
        # The payload may contain `|>`; it ends only where the next record starts.
        self.assertEqual(first["mask"], {"height": 227, "width": 93, "encoding": "!", "payload": "!!!tE|>w", "c": 0.8})
        self.assertEqual((second["id"], "mask" in second), (4, False))
        self.assertEqual(mm.parse_sam(""), [])

    def test_image_generation_disables_retrieval_tools_and_decodes_the_bytes(self):
        reply = {"created": 1, "data": [{"b64_json": "iVBORw0KGgo="}], "output_format": "png", "usage": {"total_tokens": 3}}
        client, transport = self.client([reply])
        result = client.generate_image("an empty living room", "1536x1024")
        body = transport.calls[0]["body"]
        self.assertEqual(transport.calls[0]["url"], "https://api.meta.ai/v1/images/generations")
        self.assertEqual((body["model"], body["n"], body["size"], body["output_format"]), ("muse-image-1.0", 1, "1536x1024", "png"))
        self.assertEqual(body["tool_enablement"], {"enable_image_search": False, "enable_web_search": False, "enable_shell": False})
        self.assertEqual((result["image"], result["format"], result["cached"]), (b"\x89PNG\r\n\x1a\n", "png", False))

    def test_image_edit_sends_the_picture_without_a_mask(self):
        reply = {"created": 1, "data": [{"b64_json": "iVBORw0KGgo="}], "output_format": "png"}
        client, transport = self.client([reply])
        result = client.edit_image(self.image, "add two pedestals")
        body = transport.calls[0]["body"]
        self.assertEqual(transport.calls[0]["url"], "https://api.meta.ai/v1/images/edits")
        self.assertEqual((body["prompt"], body["n"], body["images"]), ("add two pedestals", 1, [{"image_url": mm.image_data_url(self.image)}]))
        self.assertNotIn("mask", body)
        self.assertFalse(any(body["tool_enablement"].values()))
        self.assertEqual(result["image"], b"\x89PNG\r\n\x1a\n")

    def test_masks_decode_with_the_official_parser_after_a_size_check(self):
        # Published one_bit test vector (meta-sam protocol): a 5 x 5 raster.
        record = {"box": {"x1": 10, "y1": 20, "x2": 14, "y2": 24},
                  "mask": {"height": 5, "width": 5, "encoding": "!", "payload": "!!!!(QO(0lu8?"}}
        self.assertEqual(mm.mask_raster(record).ravel().tolist(),
                         [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "larger than its box"):
            mm.mask_raster({**record, "mask": {**record["mask"], "height": 10**6}})

    def test_key_comes_from_the_environment_or_the_ignored_env_file(self):
        env = self.dir / ".env"
        env.write_text("# local\nOTHER=1\nMETA_API_KEY='from-file'\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mm.api_key("META_API_KEY", env), "from-file")
            with self.assertRaisesRegex(RuntimeError, "Set BASETEN_API_KEY") as caught:
                mm.api_key("BASETEN_API_KEY", env)
            self.assertNotIn("from-file", str(caught.exception))
        self.assertEqual(mm.api_key("META_API_KEY", env), "secret-test-key")
        with self.assertRaises(ValueError):
            mm.MetaModels("unknown")
        with self.assertRaises(ValueError):
            mm.image_data_url(self.dir / "frame.gif")


if __name__ == "__main__":
    unittest.main()
