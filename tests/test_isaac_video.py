"""CPU tests for encoding, decoding, timing and partial recording rejection."""

from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isaac_video as video


def write_test_frame(path: Path, index: int, width: int = 64, height: int = 48):
    """An authored moving-square PNG fixture, requiring no imaging dependencies."""
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            color = (220, 20, 30) if index * 3 <= x < index * 3 + 12 and 8 <= y < 28 else (30, 80, 150)
            rows.extend(color)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class VideoTests(unittest.TestCase):
    def test_static_standing_video_is_explicitly_allowed_without_weakening_smoke(self):
        text = "#tb 0: 1/25\n#dimensions 0: 64x64\n0,0,0,1,100,same\n0,1,1,1,100,same\n"
        with self.assertRaises(RuntimeError):
            video.inspect_frame_checksums(text, 2, Fraction(25), 64, 64)
        self.assertEqual(video.inspect_frame_checksums(text, 2, Fraction(25), 64, 64, require_motion=False)["unique_decoded_frames"], 1)

    def test_missing_encoder_fails_explicitly(self):
        with mock.patch.object(video.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "updated Modal Isaac image"):
                video.require_ffmpeg(ROOT)

    def test_decoded_static_frames_and_wrong_timing_are_rejected(self):
        base = "#tb 0: 1/25\n#dimensions 0: 64x48\n0, 0, 0, 1, 4608, a\n"
        with self.assertRaisesRegex(RuntimeError, "changing frames"):
            video.inspect_frame_checksums(base + "0, 1, 1, 1, 4608, a\n", 2, Fraction(25), 64, 48)
        with self.assertRaisesRegex(RuntimeError, "timestamps"):
            video.inspect_frame_checksums(base + "0, 2, 2, 1, 4608, b\n", 2, Fraction(25), 64, 48)
        with self.assertRaisesRegex(RuntimeError, "frame count"):
            video.inspect_frame_checksums(base, 2, Fraction(25), 64, 48)

    def test_incomplete_png_sequence_cannot_create_a_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "frames").mkdir()
            write_test_frame(path / "frames/000000.png", 0)
            with self.assertRaisesRegex(RuntimeError, "Incomplete source"):
                video.encode_recording(path, binary="must-not-run", count=2,
                                       fps=Fraction(25), width=64, height=48)
            self.assertFalse((path / "rollout.mp4").exists())

    def test_h264_roundtrip_and_timeline_evidence(self):
        binary = shutil.which(os.environ.get("FFMPEG_BINARY", "ffmpeg"))
        if binary is None:
            self.skipTest("Set FFMPEG_BINARY to run the real CPU H.264 roundtrip")
        with tempfile.TemporaryDirectory(prefix="fmr video with spaces-") as directory:
            path = Path(directory)
            (path / "frames").mkdir()
            for index in range(8):
                write_test_frame(path / f"frames/{index:06d}.png", index)
            actual_binary = video.require_ffmpeg(path)
            result = video.encode_recording(path, binary=actual_binary, count=8,
                                            fps=Fraction(25), width=64, height=48)
            self.assertEqual(result["decoded_frames"], 8)
            self.assertGreater(result["unique_decoded_frames"], 1)
            self.assertAlmostEqual(result["duration_seconds"], 0.32)
            self.assertTrue((path / "rollout.mp4").is_file())
            self.assertFalse((path / "rollout.partial.mp4").exists())
            result["unique_source_frames"] = 8
            (path / "video.json").write_text(json.dumps(result))
            config = {"steps": 16, "recording": {"capture_every_steps": 2},
                      "camera": {"width": 64, "height": 48}}
            evidence = {"control_dt_seconds": 0.02, "video": result}
            rows = [{"step": step, "sim_time_seconds": step * 0.02,
                     "video_frame_index": step // 2 - 1 if step % 2 == 0 else None,
                     "video_time_seconds": (step // 2 - 1) / 25 if step % 2 == 0 else None}
                    for step in range(1, 17)]
            timeline = path / "trajectory.jsonl"
            timeline.write_text("".join(json.dumps(row) + "\n" for row in rows))
            video.validate_recording(path, config, evidence)
            rows[1]["video_frame_index"] = 1
            timeline.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(RuntimeError, "timeline"):
                video.validate_recording(path, config, evidence)
            with (path / "rollout.mp4").open("ab") as output:
                output.write(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                video.validate_recording(path, config, evidence)


if __name__ == "__main__":
    unittest.main()
