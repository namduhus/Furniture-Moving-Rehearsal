"""Encode self-rendered PNG frames and verify decoded MP4 timing and motion."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require_ffmpeg(directory: Path) -> str:
    binary = shutil.which(os.environ.get("FFMPEG_BINARY", "ffmpeg"))
    if binary is None:
        raise RuntimeError("Recording requires FFmpeg on PATH; use the updated Modal Isaac image")
    version = subprocess.run([binary, "-version"], capture_output=True, text=True,
                             check=True, timeout=10)
    (directory / "ffmpeg-version.txt").write_text(version.stdout)
    encoders = subprocess.run([binary, "-hide_banner", "-encoders"], capture_output=True,
                              text=True, check=True, timeout=10)
    if not re.search(r"\blibx264\b", encoders.stdout):
        raise RuntimeError("FFmpeg does not provide the required libx264 H.264 encoder")
    return binary


def inspect_frame_checksums(text: str, count: int, fps: Fraction,
                            width: int, height: int) -> dict:
    """Require every decoded frame, its dimensions, timestamp and visible changes."""
    timebase = None
    dimensions = None
    frames = []
    for line in text.splitlines():
        if line.startswith("#tb 0:"):
            timebase = Fraction(line.split(":", 1)[1].strip())
        elif line.startswith("#dimensions 0:"):
            dimensions = line.split(":", 1)[1].strip()
        elif line.strip() and not line.startswith("#"):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6 or fields[0] != "0":
                raise RuntimeError("Unexpected decoded video checksum record")
            frames.append((int(fields[2]), int(fields[3]), fields[5]))
    if timebase is None or dimensions != f"{width}x{height}" or len(frames) != count:
        raise RuntimeError("Decoded video dimensions or frame count do not match the recording")
    for index, (pts, duration, _) in enumerate(frames):
        if pts * timebase != Fraction(index, 1) / fps or duration * timebase != 1 / fps:
            raise RuntimeError("Decoded video timestamps do not match simulated-time playback")
    unique = len({checksum for _, _, checksum in frames})
    if unique < 2:
        raise RuntimeError("Decoded video has no changing frames; inspect camera rendering")
    return {"decoded_frames": count, "unique_decoded_frames": unique,
            "duration_seconds": float(Fraction(count, 1) / fps),
            "fps": str(fps), "width": width, "height": height}


def encode_recording(directory: Path, *, binary: str, count: int, fps: Fraction,
                     width: int, height: int, timeout: int = 120) -> dict:
    """Keep partial output on failure; publish rollout.mp4 only after decoding it."""
    frames = directory / "frames"
    if count < 2 or fps <= 0 or width % 2 or height % 2:
        raise ValueError("Recording needs at least two frames, positive FPS and even dimensions")
    if len(list(frames.glob("*.png"))) != count:
        raise RuntimeError("Incomplete source recording frames")
    for index in range(count):
        if not (frames / f"{index:06d}.png").is_file():
            raise RuntimeError("Source recording frames are not contiguous")
    partial = directory / "rollout.partial.mp4"
    final = directory / "rollout.mp4"
    if partial.exists() or final.exists():
        raise FileExistsError("Refusing to overwrite an existing recording")
    command = [binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-framerate", str(fps), "-start_number", "0", "-f", "image2",
               "-i", str(frames / "%06d.png"), "-frames:v", str(count), "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-threads", "2",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(partial)]
    (directory / "video-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "video-encode.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    with (directory / "video-decode.log").open("w") as log:
        decoded = subprocess.run(
            [binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2",
             "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(partial),
             "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-f", "framemd5", "-"],
            stdout=subprocess.PIPE, stderr=log, text=True, check=True, timeout=timeout,
        )
    (directory / "video-frames.md5").write_text(decoded.stdout)
    evidence = inspect_frame_checksums(decoded.stdout, count, fps, width, height)
    partial.rename(final)
    evidence.update({"status": "passed_video_decode", "file": "rollout.mp4",
                     "sha256": file_sha256(final), "codec": "libx264", "pixel_format": "yuv420p"})
    (directory / "video.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def validate_recording(directory: Path, config: dict, evidence: dict) -> None:
    """Cross-check persisted video evidence against the step-to-frame timeline."""
    count = config["steps"] // config["recording"]["capture_every_steps"]
    video = json.loads((directory / "video.json").read_text())
    if video != evidence.get("video") or video.get("status") != "passed_video_decode":
        raise RuntimeError("Incomplete video decode evidence")
    if (video.get("decoded_frames") != count or video.get("unique_decoded_frames", 0) < 2
            or video.get("unique_source_frames", 0) < 2):
        raise RuntimeError("Incomplete or static recorded video")
    if (video.get("width"), video.get("height")) != (config["camera"]["width"], config["camera"]["height"]):
        raise RuntimeError("Unexpected recorded video dimensions")
    if video.get("sha256") != file_sha256(directory / "rollout.mp4"):
        raise RuntimeError("Recorded video changed after verification")
    stride = config["recording"]["capture_every_steps"]
    fps = Fraction(video["fps"])
    if fps <= 0:
        raise RuntimeError("Invalid recorded video FPS")
    control_dt = float(evidence["control_dt_seconds"])
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise RuntimeError("Invalid recorded environment step size")
    if abs(float(1 / fps) - control_dt * stride) > 1e-8:
        raise RuntimeError("Recorded FPS does not match the environment step size")
    if (not math.isfinite(video["duration_seconds"])
            or abs(video["duration_seconds"] - config["steps"] * control_dt) > 1e-8):
        raise RuntimeError("Recorded duration does not match simulated time")
    rows = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    for index, row in enumerate(rows):
        step = index + 1
        frame_index = step // stride - 1 if step % stride == 0 else None
        if row.get("step") != step or row.get("video_frame_index") != frame_index:
            raise RuntimeError("Incomplete step-to-frame timeline")
        if (not math.isfinite(row["sim_time_seconds"])
                or abs(row["sim_time_seconds"] - step * control_dt) > 1e-8):
            raise RuntimeError("Incorrect simulation timestamp in recording")
        wanted_time = frame_index / float(fps) if frame_index is not None else None
        if row.get("video_time_seconds") != wanted_time:
            raise RuntimeError("Incorrect playback timestamp in recording")
    if len(rows) != config["steps"]:
        raise RuntimeError("Incomplete recording trajectory")
