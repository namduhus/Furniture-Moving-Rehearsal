"""Local, Git-ignored room uploads and the existing Meta photo-to-scene pipeline."""
from __future__ import annotations

import base64
import binascii
import json
import math
from pathlib import Path
import re
import struct
import threading
import uuid
import zlib

import export_replay
import meta_models
import scene_from_image

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 6 * 1024 * 1024
CONVERSION_LOCK = threading.Lock()


def settings():
    return json.loads(scene_from_image.DEFAULT_CONFIG.read_text())


def estimate():
    config = settings()
    return {"sam_calls": len(config["phrases"]), "spark_calls": 1,
            "estimated_usd": round(0.0025 * len(config["phrases"]) + 0.02, 3),
            "max_image_bytes": MAX_IMAGE_BYTES, "reference": config["reference"]}


def png_bytes(encoded):
    """Browser-normalized PNG only; reject oversized, truncated or corrupt chunks before any paid request."""
    if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError("Image must be a PNG no larger than 6 MB")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid image encoding") from None
    if len(data) > MAX_IMAGE_BYTES or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Upload a valid PNG image")
    offset, tags = 8, []
    while offset + 12 <= len(data):
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        tag, chunk = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        if offset + size + 12 > len(data) or zlib.crc32(tag + chunk) != struct.unpack(">I", data[offset + 8 + size:offset + 12 + size])[0]:
            raise ValueError("Image is truncated or corrupt")
        if not tags:
            if tag != b"IHDR" or size != 13:
                raise ValueError("Invalid PNG header")
            w, h = struct.unpack(">II", chunk[:8])
            if not (64 <= w <= 4096 and 64 <= h <= 4096 and w * h <= 8_000_000):
                raise ValueError("Image must be 64–4096 pixels per side, at most 8 megapixels")
        tags.append(tag)
        offset += size + 12
        if tag == b"IEND":
            break
    if not tags or tags[-1] != b"IEND" or b"IDAT" not in tags or offset != len(data):
        raise ValueError("Incomplete PNG image")
    return data


def validate_scene(result):
    """Do not register model output that the viewer cannot draw or the simulator cannot interpret."""
    spec = result["scene"]
    if not 1 <= len(spec["furniture"]) <= 60:
        raise ValueError("Meta did not identify 1–60 furniture items. Try a clearer room image.")
    ids = [item["id"] for item in spec["furniture"]]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", v) for v in ids):
        raise ValueError("Meta returned invalid furniture identifiers. Try the conversion again.")
    for item in spec["furniture"]:
        if not isinstance(item["category"], str) or not re.fullmatch(r"[\w ()-]{1,80}", item["category"]):
            raise ValueError("Meta returned an invalid furniture category")
        if not math.isfinite(item["yaw_deg"]):
            raise ValueError("Meta returned an invalid furniture orientation")
    checks = scene_from_image.check_scene(spec, result["detections"])
    result["checks_failed"] = checks
    # A visible item may be estimated without its own SAM mask (for example a TV on a detected TV stand).
    # Keep that warning visible; missing calibration evidence or invalid geometry still blocks registration.
    failures = [message for message in checks if not message.endswith(": no SAM detection")]
    door = spec["door"]
    limit = spec["room"]["width_m" if door["wall"] == "back" else "depth_m"]
    if not math.isfinite(door["center_m"]) or not 0 <= door["center_m"] <= limit:
        failures.append("door center outside the room")
    if failures:
        raise ValueError("Room estimate needs another conversion: " + "; ".join(failures))


def convert(body, models=None, root=None):
    root = Path(root or ROOT)
    if body.get("confirm_meta") is not True:
        raise ValueError("Confirm image submission to Meta and the estimated API cost")
    if not isinstance(body.get("reference"), dict):
        raise ValueError("Enter the known door height and width")
    reference = {"object": "door"}
    for key, low, high in (("height_m", 1.5, 3.0), ("width_m", 0.5, 1.8)):
        value = float(body.get("reference", {}).get(key, 0))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Door {key} must be between {low} and {high}")
        reference[key] = value
    data = png_bytes(body.get("image_base64"))
    if not CONVERSION_LOCK.acquire(blocking=False):
        raise RuntimeError("A room conversion is already running. Wait for it to finish.")
    try:
        scene_id = "upload-" + uuid.uuid4().hex
        folder = root / "outputs/uploads" / scene_id
        folder.mkdir(parents=True)
        image = folder / "photo.png"
        image.write_bytes(data)
        models = models or meta_models.MetaModels()
        try:
            result = scene_from_image.scene_from_image(image, {**settings(), "reference": reference}, models)
        except (KeyError, TypeError, OverflowError):
            raise ValueError("Meta returned an incomplete room estimate. Try a clearer image.") from None
        result["image"] = image.relative_to(root).as_posix()
        (folder / "estimate.json").write_text(json.dumps(result, indent=2) + "\n")
        try:
            validate_scene(result)
        except (KeyError, TypeError, OverflowError):
            raise ValueError("Meta returned an incomplete room estimate. Try a clearer image.") from None
        looks = export_replay.appearance(result, models)
        result["appearance"] = looks
        target = root / "configs/scenes" / f"{scene_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
        return {"id": scene_id, "furniture_count": len(result["scene"]["furniture"]),
                "notes": result["scene"].get("notes", ""), "warnings": result["checks_failed"], "reference": reference}
    finally:
        CONVERSION_LOCK.release()
