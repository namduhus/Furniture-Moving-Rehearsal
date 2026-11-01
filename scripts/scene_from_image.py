"""Image -> metric scene spec with Meta models: SAM 3.1 finds furniture, Muse Spark lays it out on a floor plan.

SAM 3.1 runs once per short noun phrase; its boxes, confidences and decoded mask areas are passed with the image to
Muse Spark, which returns a JSON floor plan in meters, scaled by assuming a standard door size. The result keeps the
SAM evidence next to each estimate and lists checks that failed; estimates are for user correction, not measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import meta_models

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/scene_from_image.json"

SCENE_SCHEMA = {
    "title": "SceneSpec", "type": "object", "additionalProperties": False,
    "required": ["room", "door", "furniture", "notes"],
    "properties": {
        "room": {"type": "object", "additionalProperties": False, "required": ["width_m", "depth_m", "height_m"],
                 "properties": {"width_m": {"type": "number"}, "depth_m": {"type": "number"}, "height_m": {"type": "number"}}},
        "door": {"type": "object", "additionalProperties": False, "required": ["detection_id", "wall", "center_m"],
                 "properties": {"detection_id": {"type": "string"}, "wall": {"type": "string", "enum": ["left", "right", "back"]},
                                "center_m": {"type": "number"}}},
        "furniture": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "category", "detection_id", "size_m", "center_m", "yaw_deg", "estimated_mass_kg", "mass_basis"],
            "properties": {"id": {"type": "string"}, "category": {"type": "string"}, "detection_id": {"type": "string"},
                           "size_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                           "center_m": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                           "yaw_deg": {"type": "number"}, "estimated_mass_kg": {"type": "number"},
                           "mass_basis": {"type": "string"}}}},
        "notes": {"type": "string"}}}

PROMPT = """You are building a metric floor plan of this room photo for a robot simulation.
Detections from a segmentation model (SAM 3.1), pixel boxes [x1, y1, x2, y2] in a {w}x{h} image:
{detections}
Assume the door is a standard interior door {door_h} m tall and {door_w} m wide, and use it with perspective cues to
scale everything. Floor-plan frame: origin at the back-left floor corner as seen from the camera, x to the right along
the back wall, y toward the camera, meters. Report the visible room width (x), depth (y) and ceiling height.
For each furniture item: use the matching detection id (or "none"), its size [width along its front, depth, height] in
meters, its footprint center [x, y], yaw in degrees (0 = front faces the camera), and a typical mass for such an item
with a short basis. For the door give its wall and the center position along that wall. Do not invent items that are
not visible. Put uncertainties in notes."""


def detect(models, image, phrases):
    """SAM 3.1 boxes per phrase with confidence and the decoded mask's area fraction of its box."""
    found = []
    for phrase in phrases:
        for record in models.segment(image, phrase)["records"]:
            box = record["box"]
            found.append({"id": f"{phrase.replace(' ', '_')}_{record['id']}", "phrase": phrase,
                          "box_px": [box["x1"], box["y1"], box["x2"], box["y2"]], "image_px": [box["w"], box["h"]],
                          "confidence": box.get("c"),
                          "mask_fill": float(meta_models.mask_raster(record).mean()) if "mask" in record else None})
    return found


def check_scene(scene, detections):
    """Plausibility checks of the model's floor plan; failures are reported, not corrected."""
    ids = {d["id"] for d in detections}
    room = scene["room"]
    failures = []
    if not (2.0 <= room["width_m"] <= 12.0 and 2.0 <= room["depth_m"] <= 12.0 and 2.2 <= room["height_m"] <= 4.0):
        failures.append("room size outside 2-12 m (height 2.2-4 m)")
    if scene["door"]["detection_id"] not in ids:
        failures.append("door is not tied to a SAM detection")
    for item in scene["furniture"]:
        w, d, h = item["size_m"]
        x, y = item["center_m"]
        if item["detection_id"] not in ids:
            failures.append(f"{item['id']}: no SAM detection")
        if not all(0.02 <= v <= 3.5 for v in (w, d, h)):
            failures.append(f"{item['id']}: size outside 0.02-3.5 m")
        if not (0.0 <= x <= room["width_m"] and 0.0 <= y <= room["depth_m"]):
            failures.append(f"{item['id']}: center outside the room")
        if not 0.5 <= item["estimated_mass_kg"] <= 300.0:
            failures.append(f"{item['id']}: mass outside 0.5-300 kg")
    return failures


def scene_from_image(image, config, models):
    detections = detect(models, image, config["phrases"])
    w, h = detections[0]["image_px"] if detections else (0, 0)
    ref = config["reference"]
    listing = "\n".join(f"- {d['id']} ({d['phrase']}): box {d['box_px']}, confidence {d['confidence']}" for d in detections) or "- none"
    reply = models.chat(config["chat_model"], [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(w=w, h=h, detections=listing, door_h=ref["height_m"], door_w=ref["width_m"])},
        {"type": "image_url", "image_url": {"url": meta_models.image_data_url(image)}}]}], schema=SCENE_SCHEMA)
    scene = reply["content"]
    return {"image": str(image), "models": {"segmentation": "sam-3.1", "layout": config["chat_model"]},
            "reference": ref, "detections": detections, "scene": scene, "checks_failed": check_scene(scene, detections),
            "usage": reply["usage"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/meta/scenes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "sam_calls": len(config["phrases"]), "chat_calls": 1,
                          "estimated_usd": round(0.0025*len(config["phrases"]) + 0.02, 3)}, indent=2))
        return 0
    result = scene_from_image(args.image, config, meta_models.MetaModels())
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.image.stem}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scene": str(path), "detections": [(d["id"], d["confidence"]) for d in result["detections"]],
                      "furniture": [(f["id"], f["size_m"], f["center_m"], f["estimated_mass_kg"]) for f in result["scene"]["furniture"]],
                      "room": result["scene"]["room"], "checks_failed": result["checks_failed"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
