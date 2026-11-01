import json
from pathlib import Path
import sys
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scene_from_image as sfi


class FakeModels:
    """SAM finds a door and one cabinet; Muse Spark returns the given scene."""

    def __init__(self, scene):
        self.scene, self.phrases, self.prompts = scene, [], []

    def segment(self, image, phrase):
        self.phrases.append(phrase)
        boxes = {"door": "<0f>3<|box;x1=10;y1=5;x2=30;y2=90;w=200;h=100;c=0.9|>",
                 "cabinet": "<0f>1<|box;x1=120;y1=60;x2=150;y2=90;w=200;h=100;c=0.7|><|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>"}
        text = boxes.get(phrase, "")
        return {"records": sfi.meta_models.parse_sam(text), "text": text, "cached": False}

    def chat(self, model, messages, schema=None):
        self.prompts.append(messages[0]["content"][0]["text"])
        return {"content": self.scene, "usage": {"total_tokens": 1}, "cached": False}


def scene(**changes):
    base = {"room": {"width_m": 4.0, "depth_m": 5.0, "height_m": 2.6},
            "door": {"detection_id": "door_3", "wall": "right", "center_m": 1.0},
            "furniture": [{"id": "cab", "category": "cabinet", "detection_id": "cabinet_1", "size_m": [0.5, 0.4, 0.6],
                           "center_m": [3.5, 2.0], "yaw_deg": -90, "estimated_mass_kg": 12, "mass_basis": "wood"}],
            "notes": ""}
    return {**base, **changes}


class SceneFromImageTests(unittest.TestCase):
    def test_sam_detections_go_into_the_layout_prompt_and_pass_the_checks(self):
        image = ROOT / "outputs/meta/images/living_room.png"
        config = {"chat_model": "muse-spark-1.3", "phrases": ["door", "cabinet", "bed"],
                  "reference": {"object": "door", "height_m": 2.0, "width_m": 0.8}}
        models = FakeModels(scene())
        if not image.exists():
            image = Path(__file__)
        with unittest.mock.patch.object(sfi.meta_models, "image_data_url", return_value="data:image/png;base64,AA=="):
            result = sfi.scene_from_image(image, config, models)
        self.assertEqual(models.phrases, ["door", "cabinet", "bed"])
        self.assertEqual([d["id"] for d in result["detections"]], ["door_3", "cabinet_1"])
        self.assertAlmostEqual(result["detections"][1]["mask_fill"], 10/25)   # 10 set pixels in the published vector
        self.assertIn("door_3 (door): box [10, 5, 30, 90], confidence 0.9", models.prompts[0])
        self.assertIn("200x100 image", models.prompts[0])
        self.assertEqual(result["checks_failed"], [])

    def test_checks_flag_untied_or_implausible_estimates(self):
        detections = [{"id": "door_3"}, {"id": "cabinet_1"}]
        bad = scene(room={"width_m": 1.0, "depth_m": 5.0, "height_m": 2.6},
                    furniture=[{**scene()["furniture"][0], "detection_id": "none", "size_m": [5.0, 0.4, 0.6],
                                "center_m": [9.0, 2.0], "estimated_mass_kg": 0.1}])
        self.assertEqual(sfi.check_scene(bad, detections),
                         ["room size outside 2-12 m (height 2.2-4 m)", "cab: no SAM detection", "cab: size outside 0.02-3.5 m",
                          "cab: center outside the room", "cab: mass outside 0.5-300 kg"])
        self.assertEqual(sfi.check_scene(scene(door={"detection_id": "door_9", "wall": "left", "center_m": 1}), detections),
                         ["door is not tied to a SAM detection"])
        self.assertEqual(json.loads(json.dumps(sfi.SCENE_SCHEMA))["title"], "SceneSpec")


if __name__ == "__main__":
    unittest.main()
