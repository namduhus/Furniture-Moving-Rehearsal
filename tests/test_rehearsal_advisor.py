import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rehearsal_advisor as advisor

SETTINGS = json.loads(advisor.DEFAULT_CONFIG.read_text())
REQUEST = json.loads((ROOT / "configs/rehearsal_living_room_pedestal.json").read_text())
SCENE = json.loads((ROOT / REQUEST["scene"]).read_text())


class FakeModels:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, model, messages, schema=None):
        self.calls.append((model, messages, schema))
        return {"content": self.reply, "usage": {"total_tokens": 1}, "cached": False}


def reply(strategy):
    return {"cause": "c", "evidence": ["e"], "strategy": strategy, "explanation": "x", "explanation_ko": "y",
            "confidence": "medium"}


class AdvisorRuleTests(unittest.TestCase):
    def test_carry_and_push_rules_follow_the_measured_ranges(self):
        pedestal = [0.30, 0.28, 0.90]
        self.assertTrue(advisor.carry_rule(pedestal, 3.0, SETTINGS)[0])
        self.assertFalse(advisor.carry_rule(pedestal, 8.0, SETTINGS)[0])
        self.assertFalse(advisor.carry_rule([0.7, 0.45, 0.95], 3.0, SETTINGS)[0])
        # The narrow pedestal cannot be pushed by palms 0.3 m apart; a 0.7 m wide cabinet can, but only on sliders.
        self.assertIn("narrower", advisor.push_rule(pedestal, 8.0, True, SETTINGS)[1])
        cabinet = [0.7, 0.45, 0.95]
        self.assertTrue(advisor.push_rule(cabinet, 15.0, True, SETTINGS)[0])
        self.assertIn("sliders", advisor.push_rule(cabinet, 15.0, False, SETTINGS)[1])
        self.assertIn("outside", advisor.push_rule(cabinet, 30.0, True, SETTINGS)[1])
        # Tipping: 0.1 friction x 0.8 m push height >= half of a 0.15 m depth.
        self.assertIn("tip", advisor.push_rule([0.6, 0.15, 1.2], 10.0, True, SETTINGS)[1])

    def test_decision_overrides_unsupported_model_choices(self):
        accepted = advisor.decide(reply("carry"), [0.30, 0.28, 0.90], 3.0, SETTINGS)
        self.assertEqual((accepted["strategy"], accepted["model_choice_accepted"]), ("carry", True))
        # Carrying 15 kg is outside the measured range; the push with sliders is the first feasible strategy.
        overridden = advisor.decide(reply("carry"), [0.7, 0.45, 0.95], 15.0, SETTINGS)
        self.assertEqual((overridden["strategy"], overridden["use_sliders"], overridden["model_choice_accepted"]),
                         ("push", True, False))
        # Pushes always run on sliders; a heavy narrow pedestal leaves two people.
        pushed = advisor.decide(reply("push"), [0.7, 0.45, 0.95], 15.0, SETTINGS)
        self.assertEqual((pushed["strategy"], pushed["use_sliders"], pushed["model_choice_accepted"]), ("push", True, True))
        self.assertEqual(advisor.decide(reply("push"), [0.30, 0.28, 0.90], 8.0, SETTINGS)["strategy"], "needs_two_people")
        self.assertTrue(advisor.decide(reply("needs_two_people"), [0.30, 0.28, 0.90], 3.0, SETTINGS)["model_choice_accepted"])


class AdvisorPromptTests(unittest.TestCase):
    def test_plan_and_diagnosis_prompts_carry_the_measured_evidence(self):
        models = FakeModels(reply("carry"))
        plan = advisor.advise(models, SETTINGS, REQUEST, SCENE)
        self.assertEqual((plan["mode"], plan["decision"]["strategy"]), ("plan", "carry"))
        text = models.calls[0][1][0]["content"][0]["text"]
        self.assertIn("78.9%", text)
        self.assertIn("No simulation has run yet", text)
        self.assertEqual(models.calls[0][2]["title"], "RehearsalAdvice")
        worker = {"status": "failed_carry", "mass_kg": 3.0, "phases": {
            "approach": {"outcome": "success"}, "carry": {"outcome": "object_grounded", "position_error_m": 0.05},
            "lower": {"invalid_action": False, "fallen": False}, "release": {"invalid_action": False, "fallen": True}},
            "placement": {"position_m": 0.16}}
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "000120_carry.png"
            frame.write_bytes(b"\x89PNG\r\n\x1a\n")
            diagnosis = advisor.advise(models, SETTINGS, REQUEST, SCENE, worker, [frame])
        content = models.calls[1][1][0]["content"]
        self.assertEqual(diagnosis["mode"], "diagnose")
        self.assertIn("object_grounded", content[0]["text"])
        self.assertIn('"release"', content[0]["text"])                 # only failed scripted phases are listed
        self.assertNotIn('"lower"', content[0]["text"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
