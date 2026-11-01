"""Muse Spark plans and diagnoses furniture-moving rehearsals; measured capability rules accept or override its choice.

Plan mode (a request, before any simulation) and diagnosis mode (a finished rehearsal run) send the request, the
corrected furniture, the robot's measured capabilities and, when diagnosing, the measured outcome and key frames to Muse
Spark, which returns a cause, the evidence it relied on, the next strategy (carry, push, needs_two_people, infeasible)
and a plain-language explanation. Pushes always use furniture sliders. The system then checks the strategy against the
capability rules (carry and push ranges from the simulation runs, pushed-face width, sliding friction on sliders,
tipping) and falls back to carry, then push, then needs_two_people if the model's choice is not supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import meta_models

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/rehearsal_advisor.json"

ADVICE_SCHEMA = {
    "title": "RehearsalAdvice", "type": "object", "additionalProperties": False,
    "required": ["cause", "evidence", "strategy", "explanation", "explanation_ko", "confidence"],
    "properties": {"cause": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}},
                   "strategy": {"type": "string", "enum": ["carry", "push", "needs_two_people", "infeasible"]},
                   "explanation": {"type": "string"},
                   "explanation_ko": {"type": "string"},
                   "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}}

PROMPT = """You are the planning and diagnosis step of a furniture-moving rehearsal. A simulated Unitree G1 humanoid
(GR00T whole-body walking controller plus learned room policies) moves furniture in a room reconstructed from a photo.

Request: move {target} ({category}) to {destination}. User-corrected size {size} m (width, depth, height), mass {mass} kg.
Measured robot capabilities from simulation runs (the only evidence about what the robot can do):
{capabilities}
{result}
{task} Choose the next strategy from carry, push, needs_two_people, infeasible. Prefer carry when the item is within
the carry capability; if it cannot be lifted, push it (the robot always puts furniture sliders under pushed items).
Cite only the data above (and the attached frames, if any); do not invent measurements. Write
`explanation` for a non-expert in two or three sentences, and `explanation_ko` with the same content in Korean."""


def carry_rule(size, mass, settings):
    """(feasible, reason) for lifting and carrying with the side grasp the room policies were trained with."""
    c = settings["carry"]
    if any(abs(a - b) > c["size_tolerance_m"] for a, b in zip(size, c["size_m"])):
        return False, f"size {size} m differs from the trained {c['size_m']} m cabinet by more than {c['size_tolerance_m']} m"
    if not c["mass_kg"][0] <= mass <= c["mass_kg"][1]:
        return False, f"mass {mass} kg is outside the measured carry range {c['mass_kg']} kg"
    return True, "within the measured carry range"


def push_rule(size, mass, sliders, settings):
    """(feasible, reason) for a straight push of the widest face: mass range, face width, sliding friction, tipping.

    A box pushed at height h with friction mu tips before it slides when mu * h > depth / 2 (depth along the push).
    """
    p = settings["push"]
    width, depth = max(size[0], size[1]), min(size[0], size[1])
    friction = settings["slider_friction"] if sliders else settings["floor_friction"]
    height = min(p["push_height_m"], size[2])
    if not p["mass_kg"][0] <= mass <= p["mass_kg"][1]:
        return False, f"mass {mass} kg is outside the measured push range {p['mass_kg']} kg"
    if width < p["min_face_width_m"]:
        return False, f"the widest face ({width} m) is narrower than the {p['min_face_width_m']} m the palms need"
    if friction > p["friction_max"]:
        return False, f"sliding friction {friction} exceeds the {p['friction_max']} the push policy was trained with (use sliders)"
    if friction*height >= depth/2:
        return False, f"it would tip before sliding (friction {friction} x push height {height} m >= half depth {depth/2} m)"
    return True, f"pushable on {'sliders' if sliders else 'the bare floor'} (friction {friction})"


def decide(advice, size, mass, settings):
    """Accept the model's strategy if the rules allow it, else the first feasible of carry, push (on sliders),
    needs_two_people."""
    rules = {"carry": carry_rule(size, mass, settings), "push": push_rule(size, mass, True, settings)}
    chosen = advice["strategy"]
    accepted = chosen in ("needs_two_people", "infeasible") or rules[chosen][0]
    if not accepted:
        chosen = "carry" if rules["carry"][0] else "push" if rules["push"][0] else "needs_two_people"
    return {"strategy": chosen, "use_sliders": chosen == "push", "model_choice_accepted": accepted,
            "rules": {k: {"feasible": ok, "reason": reason} for k, (ok, reason) in rules.items()}}


def run_summary(worker):
    """The measured outcome of a rehearsal run, compacted for the prompt."""
    phases = worker.get("phases", {})
    return {"status": worker.get("status"), "mass_kg": worker.get("mass_kg"),
            **{task: phases[task] for task in ("approach", "carry") if task in phases},
            "scripted_phase_failures": {k: v for k, v in phases.items() if isinstance(v, dict) and k not in ("approach", "carry")
                                        and any(v.get(flag) for flag in ("invalid_action", "fallen"))},
            "grasp": worker.get("grasp"), "lift": worker.get("lift"), "placement": worker.get("placement")}


def advise(models, settings, request, scene, worker=None, frames=()):
    """Plan (worker None) or diagnose (a rehearsal-worker.json dict) one request; returns the model reply and decision."""
    items = {f["id"]: f for f in scene["scene"]["furniture"]}
    item, fix = items[request["target"]], request["corrections"]
    capabilities = {k: settings[k] for k in ("carry", "push", "floor_friction", "slider_friction")}
    if worker is None:
        result, task = "", "No simulation has run yet. Pick the first strategy to rehearse."
    else:
        result = "Measured rehearsal result:\n" + json.dumps(run_summary(worker), indent=1)
        task = "Diagnose the outcome: what happened, and why, given the measurements."
    text = PROMPT.format(target=request["target"], category=item["category"], destination=json.dumps(request["destination"]),
                         size=fix["size_m"], mass=fix["mass_kg"], capabilities=json.dumps(capabilities, indent=1),
                         result=result, task=task)
    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": meta_models.image_data_url(f)}}
                                                   for f in frames]
    reply = models.chat(settings["chat_model"], [{"role": "user", "content": content}], schema=ADVICE_SCHEMA)
    advice = reply["content"]
    return {"mode": "plan" if worker is None else "diagnose", "model": settings["chat_model"], "frames": [str(f) for f in frames],
            "advice": advice, "decision": decide(advice, fix["size_m"], fix["mass_kg"], settings), "usage": reply["usage"],
            "cached": reply["cached"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, help="plan: a rehearsal request JSON")
    source.add_argument("--run", type=Path, help="diagnose: a finished outputs/isaac/<run>-rehearsal directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text())
    if args.run:
        request = json.loads((args.run / "request.json").read_text())
        scene = json.loads((args.run / "scene.json").read_text())
        worker = json.loads((args.run / "rehearsal-worker.json").read_text())
        frames = sorted((args.run / "rehearsal" / "key_frames").glob("*.png"))
    else:
        request = json.loads(args.request.read_text())
        scene = json.loads((ROOT / request["scene"]).read_text())
        worker, frames = None, []
    if args.dry_run:
        print(json.dumps({"status": "local_plan_only", "chat_calls": 1, "images": len(frames), "estimated_usd": 0.02}, indent=2))
        return 0
    result = advise(meta_models.MetaModels(), settings, request, scene, worker, frames)
    out = (args.run / "advice.json") if args.run else ROOT / "outputs/meta/advice" / f"{args.request.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"advice": str(out), "strategy": result["decision"]["strategy"],
                      "use_sliders": result["decision"]["use_sliders"],
                      "model_choice_accepted": result["decision"]["model_choice_accepted"],
                      "cause": result["advice"]["cause"], "explanation": result["advice"]["explanation"]},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
