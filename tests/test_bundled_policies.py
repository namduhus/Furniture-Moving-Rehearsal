"""Verify the distributed policy artifacts and their published evaluation counts without a GPU."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isaac_g1_rehearsal as rehearsal


class BundledPolicyTests(unittest.TestCase):
    def test_default_rehearsal_uses_complete_checkpoints_with_matching_configs(self):
        manifest = json.loads((ROOT / "models/manifest.json").read_text())
        config, _ = rehearsal.load_config(rehearsal.DEFAULT_CONFIG)
        self.assertEqual(set(manifest["policies"]), {"approach", "carry", "push"})
        total = 0
        for role, item in manifest["policies"].items():
            with self.subTest(role=role):
                selected = config["push"]["checkpoint"] if role == "push" else config[f"{role}_checkpoint"]
                training = config["push"]["training"] if role == "push" else config[f"{role}_training"]
                self.assertEqual(selected, item["file"])
                self.assertEqual(training, item["training_config"])
                checkpoint = ROOT / selected
                self.assertEqual(checkpoint.stat().st_size, item["bytes"])
                total += item["bytes"]
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), item["sha256"])
                self.assertEqual(hashlib.sha256((ROOT / training).read_bytes()).hexdigest(), item["training_config_sha256"])
                # Verify the container without deserializing tensors or installing PyTorch locally.
                with zipfile.ZipFile(checkpoint) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertTrue(any(name.endswith("/data.pkl") for name in archive.namelist()))
                    self.assertTrue(any("/data/" in name for name in archive.namelist()))
        self.assertEqual(total, manifest["total_checkpoint_bytes"])
        self.assertEqual(hashlib.sha256((ROOT / manifest["controller_contract"]).read_bytes()).hexdigest(),
                         manifest["controller_contract_sha256"])

    def test_reported_success_and_time_match_the_saved_episode_outcomes(self):
        manifest = json.loads((ROOT / "models/manifest.json").read_text())
        evidence = ROOT / manifest["evaluation"]["file"]
        self.assertEqual(hashlib.sha256(evidence.read_bytes()).hexdigest(), manifest["evaluation"]["sha256"])
        results = json.loads(evidence.read_text())["policies"]
        self.assertEqual(set(results), set(manifest["policies"]))
        for role, result in results.items():
            self.assertEqual(result["training_run"], manifest["policies"][role]["training_run"])
            self.assertEqual(result["scripted_eval"]["seed"], result["learned_eval"]["seed"])
            for mode in ("scripted_eval", "learned_eval"):
                with self.subTest(role=role, mode=mode):
                    metric = result[mode]
                    records = metric["per_episode"]
                    successes = [r for r in records if r["outcome"] == "success"]
                    self.assertEqual(len(records), metric["episodes"])
                    self.assertEqual(len(records), result["environment_count"])
                    self.assertEqual(len(successes), metric["successes"])
                    self.assertEqual(len(successes) / len(records), metric["success_rate"])
                    self.assertEqual(dict(Counter(r["outcome"] for r in records)), metric["outcome_counts"])
                    self.assertEqual(statistics.median(r["seconds"] for r in successes), metric["median_success_seconds"])

    def test_policy_paths_allow_bundles_and_local_runs_but_reject_escape_paths(self):
        prefixes = ("models/", "outputs/isaac/")
        for path in ("models/approach.pt", "outputs/isaac/local-run/train/model_99.pt"):
            self.assertEqual(rehearsal.repo_path(path, prefixes, ".pt"), ROOT / path)
        for path in ("models/../secret.pt", "outputs/isaac/../../secret.pt", "/tmp/weights.pt",
                     "models-backup/approach.pt", "models/approach.onnx"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                rehearsal.repo_path(path, prefixes, ".pt")


if __name__ == "__main__":
    unittest.main()
