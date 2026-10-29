"""Offline guards for the first paid Isaac runtime check."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_module("fmr_isaac_preflight", ROOT / "scripts/isaac_preflight.py")
launcher = load_module("fmr_isaac_launcher", ROOT / "scripts/modal_dev.py")


class IsaacPreflightTests(unittest.TestCase):
    def test_software_vulkan_does_not_pass_as_nvidia_rtx(self):
        with self.assertRaisesRegex(RuntimeError, "Vulkan"):
            preflight.validate_vulkan("deviceName = llvmpipe (LLVM 15.0.7)")
        with self.assertRaisesRegex(RuntimeError, "ray_tracing_pipeline"):
            preflight.validate_vulkan("NVIDIA L40S VK_KHR_acceleration_structure")
        preflight.validate_vulkan("NVIDIA L40S VK_KHR_acceleration_structure VK_KHR_ray_tracing_pipeline")

    def test_numeric_driver_comparison(self):
        self.assertGreater(preflight.driver_version("610.57.04"), preflight.driver_version("595.58.03"))
        self.assertLess(preflight.driver_version("535.230.02"), preflight.driver_version("595.58.03"))
        with self.assertRaises(ValueError):
            preflight.driver_version("unknown")

    def test_cpu_and_newton_profiles_are_rejected(self):
        original = json.loads(preflight.DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for change in ({"device": "cpu"}, {"physics": "newton_mjwarp"}, {"num_envs": 256}):
                with self.subTest(change=change):
                    path.write_text(json.dumps({**original, **change}))
                    with self.assertRaises(ValueError):
                        preflight.load_config(path)

    def test_zero_exit_or_partial_g1_checks_are_not_success(self):
        config = preflight.load_config(preflight.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(RuntimeError):
                preflight.validate_evidence("gpu", {}, path, config)
            with self.assertRaisesRegex(RuntimeError, "number of steps"):
                preflight.validate_evidence("g1", {"status": "checks_completed_before_kit_shutdown",
                                                 "steps_completed": 1}, path, config)
            with self.assertRaisesRegex(RuntimeError, "rendering evidence"):
                preflight.validate_evidence("g1", {"status": "checks_completed_before_kit_shutdown",
                                                 "steps_completed": config["steps"]}, path, config)

    def test_hung_process_is_terminated_on_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subprocess.TimeoutExpired):
                preflight.run_stage([sys.executable, "-c", "import time; time.sleep(60)"],
                                    Path(directory) / "worker.log", timeout=0.1)

    def test_clean_kit_exit_requires_complete_evidence(self):
        config = preflight.load_config(preflight.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ("initial.png", "final.png"):
                (path / name).write_bytes(b"test artifact" * 8)
            (path / "trajectory.jsonl").write_text("{}\n" * config["steps"])
            preflight.write_json(path / "g1-worker.json", {
                "status": "checks_completed_before_kit_shutdown", "steps_completed": config["steps"],
            })
            with mock.patch.object(preflight, "check_versions", return_value={}), \
                 mock.patch.object(preflight, "g1_check", side_effect=SystemExit(0)):
                self.assertEqual(preflight.run_worker("g1", config, path), 0)

    def test_early_zero_exit_is_recorded_as_failure(self):
        config = preflight.load_config(preflight.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with mock.patch.object(preflight, "check_versions", return_value={}), \
                 mock.patch.object(preflight, "g1_check", side_effect=SystemExit(0)):
                with self.assertRaises(SystemExit):
                    preflight.run_worker("g1", config, path)
            self.assertEqual(json.loads((path / "g1-worker.json").read_text())["status"], "failed")

    def test_runtime_dry_run_works_without_gpu_packages(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/isaac_preflight.py"), "--dry-run"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["status"], "local_plan_only")

    def test_source_and_lock_pins_match_the_runtime_and_probe(self):
        image = launcher.load_config(ROOT / "configs/modal_isaac.json")
        smoke = preflight.load_config(preflight.DEFAULT_CONFIG)
        self.assertEqual(image["runtime"]["commit"], smoke["isaaclab_commit"])
        self.assertEqual(image["runtime"]["lock_sha256"], smoke["isaaclab_lock_sha256"])
        self.assertIn("--frozen", "\n".join(launcher.runtime_build_commands(image)))
        self.assertIn("--extra isaacsim", "\n".join(launcher.runtime_build_commands(image)))

    def test_eula_is_required_before_any_cloud_setup(self):
        config = launcher.load_config(ROOT / "configs/modal_isaac.json")
        with mock.patch.object(launcher, "create_app") as create:
            with self.assertRaisesRegex(ValueError, "accept-isaac-eula"):
                launcher.start(config, Path("missing.pub"), None)
            create.assert_not_called()

    def test_sdk_accepts_isaac_image_graph_without_cloud_calls(self):
        try:
            import modal
        except ImportError:
            self.skipTest("Use environments/modal for SDK validation")
        app, function = launcher.create_app(launcher.load_config(ROOT / "configs/modal_isaac.json"))
        self.assertEqual(app.name, "fmr-isaac-dev")
        self.assertIsNone(app.app_id)
        self.assertIsInstance(function, modal.Function)


if __name__ == "__main__":
    unittest.main()
