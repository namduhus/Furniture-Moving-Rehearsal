"""Offline guards for the first paid Isaac runtime check."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isaac_preflight as preflight
import modal_dev as launcher


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

    def test_recording_plan_outputs_belong_to_the_project_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/isaac_preflight.py"), "--dry-run", "--stage", "g1",
                 "--config", str(ROOT / "configs/isaac_record.json")],
                cwd=directory, capture_output=True, text=True, check=True,
            )
        plan = json.loads(result.stdout)
        self.assertEqual(Path(plan["output_root"]), ROOT / "outputs/isaac")
        self.assertEqual(plan["stages"], ["gpu", "g1"])
        self.assertEqual(plan["config"]["steps"], 500)

    def test_recording_rejects_partial_intervals_excess_frames_and_odd_dimensions(self):
        original = json.loads((ROOT / "configs/isaac_record.json").read_text())
        for change in ({"steps": 501}, {"steps": 2000},
                       {"recording": {"capture_every_steps": 0}},
                       {"camera": {"width": 639, "height": 360}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    preflight.load_config(path)

    def test_observer_requires_bounded_explicit_settings(self):
        original = json.loads((ROOT / "configs/isaac_observe.json").read_text())
        for setting, bad in (("port", 0), ("max_fps", 100), ("finished_linger_seconds", 600),
                             ("realtime_pacing", "yes")):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as name:
                path = Path(name) / "config.json"
                value = {**original, "observation": {**original["observation"], setting: bad}}
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    preflight.load_config(path)
        self.assertEqual(preflight.load_config(ROOT / "configs/isaac_observe.json")["observation"]["port"], 8765)

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

    def test_image_build_cache_never_occupies_the_workspace_volume(self):
        image = mock.Mock()
        for method in ("apt_install", "pip_install", "env", "run_commands"):
            getattr(image, method).return_value = image
        app = mock.Mock()
        app.function.return_value = lambda function: function
        fake_modal = SimpleNamespace(
            Image=SimpleNamespace(from_registry=mock.Mock(return_value=image)),
            Volume=SimpleNamespace(from_name=mock.Mock(return_value=object())),
            App=mock.Mock(return_value=app),
        )
        config = launcher.load_config(ROOT / "configs/modal_isaac.json")
        with mock.patch.dict(sys.modules, {"modal": fake_modal}), \
             mock.patch.object(launcher.importlib.metadata, "version", return_value=config["modal_version"]):
            launcher.create_app(config)
        for call in image.env.call_args_list:
            for value in call.args[0].values():
                self.assertFalse(value.startswith("/workspace"), "Image build writes into the Volume mount")
        runtime_env = app.function.call_args.kwargs["env"]
        self.assertEqual(runtime_env["XDG_CACHE_HOME"], "/workspace/.cache")
        self.assertEqual(runtime_env["UV_CACHE_DIR"], "/workspace/.cache/uv")
        self.assertIn("/workspace", app.function.call_args.kwargs["volumes"])
        calls = image.method_calls
        upstream_install = next(i for i, call in enumerate(calls)
                                if call[0] == "run_commands")
        libxt_install = next(i for i, call in enumerate(calls)
                             if call[0] == "apt_install" and "libxt6t64" in call.args)
        self.assertGreater(libxt_install, upstream_install)
        self.assertIn("ffmpeg", calls[libxt_install].args)
        final_checks = image.run_commands.call_args.args
        self.assertIn('ctypes.CDLL("libXt.so.6")', final_checks[0])
        self.assertIn("-c:v libx264", final_checks[1])
        self.assertEqual(final_checks[-1], launcher.mountpoint_check_command("/workspace"))
        for command in final_checks:
            self.assertNotIn("\n", command)
            syntax = subprocess.run(["sh", "-n", "-c", command], capture_output=True, text=True)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_mount_guard_rejects_nonempty_directory_without_deleting_files(self):
        with tempfile.TemporaryDirectory(prefix="fmr-mount with spaces-") as directory:
            command = launcher.mountpoint_check_command(directory)
            empty = subprocess.run(command, shell=True, capture_output=True, text=True)
            self.assertEqual(empty.returncode, 0, empty.stderr)
            cache = Path(directory) / "cache.txt"
            cache.write_text("preserve this data")
            occupied = subprocess.run(command, shell=True, capture_output=True, text=True)
            self.assertNotEqual(occupied.returncode, 0)
            self.assertIn("Volume mountpoint must be empty", occupied.stderr)
            self.assertEqual(cache.read_text(), "preserve this data")

    def test_sdk_dockerfile_keeps_each_build_command_on_one_physical_line(self):
        try:
            from modal._image import _Image
        except ImportError:
            self.skipTest("Use the pinned Modal SDK for Dockerfile generation")
        config = launcher.load_config(ROOT / "configs/modal_isaac.json")
        commands = launcher.runtime_build_commands(config)
        base = _Image.from_registry(config["image"])
        # Capture the actual pinned SDK's Dockerfile factory without building or hydrating.
        with mock.patch.object(_Image, "_from_args") as factory:
            base.run_commands(*commands)
        dockerfile = factory.call_args.kwargs["dockerfile_function"]("2024.10")
        physical_lines = "\n".join(dockerfile.commands).splitlines()
        self.assertEqual(len(physical_lines), len(commands) + 1)
        self.assertEqual(physical_lines[0], "FROM base")
        for line in physical_lines[1:]:
            self.assertTrue(line.startswith("RUN "), line)
            self.assertNotIn("\r", line)
            syntax = subprocess.run(["sh", "-n", "-c", line[4:]], capture_output=True, text=True)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)


class Tensor:
    is_cuda = True

    def __init__(self, values=None):
        self.values = values if values is not None else [0.0]

    def is_floating_point(self):
        return False

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def sum(self):
        return self

    def any(self):
        return self

    def item(self):
        return 0


class G1ControlFlowTests(unittest.TestCase):
    def test_environment_observations_cannot_replace_viewer_settings(self):
        for profile in ("isaac_smoke", "isaac_record", "isaac_observe"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                config = json.loads((ROOT / f"configs/{profile}.json").read_text())
                config.update(steps=2, camera={"width": 64, "height": 48})
                if "recording" in config:
                    config["recording"]["capture_every_steps"] = 1
                path = directory / "test-config.json"
                path.write_text(json.dumps(config))
                config = preflight.load_config(path)
                calls = {"captures": 0, "steps": 0, "closed": False}

                class Pixels:
                    shape = (48, 64, 3)

                    def std(self):
                        return 20.0

                    def astype(self, dtype):
                        return self

                    def tobytes(self):
                        return bytes([calls["captures"]])

                class CameraTensor(Tensor):
                    def __getitem__(self, item):
                        return self

                    def numpy(self):
                        calls["captures"] += 1
                        return Pixels()

                class Image:
                    width = 64

                    def save(self, destination):
                        # Deliberately not a real image: this test checks control flow only.
                        Path(destination).write_bytes(b"control-flow fixture" * 8)

                physics_config = type("PhysxCfg", (), {})
                physics_manager = type("PhysxManager", (), {"__module__": "isaaclab_physx.physics.test"})
                cfg = SimpleNamespace(sim=SimpleNamespace(physics=physics_config(), dt=0.005),
                                      scene=SimpleNamespace())
                camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": CameraTensor()}),
                                         update=mock.Mock(), set_world_poses_from_view=mock.Mock())
                robot = SimpleNamespace(data=SimpleNamespace(joint_pos=Tensor(),
                                        root_state_w=Tensor([[0.0] * 13])))
                env = SimpleNamespace(step_dt=0.02, sim=SimpleNamespace(physics_manager=physics_manager,
                                      render=mock.Mock()), scene={"robot": robot, "fmr_camera": camera},
                                      action_space=SimpleNamespace(shape=(1, 5)))
                env.unwrapped = env
                # Actual observations intentionally contain no realtime_pacing setting.
                env.reset = lambda **kwargs: ({"policy": Tensor()}, {})

                def step(actions):
                    calls["steps"] += 1
                    return {"policy": Tensor()}, Tensor(), Tensor(), Tensor(), {}

                env.step = step
                env.close = lambda: calls.update(closed=True)
                torch = dict(Tensor=Tensor, manual_seed=mock.Mock(), tensor=lambda *a, **k: Tensor(),
                             zeros=lambda *a, **k: Tensor(), logical_or=lambda *a: Tensor(),
                             cuda=SimpleNamespace(synchronize=mock.Mock()))
                definitions = {
                    "torch": torch, "warp": {"config": SimpleNamespace(enable_backward=True)},
                    "isaaclab": {}, "isaaclab.app": {"launch_simulation": lambda *a, **k: nullcontext()},
                    "isaaclab.sim": {"PinholeCameraCfg": lambda **k: k},
                    "isaaclab.sensors": {"CameraCfg": lambda **k: k},
                    "isaaclab_physx": {}, "isaaclab_physx.physics": {"PhysxCfg": physics_config},
                    "isaaclab_physx.renderers": {"IsaacRtxRendererCfg": lambda: None},
                    "gymnasium": {"make": lambda *a, **k: env}, "isaaclab_tasks": {},
                    "isaaclab_tasks.utils": {"parse_env_cfg": lambda *a, **k: cfg},
                    "PIL": {"Image": SimpleNamespace(fromarray=lambda a: Image()),
                            "ImageDraw": SimpleNamespace(Draw=lambda image: mock.Mock())},
                }
                modules = {}
                for module_name, attributes in definitions.items():
                    module = ModuleType(module_name)
                    module.__dict__.update(attributes)
                    modules[module_name] = module
                    parent, _, child = module_name.rpartition(".")
                    if parent in modules:
                        setattr(modules[parent], child, module)
                publisher = SimpleNamespace(phase=mock.Mock(), publish_image=mock.Mock(),
                                            sequence=3, disabled=False)
                with mock.patch.dict(sys.modules, modules), \
                     mock.patch.object(preflight, "ObservationPublisher", return_value=publisher), \
                     mock.patch.object(preflight, "require_ffmpeg", return_value="unused-test-codec"), \
                     mock.patch.object(preflight, "encode_recording", return_value={"status": "test-double"}), \
                     mock.patch.object(preflight.time, "sleep") as sleep:
                    preflight.g1_check(config, directory)
                evidence = json.loads((directory / "g1-worker.json").read_text())
                self.assertEqual(calls["steps"], 2)
                self.assertTrue(calls["closed"])
                self.assertEqual(evidence["steps_completed"], 2)
                self.assertEqual(evidence["status"], "checks_completed_before_kit_shutdown")
                self.assertEqual(sleep.call_count, 2 if profile == "isaac_observe" else 0)
                if profile == "isaac_observe":
                    self.assertTrue(evidence["observation"]["realtime_pacing"])
                    self.assertGreater(publisher.publish_image.call_count, 0)

if __name__ == "__main__":
    unittest.main()
