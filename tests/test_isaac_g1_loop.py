"""Exercise the real G1 control flow with CPU-only simulator/codec interface doubles."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isaac_preflight as preflight


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
