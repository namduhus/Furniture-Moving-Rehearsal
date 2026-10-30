"""Run real controller-launch flow with CPU interfaces, not simulated physics."""

from contextlib import nullcontext
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g1_wbc
import isaac_g1_wbc as runner
from isaac_video import inspect_frame_checksums


class ControllerLaunchTests(unittest.TestCase):
    def test_static_standing_video_is_explicitly_allowed_without_weakening_smoke(self):
        text = "#tb 0: 1/25\n#dimensions 0: 64x64\n0,0,0,1,100,same\n0,1,1,1,100,same\n"
        with self.assertRaises(RuntimeError):
            inspect_frame_checksums(text, 2, Fraction(25), 64, 64)
        self.assertEqual(inspect_frame_checksums(text, 2, Fraction(25), 64, 64, require_motion=False)["unique_decoded_frames"], 1)

    def test_actual_balance_flow_initializes_frames_gains_and_contiguous_steps(self):
        profile, asset, runtime = runner.load_plan(runner.DEFAULT_PROFILE)
        profile = {**profile, "control_steps": 4,
                   "camera": {"width": 64, "height": 64, "capture_every_control_steps": 2}}
        contract, reference = g1_wbc.load_contract(ROOT / profile["controller_contract"])
        names = list(reversed(sum(asset["joint_groups"].values(), [])))

        class Tensor:
            is_cuda = True
            def __init__(self, value): self.value = np.asarray(value)
            def __getitem__(self, index): return Tensor(self.value[index])
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        class Data:
            @property
            def joint_pos(self): return Tensor([robot.q])
            @property
            def joint_vel(self): return Tensor([robot.dq])
            @property
            def default_joint_pos(self): return Tensor([np.zeros(43)])
            @property
            def joint_pos_limits(self): return Tensor([limits])
            @property
            def root_link_pose_w(self): return Tensor([robot.pose])
            @property
            def root_link_vel_w(self): return Tensor([robot.velocity])
            @property
            def body_link_pose_w(self):
                root = g1_wbc.pose_matrix(robot.pose)
                local = reference.forward(dict(zip(names, robot.q, strict=True)))
                poses = [root @ local[n] for n in robot.body_names]
                return Tensor([[np.r_[p[:3, 3], g1_wbc.matrix_quat_xyzw(p[:3, :3])] for p in poses]])

        class Robot:
            joint_names = names
            body_names = list(reference.forward({n: 0 for n in names}))
            is_fixed_base = False
            q = np.zeros(43)
            dq = np.zeros(43)
            pose = np.r_[0, 0, 0.75, g1_wbc.matrix_quat_xyzw(g1_wbc.axis_rotation([1,0,0], 0.7))]
            velocity = np.zeros(6)
            data = Data()
            def write_joint_position_to_sim_index(self, *, position): self.q = position.value[0].copy()
            def write_joint_velocity_to_sim_index(self, *, velocity): self.dq = velocity.value[0].copy()
            def write_root_pose_to_sim_index(self, *, root_pose): self.pose = root_pose.value[0].copy()
            def write_root_velocity_to_sim_index(self, *, root_velocity): self.velocity = root_velocity.value[0].copy()
            def set_joint_position_target_index(self, *, target): self.targets = target.value[0].copy()
            def set_joint_velocity_target_index(self, *, target): pass

        robot = Robot()
        limits = [[reference.limits[n]["lower"], reference.limits[n]["upper"]] for n in names]
        rgb = np.repeat(((np.indices((64, 64)).sum(axis=0) % 2)*255)[..., None], 3, axis=-1).astype(np.uint8)

        class Camera:
            data = SimpleNamespace(output={"rgb": Tensor(rgb[None])})
            def set_world_poses_from_view(self, *args): pass
            def update(self, *args, **kwargs): pass

        class Scene(dict):
            def reset(self): pass
            def update(self, dt): pass
            def write_data_to_sim(self): pass

        class Config:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
            def to_dict(self): return self.__dict__

        class Simulation:
            physics_manager = type("PhysxManager", (), {"__module__": "isaaclab_physx.physics"})
            steps = 0
            def __init__(self, config): self.config = config
            def reset(self): pass
            def forward(self): pass
            def render(self): pass
            def step(self, *, render): Simulation.steps += 1

        class Policy:
            metadata = {"input": ["batch",516], "output": ["batch",15]}
            def __init__(self, *args): pass
            def __call__(self, observation):
                self.assert_shape = observation.shape == (516,)
                if not self.assert_shape: raise AssertionError("Bad observation passed to policy")
                return np.zeros(15)

        class Image:
            @staticmethod
            def fromarray(array): return Image()
            def save(self, path): Path(path).write_bytes(b"CPU interface image double")

        candidate = Config(spawn=Config(fix_root_link=False, rigid_props=Config(disable_gravity=False)))
        instantiated = []

        def instantiate(config):
            instantiated.append(config)
            return Scene(robot=robot, camera=Camera())

        def replace(value, **kwargs):
            value = copy.deepcopy(value)
            value.__dict__.update(kwargs)
            return value

        modules = {}
        for key, fields in {
            "torch": {"float32": np.float32, "manual_seed": lambda seed: None,
                      "as_tensor": lambda value, **kwargs: Tensor(np.asarray(value, dtype=np.float32))},
            "warp": {"config": Config(enable_backward=True)}, "isaaclab": {},
            "isaaclab.app": {"launch_simulation": lambda *args: nullcontext()},
            "isaaclab.actuators": {"IdealPDActuatorCfg": Config},
            "isaaclab.assets": {"AssetBaseCfg": Config},
            "isaaclab.scene": {"InteractiveSceneCfg": Config},
            "isaaclab.sensors": {"CameraCfg": Config},
            "isaaclab.sim": {"GroundPlaneCfg": Config, "DomeLightCfg": Config, "PinholeCameraCfg": Config,
                             "SimulationCfg": Config, "SimulationContext": Simulation},
            "isaaclab.utils": {"configclass": lambda c: c, "instantiate": instantiate, "replace": replace},
            "isaaclab_assets": {"G1_29DOF_CFG": candidate}, "isaaclab_physx": {},
            "isaaclab_physx.physics": {"PhysxCfg": Config},
            "isaaclab_physx.renderers": {"IsaacRtxRendererCfg": Config}, "PIL": {"Image": Image},
        }.items():
            module = ModuleType(key)
            module.__dict__.update(fields)
            modules[key] = module

        def encode(directory, **kwargs):
            self.assertFalse(kwargs["require_motion"])
            path = directory / "rollout.mp4"
            path.write_bytes(b"CPU codec interface double")
            return {"status": "passed_video_decode", "decoded_frames": kwargs["count"],
                    "width": kwargs["width"], "height": kwargs["height"], "fps": str(kwargs["fps"]),
                    "duration_seconds": float(kwargs["count"]/kwargs["fps"]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        with tempfile.TemporaryDirectory() as name, mock.patch.dict(sys.modules, modules), \
                mock.patch.object(g1_wbc, "BalancePolicy", Policy), \
                mock.patch.object(runner, "require_ffmpeg", return_value="unused-interface-double"), \
                mock.patch.object(runner, "encode_recording", side_effect=encode):
            directory = Path(name)
            runner.balance_check(profile, asset, runtime, contract, reference, Path("unused.onnx"), directory)
            runner.validate_stage("balance", directory, profile)
            self.assertEqual(Simulation.steps, 16)
            self.assertEqual(instantiated[0].robot.actuators["reference_pd"].stiffness["waist_yaw_joint"], 250)
            self.assertTrue(np.allclose(robot.pose[3:], [0,0,0,1]))
            evidence = json.loads((directory / "balance-worker.json").read_text())
            self.assertTrue(evidence["controller_inference_verified"])
            self.assertFalse(evidence["walking_verified"])
            self.assertEqual(len(list(directory.glob("frame-input-*.json"))), 3)
            records = [json.loads(l) for l in (directory / "trajectory.jsonl").read_text().splitlines()]
            records[1]["root_height_m"] = 0.1
            (directory / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaises(RuntimeError):
                runner.validate_stage("balance", directory, profile)
            failed = directory / "fall"
            failed.mkdir()

            def falling_step(self, *, render):
                robot.pose[2] = 0.1

            with mock.patch.object(Simulation, "step", new=falling_step):
                with self.assertRaisesRegex(RuntimeError, "Balance guard failed"):
                    runner.balance_check(profile, asset, runtime, contract, reference, Path("unused.onnx"), failed)
            failure = json.loads((failed / "balance-worker.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["steps_completed"], 1)
            self.assertIn("Balance guard", failure["error"])


if __name__ == "__main__":
    unittest.main()
