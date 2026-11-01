import copy
from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import g1_wbc as wbc
import isaac_g1_rl as rl
from test_g1_wbc import contact_fixture, contact_geometry, joint_frame_fixture


class BatchAdapterTests(unittest.TestCase):
    def setUp(self):
        profile, _, _ = rl.load_plan(ROOT / "configs/isaac_g1_push_sliders.json")
        self.contract, self.reference = rl.trial_contract(profile)

    def test_batch_history_matches_single_environment_history(self):
        rng = np.random.default_rng(0)
        count = 3
        batch = wbc.BatchHistory(self.contract, count)
        singles = [wbc.ObservationHistory(self.contract) for _ in range(count)]
        rotation = wbc.axis_rotation([0, 0, 1], 0.01)
        limits = np.array([[-1.0, 1.0]]*15)
        for step in range(8):
            q, dq = rng.normal(0, 0.2, (count, 29)), rng.normal(0, 1, (count, 29))
            quats = np.array([wbc.matrix_quat_xyzw(wbc.axis_rotation(rng.normal(size=3), 0.1)) for _ in range(count)])
            gyros, commands, rpys = rng.normal(size=(count, 3)), rng.normal(0, 0.2, (count, 3)), rng.normal(0, 0.1, (count, 3))
            observed = batch.observe(q, dq, quats, gyros, rotation, commands, rpys)
            for i, single in enumerate(singles):
                np.testing.assert_allclose(observed[i], single.observe(q[i], dq[i], quats[i], gyros[i], rotation, commands[i], rpys[i]), atol=1e-6)
            actions = rng.normal(0, 0.5, (count, 15))
            targets, clipped, invalid = batch.targets(actions, limits)
            for i, single in enumerate(singles):
                wanted, count_i = single.targets(actions[i], limits)
                np.testing.assert_allclose(targets[i], wanted, atol=1e-12)
                self.assertEqual(clipped[i], count_i)
            self.assertFalse(invalid.any())
        actions[1, 4] = 9.0
        _, _, invalid = batch.targets(actions, limits)
        self.assertEqual(invalid.tolist(), [False, True, False])
        self.assertEqual(float(np.max(np.abs(batch.last_action[1]))), self.contract["action_limit"])
        quats = np.array([wbc.matrix_quat_xyzw(wbc.axis_rotation([1, 2, 3], 0.4)), [0, 0, 0, 1]])
        np.testing.assert_allclose(wbc.quat_xyzw_matrices(quats), [wbc.quat_xyzw_matrix(q) for q in quats], atol=1e-12)
        with self.assertRaises(ValueError):
            wbc.quat_xyzw_matrices([[0, 0, 0, 2.0]])

    def test_batch_policy_and_configuration_checks(self):
        policy = object.__new__(wbc.BalancePolicy)
        policy.input_name = "obs"
        policy.session = SimpleNamespace(run=lambda _, feed: [np.full((len(feed["obs"]), 15), 0.5, dtype=np.float32)])
        self.assertEqual(policy.batch(np.zeros((4, 516))).shape, (4, 15))
        with self.assertRaises(ValueError):
            policy.batch(np.zeros((4, 515)))
        config, profile, _, _ = rl.load_config(rl.DEFAULT_CONFIG)
        self.assertEqual((config["num_envs"], profile["push_object"]["friction"]), (64, 0.1))
        scaling, scaling_profile, _, _ = rl.load_config(ROOT / "configs/isaac_g1_rl_throughput_256.json")
        self.assertEqual((scaling["num_envs"], scaling["control_steps"]), (256, 100))
        self.assertEqual(scaling_profile, profile)
        raw = json.loads(rl.DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rl.json"
            for data in ({**raw, "num_envs": 0}, {**raw, "env_spacing_m": 1.0}, {**raw, "extra": 1},
                         {**raw, "push_profile": "configs/isaac_g1_crouch_start_forward.json"}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    rl.load_config(path)
        with mock.patch("builtins.print"):
            self.assertEqual(rl.main(["--dry-run"]), 0)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value, dtype=float)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def rows(env_ids, count):
    return list(range(count)) if env_ids is None else list(env_ids)


@contextmanager
def parallel_isaac(count, profile, asset):
    """Isaac Lab, scene and USD-edit doubles for `count` parallel push environments on a 4 m grid.

    Each box rests for 80 physics steps after its last pose write, then slides 0.4 mm along x per step.
    """
    contract, reference = rl.trial_contract(profile)
    names = list(reversed(asset["controller_reference"]["joint_order"]))
    lower = asset["joint_groups"]["lower_body"]
    bodies = list(reference.forward(dict.fromkeys(reference.limits, 0.0)))
    source = wbc.load_dynamics_reference()["bodies"]
    palm = wbc.palm_inertial_reference(contract)["mass_kg"]
    masses = [palm if b == "left_hand_palm_link" else source.get(b, {"mass_kg": 1.0})["mass_kg"] for b in bodies]
    origins = np.array([[4.0*(i % 2), 4.0*(i // 2), 0.0] for i in range(count)])
    steps = {"count": 0}
    push = profile["push_object"]
    box_mass, box_friction = np.full(count, float(push["mass_kg"])), np.full(count, float(push["friction"]))

    class Data:
        def __init__(self, robot): self.robot = robot
        joint_stiffness = property(lambda self: Tensor([[0.0 if n in lower else 100.0 for n in names]]*count))
        joint_damping = property(lambda self: Tensor([[0.0 if n in lower else 0.5 for n in names]]*count))
        joint_armature = property(lambda self: Tensor(np.full((count, 29), 0.01)))
        joint_friction_coeff = property(lambda self: Tensor(np.full((count, 29), 0.1)))
        joint_dynamic_friction_coeff = property(lambda self: Tensor(np.full((count, 29), 0.1)))
        joint_viscous_friction_coeff = property(lambda self: Tensor(np.full((count, 29), 0.001)))
        body_mass = property(lambda self: Tensor([masses]*count))
        joint_pos_limits = property(lambda self: Tensor([[[reference.limits[n]["lower"], reference.limits[n]["upper"]] for n in names]]*count))
        joint_pos = property(lambda self: Tensor(self.robot.q))
        joint_vel = property(lambda self: Tensor(np.zeros((count, 29))))
        root_link_pose_w = property(lambda self: Tensor(self.robot.pose))
        root_link_vel_w = property(lambda self: Tensor(np.zeros((count, 6))))

        @property
        def body_link_pose_w(self):
            out = []
            for q, pose in zip(self.robot.q, self.robot.pose):
                fk = reference.forward(dict(zip(names, q, strict=True)))
                root = wbc.pose_matrix(pose)
                out.append([np.r_[(root @ fk[b])[:3, 3], wbc.matrix_quat_xyzw((root @ fk[b])[:3, :3])] for b in bodies])
            return Tensor(out)

    class Robot:
        joint_names, body_names = names, bodies
        def __init__(self):
            self.q = np.zeros((count, 29))
            self.pose = np.c_[origins + [0, 0, 0.8], np.tile([0, 0, 0, 1.0], (count, 1))]
            self.data = Data(self)
            self.root_writes = []
        def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None):
            self.pose[rows(env_ids, count)] = root_pose.value
            self.root_writes.append(rows(env_ids, count))
        def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids=None): pass
        def write_joint_position_to_sim_index(self, *, position, env_ids=None): self.q[rows(env_ids, count)] = position.value
        def write_joint_velocity_to_sim_index(self, *, velocity, env_ids=None): pass
        def set_joint_position_target_index(self, *, target): self.targets = target.value.copy()
        def set_joint_velocity_target_index(self, *, target): pass

    class Box:
        is_initialized = False
        def __init__(self):
            self.pose = np.c_[origins + [built.hand_reach + push["gap_m"] + push["size_m"][0]/2, 0, push["size_m"][2]/2],
                              np.tile([0, 0, 0, 1.0], (count, 1))]
            self.written = np.zeros(count)
            self.root_view = SimpleNamespace(get_material_properties=lambda: Tensor(
                np.stack([box_friction.astype(np.float32)]*2 + [np.zeros(count)], axis=1)[:, None, :]))
        def slide(self): return 0.0004*np.maximum(0, steps["count"]-self.written-80)
        @property
        def data(self):
            pose = self.pose.copy()
            pose[:, 0] += self.slide()
            velocity = np.zeros((count, 6))
            velocity[:, 0] = np.where(self.slide() > 0, 0.0004/0.005, 0.0)
            return SimpleNamespace(body_mass=Tensor(box_mass[:, None]), root_link_pose_w=Tensor(pose),
                                   root_link_vel_w=Tensor(velocity))
        def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None):
            ids = rows(env_ids, count)
            self.pose[ids], self.written[ids] = root_pose.value, steps["count"]
        def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids=None): pass

    class Camera:
        """640x360 RGBA frames whose content changes with the physics step."""
        def set_world_poses_from_view(self, eyes, targets): self.eyes, self.targets = eyes.value, targets.value
        def update(self, dt, force_recompute): pass
        @property
        def data(self):
            image = (np.add.outer(np.arange(360), np.arange(640)) + steps["count"]) % 256
            return SimpleNamespace(output={"rgb": Tensor(np.tile(image[None, :, :, None], (count, 1, 1, 4)))})

    # Each of two hand links feels 3 N from the box along -x, i.e. a 6 N push.
    hand = SimpleNamespace(data=SimpleNamespace(force_matrix_w=Tensor(np.tile([-3.0, 0.0, 0.0], (count, 2, 1, 1)))))
    # Leg/torso links touching the box: none by default.
    body = SimpleNamespace(body_names=["left_knee_link", "torso_link"],
                           data=SimpleNamespace(force_matrix_w=Tensor(np.zeros((count, 2, 1, 3)))))

    class Scene(dict):
        env_origins = Tensor(origins)
        def reset(self, env_ids=None): pass
        def update(self, dt): pass
        def write_data_to_sim(self): pass

    class Simulation:
        stage = object()
        def __init__(self, config): pass
        def reset(self): self.started = True
        def forward(self): pass
        def step(self, *, render): steps["count"] += 1
        def render(self): pass

    policy_calls = []

    class Policy:
        def __init__(self, path, contract): self.role = contract.get("policy_role")
        def batch(self, observations):
            policy_calls.append((self.role, len(observations)))
            return np.zeros((len(observations), 15), dtype=np.float32)

    def author_box(stage, box, env, mass, friction):
        i = int(env.rsplit("_", 1)[1])
        box_mass[i], box_friction[i] = mass, friction

    def material(stage, path):
        value = 1.0
        if "Furniture" in path:
            value = float(np.float32(box_friction[int(path.split("env_")[1].split("/")[0])]))
        return {"path": path, "static_friction": value, "dynamic_friction": value, "friction_combine_mode": "multiply"}

    built = SimpleNamespace(cfg=SimpleNamespace(sim=None, scene=None), hand_reach=0.415)
    robot, box = Robot(), Box()
    hands = {"status": "authored_fixed_hands_before_reset", "joints": dict.fromkeys(range(14), {})}
    report, _, _ = contact_fixture()
    modules = {"torch": ModuleType("torch"), "warp": ModuleType("warp"), "isaaclab": ModuleType("isaaclab"),
               "isaaclab.app": ModuleType("isaaclab.app"), "isaaclab.sim": ModuleType("isaaclab.sim"),
               "isaaclab.utils": ModuleType("isaaclab.utils")}
    modules["torch"].__dict__.update(float32=np.float32, manual_seed=lambda seed: None,
                                     as_tensor=lambda value, **kwargs: Tensor(value))
    modules["warp"].config = SimpleNamespace(enable_backward=True)
    modules["isaaclab.app"].launch_simulation = lambda *args: nullcontext()
    modules["isaaclab.sim"].SimulationContext = Simulation
    camera = Camera()
    modules["isaaclab.utils"].instantiate = lambda cfg: Scene(robot=robot, furniture=box, hand_contact=hand, body_contact=body,
                                                              camera=camera)
    authored = []
    with ExitStack() as stack:
        for patch in (mock.patch.dict(sys.modules, modules), mock.patch.object(wbc, "BalancePolicy", Policy),
                      mock.patch.object(rl, "apply_fixed_hands", side_effect=lambda stage, r, d, env: authored.append(env) or hands),
                      mock.patch.object(rl, "apply_palm_override", return_value={"status": "authored_before_physics_reset"}),
                      mock.patch.object(rl, "apply_source_contact", return_value=copy.deepcopy(report)),
                      mock.patch.object(rl, "apply_box_properties", side_effect=author_box),
                      mock.patch.object(rl, "read_material", side_effect=material),
                      mock.patch.object(rl, "read_joint_frames", return_value=joint_frame_fixture(reference)),
                      mock.patch.object(rl, "read_collision_geometry", side_effect=lambda stage, poses: contact_geometry(poses))):
            stack.enter_context(patch)
        scene_config = stack.enter_context(mock.patch.object(rl, "scene_config", return_value=built))
        yield SimpleNamespace(robot=robot, box=box, steps=steps, authored=authored, scene_config=scene_config,
                              origins=origins, box_mass=box_mass, box_friction=box_friction, camera=camera, hand=hand,
                              body=body, policy_calls=policy_calls)


class ParallelWorkerTests(unittest.TestCase):
    def test_worker_verifies_every_environment_and_runs_the_batched_loop(self):
        config, profile, asset, runtime = rl.load_config(rl.DEFAULT_CONFIG)
        count = 3
        config = {**config, "num_envs": count, "control_steps": 30}
        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(count, profile, asset) as fake:
            directory = Path(tmp)
            rl.run_worker(config, profile, asset, runtime, Path("unused.onnx"), directory)
            result = json.loads((directory / "rl-worker.json").read_text())
        self.assertEqual(result["status"], "passed_rl_throughput")
        self.assertEqual(fake.authored, [f"/World/envs/env_{i}" for i in range(count)])
        self.assertEqual(fake.scene_config.call_args.kwargs["replicate_physics"], False)
        self.assertEqual(result["setup"]["environments"], count)
        self.assertAlmostEqual(result["initialization"]["clearance_after_m"], 0.001, places=6)
        self.assertEqual(fake.steps["count"], 30*4)
        self.assertEqual((result["fallen_envs"], result["invalid_action_envs"]), (0, 0))
        self.assertAlmostEqual(result["box_progress_m"]["median"], 0.0004*(120-80), places=6)
        self.assertAlmostEqual(result["simulated_seconds_per_env"], 0.6)
        self.assertEqual(set(result["phase_seconds"]), {"write", "step", "update", "read_observe", "policy", "command"})
        self.assertLessEqual(sum(result["phase_seconds"].values()), result["wall_seconds"])
        self.assertAlmostEqual(result["seconds_per_physics_step"], result["wall_seconds"]/120)

    def test_worker_rejects_physx_friction_that_differs_from_usd(self):
        config, profile, asset, runtime = rl.load_config(rl.DEFAULT_CONFIG)
        config = {**config, "num_envs": 2, "control_steps": 4}
        with tempfile.TemporaryDirectory() as tmp, parallel_isaac(2, profile, asset) as fake:
            fake.box.root_view.get_material_properties = lambda: Tensor(np.full((2, 1, 3), 0.3))
            with self.assertRaisesRegex(RuntimeError, "friction materials"):
                rl.run_worker(config, profile, asset, runtime, Path("unused.onnx"), Path(tmp))


if __name__ == "__main__":
    unittest.main()
