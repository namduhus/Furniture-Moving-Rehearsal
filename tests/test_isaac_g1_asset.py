"""Check rejection of incompatible assets and the isolated inventory workflow."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isaac_g1_asset as audit


class AssetInventoryTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.runtime = audit.load_profile(audit.DEFAULT_PROFILE)
        names = sum(self.profile["joint_groups"].values(), [])
        self.inventory = {
            "joint_names": list(reversed(names)), "is_fixed_base": False,
            "gravity_disabled": False, "joint_state_on_cuda": True,
            "physics_manager": "isaaclab_physx.physics.PhysxManager",
            "joint_positions": [0.0] * 43, "default_joint_positions": [0.0] * 43,
            "joint_position_limits": [[-2.0, 2.0] for _ in names],
        }

    def test_runtime_order_is_mapped_by_name_without_claiming_walking(self):
        result = audit.validate_inventory(self.inventory, self.profile)
        names = self.inventory["joint_names"]
        actual = [names[i] for i in result["controller_body_to_sim_joint_indices"]]
        self.assertEqual(actual, self.profile["controller_reference"]["joint_order"])
        self.assertEqual(result["status"], "passed_asset_inventory")
        self.assertFalse(result["controller_inference_verified"])
        self.assertFalse(result["balance_or_walking_verified"])

    def test_fixed_base_gravity_cpu_and_other_physics_cannot_pass(self):
        for field, value in (("is_fixed_base", True), ("gravity_disabled", True),
                             ("joint_state_on_cuda", False), ("physics_manager", "newton.Manager")):
            with self.subTest(field=field):
                invalid = {**self.inventory, field: value}
                with self.assertRaises(RuntimeError):
                    audit.validate_inventory(invalid, self.profile)

    def test_old_or_duplicate_joints_cannot_pass_as_a_compatible_model(self):
        for names in (self.inventory["joint_names"][:37],
                      ["torso_joint"] + self.inventory["joint_names"][1:],
                      self.inventory["joint_names"][:-1] + [self.inventory["joint_names"][0]]):
            with self.subTest(names=names[:2]):
                with self.assertRaises(RuntimeError):
                    audit.validate_inventory({**self.inventory, "joint_names": names}, self.profile)

    def test_nonfinite_states_invalid_limits_and_bad_defaults_are_rejected(self):
        for field, value in (("joint_positions", [float("nan")] * 43),
                             ("joint_position_limits", [[2.0, -2.0]] * 43),
                             ("default_joint_positions", [3.0] * 43)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                audit.validate_inventory({**self.inventory, field: value}, self.profile)

    def test_profile_rejects_controller_schema_and_duplicate_names(self):
        for change in (lambda p: p["controller_reference"].update(observation_dim=123),
                       lambda p: p["joint_groups"]["hands"].__setitem__(0, p["joint_groups"]["arms"][0]),
                       lambda p: p.update(asset_config="G129_CFG_WITH_DEX3_BASE_FIX")):
            profile = copy.deepcopy(self.profile)
            change(profile)
            with tempfile.TemporaryDirectory() as name:
                path = Path(name) / "profile.json"
                path.write_text(json.dumps(profile))
                with self.assertRaises(ValueError):
                    audit.load_profile(path)

    def test_parent_revalidates_inventory_instead_of_trusting_passed_status(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            result = audit.validate_inventory(self.inventory, self.profile)
            (directory / "asset-worker.json").write_text(json.dumps(result))
            with self.assertRaises(OSError):
                audit.validate_worker("asset", directory, self.profile)
            inventory = {**self.inventory, "is_fixed_base": True}
            (directory / "asset-inventory.json").write_text(json.dumps(inventory))
            with self.assertRaises(RuntimeError):
                audit.validate_worker("asset", directory, self.profile)

    def test_dry_run_cannot_launch_a_worker_or_import_simulation(self):
        with mock.patch.object(audit, "run_stage", side_effect=AssertionError("worker launched")), \
                mock.patch.object(audit, "inspect_asset", side_effect=AssertionError("Isaac imported")), \
                mock.patch("builtins.print"):
            self.assertEqual(audit.main(["--dry-run"]), 0)

    def test_asset_initialization_uses_candidate_and_records_evidence_without_rollout(self):
        """Exercise actual inspect_asset control flow with simulator interface doubles."""
        inventory = self.inventory

        class Array:
            is_cuda = True

            def __init__(self, values): self.values = values
            def __getitem__(self, key): return Array(self.values[key])
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.values

        class SceneCfg:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)

        class Sim:
            physics_manager = type("PhysxManager", (), {"__module__": "isaaclab_physx.physics"})
            resets = 0

            def __init__(self, cfg): self.cfg = cfg
            def reset(self): Sim.resets += 1
            def step(self): raise AssertionError("Inventory must not request a control rollout")

        robot = SimpleNamespace(
            joint_names=inventory["joint_names"], body_names=["pelvis"], is_fixed_base=False,
            data=SimpleNamespace(joint_pos=Array([inventory["joint_positions"]]),
                                 default_joint_pos=Array([inventory["default_joint_positions"]]),
                                 joint_pos_limits=Array([inventory["joint_position_limits"]])),
        )
        candidate = SimpleNamespace(
            spawn=SimpleNamespace(fix_root_link=False, rigid_props=SimpleNamespace(disable_gravity=False),
                                  usd_path="reviewed-g1.usd"), actuators={},
            init_state=SimpleNamespace(pos=(0.0, 0.0, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
        )

        def replace(value, **kwargs):
            value = copy.deepcopy(value)
            value.__dict__.update(kwargs)
            return value

        modules = {}
        for key, fields in {
            "warp": {"config": SimpleNamespace(enable_backward=True)},
            "isaaclab": {}, "isaaclab.app": {"launch_simulation": lambda *a: nullcontext()},
            "isaaclab.assets": {"AssetBaseCfg": lambda **k: SimpleNamespace(**k)},
            "isaaclab.scene": {"InteractiveSceneCfg": SceneCfg},
            "isaaclab.sim": {"GroundPlaneCfg": SimpleNamespace, "SimulationCfg": SimpleNamespace, "SimulationContext": Sim},
            "isaaclab.utils": {"configclass": lambda c: c, "instantiate": lambda cfg: {"robot": robot}, "replace": replace},
            "isaaclab_assets": {"G1_29DOF_CFG": candidate},
            "isaaclab_physx": {}, "isaaclab_physx.physics": {"PhysxCfg": SimpleNamespace},
        }.items():
            module = ModuleType(key)
            module.__dict__.update(fields)
            modules[key] = module
        with tempfile.TemporaryDirectory() as name, mock.patch.dict(sys.modules, modules):
            directory = Path(name)
            audit.inspect_asset(self.profile, self.runtime, directory)
            audit.validate_worker("asset", directory, self.profile)
            saved = json.loads((directory / "asset-inventory.json").read_text())
            self.assertEqual(saved["usd_path"], "reviewed-g1.usd")
            self.assertEqual(saved["simulation_steps_requested"], 0)
            self.assertFalse(saved["controller_weights_loaded"])
            self.assertEqual(Sim.resets, 1)


if __name__ == "__main__":
    unittest.main()
