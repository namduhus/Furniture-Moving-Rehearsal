"""Offline checks for SSH credentials, configuration and duplicate billing guards."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import modal_dev as launcher


class ModalDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fmr-ssh-tests-")
        cls.directory = Path(cls.temporary.name)
        cls.identity = cls.directory / "test_ed25519"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                        "-C", "test-comment", "-f", str(cls.identity)], check=True)
        cls.public = cls.identity.with_suffix(".pub").read_text()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_public_key_validation_discards_comments_and_rejects_private_keys(self):
        self.assertEqual(launcher.canonical_public_key(self.public),
                         " ".join(self.public.split()[:2]))
        invalid_keys = [self.identity.read_text(), self.public + self.public,
                        'command="evil" ' + self.public, "ssh-ed25519 AAAA"]
        for text in invalid_keys:
            with self.subTest(text_type=text[:16]), self.assertRaises(ValueError):
                launcher.canonical_public_key(text)

    def test_generated_config_is_accepted_by_openssh_and_pins_host_key(self):
        directory = self.directory / "connection with spaces"
        directory.mkdir(exist_ok=True)
        event = {"host": "example.modal.host", "port": 12345,
                 "host_public_key": self.public}
        config = launcher.write_connection(directory, event, self.identity)
        result = subprocess.run(["ssh", "-G", "-F", str(config), "fmr-modal"],
                                capture_output=True, text=True, check=True)
        self.assertIn("hostname example.modal.host\n", result.stdout)
        self.assertIn("port 12345\n", result.stdout)
        self.assertIn("stricthostkeychecking true\n", result.stdout)
        self.assertIn("forwardagent no\n", result.stdout)
        self.assertIn("[example.modal.host]:12345 ", (directory / "known_hosts").read_text())
        self.assertEqual(config.stat().st_mode & 0o777, 0o600)

    def test_tunnel_data_cannot_inject_ssh_directives(self):
        for host, port in (("example\nProxyCommand evil", 22), ("example", 0),
                           ("example", True), ("example", 65536)):
            with self.subTest(host=host, port=port), self.assertRaises(ValueError):
                launcher.write_connection(self.directory, {
                    "host": host, "port": port, "host_public_key": self.public,
                }, self.identity)
        with self.assertRaises(ValueError):
            launcher.ssh_quote("/tmp/identity\nProxyCommand evil")

    def test_plan_requires_no_modal_sdk_or_cloud_call(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/modal_dev.py"), "plan"],
            capture_output=True, text=True, check=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "local_plan_only")
        self.assertAlmostEqual(plan["estimated_session_compute_usd"], 0.351576)
        self.assertGreater(plan["estimated_compute_with_startup_cleanup_usd"],
                           plan["estimated_session_compute_usd"])

    def test_invalid_resource_values_fail_before_cloud_access(self):
        original = json.loads(launcher.DEFAULT_CONFIG.read_text())
        for name, value in (("session_seconds", 0), ("session_seconds", 10801),
                            ("memory_mib", True), ("cpu", -1), ("gpu", "H100")):
            with self.subTest(name=name, value=value):
                path = self.directory / "invalid.json"
                path.write_text(json.dumps({**original, name: value}))
                with self.assertRaises(ValueError):
                    launcher.load_config(path)

    def test_duplicate_launcher_is_rejected(self):
        with mock.patch.object(launcher, "ROOT", self.directory):
            with launcher.session_lock():
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with launcher.session_lock():
                        self.fail("Duplicate session acquired the lock")
            with launcher.session_lock():
                pass

    def test_private_key_path_is_rejected_before_sdk_access(self):
        with mock.patch.object(launcher, "create_app") as create:
            with self.assertRaisesRegex(ValueError, r"\.pub"):
                launcher.start({}, self.identity, None)
            create.assert_not_called()

    def test_sdk_accepts_the_graph_without_hydrating_cloud_resources(self):
        try:
            import modal
        except ImportError:
            self.skipTest("Run in environments/modal to validate the SDK graph")
        app, function = launcher.create_app(launcher.load_config(launcher.DEFAULT_CONFIG))
        self.assertEqual(app.name, "fmr-ssh-dev")
        self.assertIsNone(app.app_id)
        self.assertIsInstance(function, modal.Function)


if __name__ == "__main__":
    unittest.main()
