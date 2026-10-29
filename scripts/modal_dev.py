"""Start a bounded Modal SSH development session; default to a local cost plan."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/modal_dev.json"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    for key in ("cpu", "memory_mib", "session_seconds", "cleanup_seconds",
                "startup_seconds", "commit_interval_seconds"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["gpu"] != "L40S":
        raise ValueError("This launcher supports the selected L40S only")
    if not 60 <= config["session_seconds"] <= 7200:
        raise ValueError("session_seconds must be between 60 and 7200")
    if config["commit_interval_seconds"] >= config["session_seconds"]:
        raise ValueError("Commit interval must be shorter than the session")
    for key in ("app_name", "volume_name"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", config[key]):
            raise ValueError(f"Invalid {key}")
    for value in config["rates_usd_per_second"].values():
        if type(value) not in (int, float) or not 0 < value < 1:
            raise ValueError("Invalid resource price")
    if runtime := config.get("runtime"):
        if runtime["kind"] != "isaaclab" or runtime["path"] != "/opt/isaaclab":
            raise ValueError("Unsupported runtime profile")
        if runtime["repository"] != "https://github.com/isaac-sim/IsaacLab.git":
            raise ValueError("Use the official Isaac Lab repository")
        if not re.fullmatch(r"[0-9a-f]{40}", runtime["commit"]):
            raise ValueError("Pin an immutable Isaac Lab commit")
        if not re.fullmatch(r"[0-9a-f]{64}", runtime["lock_sha256"]):
            raise ValueError("Pin the upstream lock SHA-256")
        if runtime["extras"] != ["isaacsim", "rsl-rl"]:
            raise ValueError("The selected runtime requires the Isaac Sim and RSL-RL extras")
    return config


def cost_plan(config: dict) -> dict:
    rates = config["rates_usd_per_second"]
    per_second = (rates["gpu"] + config["cpu"] * rates["cpu_core"]
                  + config["memory_mib"] / 1024 * rates["memory_gib"])
    seconds = config["session_seconds"]
    return {
        "status": "local_plan_only",
        "gpu": config["gpu"],
        "cpu_cores_request_and_limit": config["cpu"],
        "memory_mib_request_and_limit": config["memory_mib"],
        "session_seconds": seconds,
        "function_timeout_seconds": seconds + config["cleanup_seconds"],
        "startup_timeout_seconds": config["startup_seconds"],
        "estimated_session_compute_usd": round(per_second * seconds, 6),
        "estimated_compute_with_startup_cleanup_usd": round(
            per_second * (seconds + config["cleanup_seconds"]
                          + config["startup_seconds"] + 2), 6),
        "rates_reviewed": config["rates_reviewed"],
        "pricing_source": "https://modal.com/pricing",
        "exclusions": "Image build, storage, network and other jobs; not a dollar cap.",
        "persistence": f"Modal Volume v2 {config['volume_name']} at /workspace",
        "runtime_scope": (f"SSH + Isaac Lab {config['runtime']['release']} / Isaac Sim; not validated"
                          if config.get("runtime") else
                          "SSH, uv and GPU inventory only; Isaac is not installed."),
        "isaac_eula_required": bool(config.get("runtime")),
    }


def canonical_public_key(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1 or "PRIVATE KEY" in text or len(text) > 16384:
        raise ValueError("Provide exactly one OpenSSH public key, never a private key")
    fields = lines[0].split()
    if len(fields) < 2 or fields[0] not in {
        "ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256",
        "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521",
    }:
        raise ValueError("Unsupported public key; an Ed25519 public key is recommended")
    try:
        raw = base64.b64decode(fields[1], validate=True)
        size = struct.unpack(">I", raw[:4])[0]
        if raw[4:4 + size].decode("ascii") != fields[0]:
            raise ValueError("Mismatched SSH key type")
    except (ValueError, UnicodeError, struct.error) as exc:
        raise ValueError("Invalid SSH public key encoding") from exc
    # Discard comments, which can contain personal identifiers.
    key = " ".join(fields[:2])
    with tempfile.TemporaryDirectory(prefix="fmr-public-key-") as directory:
        path = Path(directory) / "key.pub"
        path.write_text(key + "\n")
        result = subprocess.run(["ssh-keygen", "-l", "-f", str(path)],
                                capture_output=True, timeout=5)
        if result.returncode:
            raise ValueError("ssh-keygen rejected the public key")
    return key


def ssh_quote(value: str) -> str:
    if any(ord(char) < 32 or ord(char) == 127 for char in value) or "%" in value:
        raise ValueError("SSH configuration values cannot contain control characters or %")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_connection(directory: Path, event: dict, identity: Path) -> Path:
    host, port = event["host"], event["port"]
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host):
        raise ValueError("Invalid tunnel hostname")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid tunnel port")
    key = canonical_public_key(event["host_public_key"])
    known_hosts = directory / "known_hosts"
    host_token = host if port == 22 else f"[{host}]:{port}"
    known_hosts.write_text(f"{host_token} {key}\n")
    known_hosts.chmod(0o600)
    config_path = directory / "ssh_config"
    config_path.write_text(
        "Host fmr-modal\n"
        f"    HostName {host}\n    Port {port}\n    User root\n"
        f"    IdentityFile {ssh_quote(str(identity))}\n"
        f"    UserKnownHostsFile {ssh_quote(str(known_hosts))}\n"
        "    StrictHostKeyChecking yes\n    IdentitiesOnly yes\n"
        "    ForwardAgent no\n    ServerAliveInterval 30\n"
        "    ServerAliveCountMax 3\n"
    )
    config_path.chmod(0o600)
    return config_path


@contextmanager
def session_lock():
    """Prevent duplicate GPU sessions started from this local checkout."""
    import fcntl

    directory = ROOT / "outputs/modal"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".launcher.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A launcher is already running in this checkout") from exc
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def serve_ssh(public_key: str, session_id: str, settings: dict):
    """Run inside Modal; send connection data through the authenticated SDK channel."""
    import json
    import os
    from pathlib import Path
    import socket
    import subprocess
    import tempfile
    import time
    import sys

    import modal

    deadline = time.monotonic() + settings["session_seconds"]
    workspace = Path("/workspace")
    volume = modal.Volume.from_name(settings["volume_name"])
    metadata = workspace / ".fmr"
    metadata.mkdir(exist_ok=True)
    marker = metadata / "last_session.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    for name in ("repo", "outputs", "models", ".cache", ".vscode-server"):
        (workspace / name).mkdir(exist_ok=True)
    # VS Code installs its server here by default. Keep it across sessions.
    vscode = Path("/root/.vscode-server")
    if not vscode.exists() and not vscode.is_symlink():
        vscode.symlink_to(workspace / ".vscode-server", target_is_directory=True)
    with Path("/root/.bashrc").open("a") as file:
        file.write('\nexport UV_CACHE_DIR=/workspace/.cache/uv\n'
                   'if [ -n "${SSH_CONNECTION:-}" ]; then cd /workspace; fi\n')
        if settings.get("runtime"):
            file.write('source /opt/isaaclab/.venv/bin/activate\n')
    stop_path = Path("/run/fmr-stop")
    stop_path.unlink(missing_ok=True)
    for name, body in {
        "fmr-save": 'sync /workspace\necho "Workspace committed."\n',
        "fmr-stop": 'sync /workspace\ntouch /run/fmr-stop\n'
                    'echo "Workspace committed; shutdown requested."\n',
    }.items():
        path = Path("/usr/local/bin") / name
        path.write_text("#!/bin/sh\nset -eu\n" + body)
        path.chmod(0o755)
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    if "L40S" not in inventory:
        raise RuntimeError("Expected an L40S; refusing to start SSH")
    versions = {
        "python": sys.version,
        "uv": subprocess.run(["uv", "--version"], capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip(),
        "system_packages": subprocess.run(
            ["dpkg-query", "-W", "-f=${Package}=${Version}\\n",
             "openssh-server", "git", "libstdc++6"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip(),
    }
    marker.write_text(json.dumps({"session_id": session_id, "gpu_inventory": inventory,
                                  "versions": versions}))
    volume.commit()
    reason = "session_timeout"
    daemon = None
    # StrictModes rejects authorized_keys beneath world-writable /tmp.
    with tempfile.TemporaryDirectory(prefix="fmr-sshd-", dir="/run") as directory:
        runtime = Path(directory)
        runtime.chmod(0o700)
        authorized = runtime / "authorized_keys"
        authorized.write_text(public_key + "\n")
        authorized.chmod(0o600)
        host_key = runtime / "host_ed25519"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                        "-f", str(host_key)], check=True, timeout=10)
        Path("/run/sshd").mkdir(exist_ok=True)
        config = runtime / "sshd_config"
        session_env = {"UV_CACHE_DIR": "/workspace/.cache/uv"}
        if settings.get("runtime"):
            session_env.update({"OMNI_KIT_ACCEPT_EULA": "yes", "ISAACSIM_ACCEPT_EULA": "Y",
                                "MUJOCO_GL": "disable", "XDG_CACHE_HOME": "/workspace/.cache"})
        # sshd constructs a fresh environment; preserve only driver/library routing here.
        for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"):
            if value := os.environ.get(name):
                if any(char.isspace() or char in '"\\' for char in value):
                    raise RuntimeError(f"Cannot safely pass {name} to the SSH environment")
                session_env[name] = value
        ssh_environment = "SetEnv " + " ".join(f'{key}="{value}"' for key, value in session_env.items()) + "\n"
        config.write_text(
            "Port 2222\nListenAddress 0.0.0.0\n"
            f"HostKey {host_key}\nAuthorizedKeysFile {authorized}\n"
            f"PidFile {runtime / 'sshd.pid'}\n"
            "PermitRootLogin prohibit-password\nPasswordAuthentication no\n"
            "KbdInteractiveAuthentication no\nAuthenticationMethods publickey\n"
            "PubkeyAuthentication yes\nUsePAM yes\nStrictModes yes\n"
            "AllowAgentForwarding no\nAllowTcpForwarding yes\nGatewayPorts no\n"
            "X11Forwarding no\nPermitTunnel no\nPermitTTY yes\n"
            "ClientAliveInterval 30\nClientAliveCountMax 3\n"
            "Subsystem sftp internal-sftp\nLogLevel ERROR\n" + ssh_environment
        )
        subprocess.run(["/usr/sbin/sshd", "-t", "-f", str(config)],
                       check=True, timeout=10)
        log_path = metadata / f"sshd-{session_id}.log"
        with log_path.open("a") as log:
            try:
                daemon = subprocess.Popen(
                    ["/usr/sbin/sshd", "-D", "-e", "-f", str(config)],
                    stdout=log, stderr=log,
                )
                ready_deadline = min(deadline, time.monotonic() + 15)
                while True:
                    if daemon.poll() is not None:
                        raise RuntimeError(f"SSH server exited; inspect {log_path}")
                    try:
                        with socket.create_connection(("127.0.0.1", 2222), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() >= ready_deadline:
                            raise RuntimeError("SSH server did not become ready")
                        time.sleep(0.2)
                with modal.forward(2222, unencrypted=True) as tunnel:
                    host, port = tunnel.tcp_socket
                    yield {
                        "status": "ssh_server_ready",
                        "host": host, "port": port,
                        "host_public_key": host_key.with_suffix(".pub").read_text(),
                        "gpu_inventory": inventory,
                        "versions": versions,
                        "previous_session": previous,
                        "session_id": session_id,
                        "remaining_seconds": max(0, int(deadline - time.monotonic())),
                        "isaac_status": ("installed_not_validated" if settings.get("runtime")
                                         else "not_installed"),
                    }
                    last_commit = time.monotonic()
                    while time.monotonic() < deadline:
                        if stop_path.exists():
                            reason = "requested_stop"
                            break
                        if daemon.poll() is not None:
                            raise RuntimeError(f"SSH server exited; inspect {log_path}")
                        if time.monotonic() - last_commit >= settings["commit_interval_seconds"]:
                            volume.commit()
                            last_commit = time.monotonic()
                        time.sleep(1)
            finally:
                try:
                    volume.commit()
                finally:
                    if daemon is not None and daemon.poll() is None:
                        daemon.terminate()
                        try:
                            daemon.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            daemon.kill()
                            daemon.wait(timeout=5)
    yield {"status": "session_finished", "reason": reason, "session_id": session_id}


def mountpoint_check_command(path: str) -> str:
    """Fail during image build if a future Volume mount would hide image files."""
    # Image.run_commands inserts this directly after Dockerfile RUN. Shell
    # quoting does not protect physical newlines from the Dockerfile parser.
    code = ("import sys; from pathlib import Path; "
            f"p = Path({path!r}); "
            "sys.exit('Volume mountpoint must be empty in the image; keep build caches elsewhere') "
            "if p.exists() and (not p.is_dir() or any(p.iterdir())) else None")
    return shlex.join(["python3", "-c", code])


def runtime_build_commands(config: dict) -> list[str]:
    """Use the pinned upstream checkout and its frozen lock without local project uploads."""
    runtime = config.get("runtime")
    if not runtime:
        return []
    path = runtime["path"]
    commit = runtime["commit"]
    extras = " ".join(f"--extra {shlex.quote(extra)}" for extra in runtime["extras"])
    return [
        f"git init {path} && git -C {path} remote add origin {runtime['repository']} "
        f"&& git -C {path} fetch --depth 1 origin {commit} "
        f"&& git -C {path} checkout --detach {commit}",
        f"test \"$(git -C {path} rev-parse HEAD)\" = {commit}",
        f"echo '{runtime['lock_sha256']}  {path}/uv.lock' | sha256sum -c -",
        f"timeout --signal=TERM --kill-after=30s 1800 "
        f"uv sync --project {path} --frozen {extras} --python 3.12",
        mountpoint_check_command("/workspace"),
    ]


def create_app(config: dict):
    import modal

    installed = importlib.metadata.version("modal")
    if installed != config["modal_version"]:
        raise RuntimeError(f"Use modal=={config['modal_version']}; found {installed}")
    image = (modal.Image.from_registry(config["image"], add_python=config["python_version"])
             .apt_install("openssh-server", "git", "ca-certificates", "curl", "procps",
                          "libstdc++6", "tar", "gzip")
             .pip_install(f"uv=={config['uv_version']}"))
    if config.get("runtime"):
        image = (image.apt_install(
            "build-essential", "cmake", "pkg-config", "libglib2.0-0", "libsm6", "libxext6",
            "libxrender1", "libgl1", "libegl1", "libvulkan1", "vulkan-tools", "libxi6",
            "libxrandr2", "libxcursor1", "libxinerama1", "libgomp1",
        ).env({
            "OMNI_KIT_ACCEPT_EULA": "yes", "ISAACSIM_ACCEPT_EULA": "Y",
            "NVIDIA_DRIVER_CAPABILITIES": "all", "MUJOCO_GL": "disable",
            "XDG_CACHE_HOME": "/opt/fmr-build-cache", "UV_HTTP_TIMEOUT": "120",
        }).run_commands(*runtime_build_commands(config)))
    volume = modal.Volume.from_name(config["volume_name"], create_if_missing=True, version=2)
    app = modal.App(config["app_name"])
    function = app.function(
        image=image, gpu=config["gpu"], cpu=(config["cpu"], config["cpu"]),
        memory=(config["memory_mib"], config["memory_mib"]),
        volumes={"/workspace": volume}, max_containers=1, min_containers=0,
        buffer_containers=0, scaledown_window=2,
        timeout=config["session_seconds"] + config["cleanup_seconds"],
        startup_timeout=config["startup_seconds"],
        single_use_containers=True, serialized=True, include_source=False,
        env={"UV_CACHE_DIR": "/workspace/.cache/uv", "XDG_CACHE_HOME": "/workspace/.cache"},
    )(serve_ssh)
    return app, function


def start(config: dict, public_path: Path, identity: Path | None,
          accept_isaac_eula: bool = False) -> int:
    if config.get("runtime") and not accept_isaac_eula:
        raise ValueError("Review the NVIDIA Omniverse EULA linked in the runtime profile, "
                         "then pass --accept-isaac-eula to start that image")
    public_path = public_path.expanduser().resolve()
    if public_path.suffix != ".pub":
        raise ValueError("--public-key must name a .pub file; private keys are never uploaded")
    public_key = canonical_public_key(public_path.read_text())
    identity = (identity.expanduser().resolve() if identity else public_path.with_suffix(""))
    if not identity.is_file():
        raise ValueError("Matching local identity file missing; pass --identity-file explicitly")
    ssh_quote(str(identity))
    app, function = create_app(config)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = ROOT / "outputs/modal" / session_id
    directory.mkdir(parents=True, mode=0o700)
    summary = {"status": "starting", "session_id": session_id, "plan": cost_plan(config)}
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        import modal
        with modal.enable_output(), app.run():
            summary["app_id"] = app.app_id
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            emergency = (f"uv run --project environments/modal --locked modal app stop {app.app_id}")
            print(f"Session output: {directory}\nEmergency stop: {emergency}", flush=True)
            with (directory / "events.jsonl").open("a") as log:
                for event in function.remote_gen(public_key, session_id, config):
                    log.write(json.dumps(event) + "\n")
                    log.flush()
                    summary["status"] = event["status"]
                    if event["status"] == "ssh_server_ready":
                        connection = write_connection(directory, event, identity)
                        current_config = directory.parent / "ssh_config"
                        current_config.write_text(connection.read_text())
                        current_config.chmod(0o600)
                        summary["ssh_config"] = str(connection)
                        summary["gpu_inventory"] = event["gpu_inventory"]
                        summary["versions"] = event["versions"]
                        summary["previous_session"] = event["previous_session"]
                        print(f"\nSSH: ssh -F {shlex.quote(str(connection))} fmr-modal\n"
                              f"VS Code SSH configuration file: {current_config}\n"
                              "Open remote folder: /workspace/repo\n"
                              "Save: fmr-save | Save and stop: fmr-stop\n"
                              f"Remaining seconds: {event['remaining_seconds']}\n"
                              "Keep this terminal open. Closing VS Code does not stop the GPU.\n"
                              f"Isaac status: {event['isaac_status']}\n",
                              flush=True)
                    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return 0
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error_type"] = type(exc).__name__
        (directory / "error.log").write_text(traceback.format_exc())
        if app_id := summary.get("app_id"):
            command = ("uv run --project environments/modal --locked "
                       f"modal app logs {app_id} --tail 200 --timestamps")
            summary["log_command"] = command
            print(f"App log query (does not start a worker): {command}", file=sys.stderr, flush=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "start"), nargs="?", default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--public-key", type=Path, default=Path("~/.ssh/id_ed25519.pub"))
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--accept-isaac-eula", action="store_true",
                        help="Accept the NVIDIA EULA for the optional Isaac image")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    print(json.dumps(cost_plan(config), indent=2), flush=True)
    if args.action == "plan":
        return 0
    with session_lock():
        return start(config, args.public_key, args.identity_file, args.accept_isaac_eula)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
