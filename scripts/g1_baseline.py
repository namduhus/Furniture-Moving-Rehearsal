"""Check a single NVIDIA GPU and G1 environment, or launch a Brax PPO pilot."""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def train_command(config, run_dir):
    flags = {
        **config["train_flags"],
        "env_name": config["environment"],
        "impl": config["implementation"],
        "seed": config["seed"],
        "logdir": str(run_dir / "training"),
    }
    if flags["batch_size"] * flags["num_minibatches"] % flags["num_envs"]:
        raise ValueError("batch_size * num_minibatches must be divisible by num_envs")
    command = [sys.executable, "-u", "-c", "from learning.train_jax_ppo import run; run()"]
    for key, value in flags.items():
        value = str(value).lower() if isinstance(value, bool) else str(value)
        command.append(f"--{key}={value}")
    return command


def check_gpu(summary):
    if sys.platform != "linux":
        raise RuntimeError("Run this preflight on the Linux NVIDIA server; use smoke_mujoco.py on Mac")
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    summary["nvidia_smi"] = gpu_info.stdout.strip().splitlines()
    summary["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "brax", "mujoco", "mujoco-mjx", "playground")
    }

    import jax
    import jax.numpy as jp

    devices = jax.devices()
    summary["jax_devices"] = [str(device) for device in devices]
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected exactly one visible GPU, found {devices}")
    x = jp.ones((256, 256), dtype=jp.float32)
    multiply = jax.jit(lambda value: value @ value)
    started = time.perf_counter()
    result = multiply(x).block_until_ready()
    summary["gpu_probe_compile_and_first_call_s"] = time.perf_counter() - started
    started = time.perf_counter()
    result = multiply(x).block_until_ready()
    summary["gpu_probe_second_call_s"] = time.perf_counter() - started
    if not bool(jp.all(result == 256.0)):
        raise RuntimeError("GPU matrix multiplication returned unexpected values")
    summary["checks"]["gpu_compute"] = True
    return jax, jp


def check_environment(config, run_dir, summary, jax, jp):
    from mujoco_playground import registry

    env = registry.load(
        config["environment"],
        config_overrides={"impl": config["implementation"]},
    )
    summary["environment"] = config["environment"]
    summary["implementation"] = config["implementation"]
    summary["action_size"] = int(env.action_size)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    started = time.perf_counter()
    state = jax.block_until_ready(reset(jax.random.PRNGKey(config["seed"])))
    summary["reset_compile_and_first_call_s"] = time.perf_counter() - started
    initial_time = float(state.data.time)
    action = jp.zeros(env.action_size)
    terminations = 0
    with (run_dir / "trajectory.jsonl").open("w") as trajectory:
        for index in range(config["smoke_steps"]):
            started = time.perf_counter()
            state = jax.block_until_ready(step(state, action))
            elapsed = time.perf_counter() - started
            if index == 0:
                summary["step_compile_and_first_call_s"] = elapsed
            finite = all(math.isfinite(float(v)) for v in state.data.qpos)
            finite = finite and all(math.isfinite(float(v)) for v in state.data.qvel)
            if not finite:
                raise RuntimeError("Non-finite G1 physics state")
            terminations += int(state.done)
            trajectory.write(json.dumps({
                "step": index + 1,
                "time_s": float(state.data.time),
                "reward": float(state.reward),
                "done": bool(state.done),
                "qpos": state.data.qpos.tolist(),
                "qvel": state.data.qvel.tolist(),
                "wall_s": elapsed,
            }) + "\n")
    summary["simulation_time_s"] = float(state.data.time) - initial_time
    summary["terminations_observed"] = terminations
    if summary["simulation_time_s"] <= 0:
        raise RuntimeError("G1 simulation time did not advance")
    summary["checks"]["environment_steps_finite"] = True
    summary["policy"] = "zero actions; environment smoke only, not a walking policy"
    summary["rendering_checked"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gpu", "env", "train"), default="gpu")
    parser.add_argument("--gpu", type=int, default=1, help="Physical NVIDIA GPU index; expose only this GPU")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g1_baseline.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/g1")
    parser.add_argument("--dry-run", action="store_true", help="Print the training command without running it")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.output_dir.resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "failed", "stage": args.stage, "python": platform.python_version(),
        "platform": platform.system(), "machine": platform.machine(),
        "physical_gpu_index": args.gpu, "checks": {},
    }
    try:
        config = json.loads(args.config.read_text())
        if config["implementation"] != "jax":
            raise ValueError("This baseline requires the explicitly selected JAX implementation")
        if config["smoke_steps"] < 1:
            raise ValueError("smoke_steps must be positive")
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        summary["seed"] = config["seed"]
        command = train_command(config, run_dir)
        if args.dry_run:
            print(shlex.join(command))
            summary["command"] = command
            summary["status"] = "dry_run_only"
            return 0

        # Set GPU selection before importing JAX or loading CUDA libraries.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["MUJOCO_GL"] = "egl"
        if os.environ.get("LD_LIBRARY_PATH"):
            raise RuntimeError(
                "LD_LIBRARY_PATH is set and may override wheel-provided CUDA libraries. "
                "Run with env -u LD_LIBRARY_PATH as documented."
            )
        jax, jp = check_gpu(summary)
        if args.stage == "env":
            check_environment(config, run_dir, summary, jax, jp)
        elif args.stage == "train":
            if not shutil.which("ffmpeg"):
                raise RuntimeError("The upstream rollout video requires ffmpeg on PATH")
            summary["command"] = command
            with (run_dir / "train.log").open("w") as log:
                started = time.perf_counter()
                with subprocess.Popen(
                    command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                ) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                summary["train_process_wall_s"] = time.perf_counter() - started
            summary["train_returncode"] = returncode
            if returncode:
                raise RuntimeError(f"Upstream training exited with code {returncode}; inspect train.log")
            summary["status"] = "pilot_process_completed"
            return 0
        summary["status"] = f"passed_{args.stage}_smoke"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        (run_dir / "error.log").write_text(traceback.format_exc())
        print(summary["error"], file=sys.stderr)
    finally:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"status": summary["status"], "run_dir": str(run_dir)}))
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
