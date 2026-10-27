# Furniture-Moving Rehearsal

A simulated humanoid physically rehearses moving furniture through hallways and corners before a person attempts the move. Built for the **Push the Frontier** theme at the Meta Global AI Developer Hackathon.

The intended pipeline combines Meta models for scene interpretation and failure diagnosis with a physics simulation that checks proposed moving strategies. The target is a **VLM-guided humanoid controller**.

## Documentation

See [AGENTS.md](AGENTS.md) for AI collaboration and development guidelines.

## Local MuJoCo smoke check

Python is pinned to 3.12. MuJoCo is pinned to 3.17.0 in `pyproject.toml`, and `uv.lock` records the resolved dependencies and wheel hashes.

To reproduce the smoke check, run the following from a terminal with PyPI access:

```sh
uv sync --locked --no-build --default-index https://pypi.org/simple
uv run --locked python scripts/smoke_mujoco.py
```

Keep `uv.lock` with the dependency change. The script loads a newly authored box-and-floor scene, advances 1,000 CPU physics steps at 0.002 seconds per step, and checks finite states, simulation warnings, elapsed simulated time, floor contact, and the box's resting height and velocity. The fixed scene has no random sampling; the seed is recorded for future experiments.

Each run creates a separate directory under `outputs/smoke/` containing the scene, configuration, `trajectory.jsonl`, and `summary.json`. A full successful run also writes `initial.png` and `final.png` and checks that the RGB frames are non-uniform and changed. The process exits nonzero on failure and writes `error.log`. Only `"status": "passed"` indicates a complete smoke-check pass.

For physics-only diagnosis, run `uv run --locked python scripts/smoke_mujoco.py --physics-only`. A physics-only result does not verify rendering. Offscreen rendering is the default; no interactive viewer is opened. A remote Linux machine may need a suitable `MUJOCO_GL` backend.

To open the scene in the interactive viewer, run:

```sh
uv run --locked python -m mujoco.viewer --mjcf=assets/scenes/smoke.xml
```

## GPU JAX/Brax workflow

Run these commands from the cloned repository root on Linux x86_64. The training environment is a separate uv project in `environments/gpu/`; the root project is for local CPU MuJoCo. Both use the scripts and configurations in this repository.

The training project uses the JAX CUDA 12 wheels with their CUDA/cuDNN runtime dependencies. JAX's documented minimum Linux driver for CUDA 12 is 525; CUDA 13 requires 580 or newer. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Runtime GPU compatibility is checked by the preflight rather than inferred from `nvidia-smi` alone.

Install the environment:

```sh
uv sync --project environments/gpu --no-build --default-index https://pypi.org/simple
```

The first sync generates `environments/gpu/uv.lock`; retain that file with the training environment. Once it exists, use `uv sync --project environments/gpu --locked --no-build` for subsequent installations. The root `uv.lock` belongs to the local CPU project.

Check GPU computation, then G1 loading and physics steps:

```sh
env -u LD_LIBRARY_PATH uv run --project environments/gpu --locked python scripts/g1_baseline.py --stage gpu --gpu 1
env -u LD_LIBRARY_PATH uv run --project environments/gpu --locked python scripts/g1_baseline.py --stage env --gpu 1
```

Change `--gpu` to the physical GPU index assigned to the run. The script exposes only that GPU before importing JAX, so it appears as logical device 0. Removing `LD_LIBRARY_PATH` applies only to the command and prevents a local CUDA installation from overriding wheel-provided libraries. Results and errors go into a unique directory under `outputs/g1/`. The environment stage uses zero actions to check compilation and finite physics states; it does not demonstrate walking or check rendering. The first G1 load downloads Menagerie assets using Git and needs GitHub access.

Inspect the PPO pilot command without launching it:

```sh
uv run python scripts/g1_baseline.py --stage train --dry-run
```

After the GPU and environment checks pass, launch the pilot:

```sh
env -u LD_LIBRARY_PATH uv run --project environments/gpu --locked python scripts/g1_baseline.py --stage train --gpu 1
```

The pilot uses [configs/g1_baseline.json](configs/g1_baseline.json): JAX/Brax PPO, `G1JoystickFlatTerrain`, seed 0, 256 environments, and 1 million requested environment steps. These are initial validation settings, not a claim of sufficient training or a measured memory fit. The upstream training process requires `ffmpeg` on `PATH` and EGL rendering support to save its rollout video. Its log, checkpoints, and video are placed under the run's `training/` directory. `pilot_process_completed` means the upstream process exited successfully; inspect the rollout and evaluation results before claiming successful walking.

## Dependencies and licenses

Google DeepMind maintains the canonical [`mujoco` PyPI package](https://pypi.org/project/mujoco/). Its engine and Python bindings use [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/main/LICENSE); prebuilt wheels also include third-party notices documented on the package page. Dependency versions and wheel SHA-256 hashes are recorded in `uv.lock`, using the official PyPI registry.

Transitive package licenses: absl-py and etils (Apache-2.0), fsspec and websockets (BSD-3-Clause), glfw and zipp (MIT), PyOpenGL (BSD), typing-extensions (PSF-2.0), and NumPy (BSD-3-Clause, 0BSD, MIT, Zlib, and CC0-1.0). Retain the installed distributions' bundled license notices when redistributing their binaries.

The training project pins JAX 0.9.2, Brax 0.14.2, MuJoCo/MJX 3.17.0, and MuJoCo Playground 0.2.0 from official PyPI distributions. Brax 0.14.2 uses a replication API removed in JAX 0.10; retain the compatible JAX pin and resolved Flax dependency in the training lock. [JAX](https://github.com/jax-ml/jax), [Brax](https://github.com/google/brax), [MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx), and [Playground](https://github.com/google-deepmind/mujoco_playground) use Apache-2.0. Use Brax's maintained training library with Playground environments. The [Menagerie G1 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) has a separate BSD-3-Clause license. CUDA/cuDNN wheels and the system `ffmpeg` distribution retain their own license notices and terms.

## Original contributions and third-party components

Keep this table current in the same commit as each new module or third-party addition. It identifies the origin and purpose of repository components.

| Component | Origin | Purpose | Source / license |
| --- | --- | --- | --- |
| Collaboration instructions | Project planning | Agent guidance | [AGENTS.md](AGENTS.md) |
| README and ignore rules | Created during the hackathon | Usage documentation and local artifact exclusions | [README](README.md), [.gitignore](.gitignore) |
| CPU smoke script, box-drop scene, and configuration | Created during the hackathon | Check scene loading, CPU physics, contact, and rendering | [Script](scripts/smoke_mujoco.py), [scene](assets/scenes/smoke.xml), [configuration](configs/smoke.json) |
| MuJoCo 3.17.0 and its Python dependencies | Third-party software | Physics engine, Python bindings, and rendering support | [Official package](https://pypi.org/project/mujoco/), [Apache-2.0 and bundled notices](https://pypi.org/project/mujoco/#license-and-disclaimer), [lock file](uv.lock) |
| G1 preflight and PPO pilot launcher | Created during the hackathon | GPU and environment checks, configured upstream training, and local run logs | [Launcher](scripts/g1_baseline.py), [configuration](configs/g1_baseline.json), [training environment](environments/gpu/pyproject.toml) |
| JAX, Brax, MJX, and MuJoCo Playground | Third-party software | GPU computation, PPO training, and the G1 locomotion environment | Official sources and Apache-2.0 licenses linked above |
| Menagerie G1 robot assets | Third-party assets | Robot geometry, dynamics, and collision model fetched by Playground | [Source and BSD-3-Clause license](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) |

## Assets and credentials

The box-drop scene and its procedural checker texture were authored during the hackathon; they use no downloaded image or robot assets. Use self-rendered or explicitly licensed assets and record each source and license here when added. Do not use real home photos or personal information.

Keep API keys and tokens in environment variables. Do not include them in source files, prompts, logs, or commits. Local environment files, caches, model weights, and run artifacts are ignored by default.

## Validation and submission

Report only measured results. Record seeds and configurations for experiments, distinguish simulated outcomes from real-world claims, and cache model responses without retaining credentials.

Submission requirements include a public repository, a demo video of at most three minutes, clear identification of new hackathon work, verified asset licenses, and meaningful Meta model usage.
