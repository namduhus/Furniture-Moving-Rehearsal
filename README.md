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

## Isaac deployment target

The selected stack is Isaac Sim / Isaac Lab with RSL-RL PPO and PyTorch, running on **Modal L40S**. Glimmer inference also targets Modal. Select compatible releases against the actual Modal worker and validate headless graphics and Isaac startup before training. A working CUDA probe alone does not validate Isaac graphics. The MuJoCo and JAX/Brax workflows below remain reference tools; the Isaac runtime has not yet been validated.

## Isaac runtime profile and G1 smoke check

The prepared runtime follows the official [Isaac Lab v3.0.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0): **Isaac Sim 6.2.0.0, Python 3.12, PyTorch 2.11.0 CUDA 13.0, Torchvision 0.26.0, RSL-RL 5.5.1 and Warp 1.17.0**. The source checkout is pinned to commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. The image verifies its upstream `uv.lock` SHA-256 and uses `uv sync --frozen --extra isaacsim --extra rsl-rl`. Dependencies are installed during the image build, before the interactive GPU session starts. No GPU is requested for image build steps. The upstream dependency-install command is limited to 30 minutes; build, provisioning and first-launch timings have not been measured here.

This is a source-checkout installation using the release's complete dependency graph and overrides. Its internal `isaaclab` workspace distribution reports version **28.0.0**, distinct from the public release tag **v3.0.0**; the preflight checks both source identity and distribution versions. Do not combine a different package-index wheel with this checkout. See the [versioned installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html).

Inspect both plans on your Mac without GPU calls:

```sh
python3 scripts/modal_dev.py plan --config configs/modal_isaac.json
python3 scripts/isaac_preflight.py --dry-run
```

The Isaac profile requests **one L40S, four physical CPU cores and 32 GiB RAM**, with requests equal to limits, for a maximum 30-minute session including SSH setup. At the reviewed prices this is approximately **$1.20** for 30 minutes of compute, or an illustrative **$1.36** including startup/cleanup/scale-down allowances. Image builds, storage, network and other jobs are additional; neither figure is a hard account cap or a measured training requirement.

After reviewing NVIDIA's [Omniverse EULA](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html) and the cost plan, start this specific profile from the local repository root if you accept the terms:

```sh
uv run --project environments/modal --locked python scripts/modal_dev.py start --config configs/modal_isaac.json --accept-isaac-eula
```

The explicit flag enables NVIDIA's noninteractive EULA setting. No optional telemetry-consent flag is set. The plain `start` command still selects the lightweight SSH-only profile. Both profiles reuse the existing workspace Volume and generated `fmr-modal` SSH alias; run only one session at a time.

Connect through VS Code and clone this project under `/workspace/repo`, or run `git pull --ff-only` in an existing checkout after pushing your local changes. In the remote **project root**, execute:

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_preflight.py --stage all
```

The image contains the upstream Isaac Lab runtime under `/opt/isaaclab`; your project checkout contains the editable rehearsal and verification code. Editing project scripts and pulling them updates the next script run. Changing installed dependencies requires updating and rebuilding the runtime image. The SSH shell activates the runtime interpreter; its absolute path above also works from noninteractive SSH commands.

The smoke check first verifies the pinned package/source versions, driver requirement, synchronized CUDA computation and an NVIDIA L40S Vulkan device with RTX extensions. `NVIDIA_DRIVER_CAPABILITIES=all` requests graphics exposure; it cannot make an unsupported Modal worker provide Vulkan. Software Vulkan rendering is rejected, and the script does not install drivers, disable compatibility checks or silently switch physics backends.

The second stage explicitly selects **`physics=isaacsim_physx`**, loads **`Isaac-Velocity-Flat-G1`** (the v3 task ID has no `-v0` suffix), and runs one environment for 64 zero-action steps with seed 0. It checks finite observations/rewards/joint and root states, CUDA joint-state storage, the actual PhysX manager and two non-uniform RTX camera frames. Isaac Lab 3.0 installs Newton as an upstream dependency, but this check requires the Isaac Sim PhysX backend. It does not demonstrate trained walking, pushing or furniture movement.

Each stage runs in a separate subprocess with a ten-minute timeout covering startup, execution and shutdown; a timeout terminates its process group. Results go to `/workspace/outputs/isaac/<run-id>/`, including `summary.json`, configuration/version records, GPU/Vulkan logs, per-stage stdout/error logs, `trajectory.jsonl`, `initial.png` and `final.png`. Only `passed_g1_smoke` plus inspection of the artifacts establishes this preflight's success. Empty/partial evidence or a failed shutdown cannot pass. Use `fmr-save` and `fmr-stop` after the check. The image build and this complete check remain unverified on Modal until a real run is reviewed.

## Modal SSH development

The initial developer image uses Ubuntu 24.04, Python 3.12, OpenSSH, Git and uv 0.11.7. It provides an L40S SSH session for VS Code Remote SSH; **it does not install or validate Isaac, PyTorch or Glimmer**. The local orchestrator is isolated in `environments/modal`, with Modal 1.6.3 and a separate dependency lock. The image's OS packages resolve at build time; their installed versions and the worker's GPU/driver inventory are recorded when the server starts. This is not a fully immutable Isaac image.

Inspect the resource and cost plan without installing packages or contacting Modal:

```sh
python3 scripts/modal_dev.py plan
```

The default configuration requests one L40S, two physical CPU cores and 8 GiB host RAM, with CPU/RAM limits equal to their requests. The SSH session lasts at most 600 seconds from function entry, including its setup. The function timeout is 660 seconds, with a separate 90-second startup timeout. Based on the prices reviewed on October 29, 2026, 600 seconds of requested compute is about **$0.35**; allowing for startup, cleanup and scale-down gives an illustrative **$0.44**. Image builds, storage, network and other workloads are additional. These estimates are not an account spending cap. See [Modal pricing](https://modal.com/pricing).

Modal requires a valid payment method for GPU functions, including when promotional credits are available. In the workspace used by your CLI profile, check **Settings → Usage & Billing → Manage payment details**, the credit balance and the usage/spend limits before starting. See [Modal's GPU payment requirement](https://modal.com/docs/guide) and [billing instructions](https://modal.com/docs/guide/billing). The error `Please add a payment method to use L40S GPU functions` is an account prerequisite failure during Function creation; adding or changing Python dependencies will not resolve it.

After reviewing that plan, authenticating Modal and configuring the workspace payment method, run from the repository root:

```sh
uv sync --project environments/modal --locked --no-build
uv run --project environments/modal --locked python scripts/modal_dev.py start --public-key ~/.ssh/id_ed25519.pub
```

The launcher reads only the public key, strips its comment and sends it through the SDK. It never reads or uploads the matching local private key; `--identity-file` can select a different local identity path. If you have no SSH key, create one yourself with `ssh-keygen -t ed25519` and retain the private key on your Mac. No repository contents or local credential files are automatically uploaded. Clone or copy the project separately after connecting; authentication to a private Git repository is a separate step.

When the server reports ready:

1. Set VS Code's `remote.SSH.configFile` to the absolute path of `outputs/modal/ssh_config` printed by the launcher. This generated file is updated on each launch; no file under your home directory is modified.
2. Use **Remote-SSH: Connect to Host**, select **fmr-modal**, and open `/workspace/repo`.
3. In the remote terminal, run `nvidia-smi` and `uv --version`. Create `echo persistence-check > /workspace/repo/persistence-check.txt`, then run `fmr-save` and `fmr-stop`.
4. Launch a second session and check that file with `cat /workspace/repo/persistence-check.txt`. The session metadata also records the previous session marker. VS Code connectivity and storage durability are only verified after these checks succeed.

The launcher also prints a direct `ssh -F ... fmr-modal` command. It obtains the server's public host key through Modal and writes a dedicated `known_hosts` file with strict checking enabled. SSH uses a temporary TCP tunnel, so the hostname/port can change on each launch. The underlying [Modal forwarding API](https://modal.com/docs/sdk/py/latest/forward) is experimental.

To show Modal alongside existing hosts in your normal `~/.ssh/config`, add an `Include` directive at the **top of that file, before any Host or Match block**, using the absolute path of the generated `outputs/modal/ssh_config`. Keep VS Code configured to use your normal SSH config in this arrangement. The `fmr-modal` alias will resolve through the generated file, which is refreshed on each launch. After setup, `ssh fmr-modal` and `ssh fmr-modal /usr/local/bin/fmr-stop` work from any directory while the session is running. See [OpenSSH Include](https://man.openbsd.org/ssh_config#Include). The launcher never edits your home SSH config.

Keep the launcher terminal open during this initial session. Closing VS Code does not stop GPU allocation. In the remote terminal, **`fmr-save`** commits `/workspace`; **`fmr-stop`** commits it and requests shutdown. Stop writers before saving a consistent checkpoint. An emergency `modal app stop` command with the session's App ID is printed at startup; forced termination can lose recent uncommitted writes. The launcher additionally requests a commit every 30 seconds and during normal shutdown. Preemption or forced termination can bypass application cleanup.

The persistent v2 Volume is named `fmr-dev-workspace-v2`. Files under `/workspace`, including the VS Code server installation, survive across committed sessions; installs elsewhere in the container are temporary. Configure reproducible packages in the image or keep project virtual environments under `/workspace`. The 8 GiB allocation is for the SSH check, not an Isaac training configuration. Keep Isaac and Glimmer sizing separate until measured.

Each launch writes `summary.json`, `events.jsonl`, `ssh_config` and `known_hosts` under a unique `outputs/modal/` directory. Server logs live in `/workspace/.fmr/`. A local file lock prevents duplicate launchers from the same checkout, and each App allows one container. This is not a workspace-wide or cross-machine concurrency lock; do not launch competing sessions against the same Volume. No training, model download, persistent deployment or automatic retry loop is started by this launcher.

Run the offline checks with:

```sh
uv run --project environments/modal --locked python -m unittest discover -s tests -v
```

## Reference GPU JAX/Brax workflow

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

For a 10-million-step continuation experiment, use `--stage train --config configs/g1_continue_10m.json --checkpoint-dir <checkpoint-parent>`. The checkpoint parent must contain the numeric step directories; the latest is selected by the upstream loader. Restoring a checkpoint warm-starts the policy, critic, and observation-normalization statistics with a fresh optimizer and step counter. The requested step budget applies to the new run. This configuration requires a checkpoint, retains the pilot's PPO and environment settings, evaluates about every million steps, and enables TensorBoard event logging through the upstream TensorBoardX dependency.

## Dependencies and licenses

Google DeepMind maintains the canonical [`mujoco` PyPI package](https://pypi.org/project/mujoco/). Its engine and Python bindings use [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/main/LICENSE); prebuilt wheels also include third-party notices documented on the package page. Dependency versions and wheel SHA-256 hashes are recorded in `uv.lock`, using the official PyPI registry.

Transitive package licenses: absl-py and etils (Apache-2.0), fsspec and websockets (BSD-3-Clause), glfw and zipp (MIT), PyOpenGL (BSD), typing-extensions (PSF-2.0), and NumPy (BSD-3-Clause, 0BSD, MIT, Zlib, and CC0-1.0). Retain the installed distributions' bundled license notices when redistributing their binaries.

The training project pins JAX 0.9.2, Brax 0.14.2, MuJoCo/MJX 3.17.0, and MuJoCo Playground 0.2.0 from official PyPI distributions. Brax 0.14.2 uses a replication API removed in JAX 0.10; retain the compatible JAX pin and resolved Flax dependency in the training lock. [JAX](https://github.com/jax-ml/jax), [Brax](https://github.com/google/brax), [MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx), and [Playground](https://github.com/google-deepmind/mujoco_playground) use Apache-2.0. Use Brax's maintained training library with Playground environments. The [Menagerie G1 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) has a separate BSD-3-Clause license. CUDA/cuDNN wheels and the system `ffmpeg` distribution retain their own license notices and terms.

The SSH development workflow uses the official [Modal SDK](https://github.com/modal-labs/modal-client) (Apache-2.0), [uv](https://github.com/astral-sh/uv) (MIT or Apache-2.0), and [Ubuntu's official image](https://hub.docker.com/_/ubuntu). [OpenSSH](https://github.com/openssh/openssh-portable/blob/master/LICENCE) has permissive component licenses; [Git](https://github.com/git/git/blob/master/COPYING) uses GPL-2.0. Retain package-specific copyright/license notices under `/usr/share/doc` and the Python distributions' bundled notices. The SDK's transitive Python dependencies and distribution hashes are recorded in `environments/modal/uv.lock`; this attribution makes no claim of original infrastructure or SSH implementation.

The optional Isaac profile uses the official [Isaac Lab v3.0.0 source and BSD-3-Clause core license](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0), [RSL-RL 5.5.1](https://pypi.org/project/rsl-rl-lib/5.5.1/) (BSD-3-Clause), [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) / Torchvision (BSD-style notices), and NVIDIA Warp (Apache-2.0 and bundled notices). Isaac Sim/Kit and their dependencies remain subject to their [applicable NVIDIA licenses](https://docs.isaacsim.omniverse.nvidia.com/latest/common/legal.html), including the Omniverse EULA. The upstream G1 USD assets are fetched from NVIDIA's asset service at runtime; retain the [additional software/materials terms](https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-isaac-sim-additional.html) and asset-specific notices. The earlier Menagerie asset license does not establish the license of these separate USD assets. No G1 USD asset is authored or bundled by this repository.

## Original contributions and third-party components

Keep this table current in the same commit as each new module or third-party addition. It identifies the origin and purpose of repository components.

| Component | Origin | Purpose | Source / license |
| --- | --- | --- | --- |
| Collaboration instructions | Project planning | Agent guidance | [AGENTS.md](AGENTS.md) |
| README and ignore rules | Created during the hackathon | Usage documentation and local artifact exclusions | [README](README.md), [.gitignore](.gitignore) |
| CPU smoke script, box-drop scene, and configuration | Created during the hackathon | Check scene loading, CPU physics, contact, and rendering | [Script](scripts/smoke_mujoco.py), [scene](assets/scenes/smoke.xml), [configuration](configs/smoke.json) |
| MuJoCo 3.17.0 and its Python dependencies | Third-party software | Physics engine, Python bindings, and rendering support | [Official package](https://pypi.org/project/mujoco/), [Apache-2.0 and bundled notices](https://pypi.org/project/mujoco/#license-and-disclaimer), [lock file](uv.lock) |
| G1 preflight and PPO launcher | Created during the hackathon | GPU and learner API checks, pilot and checkpoint continuation configuration, and local run logs | [Launcher](scripts/g1_baseline.py), [pilot](configs/g1_baseline.json), [continuation](configs/g1_continue_10m.json), [training environment](environments/gpu/pyproject.toml) |
| Modal SSH launcher and offline checks | Created during the hackathon | Bounded development sessions, public-key validation, pinned host-key configuration, persistent workspace, save/stop helpers and local duplicate-session guard | [Launcher](scripts/modal_dev.py), [configuration](configs/modal_dev.json), [environment](environments/modal/pyproject.toml), [checks](tests/test_modal_dev.py) |
| Isaac runtime profile and G1 preflight | Created during the hackathon | Pinned source/lock verification, CUDA/graphics guards, bounded PhysX G1 execution, finite-state/render evidence and shutdown checks | [Profile](configs/modal_isaac.json), [smoke configuration](configs/isaac_smoke.json), [preflight](scripts/isaac_preflight.py), [offline checks](tests/test_isaac_preflight.py) |
| Isaac Sim, Isaac Lab, RSL-RL, PyTorch and G1 USD assets | Third-party software and assets | Simulation, robot task/assets, PPO learner and tensor operations; no original algorithm or robot-asset claim | Official releases, asset and license references linked above; resolved runtime graph comes from the pinned upstream lock |
| Modal SDK, Ubuntu, OpenSSH, Git and uv | Third-party software and services | Cloud execution, base OS, SSH access, version control and Python environment management | Official sources and license notices linked above |
| JAX, Brax, MJX, and MuJoCo Playground | Third-party software | GPU computation, PPO training, and the G1 locomotion environment | Official sources and Apache-2.0 licenses linked above |
| Menagerie G1 robot assets | Third-party assets | Robot geometry, dynamics, and collision model fetched by Playground | [Source and BSD-3-Clause license](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) |

## Assets and credentials

The box-drop scene and its procedural checker texture were authored during the hackathon; they use no downloaded image or robot assets. Use self-rendered or explicitly licensed assets and record each source and license here when added. Do not use real home photos or personal information.

Keep API keys and tokens in environment variables. Do not include them in source files, prompts, logs, or commits. Local environment files, caches, model weights, and run artifacts are ignored by default.

## Validation and submission

Report only measured results. Record seeds and configurations for experiments, distinguish simulated outcomes from real-world claims, and cache model responses without retaining credentials.

Submission requirements include a public repository, a demo video of at most three minutes, clear identification of new hackathon work, verified asset licenses, and meaningful Meta model usage.
