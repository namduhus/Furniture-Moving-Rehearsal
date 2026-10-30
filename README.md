# Furniture-Moving Rehearsal

A simulated humanoid physically rehearses moving furniture through hallways and corners before a person attempts the move. Built for the **Push the Frontier** theme at the Meta Global AI Developer Hackathon.

The intended pipeline combines Meta models for scene interpretation and failure diagnosis with a physics simulation that checks proposed moving strategies. The target is a **VLM-guided humanoid controller**.

## Documentation

See [AGENTS.md](AGENTS.md) for AI collaboration and development guidelines.

## Local checks

Python is pinned to 3.12. The root environment contains NumPy 2.5.4 for controller math and offline tests; `uv.lock` retains the existing package version and distribution hashes. The Modal orchestrator and remote Isaac runtime have separate environments.

```sh
uv sync --locked --no-build
uv run --locked python -m unittest discover -s tests -v
```

These checks cover numerical conventions, runtime guards, control flow, recording and observation. Hardware-dependent Isaac execution is validated separately on the selected Modal runtime. SDK and codec checks require their respective existing environments/tools.

## Isaac deployment target

The selected stack is Isaac Sim / Isaac Lab with RSL-RL PPO and PyTorch, running on **Modal L40S**. Glimmer inference also targets Modal. Select compatible releases against the actual Modal worker and validate headless graphics and Isaac startup before training. A working CUDA probe alone does not validate Isaac graphics. Use the Isaac smoke check before training.

## Isaac runtime profile and G1 smoke check

The prepared runtime follows the official [Isaac Lab v3.0.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0): **Isaac Sim 6.2.0.0, Python 3.12, PyTorch 2.11.0 CUDA 13.0, Torchvision 0.26.0, RSL-RL 5.5.1 and Warp 1.17.0**. The source checkout is pinned to commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. The image verifies its upstream `uv.lock` SHA-256 and uses `uv sync --frozen --extra isaacsim --extra rsl-rl`. Dependencies are installed during the image build, before the interactive GPU session starts. No GPU is requested for image build steps. The upstream dependency-install command is limited to 30 minutes; build, provisioning and first-launch timings have not been measured here.

This is a source-checkout installation using the release's complete dependency graph and overrides. Its internal `isaaclab` workspace distribution reports version **28.0.0**, distinct from the public release tag **v3.0.0**; the preflight checks both source identity and distribution versions. Do not combine a different package-index wheel with this checkout. See the [versioned installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html).

Image-build caches stay under `/opt/fmr-build-cache`. The `/workspace` mountpoint must remain empty in the image; a final build check rejects existing image files there without deleting them. Only the running Function and SSH sessions use `/workspace/.cache`, after the persistent Volume has been mounted. Keep build-time cache paths separate when editing this image. Build-check commands occupy one physical Dockerfile `RUN` line; shell quoting alone does not make embedded newlines valid Dockerfile syntax.

The Isaac image includes Ubuntu 24.04's `libxt6t64` for the `libXt.so.6` dependency used by MaterialX plugins. It is installed in a separate layer after the pinned upstream environment so that this OS-library change can reuse the preceding installation cache. The final CPU build check loads `libXt.so.6` and checks the empty Volume mountpoint; plugin and renderer behavior must still be checked on the GPU worker. An already running session does not receive image changes; use the updated image on the next explicit start.

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

Each stage runs in a separate subprocess with a ten-minute timeout covering startup, execution and shutdown; a timeout terminates its process group. Results go to this checkout's `outputs/isaac/<run-id>/`, regardless of the shell's current working directory. With a checkout at `/workspace/repo/Furniture-Moving-Rehearsal`, the full path is `/workspace/repo/Furniture-Moving-Rehearsal/outputs/isaac/<run-id>/`. Override it explicitly with `--output-root` when needed. Existing runs under `/workspace/outputs/isaac` are not moved or deleted. Artifacts include `summary.json`, configuration/version records, GPU/Vulkan logs, per-stage stdout/error logs, `trajectory.jsonl`, `initial.png` and `final.png`. Only `passed_g1_smoke` plus inspection of the artifacts establishes this preflight's success. Empty/partial evidence or a failed shutdown cannot pass. Use `fmr-save` and `fmr-stop` after the check. Review each run's logs and images before proceeding to locomotion or manipulation; smoke-check success does not establish task performance.

### Inspect the G1 manipulation asset

The separate [asset profile](configs/isaac_g1_asset.json) selects Isaac Lab's existing `G1_29DOF_CFG` as a candidate for pretrained whole-body control. It expects 29 body joints and 14 three-finger-hand joints, matching the joint names in the [pinned GR00T-WBC robot description](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf). The profile references the controller's 15-action / 516-observation contract; it does not download or run controller weights. A matching name or joint count does not establish matching axes, dynamics or policy compatibility.

Inspect the plan locally using Python 3.12:

```sh
uv run --locked python scripts/isaac_g1_asset.py --dry-run
```

After pulling the code in an existing Isaac session, run from the remote project root:

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_asset.py
fmr-save
```

This uses the existing runtime image and starts no additional Modal session. Isaac may fetch its referenced USD assets on first use. The command checks the existing pinned runtime, CUDA and Vulkan, initializes one candidate robot in PhysX, and records its actual joint names/order, limits, default positions, actuator settings, root-pose configuration and floating-base status. It requests no control rollout, policy inference or training. Each subprocess has a ten-minute limit; the existing Modal session deadline and compute charges still apply.

Artifacts appear under `outputs/isaac/<run-id>-asset/`. Inspect `asset-inventory.json`, `asset-worker.json`, stage logs and `summary.json`. A fixed base, disabled gravity, different joint set, invalid limits or non-CUDA state fails the check; no alternative asset is silently substituted. `passed_asset_inventory` means structural inspection only. Balance, walking, contact behavior and pretrained-controller execution require separate validation. Use `fmr-stop` when finished; command completion does not release the GPU.

### Pretrained G1 standing trial

The separate [standing profile](configs/isaac_g1_balance.json) transfers NVIDIA's pretrained **GR00T-WBC balance** policy to the inspected G1 asset. This is an experimental, stand-only controller check: navigation commands are zero, and the script performs no training. It uses the existing runtime's locked `onnxruntime==1.26.0` CPU dependency; no additional package or image change is required. A missing or different dependency fails explicitly.

The [controller contract](configs/g1_wbc_reference.json) pins the source revision, model size and SHA-256. The first `--prepare-model` run downloads one 1,886,682-byte checkpoint into ignored `models/g1_wbc/` and retains its license. Subsequent runs verify the cache. Model weights use the [NVIDIA Open Model License](licenses/GR00T-WBC.txt): **Licensed by NVIDIA Corporation under the NVIDIA Open Model License.** The [reference URDF](assets/references/g1_wbc.urdf) is used for numerical kinematic checks; the simulation continues to load Isaac's existing G1 USD. Source and Unitree-derived asset notices are retained in [licenses/](licenses/).

Inspect the plan on the Mac without downloads or a worker:

```sh
uv run --locked python scripts/isaac_g1_wbc.py --stage balance --prepare-model --dry-run
```

After updating the remote project checkout, run in the existing Isaac session:

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --stage balance --prepare-model
fmr-save
```

The command first checks checkpoint integrity and its actual 516-input / 15-output ONNX interface, then CUDA/Vulkan, then initializes one G1 for the bounded standing trial. Use `--stage policy --prepare-model` for checkpoint inference only. This does not start Isaac or a new Modal session, but an existing GPU session remains billable.

Before applying actions, the standing worker compares measured link poses against URDF forward kinematics, including two static joint-position probes. It rejects incompatible geometry/axes rather than guessing a root rotation. Isaac 3 pose buffers and root-pose writers use **xyzw**; configuration rotations use **wxyz**. The worker establishes and verifies an explicit upright root pose, joint positions and zero velocities. Lower-body PD gains come from the reviewed controller configuration, with explicit PD actuators and URDF effort/velocity bounds. Upper-body and finger holding gains are project experiment parameters, identified separately in the contract; matched names and gains do not prove equal dynamics.

The trial requests **250 control steps at 50 Hz**, or five simulated seconds, with physics at 200 Hz. It stops on non-finite state, out-of-bound policy actions, root height below 0.50 m or tilt above 0.6 rad. It performs no automatic episode reset or retry. The policy and GPU subprocesses have 60/120-second limits, and the standing subprocess has a ten-minute limit covering startup and shutdown. These limits do not extend the Modal session. The existing resource/cost plan applies; stop the session with `fmr-stop` when finished.

Outputs go to `outputs/isaac/<run-id>-wbc/`: source/runtime contracts, stage/error logs, actuator settings, frame-calibration/probe inputs, explicit initialization evidence, a per-step trajectory, initial/final PNGs and an MP4. Successful full recording contains 125 frames at 25 FPS. Static frames are allowed for this standing video because a steady stance can be visually static; physical advancement is checked through the complete timed trajectory. Existing motion-smoke video checks still require changing frames by default. Failed trials preserve partial evidence and videos when enough frames exist.

Only `passed_balance_rollout` with reviewed trajectory/video establishes this standing trial's success. `passed_policy_contract` verifies checkpoint execution only. Neither status establishes walking, furniture manipulation or VLA capability. Inspect errors and frame/initialization evidence before changing transfer settings.

### Record a G1 motion check

The separate [recording profile](configs/isaac_record.json) uses one environment, seed 0, zero actions, a 640 x 360 camera and 500 environment steps. It captures every two steps. At this task's 0.02 s control step, that produces 250 frames at 25 FPS covering 10 simulated seconds. Video timing comes from the actual environment step size, not wall-clock execution speed. This is a recording/runtime check, not a trained walking policy.

The updated Isaac image installs Ubuntu's official `ffmpeg` package after the upstream Python environment and checks CPU H.264 encoding during build. The recording profile requires FFmpeg with the `libx264` encoder on the remote runtime's PATH; the worker checks this before initializing Isaac. Existing running sessions do not receive image changes. Start the updated image for this profile. Inspect the plan on the Mac without starting a worker:

```sh
python3 scripts/isaac_preflight.py --config configs/isaac_record.json --dry-run
```

After starting the updated Isaac profile and pulling the code, run in the remote project root:

```sh
cd /workspace/repo/Furniture-Moving-Rehearsal
/opt/isaaclab/.venv/bin/python scripts/isaac_preflight.py --config configs/isaac_record.json --stage all
fmr-save
```

Successful recording returns `passed_g1_recording`. The run directory additionally contains `frames/`, `rollout.mp4`, `video.json`, `video-frames.md5`, FFmpeg version/command records and encode/decode logs. Captions identify zero-action control, step, simulated time and resets. The trajectory maps each captured step to its zero-based video frame and playback timestamp. A reset marks an episode boundary; the video does not imply uninterrupted balance. The first captured frame is after step 2 (0.04 s), at playback time 0; the last frame is after step 500. Playback spans 10 seconds at the configured capture cadence.

The encoder produces H.264/yuv420p MP4 with fast-start metadata using two CPU threads. It decodes every frame to verify dimensions, frame count, playback timestamps and image changes. Raw camera images must change before captions are added. Partial/failed recordings do not receive a passed status. Original PNGs are retained for diagnosis; recording requires extra rendering, encoding and storage compared with the smoke check.

From the local project root on the Mac, replace `RUN_ID` with the reported run directory name and copy the result while the SSH session is active:

```sh
mkdir -p outputs/isaac
scp -F outputs/modal/ssh_config -r fmr-modal:/workspace/repo/Furniture-Moving-Rehearsal/outputs/isaac/RUN_ID outputs/isaac/
open outputs/isaac/RUN_ID/rollout.mp4
```

Save the remote workspace before stopping it. The existing 30-minute Isaac resource profile and cost estimate apply; the 10-second video duration is simulated time and does not set the GPU session length. A live browser camera viewer is a later step.

### Viewing remote simulation from a Mac

The selected development workflow is headless Isaac execution on Modal with motion observation from the Mac. Scene, physical-property and camera changes belong in reproducible code/configuration. The smoke check exports two PNGs and a trajectory; use the separate recording profile above for continuous MP4 output. Copy artifacts over SSH for local inspection and keep training independent of viewer connectivity. Live browser camera delivery remains a separate feature.

The [observation profile](configs/isaac_observe.json) adds an optional read-only browser camera viewer. It publishes the latest JPEG and matching step/time/reset metadata, then serves HTTP/MJPEG on remote loopback only through the existing authenticated SSH/TCP connection. Python stdlib and the runtime's existing Pillow are used; no extra frontend framework or Python dependency is added. This is a separate implementation from NVIDIA's [WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/manual_livestream_clients.html), which requires a UDP media path that the current [Modal TCP tunnel](https://modal.com/docs/guide/tunnels) does not provide. Full remote editor access is outside the current development workflow.

After starting the existing Isaac session and pulling the updated code, open a separate **Mac terminal** for the viewing tunnel and leave it running:

```sh
ssh -N -L 8765:127.0.0.1:8765 fmr-modal
```

In the **remote project terminal**, start observation:

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_preflight.py --config configs/isaac_observe.json --stage all
```

Once the command prints `Observer: http://127.0.0.1:8765`, open [the local viewer](http://127.0.0.1:8765) on the Mac. The server starts before GPU/Isaac checks; the page initially shows initialization and then camera frames. It shows simulated time, reset count, frame freshness and zero-action control explicitly. Display publication is capped at 10 wall-clock FPS; actual delivery/latency must be measured on the user's connection. Optional wall-clock pacing is enabled only in this observation profile to make the short rollout inspectable; the smoke/recording profiles are not paced and a paced observation is not a throughput benchmark.

The run still performs 500 zero-action steps and verifies its MP4; it does not train a policy. After the final result, the viewer retains the last frame for 30 seconds and then closes. Browser closure, tunnel closure, run completion and Modal GPU termination are separate events: the simulation does not depend on a viewer, and a finished run does not release the GPU session. Keep the command/launcher terminals alive during execution; use `fmr-save` and `fmr-stop` when work is finished. `fmr-save` syncs files and does not create a training checkpoint. A future training command will have its own step/time/checkpoint limits in addition to the Modal session deadline.

Observation writes `observation.json` (one atomic JPEG/state snapshot), phase/error records and observation counts within the run directory. A slow/disconnected browser receives the latest available frame on reconnect without queuing every old frame. Preview encoding errors are reported separately and do not turn a valid physics/recording result into a preview success claim. To inspect the retained final snapshot later, serve that run explicitly with `python scripts/isaac_observe.py --run-dir outputs/isaac/RUN_ID --seconds 600`; this views saved state and does not restart simulation.

The product interface is an interactive WebGL room: users select furniture and its destination, then observe robot and furniture motion from simulation states and receive explanations. The developer camera viewer does not replace this WebGL interaction.

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

On a local or remote failure the launcher also saves `error.log` and, when an App ID is available, prints a read-only `modal app logs` command. An empty `RemoteError` alone does not identify the underlying failure: use the runner's App logs, which can report container setup errors before any SSH event is emitted.

Run the offline checks with:

```sh
uv run --project environments/modal --locked python -m unittest discover -s tests -v
```

## Dependencies and licenses

The local numerical checks use [NumPy](https://numpy.org/) with its BSD-3-Clause, 0BSD, MIT, Zlib and CC0-1.0 bundled notices. Retain the installed distribution's license files when redistributing its binaries. The existing package version and wheel hashes are recorded in the root `uv.lock`.

The SSH development workflow uses the official [Modal SDK](https://github.com/modal-labs/modal-client) (Apache-2.0), [uv](https://github.com/astral-sh/uv) (MIT or Apache-2.0), and [Ubuntu's official image](https://hub.docker.com/_/ubuntu). [OpenSSH](https://github.com/openssh/openssh-portable/blob/master/LICENCE) has permissive component licenses; [Git](https://github.com/git/git/blob/master/COPYING) uses GPL-2.0. Retain package-specific copyright/license notices under `/usr/share/doc` and the Python distributions' bundled notices. The SDK's transitive Python dependencies and distribution hashes are recorded in `environments/modal/uv.lock`; this attribution makes no claim of original infrastructure or SSH implementation.

The optional Isaac profile uses the official [Isaac Lab v3.0.0 source and BSD-3-Clause core license](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0), [RSL-RL 5.5.1](https://pypi.org/project/rsl-rl-lib/5.5.1/) (BSD-3-Clause), [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) / Torchvision (BSD-style notices), and NVIDIA Warp (Apache-2.0 and bundled notices). Isaac Sim/Kit and their dependencies remain subject to their [applicable NVIDIA licenses](https://docs.isaacsim.omniverse.nvidia.com/latest/common/legal.html), including the Omniverse EULA. The upstream G1 USD assets are fetched from NVIDIA's asset service at runtime; retain the [additional software/materials terms](https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-isaac-sim-additional.html) and asset-specific notices. No G1 USD asset is authored or bundled by this repository.

The Isaac image's [Ubuntu libxt6t64 package](https://packages.ubuntu.com/noble/libxt6t64) comes from the distribution's official repositories and the X.Org libXt source. Its [upstream release](https://www.x.org/releases/individual/lib/libXt-1.2.1.tar.gz) includes MIT/X11-style permission notices in `COPYING`; retain the installed package's `/usr/share/doc/libxt6t64/copyright` and dependency notices. OS package versions resolve at image build time, separately from the pinned Python lock.

MP4 recording invokes [FFmpeg](https://ffmpeg.org/) with [libx264](https://www.videolan.org/developers/x264.html). FFmpeg's [license depends on the configured components](https://github.com/FFmpeg/FFmpeg/blob/n6.1.1/LICENSE.md), including GPL terms for libx264-enabled builds. Retain the selected distribution's `/usr/share/doc/ffmpeg/copyright` and linked-library notices. This project invokes the encoder as a separate process and does not claim authorship of the codec. Each recording stores the executable's version/build configuration; its resolved system packages are separate from the upstream Python lock.

## Original contributions and third-party components

Keep this table current in the same commit as each new module or third-party addition. It identifies the origin and purpose of repository components.

| Component | Origin | Purpose | Source / license |
| --- | --- | --- | --- |
| Collaboration instructions | Project planning | Agent guidance | [AGENTS.md](AGENTS.md) |
| README and ignore rules | Created during the hackathon | Usage documentation and local artifact exclusions | [README](README.md), [.gitignore](.gitignore) |
| NumPy | Third-party software | Local controller mathematics and offline numerical checks | [Official source](https://github.com/numpy/numpy), bundled licenses linked above; version and hashes in [uv.lock](uv.lock) |
| Modal SSH launcher and offline checks | Created during the hackathon | Bounded development sessions, public-key validation, pinned host-key configuration, persistent workspace, save/stop helpers and local duplicate-session guard | [Launcher](scripts/modal_dev.py), [configuration](configs/modal_dev.json), [environment](environments/modal/pyproject.toml), [checks](tests/test_modal_dev.py) |
| Isaac runtime profile, G1 preflight and recording | Created during the hackathon | Pinned source/lock verification, OS shared-library/build-cache guards, CUDA/graphics guards, bounded PhysX execution, frame/state timing, MP4 verification and shutdown checks | [Profile](configs/modal_isaac.json), [smoke configuration](configs/isaac_smoke.json), [recording configuration](configs/isaac_record.json), [preflight](scripts/isaac_preflight.py), [video verification](scripts/isaac_video.py), [preflight checks](tests/test_isaac_preflight.py), [video checks](tests/test_isaac_video.py) |
| Browser motion observer | Created during the hackathon | Latest-frame/state publication, read-only loopback MJPEG viewer, lifecycle/freshness labels, reconnect handling and observation-only pacing | [Observation profile](configs/isaac_observe.json), [viewer](scripts/isaac_observe.py), [HTTP/state checks](tests/test_isaac_observe.py) |
| G1 manipulation-asset inventory | Created during the hackathon | Separate floating-base candidate profile, explicit joint-order mapping, finite state/limit guards, bounded inventory subprocess and evidence validation | [Profile](configs/isaac_g1_asset.json), [script](scripts/isaac_g1_asset.py), [checks](tests/test_isaac_g1_asset.py) |
| Pretrained G1 controller adapter and bounded standing trial | Created during the hackathon | Explicit quaternion/frame conversion, URDF kinematic probes, initialization, observation history, joint/actuator mapping, transfer guards and recorded evaluation | [Adapter](scripts/g1_wbc.py), [runner](scripts/isaac_g1_wbc.py), [profile](configs/isaac_g1_balance.json), [contract](configs/g1_wbc_reference.json), [numerical and flow checks](tests/test_g1_wbc.py) |
| GR00T-WBC balance checkpoint and G1 reference kinematics | Third-party model and robot data | Downloaded pretrained balance inference and reference-only frame checks; no original robot or pretrained-policy claim | [Pinned NVIDIA source](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc), [reference URDF](assets/references/g1_wbc.urdf), [NVIDIA code/model terms](licenses/GR00T-WBC.txt), [Unitree BSD notice](licenses/UNITREE-BSD-3-CLAUSE.txt); weights remain Git-ignored |
| FFmpeg and libx264 | Third-party software | MP4 encoding and decoded-frame verification for self-rendered rollouts | Official upstream sources and component license notices linked above |
| Isaac Sim, Isaac Lab, RSL-RL, PyTorch and G1 USD assets | Third-party software and assets | Simulation, robot task/assets, PPO learner and tensor operations; no original algorithm or robot-asset claim | Official releases, asset and license references linked above; resolved runtime graph comes from the pinned upstream lock |
| Modal SDK, Ubuntu, X.Org libXt, OpenSSH, Git and uv | Third-party software and services | Cloud execution, base OS/shared libraries, SSH access, version control and Python environment management | Official sources and license notices linked above |

## Assets and credentials

The G1 reference URDF is imported robot data used for kinematic checks; its original bytes and license notices are retained. Simulation uses the separately licensed Isaac USD asset. Use self-rendered or explicitly licensed images and record each source and license here when added. Do not use real home photos or personal information.

Keep API keys and tokens in environment variables. Do not include them in source files, prompts, logs, or commits. Local environment files, caches, model weights, and run artifacts are ignored by default.

## Validation and submission

Report only measured results. Record seeds and configurations for experiments, distinguish simulated outcomes from real-world claims, and cache model responses without retaining credentials.

Submission requirements include a public repository, a demo video of at most three minutes, clear identification of new hackathon work, verified asset licenses, and meaningful Meta model usage.
