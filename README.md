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

The Isaac profile requests **one L40S, four physical CPU cores and 32 GiB RAM**, with requests equal to limits, for a maximum 60-minute session including SSH setup, long enough for the corridor carry training (~43 min). At the reviewed prices this is approximately **$2.40** for 60 minutes of compute, or an illustrative **$2.56** including startup/cleanup/scale-down allowances. Image builds, storage, network and other jobs are additional; neither figure is a hard account cap or a measured training requirement.

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

The [controller contract](configs/g1_wbc_reference.json) pins the source revision, model size and SHA-256. The first `--prepare-model` run downloads one 1,886,682-byte checkpoint into ignored `models/g1_wbc/` and retains its license. Subsequent runs verify the cache. Model weights use the [NVIDIA Open Model License](licenses/GR00T-WBC.txt): **Licensed by NVIDIA Corporation under the NVIDIA Open Model License.** The [reference URDF](assets/references/g1_wbc.urdf) is the unmodified upstream [G1 rev 1.0 model](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf). Git attributes preserve its original line endings for the SHA-256 check. It supplies reference kinematics consistent with the common body frames in the published [controller deployment model](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.xml); it does not replace Isaac's existing G1 USD or establish identical dynamics. Source and Unitree-derived asset notices are retained in [licenses/](licenses/).

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

Before applying actions, the standing worker compares measured joint pivots and body orientations against URDF forward kinematics, including two static joint-position probes. A USD rigid-body origin need not coincide with the URDF child-link origin at the joint: the worker reads the authored `physics:localPos1` and joint-axis rotation from the [USD joint schema](https://openusd.org/release/api/class_usd_physics_joint.html), verifies all 43 parent/child bindings and axes, and converts measured body poses into joint-pivot poses. It does not fit per-link translations to make a mismatch pass. Root-frame rotation comes from corresponding orientations; the position tolerance remains 5 mm. Parent validation recomputes the checks from saved raw poses and authored metadata. It rejects incompatible pivots/axes rather than changing tolerances. Isaac 3 pose buffers and root-pose writers use **xyzw**; configuration rotations use **wxyz**. The worker establishes and verifies an explicit upright root pose, joint positions and zero velocities. Lower-body PD gains come from the reviewed controller configuration. Motor effort/velocity bounds are separate experiment settings: the contract explicitly preserves the trial's bounds where they differ from the kinematic URDF revision, and records their origin in `actuators.json`. Upper-body and finger holding gains are also project experiment parameters; matched names and gains do not prove equal dynamics.

The 15 lower-body joints use explicit PD with the published deployment model's armature of 0.01 kg m². The arms and hands use Isaac's implicit PD for posture holding, with its G1 configuration's armature of 0.001 kg m². Armature adds joint-space inertia; implicit holding avoids applying a sampled explicit damping controller to the small wrist/finger inertias. Holding gains, motor bounds and policy scales remain separately recorded experiment settings. The worker reads all 43 applied armature values before actions and the parent revalidates their evidence. This configuration still requires physical transfer validation on the selected worker.

The trial requests **250 control steps at 50 Hz**, or five simulated seconds, with physics at 200 Hz. It stops on non-finite state, out-of-bound policy actions, root height below 0.50 m or tilt above 0.6 rad. It performs no automatic episode reset or retry. The policy and GPU subprocesses have 60/120-second limits, and the standing subprocess has a ten-minute limit covering startup and shutdown. These limits do not extend the Modal session. The existing resource/cost plan applies; stop the session with `fmr-stop` when finished.

Outputs go to `outputs/isaac/<run-id>-wbc/`: source/runtime contracts, stage/error logs, actuator configuration and applied armature, `joint-frames.json`, raw frame-calibration/probe inputs, per-link `frame-comparison-*.json` reports, explicit initialization evidence, a per-step trajectory, initial/final PNGs and an MP4. Failures inside the simulation block are saved before Kit shutdown can terminate the process; the parent also reports a bounded log tail. A rejected policy target writes `rejected-action.json` with the complete 516-value observation, raw action, state, joint mapping and failing step before applying it; the raw-action bound remains 5. Successful full recording contains 125 frames at 25 FPS. Static frames are allowed for this standing video because a steady stance can be visually static; physical advancement is checked through the complete timed trajectory. Existing motion-smoke video checks still require changing frames by default. Failed trials preserve partial evidence and videos when enough frames exist.

A reviewed `passed_balance_rollout` means the bounded non-fall, action and recording checks completed; it does not assess stationary position holding. `passed_policy_contract` verifies checkpoint execution only. Neither status establishes walking, furniture manipulation or VLA capability. Inspect errors and frame/initialization evidence before changing transfer settings.

### Diagnose zero-command drift

The [diagnostic profile](configs/isaac_g1_diagnostics.json) extends the standing profile with instrumentation. The checkpoint, zero navigation command, gains, armatures, motor limits, time steps and recording remain the same. Contact reporting is enabled on the existing robot; no friction coefficients or controller parameters are tuned. This profile uses the selected Isaac Sim 6.2 [contact-sensor API](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py), including aggregate normal, friction and total forces.

Preview on the Mac, then run once in an existing remote Isaac session after synchronizing the code:

```sh
uv run --locked python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_diagnostics.json --stage balance --dry-run
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_diagnostics.json --stage balance --prepare-model
```

No additional image, dependency or Modal session is required. The existing single-L40S session/cost limits apply; five simulated seconds is not the GPU session duration. There is no automatic experiment loop.

The run records 1,000 physics samples at 200 Hz in `contacts.jsonl`: both foot-body world poses/twists, aggregate normal/friction/total contact forces, and the lower-body collection's computed/applied effort staging buffers. The control trajectory additionally records the 250 complete policy inputs and their source states. `contact-materials.json` stores authored collision-material bindings and coefficients; missing bindings/combine modes remain unresolved instead of assuming a value. `contact-metadata.json` records body mapping and sampling semantics. The parent requires complete finite samples, force-component consistency, some supporting foot-force evidence and consistent recorded policy inputs. Partial logs remain available on failure.

`diagnostics-summary.json` reports planar drift/speed, loaded-foot motion and differences between the logged computed/applied effort staging buffers. These effort counts do not establish actual clipping when native actuators execute outside those buffers. The 1 N load threshold is an aggregation setting that does not alter control or define a standing-success criterion. Foot-body origin motion can include rolling; it is not contact-point slip by itself. Authored material properties alone do not establish an effective pair coefficient or a drift cause. A non-fall pass also does not establish stationary standing.

After copying a run locally, analyze it without Isaac, ONNX inference or cloud access:

```sh
uv run --locked python scripts/isaac_g1_wbc.py --analyze-run outputs/isaac/RUN_ID
```

This command prints a derived report without rewriting the original artifacts. Older runs without the new telemetry receive a motion-only report; missing contact evidence is not reconstructed as measured force. Policy input replay checks state/history consistency with the adapter and does not prove equivalence to upstream training observations.

### Compare initial foot placement

The optional [placement profile](configs/isaac_g1_placement.json) preserves the diagnostic trial while separating initial placement from the policy's 0.74 m height command. Before the first physics step, it measures the enabled foot collision spheres against the horizontal ground Plane, then translates the root upward only as far as needed to establish 1 mm geometric clearance. It rejects a required lift above 10 cm, unsupported collider types, nonuniform sphere scales, nonhorizontal ground or a non-meter/Z-up stage. The clearance and lift bounds are project experiment settings, not vendor recommendations.

Preview locally, then run one comparison in the existing remote image after synchronizing the code:

```sh
uv run --locked python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_placement.json --stage balance --dry-run
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_placement.json --stage balance --prepare-model
```

`initial-placement.json` records measured body poses, sphere radii and body-relative transforms, ground geometry, before/after clearances, root poses and the declared vertical translation. Raw geometry is saved before clearance validation so a rejected shape can be inspected without another simulation. The local USD radius is multiplied by its composed uniform scale; the 10-cm collider-radius bound applies to the resulting world radius, including for scaled unit spheres. Foot poses come from current Isaac buffers; only shape-relative transforms come from USD, using its [transform cache](https://openusd.org/dev/api/class_usd_geom_xform_cache.html) and [sphere radius](https://openusd.org/dev/api/class_usd_geom_sphere.html). This avoids treating an authored USD world body pose as a current simulation measurement. The parent and CPU analyzer recompute clearance and validate the saved translation. Baseline and diagnostic profiles retain their original placement and do not invoke this geometry reader.

No settling/pre-roll steps, gain/friction changes, checkpoint changes or extra workers are added. Compare the new run's contact forces, displacement, tilt and final speed with an unchanged diagnostic baseline. Geometric clearance does not measure solver contact offsets or dynamic settling, and a reduced initial force alone would not establish stationary control. Native geometry collection and the effect of this new placement require a GPU run; local interface checks do not establish those outcomes. Existing session deadlines and billing apply, and run completion does not release the GPU.

### Inspect G1 dynamics without a rollout

Use the placement profile with the static audit stage to capture the current PhysX articulation before trial control. It shares the standing runner's asset, actuator configuration, frame checks and initial placement. It performs no trial physics steps, model download/inference, camera recording or training. Normal simulator startup/reset still occurs, and the Modal GPU remains allocated until the session stops.

```sh
# Mac preview; no cloud operation or model access
uv run --locked python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_placement.json --stage dynamics --dry-run

# Once in the existing remote Isaac image; --prepare-model is not used
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_placement.json --stage dynamics
```

The bounded stages are GPU preflight and a static audit. No image or dependency change is needed. The output contains `dynamics-runtime.json` with actual body masses, local COM poses, current link poses, raw inertia tensors, solver drive parameters, armatures, effort limits, target positions/velocities and resolved actuator classes/gains. Physical data is saved before actuator introspection and after each group; a query failure preserves a partial snapshot and its collection phase/error, which the analyzer rejects as a completed audit. Native owners can lack Lab model attributes or be merged/shared between groups. Their live gains are read in resolved joint order through the official [group parameter API](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.actuators.html); Lab-owned models retain their model-property path. `dynamics-reference.json` snapshots the hash-checked [derived deployment reference](assets/references/g1_wbc_dynamics.json), including source URLs and hashes. This data extraction does not execute the original simulator or establish the original training dynamics.

`dynamics-comparison.json` and `dynamics-comparison.md` compare total mass/COM, same-name bodies, principal inertias and per-joint control conditions. Unmatched bodies remain explicit and still contribute to runtime totals. The source fingers are fixed, while the current articulation has additional actuated hand joints; same-name link differences can reflect split/merged mass definitions. Inertia comparison uses eigenvalues only, preserving the raw tensors without assuming their backend frame orientation. See the [PhysX tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html) and [pinned Isaac data implementation](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py).

Analyze a copied audit on the Mac with the same `--analyze-run outputs/isaac/RUN_ID` command. `passed_dynamics_audit` means static collection and evidence validation completed; it does not establish balance, walking or a drift cause. Native collection must be validated on the selected worker before relying on the comparison. Existing deadlines/billing apply, with no automatic retry or parameter change.

### Compare left-palm inertial properties

The separate [palm profile](configs/isaac_g1_palm.json) preserves the placement profile's policy, observation history, controllers/gains, joint limits, fingers, armatures, contacts, seed and timing. It changes only the left palm's mass, body-local COM and full inertia to the existing hash-pinned rev 1.0 URDF. It derives principal moments and a proper principal-axis rotation from the left tensor, including its mirrored off-diagonal terms. The right-hand values are not copied.

The runner applies [USD mass properties](https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html) in the stage's session layer after spawning and before physics reset. It rejects an already initialized articulation, unexpected palm paths/proxies or stage units other than meters/kilograms. It does not save changes into the vendor asset. `palm-inertial-override.json` retains source/hash, before/after authored properties and actual runtime mass/COM/inertia for both palms, including values from a failed application. Right-palm authored values must remain unchanged.

First perform one static audit in the existing remote Isaac image:

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_palm.json --stage dynamics
```

Inspect `palm-inertial-override.json` for `passed_palm_inertial_override` and the run summary for `passed_dynamics_audit`. Only after this succeeds, perform one bounded five-second comparison:

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_palm.json --stage balance
```

The verification checks mass within 1e-6 kg, COM components within 1e-6 m and full inertia entries within 1e-8 kg m². These are project float-buffer acceptance tolerances. For this focused PhysX check, the [tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_inertias) specifies column-major inertia at COM in the rigid-body-prim frame; the pinned Isaac implementation forwards that data in public body order. The comparison therefore checks tensor orientation as well as moments. Historical general dynamics reports retain their eigenvalue-only comparison. An application mismatch stops before kinematic probes or policy actions. CPU replay rechecks the saved override and, for static audits, its agreement with the full raw snapshot.

Compare displacement, final speed, tilt, foot contacts and arm motion against the placement profile; matching physical properties alone does not establish stationary control. The static audit and rollout are separate manual runs with existing deadlines and billing, not an automatic loop. No image rebuild or dependency change is required. Actual application and motion still require GPU validation. Stop the Modal session with `fmr-stop` when finished.

### Qualify the complete GR00T deployment contract

Use the [deployment standing profile](configs/isaac_g1_deployment.json) to evaluate a coherent transfer of the pinned standalone GR00T deployment conditions. The earlier balance/diagnostic/placement/palm profiles remain available as historical comparisons. The [derived deployment reference](assets/references/g1_wbc_deployment.json) records source paths, hashes, model identities, joint effort caps and unresolved differences; it is reference data, not an original simulator or proof of the training distribution.

This profile combines the following conditions:

- Exactly 29 active body joints, with 14 finger joints changed to [fixed joints](https://openusd.org/dev/api/class_usd_physics_fixed_joint.html) in the session layer before reset. Joint body bindings and frames are retained; the vendor asset is not saved.
- Explicit PD for all body joints, published arm gains 100/0.5, armature 0.01 and source effort caps. Source passive terms are mapped to static/dynamic friction effort 0.1 N m and viscous friction 0.001 N m s/rad using the pinned Isaac Sim 6.2 / Lab 3 APIs. Actual native gains, armature, effort caps and passive values are checked before trial control; a mismatch stops the run.
- Source startup height 0.793 m and zero initial body-joint angles. The default lower-joint targets are applied for the first four physics steps; the first inference occurs after 0.02 simulated seconds. Subsequent observations are sampled after each four-step interval, with chronological zero-padded history and the previous policy action. The source pose stays fixed and requires nonnegative geometric foot clearance, with the existing 10-micrometer numerical tolerance. The separate automatic-placement profile retains its 1 mm clearance target. A clearance failure records measured and required values before stopping.
- The existing pinned left-palm correction, seed, physics/control timing, camera and safety guards. Named Balance/Walk checkpoints are selected by command magnitude, with their individual hashes pinned. Standalone source YAML aliases `ft92`/`ft109` are recorded as unresolved names; they are not asserted to be byte-identical aliases of the named releases.

Preview on the Mac without cloud or weight access:

```sh
uv run --locked python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_deployment.json --stage balance --dry-run
```

Then run **one** standing qualification in the existing remote Isaac image. It includes runtime validation and the bounded rollout in the same process; no separate static run or image rebuild is required. Existing cached Balance weights are reused.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_deployment.json --stage balance
```

Each case lasts five simulated seconds. All cases require maximum tilt at most 20 degrees, in addition to the existing fall/action guards. The following are **project acceptance criteria**, not vendor specifications or a robustness claim:

| Case | Command | Functional acceptance |
| --- | --- | --- |
| Stand | Zero navigation | Each foot moves at most 0.02 m; final planar speed at most 0.10 m/s |
| Forward | +0.2 m/s in x | Forward progress at least 0.30 m; maximum lateral drift at most 0.15 m; final-second mean x velocity 0.10–0.30 m/s; maximum heading change at most 0.20 rad |
| Turn | +0.2 rad/s yaw | Positive heading change at least 0.30 rad; maximum displacement at most 0.20 m; final speed at most 0.15 m/s; final-second mean heading rate 0.10–0.30 rad/s |

The parent reports `passed_deployment_stand`, `passed_deployment_forward` or `passed_deployment_turn` only when the corresponding functional gate passes. A completed non-fall worker rollout alone is insufficient. A failed gate makes the parent fail and retains `diagnostics-summary.json` with its individual checks. Inspect the whole-contract evidence before making another change; there is no automatic retry, parameter sweep or framework switch.

Only after standing succeeds, use its verified run as a prerequisite for separate manual movement trials. The existing `--stage balance` flag means a bounded control rollout; the profile selects the model and motion. The first movement invocation with `--prepare-model` downloads the single 1,886,682-byte Walk checkpoint; subsequent calls verify/reuse it.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_deployment_forward.json --stage balance --stand-run outputs/isaac/STAND_RUN_ID --prepare-model
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_deployment_turn.json --stage balance --stand-run outputs/isaac/STAND_RUN_ID
```

#### Source foot-contact profile

The [contact profile](configs/isaac_g1_contact.json) adds the pinned source foot spheres and friction 1.0 ([reference](assets/references/g1_wbc_contact.json)) to the deployment profile. Forward/turn variants require a standing run with the same profile.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_contact.json --stage balance
```

#### Implicit arm-hold profile

The [arm-hold profile](configs/isaac_g1_arm_hold.json) keeps the contact profile but holds the 14 arm joints with implicit PhysX PD at the same gains (100/0.5). Forward/turn variants require a standing run with the same profile.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_arm_hold.json --stage balance
```

#### Crouch-start profile

The [crouch-start profile](configs/isaac_g1_crouch_start.json) keeps the arm-hold profile but starts in the policy's default crouch with 1 mm foot clearance instead of the source zero-angle pose. Forward/turn variants require a standing run with the same profile.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_crouch_start.json --stage balance
```

#### Box push

The [push profile](configs/isaac_g1_push.json) adds a 0.7 x 0.8 x 1.0 m, 10 kg box (friction 0.3) 0.05 m beyond the fingertips and walks forward with a 0.2 rad forward torso-pitch command, arms in the hold pose. Hand-box contact forces are recorded. The gate requires the box to stay still for the first 0.2 s, then move at least 0.20 m without tipping past 10 degrees. Ground friction combines by multiplication so the feet keep 1.0 and the box keeps its own coefficient.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_wbc.py --profile configs/isaac_g1_push.json --stage balance --stand-run outputs/isaac/STAND_RUN_ID
```

The [slider profile](configs/isaac_g1_push_sliders.json) is identical except for box-floor friction 0.1, representing furniture sliders; slider placement itself is not simulated.

#### Parallel push environments (RL stage 2)

[`scripts/isaac_g1_rl.py`](scripts/isaac_g1_rl.py) builds parallel copies of the slider push with the frozen GR00T walk policy, checks that every copy carries the qualified setup, and measures throughput with per-phase timing ([256 environments](configs/isaac_g1_rl_throughput_256.json)). It does not train.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_rl.py --config configs/isaac_g1_rl_throughput_256.json
```

#### High-level PPO push policy (RL stage 3)

[`scripts/isaac_g1_ppo.py`](scripts/isaac_g1_ppo.py) trains an RSL-RL PPO policy that sets the frozen GR00T walk command (forward speed, yaw rate, forward lean) every 0.1 s to push the box 1.0 m. The [config](configs/isaac_g1_ppo_push.json) randomizes box mass and friction per environment and the start gap and yaw per episode. The scripted slider-push command and the learned policy are evaluated on the same seeded starts.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_ppo.py --config configs/isaac_g1_ppo_push.json
```

[`scripts/isaac_g1_ppo_eval.py`](scripts/isaac_g1_ppo_eval.py) evaluates a checkpoint on boxes with held-out mass and friction against the scripted command, and records [showcase boxes](configs/isaac_g1_ppo_eval.json) on camera with per-decision command traces. When the hand push stays high without box progress, a stall monitor stops the push policy and GR00T Balance holds the robot, following the source rule that a command norm of at most 0.05 selects Balance.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_ppo_eval.py --config configs/isaac_g1_ppo_eval.json
```

#### Lift and carry light furniture

[`scripts/isaac_g1_lift.py`](scripts/isaac_g1_lift.py) attaches a light [cabinet](configs/isaac_g1_lift.json) by its side faces to both palms with fixed joints (a grasp abstraction; hand collisions with it are filtered), raises it from the floor with arm IK under GR00T Balance, carries it with GR00T Walk, sets it down and tests releasing the joints at runtime. One environment per case sweeps the cabinet mass (optionally with static load feedforward on the arm targets) and logs arm tracking, the implied drive torque and the cabinet's contact on the legs and torso to show what limits lifting.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_lift.py --config configs/isaac_g1_lift.json
```

#### Corridor carry PPO policy

[`scripts/isaac_g1_carry.py`](scripts/isaac_g1_carry.py) trains an RSL-RL PPO policy that sets the GR00T walk forward speed and yaw rate every 0.1 s to carry the attached cabinet through an L-shaped corridor. Each environment has its own [corridor](configs/isaac_g1_carry_speed.json) width, corner distance, turn direction and cabinet mass (2-4 kg), and walking faster than 0.3 m/s is penalized. A pure-pursuit command is evaluated on the same seeded starts. A torso lean action is available when the config bounds `lean_pitch_rad`.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_carry.py --config configs/isaac_g1_carry_speed.json
```

[`scripts/isaac_g1_carry_eval.py`](scripts/isaac_g1_carry_eval.py) evaluates a checkpoint on held-out corridors and masses against pure pursuit, logging walking speed, and records [showcase corridors](configs/isaac_g1_carry_speed_eval.json) driven by pursuit and by the checkpoint side by side.

```sh
/opt/isaaclab/.venv/bin/python scripts/isaac_g1_carry_eval.py --config configs/isaac_g1_carry_speed_eval.json
```

`deployment-joints.json`, `deployment-runtime.json` and `deployment-audit.json` retain authored topology and actual control properties. The effective 29-joint asset/controller snapshots, recorded policy phase/applied action and functional checks are replayed by the existing `--analyze-run` command. Actual GPU topology/control application and functional performance require validation on the selected worker; local doubles and USD composition checks do not establish those outcomes. Contact/solver semantics, collider/body decomposition and the retained experiment velocity safety caps remain explicit residual differences. Do not claim engine or training equivalence. The same single-worker session deadlines/billing apply, and `fmr-stop` releases the session when finished.

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

Save the remote workspace before stopping it. The existing 60-minute Isaac resource profile and cost estimate apply; the 10-second video duration is simulated time and does not set the GPU session length. A live browser camera viewer is a later step.

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

Meta model calls go to the [Meta Model API](https://dev.meta.ai/docs/models) (Muse Spark, SAM 3.1, Muse Image) under Meta's terms; the API key is read from the environment or a Git-ignored `.env` (see `.env.example`). SAM 3.1 mask payloads are decoded with Meta's [meta-sam-parser](https://github.com/meta-models/meta-sam) 0.1.0 (SAM License, pinned with hashes in `uv.lock`). Demo interiors are generated with Muse Image with its image/web search tools disabled; they show no real homes or people.

The browser view uses [three.js](https://github.com/mrdoob/three.js) r186 (npm 0.186.1, MIT), vendored unmodified under `web/vendor/three/` with its LICENSE and a SHA-256 manifest (`vendor.json`).

The GR00T-WBC Balance/Walk checkpoints are downloaded from the [pinned NVIDIA source](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc) under the [NVIDIA Open Model License](licenses/GR00T-WBC.txt) and stay Git-ignored. The [reference URDF](assets/references/g1_wbc.urdf) is the unmodified [upstream rev 1.0 model](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf). The derived [deployment](assets/references/g1_wbc_deployment.json), [dynamics](assets/references/g1_wbc_dynamics.json) and [contact](assets/references/g1_wbc_contact.json) references come from the same pinned source. These carry the NVIDIA terms and the [Unitree BSD notice](licenses/UNITREE-BSD-3-CLAUSE.txt).

## Original contributions and third-party components

Update this table in the same commit as each new module or third-party addition. Licenses are listed above.

| Built during the hackathon | Third-party (imported) |
| --- | --- |
| [Modal launcher](scripts/modal_dev.py), Isaac [preflight](scripts/isaac_preflight.py)/[recording](scripts/isaac_video.py)/[observer](scripts/isaac_observe.py)/[G1 asset inventory](scripts/isaac_g1_asset.py) | Isaac Sim, Isaac Lab, RSL-RL, PyTorch, G1 USD assets |
| GR00T-to-Isaac controller transfer: [adapter](scripts/g1_wbc.py), [runner](scripts/isaac_g1_wbc.py), [parallel environments](scripts/isaac_g1_rl.py), [PPO push training](scripts/isaac_g1_ppo.py)/[evaluation](scripts/isaac_g1_ppo_eval.py), [lift probe](scripts/isaac_g1_lift.py), [corridor carry training](scripts/isaac_g1_carry.py)/[evaluation](scripts/isaac_g1_carry_eval.py), [profiles](configs/) | GR00T-WBC Balance/Walk checkpoints, [G1 URDF](assets/references/g1_wbc.urdf) |
| Contact/dynamics diagnostics and CPU analysis | Derived GR00T deployment/dynamics/contact [reference data](assets/references/) |
| [Meta model client](scripts/meta_models.py) (Muse Spark chat, SAM 3.1 segmentation, Muse Image generation, response cache), [interior prompts](configs/home_images.json) | Meta Model API (Muse Spark, SAM 3.1, Muse Image), meta-sam-parser, [three.js](web/vendor/three/) |
| [Tests](tests/), README, [AGENTS.md](AGENTS.md) | NumPy, FFmpeg/libx264, Modal SDK, Ubuntu/libXt, OpenSSH, Git, uv |

## Assets and credentials

The G1 reference URDF is imported robot data used for kinematic checks; its original bytes and license notices are retained. Simulation uses the separately licensed Isaac USD asset. Use self-rendered or explicitly licensed images and record each source and license here when added. Do not use real home photos or personal information.

Keep API keys and tokens in environment variables. Do not include them in source files, prompts, logs, or commits. Local environment files, caches, model weights, and run artifacts are ignored by default.