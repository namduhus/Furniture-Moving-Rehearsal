# Furniture-Moving Rehearsal

A simulated humanoid physically rehearses moving furniture through hallways and corners before a person attempts the move. Built for the **Push the Frontier** theme at the Meta Global AI Developer Hackathon.

The intended pipeline combines Meta models for scene interpretation and failure diagnosis with a physics simulation that checks proposed moving strategies. The target is a **VLM-guided humanoid controller**.

## Status

Project kickoff: October 27, 2026. This repository currently contains planning documents and repository setup. No simulation, trained controller, API integration, or performance result has been implemented or verified yet.

See [the project design](docs/idea.md) for scope, milestones, and open decisions, and [the collaboration instructions](AGENTS.md) for development rules. The [saved participant guide](docs/detail.md) is a reference copy; confirm current official rules before submission.

## First runnable milestone

Produce a reproducible simulation rollout in which a robot pushes a box through a straight hallway. Save the configuration, seed, rollout log, and video so the result can be inspected.

The target robot is Unitree G1. The starting order between G1 and a simpler pusher remains to be reconciled between the planning documents. The documented fallback is a pusher if G1 cannot complete the straight-hallway task within the first two implementation days.

Before installing dependencies, review their official sources, licenses, maintenance, and supply-chain risks, and obtain approval. Setup and execution commands will be added after they have been run successfully.

## Original contributions and third-party components

Keep this table current in the same commit as each new module or third-party addition. Planned components are listed separately from implemented work.

| Component | Origin | Current state | Source / license |
| --- | --- | --- | --- |
| Project design and collaboration instructions | Project planning | Present in the repository | [Design](docs/idea.md), [instructions](AGENTS.md) |
| README and ignore rules | Created during the hackathon | Repository setup only | [README](README.md), [.gitignore](.gitignore) |
| Participant guide reference copy | Third-party event material | Local reference only; not a project implementation | [Saved guide](docs/detail.md); redistribution rights not yet verified |

| Planned component | Intended origin | Current state |
| --- | --- | --- |
| Scene specification, scale calibration, and scene generation | New hackathon work | Not implemented |
| High-level controller, uncertainty randomization, and planner comparison | New hackathon work | Not implemented |
| Meta model adapter, failure diagnosis, and result explanation | New hackathon work | Not implemented |
| MuJoCo, MJX, and MuJoCo Playground | Third-party software | Not installed by this project; license review pending |
| Unitree G1 robot assets and pretrained locomotion policy | Third-party assets / model | Not added; asset and policy licenses must be checked separately |
| Meta models and API | Third-party service / model | Not integrated; model access and applicable terms must be verified |

## Assets and credentials

No demo image, video, or robot asset has been added. Use self-rendered or explicitly licensed assets and record each source and license here when added. Do not use real home photos or personal information.

Keep API keys and tokens in environment variables. Do not include them in source files, prompts, logs, or commits. Local environment files, caches, model weights, and run artifacts are ignored by default.

## Validation and submission

Report only measured results. Record seeds and configurations for experiments, distinguish simulated outcomes from real-world claims, and cache model responses without retaining credentials.

Submission preparation includes a public repository, a demo video of at most three minutes, clear identification of new hackathon work, verified asset licenses, and meaningful Meta model usage. Public publishing has not been performed.
