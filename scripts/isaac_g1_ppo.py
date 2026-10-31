"""High-level PPO push policy over the frozen GR00T walk policy (RL stage 3, task A: straight push to a goal).

Every 0.1 s the learned policy chooses a forward speed, yaw rate and forward lean; GR00T Walk turns that command
into joint targets on the source schedule. Box mass and floor friction are fixed per environment (authored before
reset and checked in PhysX); start gap and box yaw are resampled at every reset. The scripted command of the
qualified slider push is evaluated on the same seeded starts before training, and the learned policy after it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import isaac_g1_rl as rl
from isaac_g1_wbc import ROOT, load_plan, trial_contract
from isaac_preflight import check_versions, run_stage, write_json


DEFAULT_CONFIG = ROOT / "configs/isaac_g1_ppo_push.json"
ACTION_KEYS = ("forward_mps", "yaw_rate_rps", "lean_pitch_rad")
FAILURES = ("invalid_action", "fallen", "box_tipped", "off_course", "lost_box")
OUTCOMES = (*FAILURES, "success", "time_out")
PPO_KEYS = {"num_steps_per_env", "max_iterations", "save_interval", "hidden_dims", "init_std", "learning_rate",
            "entropy_coef", "num_learning_epochs", "num_mini_batches", "gamma", "lam", "desired_kl", "clip_param"}


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"push_profile", "num_envs", "env_spacing_m", "decision_control_steps", "episode_seconds",
                       "goal_distance_m", "randomization", "action_bounds", "termination", "reward",
                       "scripted_action", "ppo", "seed", "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed PPO configuration keys")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    def interval(value, low, high):
        return isinstance(value, list) and len(value) == 2 and all(number(v, low, high) for v in value) and value[0] < value[1]

    for key, (low, high) in {"num_envs": (1, 4096), "decision_control_steps": (1, 50), "seed": (0, 2**31-1),
                             "stage_timeout_seconds": (60, 10800)}.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid {key}")
    if not (number(config["env_spacing_m"], 4.0, 10.0) and number(config["episode_seconds"], 1.0, 60.0) and
            number(config["goal_distance_m"], 0.1, 2.0)):
        raise ValueError("Invalid environment spacing, episode length or goal distance")
    profile, asset, runtime = load_plan(ROOT / config["push_profile"])
    if profile.get("functional_case") != "push" or profile.get("arm_hold") != "implicit":
        raise ValueError("PPO environments must start from a reviewed implicit-arm push profile")
    r = config["randomization"]
    if (set(r) != {"mass_kg", "friction", "gap_m", "yaw_rad"} or not interval(r["mass_kg"], 0.5, 50.0) or
            not interval(r["friction"], 0.01, 1.0) or not interval(r["gap_m"], 0.0, 0.5) or
            not interval(r["yaw_rad"], -0.5, 0.5)):
        raise ValueError("Invalid randomization ranges")
    # The near box corner must stay clear of the fingertips at the smallest gap and largest yaw.
    import numpy as np
    size, yaw = profile["push_object"]["size_m"], max(abs(v) for v in r["yaw_rad"])
    if r["gap_m"][0] - size[1]/2*np.sin(yaw) - size[0]/2*(1-np.cos(yaw)) < 0.02:
        raise ValueError("Randomized box start can touch the hands")
    b = config["action_bounds"]
    # Forward speed stays above the GR00T walk/balance switch norm (0.05), so the walk policy is always valid;
    # lean stays inside the reviewed pitch command bound.
    if (set(b) != set(ACTION_KEYS) or not interval(b["forward_mps"], 0.1, 0.5) or
            not interval(b["yaw_rate_rps"], -0.5, 0.5) or not interval(b["lean_pitch_rad"], 0.0, 0.3)):
        raise ValueError("Invalid action bounds")
    s = config["scripted_action"]
    if set(s) != set(ACTION_KEYS) or not all(number(s[k], *b[k]) for k in ACTION_KEYS):
        raise ValueError("Scripted action must lie inside the action bounds")
    t = config["termination"]
    if set(t) != {"box_lateral_m", "box_yaw_rad", "box_tilt_rad", "robot_box_distance_m"} or not all(
            number(v, 0.01, 3.0) for v in t.values()):
        raise ValueError("Invalid termination limits")
    w = config["reward"]
    if set(w) != {"progress", "lateral", "yaw", "action_rate", "failure", "success"} or not all(
            number(v, 0.0, 100.0) for v in w.values()):
        raise ValueError("Invalid reward weights")
    p = config["ppo"]
    if (set(p) != PPO_KEYS or not all(type(p[k]) is int and p[k] >= 1 for k in
                                      ("num_steps_per_env", "max_iterations", "save_interval", "num_learning_epochs",
                                       "num_mini_batches")) or
            not (isinstance(p["hidden_dims"], list) and p["hidden_dims"] and all(type(v) is int and v > 0 for v in p["hidden_dims"])) or
            not all(number(p[k], 0.0, 1.0) for k in ("init_std", "learning_rate", "entropy_coef", "gamma", "lam",
                                                    "desired_kl", "clip_param"))):
        raise ValueError("Invalid PPO settings")
    return config, profile, asset, runtime


def bound_arrays(bounds):
    import numpy as np
    return tuple(np.array([bounds[k][i] for k in ACTION_KEYS], dtype=float) for i in (0, 1))


def action_commands(actions, bounds):
    """Map normalized actions in [-1, 1] to GR00T commands [vx, 0, yaw rate] and lean rpy commands [0, pitch, 0]."""
    import numpy as np
    low, high = bound_arrays(bounds)
    values = low + (np.clip(np.asarray(actions, dtype=float), -1.0, 1.0) + 1.0)/2*(high - low)
    zeros = np.zeros(len(values))
    return np.c_[values[:, 0], zeros, values[:, 1]], np.c_[zeros, values[:, 2], zeros]


def normalized_action(values, bounds):
    import numpy as np
    low, high = bound_arrays(bounds)
    return 2*(np.asarray(values, dtype=float) - low)/(high - low) - 1.0


def yaw_of(quats_xyzw):
    import numpy as np
    x, y, z, w = np.asarray(quats_xyzw, dtype=float).T
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def task_observation(robot_pose, robot_velocity, box_pose, box_velocity, origins, goal_x, gravity_b, last_action,
                     time_fraction):
    """Task state in the robot heading frame plus course-line offsets; world velocities are [linear, angular]."""
    import numpy as np
    heading = yaw_of(robot_pose[:, 3:])
    c, s = np.cos(heading), np.sin(heading)

    def local(v):
        return np.c_[c*v[:, 0] + s*v[:, 1], -s*v[:, 0] + c*v[:, 1]]

    box_yaw = yaw_of(box_pose[:, 3:])
    relative_yaw = np.arctan2(np.sin(box_yaw - heading), np.cos(box_yaw - heading))
    return np.c_[local(box_pose[:, :2] - robot_pose[:, :2]), relative_yaw, heading,
                 robot_pose[:, 1] - origins[:, 1], box_pose[:, 1] - origins[:, 1], box_yaw, goal_x - box_pose[:, 0],
                 box_velocity[:, :2], box_velocity[:, 5], local(robot_velocity[:, :2]), robot_velocity[:, 5],
                 gravity_b, last_action, time_fraction].astype(np.float32)


def task_step(box_pose, robot_pose, origins, start_x, previous_x, actions, last_actions, low_level, config, decision_dt):
    """Reward, termination and outcome index (into OUTCOMES, -1 while running) for one high-level decision."""
    import numpy as np
    from g1_wbc import quat_xyzw_matrices
    t, w = config["termination"], config["reward"]
    progress = box_pose[:, 0] - start_x
    lateral = np.abs(box_pose[:, 1] - origins[:, 1])
    yaw = np.abs(yaw_of(box_pose[:, 3:]))
    tilt = np.arccos(np.clip(quat_xyzw_matrices(box_pose[:, 3:])[:, 2, 2], -1.0, 1.0))
    flags = np.stack([low_level["invalid_action"], low_level["fallen"], tilt > t["box_tilt_rad"],
                      (lateral > t["box_lateral_m"]) | (yaw > t["box_yaw_rad"]),
                      np.linalg.norm(box_pose[:, :2] - robot_pose[:, :2], axis=1) > t["robot_box_distance_m"]])
    failed = flags.any(axis=0)
    success = ~failed & (progress >= config["goal_distance_m"])
    outcome = np.where(failed, np.argmax(flags, axis=0), np.where(success, OUTCOMES.index("success"), -1))
    reward = (w["progress"]*(box_pose[:, 0] - previous_x)/decision_dt - w["lateral"]*lateral - w["yaw"]*yaw
              - w["action_rate"]*np.sum((actions - last_actions)**2, axis=1) - w["failure"]*failed + w["success"]*success)
    return reward, failed | success, outcome, progress


def episode_summary(records):
    import numpy as np
    count = len(records)
    rates = {name: (sum(r["outcome"] == name for r in records)/count if count else None) for name in OUTCOMES}
    progress = np.array([r["progress_m"] for r in records])
    successes = [r["seconds"] for r in records if r["outcome"] == "success"]
    return {"episodes": count, "success_rate": rates["success"], "outcome_rates": rates,
            "progress_m": {"mean": float(progress.mean()), "median": float(np.median(progress))} if count else None,
            "median_success_seconds": float(np.median(successes)) if successes else None}


class ProgressLog:
    """Aggregate episodes that finish during each PPO rollout into one JSON line per iteration."""

    def __init__(self, path, steps_per_iteration, env_steps_per_decision):
        self.path, self.steps_per_iteration, self.env_steps_per_decision = path, steps_per_iteration, env_steps_per_decision
        self.records, self.decisions, self.iteration, self.tic = [], 0, 0, time.perf_counter()

    def __call__(self, completed, num_envs):
        self.records += completed.values()
        self.decisions += 1
        if self.decisions % self.steps_per_iteration == 0:
            wall = time.perf_counter() - self.tic
            line = {"iteration": self.iteration, "rollout_and_update_seconds": wall,
                    "env_steps_per_second": num_envs*self.steps_per_iteration*self.env_steps_per_decision/wall,
                    **episode_summary(self.records)}
            with self.path.open("a") as file:
                file.write(json.dumps(line) + "\n")
            self.records, self.iteration, self.tic = [], self.iteration + 1, time.perf_counter()


class PushTaskEnv:
    """RSL-RL vectorized environment over the prepared parallel push scene."""

    def __init__(self, scene, policy, contract, profile, asset, runtime, config, masses, frictions, observations):
        import numpy as np
        from g1_wbc import BatchHistory
        self.s, self.policy, self.profile, self.asset, self.config = scene, policy, profile, asset, config
        self.masses, self.frictions, self.observations = np.asarray(masses), np.asarray(frictions), observations
        self.num_envs, self.num_actions, self.device, self.cfg = len(scene.origins), len(ACTION_KEYS), runtime["device"], config
        self.decision_dt = config["decision_control_steps"]*asset["control_decimation"]*asset["physics_dt"]
        self.max_episode_length = int(round(config["episode_seconds"]/self.decision_dt))
        self.history = BatchHistory(contract, self.num_envs)
        self.targets = np.tile(scene.defaults, (self.num_envs, 1))
        self.last_action = np.zeros((self.num_envs, self.num_actions))
        self.length = np.zeros(self.num_envs, dtype=int)
        self.start_x, self.previous_x, self.goal_x = (np.zeros(self.num_envs) for _ in range(3))
        self.rng = np.random.default_rng(config["seed"])
        self.recorder, self.last_completed, self.reset_on_done = None, {}, True
        self.last_command = np.zeros((self.num_envs, self.num_actions))
        # Stopped environments get a zero command, which hands them to GR00T Balance under the source rule.
        self.stopped = np.zeros(self.num_envs, dtype=bool)
        self.reset_envs(np.arange(self.num_envs))
        self.obs = self.observe()

    @property
    def episode_length_buf(self):
        import torch
        return torch.as_tensor(self.length, device=self.device)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        import numpy as np
        self.length = np.asarray(value.detach().cpu().numpy(), dtype=int).copy()

    def get_observations(self):
        return self.obs

    def reset_envs(self, ids, rng=None):
        import numpy as np
        s, push, r = self.s, self.profile["push_object"], self.config["randomization"]
        rng = self.rng if rng is None else rng
        n, env_ids = len(ids), ids.tolist()
        gap, yaw = rng.uniform(*r["gap_m"], n), rng.uniform(*r["yaw_rad"], n)
        s.scene.reset(env_ids)
        s.robot.write_root_pose_to_sim_index(root_pose=s.device(s.root[ids]), env_ids=env_ids)
        s.robot.write_root_velocity_to_sim_index(root_velocity=s.device(np.zeros((n, 6))), env_ids=env_ids)
        s.robot.write_joint_position_to_sim_index(position=s.device(np.tile(s.defaults, (n, 1))), env_ids=env_ids)
        s.robot.write_joint_velocity_to_sim_index(velocity=s.device(np.zeros((n, 29))), env_ids=env_ids)
        pose = np.c_[s.origins[ids] + np.c_[s.hand_reach + gap + push["size_m"][0]/2, np.zeros(n), np.full(n, push["size_m"][2]/2)],
                     np.zeros(n), np.zeros(n), np.sin(yaw/2), np.cos(yaw/2)]
        s.box.write_root_pose_to_sim_index(root_pose=s.device(pose), env_ids=env_ids)
        s.box.write_root_velocity_to_sim_index(root_velocity=s.device(np.zeros((n, 6))), env_ids=env_ids)
        # Source schedule restarts: default targets for the first physics interval and an empty history.
        self.history.reset(ids)
        self.targets[ids] = s.defaults
        s.robot.set_joint_position_target_index(target=s.device(self.targets))
        self.last_action[ids] = 0.0
        self.stopped[ids] = False
        self.length[ids] = 0
        self.start_x[ids] = self.previous_x[ids] = pose[:, 0]
        self.goal_x[ids] = pose[:, 0] + self.config["goal_distance_m"]

    def observe(self):
        import numpy as np
        import torch
        from g1_wbc import finite, quat_xyzw_matrices
        s, n = self.s, self.num_envs
        robot = finite(s.batch(s.robot.data.root_link_pose_w), (n, 7))
        gravity = -(quat_xyzw_matrices(robot[:, 3:]) @ s.rotation)[:, 2, :]
        task = task_observation(robot, finite(s.batch(s.robot.data.root_link_vel_w), (n, 6)),
                                finite(s.batch(s.box.data.root_link_pose_w), (n, 7)),
                                finite(s.batch(s.box.data.root_link_vel_w), (n, 6)), s.origins, self.goal_x, gravity,
                                self.last_action, self.length[:, None]/self.max_episode_length)
        r = self.config["randomization"]
        # The critic also sees the per-environment box mass and friction, normalized to their ranges.
        privileged = np.c_[(self.masses - r["mass_kg"][0])/(r["mass_kg"][1] - r["mass_kg"][0]),
                           (self.frictions - r["friction"][0])/(r["friction"][1] - r["friction"][0])].astype(np.float32)

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        return self.observations({"policy": tensor(task), "critic": tensor(np.c_[task, privileged])})

    def step(self, actions):
        import numpy as np
        import torch
        from g1_wbc import finite, quat_xyzw_matrices
        s, n, profile = self.s, self.num_envs, self.profile
        action = np.clip(np.asarray(actions.detach().cpu().numpy(), dtype=float), -1.0, 1.0)
        commands, rpys = action_commands(action, self.config["action_bounds"])
        commands[self.stopped], rpys[self.stopped] = 0.0, 0.0
        self.last_command = np.c_[commands[:, 0], commands[:, 2], rpys[:, 1]]
        low_level = {"invalid_action": np.zeros(n, dtype=bool), "fallen": np.zeros(n, dtype=bool)}
        for _ in range(self.config["decision_control_steps"]):
            for _ in range(self.asset["control_decimation"]):
                s.scene.write_data_to_sim()
                s.sim.step(render=False)
                s.scene.update(self.asset["physics_dt"])
            pose = finite(s.batch(s.robot.data.root_link_pose_w), (n, 7))
            observation = self.history.observe(s.batch(s.robot.data.joint_pos)[:, s.indices],
                                               s.batch(s.robot.data.joint_vel)[:, s.indices], pose[:, 3:],
                                               s.batch(s.robot.data.root_link_vel_w)[:, 3:], s.rotation, commands, rpys)
            lower_targets, _, invalid = self.history.targets(self.policy.batch(observation, commands), s.limits)
            self.targets[:, s.lower_indices] = lower_targets
            s.robot.set_joint_position_target_index(target=s.device(self.targets))
            gravity = -(quat_xyzw_matrices(pose[:, 3:]) @ s.rotation)[:, 2, :]
            low_level["invalid_action"] |= invalid
            low_level["fallen"] |= ((pose[:, 2] < profile["minimum_base_height_m"]) |
                                    (np.arccos(np.clip(-gravity[:, 2], -1, 1)) > profile["maximum_tilt_rad"]))
        box = finite(s.batch(s.box.data.root_link_pose_w), (n, 7))
        reward, terminated, outcome, progress = task_step(box, pose, s.origins, self.start_x, self.previous_x, action,
                                                          self.last_action, low_level, self.config, self.decision_dt)
        self.previous_x = box[:, 0].copy()
        self.last_action = action
        self.length += 1
        time_out = ~terminated & (self.length >= self.max_episode_length)
        ids = np.flatnonzero(terminated | time_out)
        self.last_completed = {int(i): {"outcome": OUTCOMES[outcome[i]] if terminated[i] else "time_out",
                                        "progress_m": float(progress[i]), "seconds": float(self.length[i]*self.decision_dt)}
                               for i in ids}
        extras = {"time_outs": torch.as_tensor(time_out, device=self.device)}
        if len(ids):
            records = list(self.last_completed.values())
            extras["log"] = {f"/episode/{name}": torch.as_tensor([r["outcome"] == name for r in records], dtype=torch.float32)
                             for name in ("success", "fallen", "off_course", "time_out")}
            extras["log"]["/episode/progress_m"] = torch.as_tensor([r["progress_m"] for r in records], dtype=torch.float32)
            if self.reset_on_done:
                self.reset_envs(ids)
        if self.recorder is not None:
            self.recorder(self.last_completed, n)
        self.obs = self.observe()
        return (self.obs, torch.as_tensor(reward, dtype=torch.float32, device=self.device),
                torch.as_tensor(terminated | time_out, dtype=torch.long, device=self.device), extras)


def hand_push(env):
    """Box push sensed on the hand links: the negative world-x reaction, summed over links (wrist/forearm unsensed)."""
    return -env.s.batch(env.s.scene["hand_contact"].data.force_matrix_w).sum(axis=(1, 2))[:, 0]


def command_summary(commands, pushes):
    import numpy as np
    commands, pushes = np.asarray(commands, dtype=float), np.asarray(pushes, dtype=float)
    contact = pushes > 1.0
    return {"decisions": len(commands), "forward_mean_mps": float(commands[:, 0].mean()),
            "forward_max_mps": float(commands[:, 0].max()), "yaw_rate_abs_mean_rps": float(np.abs(commands[:, 1]).mean()),
            "lean_mean_rad": float(commands[:, 2].mean()), "lean_max_rad": float(commands[:, 2].max()),
            "hand_contact_fraction": float(contact.mean()),
            "hand_push_mean_during_contact_n": float(pushes[contact].mean()) if contact.any() else 0.0}


class StallMonitor:
    """Flags an environment pushing hard without moving the box: over the last window, the sensed hand push averages
    at least push_threshold_n while the box advances less than max_progress_m. The thresholds are calibrated on
    recorded traces of the learned policy (~20 N capacity); they are not a property of the robot."""

    def __init__(self, num_envs, settings, decision_dt):
        import numpy as np
        self.settings, self.size = settings, int(round(settings["window_s"]/decision_dt))
        self.push, self.progress = np.zeros((num_envs, self.size)), np.zeros((num_envs, self.size + 1))
        self.count = np.zeros(num_envs, dtype=int)

    def update(self, push, progress):
        """Add one decision sample per environment (episodes start at zero progress)."""
        import numpy as np
        self.push = np.roll(self.push, -1, axis=1)
        self.progress = np.roll(self.progress, -1, axis=1)
        self.push[:, -1], self.progress[:, -1] = push, progress
        self.count += 1
        mean_push, advance = self.push.mean(axis=1), self.progress[:, -1] - self.progress[:, 0]
        stalled = ((self.count >= self.size) & (mean_push >= self.settings["push_threshold_n"]) &
                   (advance < self.settings["max_progress_m"]))
        return stalled, mean_push, advance


def box_progress(env):
    return env.s.batch(env.s.box.data.root_link_pose_w)[:, 0] - env.start_x


def evaluate(env, act, seed, stall=None):
    """One episode per environment from seeded starts; outcomes are grouped by the force needed to slide the box.

    Commands and hand push are sampled once per high-level decision until each environment's episode ends. With
    stall settings, a StallMonitor only observes (the episode continues) so detections can be scored on outcomes.
    """
    import numpy as np
    env.reset_envs(np.arange(env.num_envs), np.random.default_rng(seed))
    env.obs = env.observe()
    first, commands, pushes = {}, [[] for _ in range(env.num_envs)], [[] for _ in range(env.num_envs)]
    monitor, detected = (StallMonitor(env.num_envs, stall, env.decision_dt) if stall else None), {}
    for _ in range(env.max_episode_length):
        active = [i for i in range(env.num_envs) if i not in first]
        env.step(act(env.obs))
        push = hand_push(env)
        for i in active:
            commands[i].append(env.last_command[i])
            pushes[i].append(push[i])
        if monitor is not None:
            stalled = monitor.update(push, box_progress(env))[0]
            for i in active:
                if stalled[i] and i not in env.last_completed:
                    detected.setdefault(i, float(env.length[i]*env.decision_dt))
        for i, record in env.last_completed.items():
            first.setdefault(i, record)
        if len(first) == env.num_envs:
            break
    records = [{**first[i], "commands": command_summary(commands[i], pushes[i]), "stall_detected_s": detected.get(i)}
               for i in range(env.num_envs)]
    force = env.frictions*env.masses*9.81
    groups = {}
    for low, high in ((0.0, 10.0), (10.0, 20.0), (20.0, float("inf"))):
        chosen = [r for r, f in zip(records, force) if low <= f < high]
        groups[f"{low:g}-{high:g}_n"] = episode_summary(chosen)
    forward = np.concatenate([np.asarray(c, dtype=float)[:, 0] for c in commands])
    stalls = None
    if monitor is not None:
        failures = [r for r in records if r["outcome"] != "success"]
        times = [r["stall_detected_s"] for r in failures if r["stall_detected_s"] is not None]
        stalls = {"settings": stall, "flagged_successes": sum(r["stall_detected_s"] is not None for r in records
                                                              if r["outcome"] == "success"),
                  "failures": len(failures), "flagged_failures": len(times),
                  "median_detection_s": float(np.median(times)) if times else None}
    return {**episode_summary(records), "by_slide_force": groups, "seed": seed, "stall": stalls,
            "forward_command": {"max_mps": float(forward.max()), "mean_mps": float(forward.mean()),
                                "fraction_above_reviewed_0_2_mps": float((forward > 0.2 + 1e-9).mean())},
            "per_env": [{"mass_kg": float(m), "friction": float(f), **r} for m, f, r in zip(env.masses, env.frictions, records)]}


def train_config(config):
    p = config["ppo"]
    return {"num_steps_per_env": p["num_steps_per_env"], "save_interval": p["save_interval"], "check_for_nan": True,
            "logger": "tensorboard", "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
            "actor": {"class_name": "MLPModel", "hidden_dims": p["hidden_dims"], "activation": "elu",
                      "obs_normalization": True,
                      "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": p["init_std"]}},
            "critic": {"class_name": "MLPModel", "hidden_dims": p["hidden_dims"], "activation": "elu",
                       "obs_normalization": True},
            "algorithm": {"class_name": "PPO", "num_learning_epochs": p["num_learning_epochs"],
                          "num_mini_batches": p["num_mini_batches"], "clip_param": p["clip_param"], "gamma": p["gamma"],
                          "lam": p["lam"], "value_loss_coef": 1.0, "entropy_coef": p["entropy_coef"],
                          "learning_rate": p["learning_rate"], "max_grad_norm": 1.0, "use_clipped_value_loss": True,
                          "schedule": "adaptive", "desired_kl": p["desired_kl"]}}


def gr00t_policy(contract, model_paths):
    from g1_wbc import BalancePolicy, SwitchingPolicy, role_contract
    walk, balance = model_paths
    return SwitchingPolicy(BalancePolicy(walk, contract), BalancePolicy(balance, role_contract(contract, "balance")))


def gr00t_models(contract):
    from g1_wbc import prepare_model, role_contract
    return prepare_model(contract), prepare_model(role_contract(contract, "balance"))


def run_worker(config, profile, asset, runtime, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp

    contract, reference = trial_contract(profile)
    policy = gr00t_policy(contract, model_paths)
    count, r = config["num_envs"], config["randomization"]
    torch.manual_seed(config["seed"])
    wp.config.enable_backward = False
    rng = np.random.default_rng(config["seed"])
    masses, frictions = rng.uniform(*r["mass_kg"], count), rng.uniform(*r["friction"], count)
    from isaaclab.app import launch_simulation

    built = rl.scene_config(profile, asset, runtime, contract, reference, with_camera=False, num_envs=count,
                            env_spacing=config["env_spacing_m"], replicate_physics=False)
    evidence = directory / "ppo-worker.json"
    result = {"status": "running", "num_envs": count, "box_properties": {"masses_kg": masses.tolist(),
                                                                         "frictions": frictions.tolist()}}
    write_json(evidence, result)
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "device": runtime["device"]}):
        try:
            s = rl.prepare_push_scene(built, count, profile, asset, runtime, contract, reference, directory, result,
                                      evidence, (masses, frictions))
            from tensordict import TensorDict
            env = PushTaskEnv(s, policy, contract, profile, asset, runtime, config, masses, frictions,
                              lambda groups: TensorDict(groups, batch_size=[count]))
            scripted = normalized_action([config["scripted_action"][k] for k in ACTION_KEYS], config["action_bounds"])
            result["scripted_eval"] = evaluate(
                env, lambda obs: torch.as_tensor(np.tile(scripted, (count, 1)), dtype=torch.float32), config["seed"] + 1)
            write_json(evidence, result)

            from rsl_rl.runners import OnPolicyRunner
            p = config["ppo"]
            env.recorder = ProgressLog(directory / "ppo-progress.jsonl", p["num_steps_per_env"],
                                       config["decision_control_steps"])
            runner = OnPolicyRunner(env, train_config(config), log_dir=str(directory / "train"), device=runtime["device"])
            tic = time.perf_counter()
            runner.learn(num_learning_iterations=p["max_iterations"], init_at_random_ep_len=True)
            wall = time.perf_counter() - tic
            env.recorder = None
            decisions = count*p["num_steps_per_env"]*p["max_iterations"]
            result["training"] = {"iterations": p["max_iterations"], "decisions": decisions, "wall_seconds": wall,
                                  "env_steps_per_second": decisions*config["decision_control_steps"]/wall}
            write_json(evidence, result)
            actor = runner.get_inference_policy(device=runtime["device"])
            with torch.inference_mode():
                result["learned_eval"] = evaluate(env, actor, config["seed"] + 1)
            result["status"] = "passed_ppo_training"
            write_json(evidence, result)
        except BaseException as error:
            result.update(status="failed", error=str(error) or repr(error), error_type=type(error).__name__)
            write_json(evidence, result)
            (directory / "ppo-error.log").write_text(traceback.format_exc())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config, profile, asset, runtime = load_config(args.config)
    contract, _ = trial_contract(profile)
    if args.dry_run:
        p = config["ppo"]
        decisions = config["num_envs"]*p["num_steps_per_env"]*p["max_iterations"]
        print(json.dumps({"status": "local_plan_only", "num_envs": config["num_envs"], "iterations": p["max_iterations"],
                          "decisions": decisions, "env_steps": decisions*config["decision_control_steps"],
                          "policy_role": contract["policy_role"], "training": True}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(runtime))
        run_worker(config, profile, asset, runtime, gr00t_models(contract), args.run_dir)
        return 0
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-ppo")
    directory.mkdir(parents=True)
    for name, value in (("ppo-config.json", config), ("profile.json", profile), ("runtime-config.json", runtime),
                        ("controller-contract.json", contract)):
        write_json(directory / name, value)
    gr00t_models(contract)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--worker", "--run-dir", str(directory)]
    code = run_stage(command, directory / "ppo.log", config["stage_timeout_seconds"])
    evidence_path = directory / "ppo-worker.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {"status": "missing"}

    def brief(name):
        e = evidence.get(name) or {}
        return {k: e.get(k) for k in ("success_rate", "progress_m", "median_success_seconds")} if e else None

    summary = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code, "run_dir": str(directory),
               "scripted_eval": brief("scripted_eval"), "learned_eval": brief("learned_eval"),
               "training": evidence.get("training"), "error": evidence.get("error")}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "passed_ppo_training" else 1


if __name__ == "__main__":
    raise SystemExit(main())
