"""Goal-pose PPO policies in random furnished rooms over frozen GR00T Walk/Balance: approach and carry.

Each environment has its own rectangular room with 1-3 furniture boxes (static colliders authored before reset). Every
0.1 s the policy sets the GR00T walk command (forward, lateral, yaw rate) from the goal pose and look-ahead points of an
A* path in its own heading frame, plus local obstacle distances. Paths end with a straight staging segment along the goal
heading. With residual_scale > 0 the policy output is a residual added to the scripted docking action. init_checkpoint
continues training from a saved model (weights and optimizer) for another max_iterations.

- `approach`: no load. The cabinet stands on the floor; the goal is the pre-grasp pose (robot root `reach` metres behind
  the cabinet centre, facing it). Success: within the position/yaw tolerance and nearly stopped, without pushing it.
- `carry`: the cabinet is held (palm attachment) from a crouch start, lifted by arm IK under Balance, then carried so that
  the robot reaches the pre-grasp pose of a target cabinet pose. Success as above.

A scripted pursuit-and-docking controller is evaluated on the same seeded starts before training, the policy after it.
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

import isaac_g1_carry as carry
import isaac_g1_lift as lift
import isaac_g1_ppo as ppo
import isaac_g1_rl as rl
import room_geometry as rg
from isaac_g1_wbc import ROOT, load_plan, trial_contract
from isaac_preflight import check_versions, run_stage, write_json


DEFAULT_CONFIG = ROOT / "configs/isaac_g1_room_approach.json"
ACTION_KEYS = ("forward_mps", "lateral_mps", "yaw_rate_rps")
FAILURES = ("invalid_action", "fallen", "wall_contact", "object_tipped", "object_grounded", "object_pushed")
OUTCOMES = (*FAILURES, "success", "time_out")
LOOKAHEAD_M = (0.3, 0.6, 1.0)
RAY_RADIUS_M = 0.45
OBSERVATIONS = 34


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"push_profile", "task", "object", "mass_kg", "arm_lift_m", "lift_prefix_seconds", "room", "goal",
                       "num_envs", "env_spacing_m", "decision_control_steps", "episode_seconds", "action_bounds",
                       "residual_scale", "termination", "reward", "ppo", "init_checkpoint", "seed", "stage_timeout_seconds",
                       "scope"}:
        raise ValueError("Unreviewed room configuration keys")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    def interval(value, low, high):
        return isinstance(value, list) and len(value) == 2 and all(number(v, low, high) for v in value) and value[0] < value[1]

    if config["task"] not in ("approach", "carry"):
        raise ValueError("task must be approach or carry")
    for key, (low, high) in {"num_envs": (1, 4096), "decision_control_steps": (1, 50), "seed": (0, 2**31-1),
                             "stage_timeout_seconds": (60, 10800)}.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid {key}")
    if not number(config["episode_seconds"], 5.0, 120.0) or not number(config["residual_scale"], 0.0, 1.0):
        raise ValueError("Invalid episode length or residual scale")
    profile, asset, runtime = load_plan(ROOT / config["push_profile"])
    if (profile.get("functional_case") != "push" or profile.get("arm_hold") != "implicit" or
            profile.get("initialization", {}).get("mode") != "collision_clearance"):
        raise ValueError("Room tasks start from the reviewed implicit-arm crouch-start push profile")
    o = config["object"]
    if (set(o) != {"size_m", "friction", "start_clearance_m", "near_face_ahead_of_palm_m"} or
            not isinstance(o["size_m"], list) or len(o["size_m"]) != 3 or not all(number(v, 0.1, 1.5) for v in o["size_m"]) or
            not number(o["friction"], 0.05, 1.5) or not number(o["start_clearance_m"], 0.0, 0.02) or
            not number(o["near_face_ahead_of_palm_m"], 0.0, 0.1)):
        raise ValueError("Invalid object")
    # The carry policy's measured range (20261101T024024, 100% on held-out corridors) is 2-4 kg.
    if not interval(config["mass_kg"], 0.5, 4.0) or not number(config["arm_lift_m"], 0.04, 0.12):
        raise ValueError("Masses must stay within 4 kg and the arm lift within 4-12 cm")
    pre = config["lift_prefix_seconds"]
    if set(pre) != {"settle", "lift", "hold"} or not all(number(v, 0.1, 5.0) for v in pre.values()):
        raise ValueError("Invalid lift prefix")
    r = config["room"]
    if (set(r) != {"width_m", "depth_m", "obstacles", "obstacle_size_m", "obstacle_gap_m", "grid_m", "inflation_m",
                   "start_clearance_m", "wall_thickness_m", "wall_height_m"} or not interval(r["width_m"], 2.5, 8.0) or
            not interval(r["depth_m"], 2.5, 8.0) or not isinstance(r["obstacles"], list) or len(r["obstacles"]) != 2 or
            not all(type(v) is int and 0 <= v <= 5 for v in r["obstacles"]) or r["obstacles"][0] > r["obstacles"][1] or
            not isinstance(r["obstacle_size_m"], list) or len(r["obstacle_size_m"]) != 3 or
            not all(interval(v, 0.1, 3.0) for v in r["obstacle_size_m"]) or not number(r["obstacle_gap_m"], 0.3, 2.0) or
            not number(r["grid_m"], 0.05, 0.2) or not number(r["inflation_m"], 0.2, 0.8) or
            not number(r["start_clearance_m"], r["inflation_m"], 1.5) or
            not number(r["wall_thickness_m"], 0.02, 0.5) or not number(r["wall_height_m"], 0.5, 3.0)):
        raise ValueError("Invalid room settings")
    if max(r["width_m"][1], r["depth_m"][1]) + 2*r["wall_thickness_m"] + 1.0 > config["env_spacing_m"]:
        raise ValueError("Rooms overlap at this environment spacing")
    g = config["goal"]
    if (set(g) != {"min_distance_m", "max_distance_m", "position_tolerance_m", "yaw_tolerance_deg", "stop_speed_mps",
                   "object_inflation_m", "object_margin_m", "staging_m"} or not number(g["min_distance_m"], 0.3, 5.0) or
            not number(g["max_distance_m"], g["min_distance_m"], 8.0) or not number(g["position_tolerance_m"], 0.01, 0.3) or
            not number(g["yaw_tolerance_deg"], 1.0, 30.0) or not number(g["stop_speed_mps"], 0.01, 0.2) or
            not number(g["object_inflation_m"], 0.0, 0.5) or not number(g["object_margin_m"], 0.0, 0.5) or
            not number(g["staging_m"], 0.1, 1.0)):
        raise ValueError("Invalid goal settings")
    b = config["action_bounds"]
    # Each range contains zero so the policy can stop: a command norm of at most 0.05 selects GR00T Balance.
    if (set(b) != set(ACTION_KEYS) or not interval(b["forward_mps"], -0.2, 0.4) or not interval(b["lateral_mps"], -0.2, 0.2) or
            not interval(b["yaw_rate_rps"], -0.4, 0.4) or not all(v[0] <= 0.0 <= v[1] for v in b.values())):
        raise ValueError("Invalid action bounds")
    t = config["termination"]
    if (set(t) != {"robot_link_radius_m", "wall_contact_m", "object_tilt_rad", "object_floor_clearance_m", "object_push_m"} or
            not number(t["robot_link_radius_m"], 0.0, 0.3) or not number(t["wall_contact_m"], 0.0, 0.05) or
            not number(t["object_tilt_rad"], 0.05, 1.0) or not number(t["object_floor_clearance_m"], 0.0, 0.05) or
            not number(t["object_push_m"], 0.005, 0.2)):
        raise ValueError("Invalid termination limits")
    w = config["reward"]
    if (set(w) != {"progress", "pose", "wall", "safe_clearance_m", "action_rate", "speed_limit_mps", "overspeed", "time",
                   "failure", "success"} or not all(number(v, 0.0, 100.0) for v in w.values()) or
            not number(w["safe_clearance_m"], 0.01, 0.5) or not number(w["speed_limit_mps"], 0.1, 1.0)):
        raise ValueError("Invalid reward weights")
    if not ppo.valid_ppo_settings(config["ppo"]):
        raise ValueError("Invalid PPO settings")
    start = config["init_checkpoint"]
    if start is not None and (not isinstance(start, str) or not start.startswith("outputs/isaac/") or
                              not start.endswith(".pt") or ".." in Path(start).parts):
        raise ValueError("init_checkpoint must be null or a .pt path under outputs/isaac/")
    return config, profile, asset, runtime


def footprint_clear(room, centre, yaw, size_xy, margin):
    """True when a rotated footprint's corners and edge midpoints keep `margin` from walls and furniture."""
    import numpy as np
    c, s = np.cos(yaw), np.sin(yaw)
    offsets = np.array([[x, y] for x in (-0.5, 0.0, 0.5) for y in (-0.5, 0.0, 0.5) if x or y])*np.asarray(size_xy)
    points = np.asarray(centre) + offsets @ np.array([[c, s], [-s, c]])
    return bool(rg.obstacle_distance(points, room).min() >= margin)


def sample_episode(task, room, settings, goal, reach, object_size, rng, tries=200):
    """Start pose, cabinet pose, goal root pose and staged A* path for one episode (None if no valid layout is found).

    approach: the cabinet stands on the floor and the goal is its pre-grasp pose; carry: the robot holds the cabinet at
    `reach` metres ahead and the goal is the pre-grasp pose of a sampled target cabinet pose. The start keeps
    start_clearance_m from walls and furniture (the fingers reach 0.40 m from the root and the settle drifts it). The
    approach path keeps the room inflation from the cabinet too (run 20261101T085012 pushed it only on paths passing
    within 0.35 m); only its final straight segment uses the smaller object_inflation_m.
    """
    import numpy as np
    w, d = room["width_m"]/2, room["depth_m"]/2
    size_xy = object_size[:2]
    static = rg.occupancy(room, settings["grid_m"], settings["inflation_m"])

    def free_point():
        return rng.uniform([-w, -d], [w, d])

    def root_clear(point, clearance=settings["inflation_m"]):
        return float(rg.obstacle_distance(point, room)) >= clearance

    for _ in range(tries):
        if task == "approach":
            cabinet = np.r_[free_point(), rng.uniform(-np.pi, np.pi)]
            target = rg.pregrasp_pose(cabinet[:2], cabinet[2], reach)
            start = np.r_[free_point(), rng.uniform(-np.pi, np.pi)]
            if (not footprint_clear(room, cabinet[:2], cabinet[2], size_xy, goal["object_margin_m"]) or not root_clear(target[:2]) or
                    not root_clear(start[:2], settings["start_clearance_m"]) or
                    rg.rotated_distance(start[:2], cabinet[:2], size_xy, cabinet[2]) < 0.6):
                continue
            grid = rg.occupancy(room, settings["grid_m"], settings["inflation_m"],
                                (cabinet[:2], size_xy, cabinet[2], settings["inflation_m"]))
            final = rg.occupancy(room, settings["grid_m"], settings["inflation_m"],
                                 (cabinet[:2], size_xy, cabinet[2], goal["object_inflation_m"]))
        else:
            start = np.r_[free_point(), rng.uniform(-np.pi, np.pi)]
            held = rg.pregrasp_pose(start[:2], start[2] + np.pi, reach)      # reach metres ahead of the root
            placed = np.r_[free_point(), rng.uniform(-np.pi, np.pi)]
            target = rg.pregrasp_pose(placed[:2], placed[2], reach)
            if (not root_clear(start[:2], settings["start_clearance_m"]) or not root_clear(target[:2]) or
                    not footprint_clear(room, held[:2], start[2], size_xy, goal["object_margin_m"]) or
                    not footprint_clear(room, placed[:2], placed[2], size_xy, goal["object_margin_m"])):
                continue
            cabinet, grid, final = np.r_[held[:2], start[2]], static, None
        distance = float(np.linalg.norm(target[:2] - start[:2]))
        if not goal["min_distance_m"] <= distance <= goal["max_distance_m"]:
            continue
        path = rg.staged_path(grid, start[:2], target, goal["staging_m"], final)
        if path is not None:
            return {"start": start, "cabinet": cabinet, "goal": target, "path": path,
                    "placed": placed if task == "carry" else None}
    return None


def feasible_room(config, rng, reach=0.411, probes=10):
    """A random room in which `probes` consecutive episode samplings all succeed; with the start clearance and staging
    segment a few percent of random rooms leave no valid layout, and resets would then fail during training."""
    while True:
        candidate = rg.sample_room(config["room"], rng)
        if all(sample_episode(config["task"], candidate, config["room"], config["goal"], reach, config["object"]["size_m"], rng)
               is not None for _ in range(probes)):
            return candidate


def wrap(angle):
    import numpy as np
    return np.arctan2(np.sin(angle), np.cos(angle))


def room_state(robot_pose, robot_velocity, object_pose, body_xyz, origins, padded, goals, paths, object_size, link_radius):
    """Goal pose and path in the robot heading frame, ray distances, clearances and cabinet state per environment."""
    import numpy as np
    from g1_wbc import quat_xyzw_matrices
    n = len(robot_pose)
    widths, depths, boxes = padded
    xy = robot_pose[:, :2] - origins[:, :2]
    heading = ppo.yaw_of(robot_pose[:, 3:])
    c, s = np.cos(heading), np.sin(heading)

    def local(v):
        return np.stack([c[:, None]*v[..., 0] + s[:, None]*v[..., 1], -s[:, None]*v[..., 0] + c[:, None]*v[..., 1]], axis=-1)

    remaining, ahead = np.zeros(n), np.zeros((n, len(LOOKAHEAD_M), 2))
    for i in range(n):
        remaining[i], ahead[i] = rg.path_lookup(paths[i], xy[i], LOOKAHEAD_M)
    goal_local = local((goals[:, :2] - xy)[:, None])[:, 0]
    yaw_error = wrap(goals[:, 2] - heading)
    angles = heading[:, None] + np.arange(8)*np.pi/4
    rays = xy[:, None] + RAY_RADIUS_M*np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    corners = carry.box_corners(object_pose, object_size)
    rotation = quat_xyzw_matrices(object_pose[:, 3:])
    return {"goal_local": goal_local, "yaw_error": yaw_error, "position_error": np.linalg.norm(goals[:, :2] - xy, axis=1),
            "remaining_m": remaining, "lookahead": local(ahead - xy[:, None]).reshape(n, -1),
            "rays": np.clip(rg.batched_distance(rays, widths, depths, boxes), -0.5, 1.5),
            "robot_clearance_m": rg.batched_distance(body_xyz[..., :2] - origins[:, None, :2], widths, depths, boxes).min(axis=1) - link_radius,
            "object_clearance_m": rg.batched_distance(corners[..., :2] - origins[:, None, :2], widths, depths, boxes).min(axis=1),
            "bottom_m": corners[..., 2].min(axis=1), "tilt_rad": np.arccos(np.clip(rotation[:, 2, 2], -1.0, 1.0)),
            "object_up": local(rotation[:, None, :2, 2])[:, 0], "object_xy": object_pose[:, :2] - origins[:, :2],
            "velocity": np.c_[local(robot_velocity[:, None, :2])[:, 0], robot_velocity[:, 5]]}


def task_observation(state, carrying, gravity_b, last_action, time_fraction, prefix):
    """Policy input (34 values); cabinet clearance and state are zero when nothing is carried."""
    import numpy as np
    held = np.asarray(carrying, dtype=float)[:, None]
    return np.c_[state["goal_local"], np.sin(state["yaw_error"]), np.cos(state["yaw_error"]), state["lookahead"],
                 state["rays"], state["robot_clearance_m"], held[:, 0]*np.clip(state["object_clearance_m"], -0.5, 1.5),
                 state["velocity"], gravity_b, held*np.c_[state["bottom_m"], state["object_up"]], last_action,
                 time_fraction, prefix].astype(np.float32)


def task_step(state, previous_remaining, prefix, carrying, pushed_m, speed, actions, last_actions, low_level, config,
              decision_dt):
    """Reward, termination and outcome index (into OUTCOMES, -1 while running) for one high-level decision."""
    import numpy as np
    t, w, g = config["termination"], config["reward"], config["goal"]
    clearance = np.where(carrying, np.minimum(state["robot_clearance_m"], state["object_clearance_m"]), state["robot_clearance_m"])
    flags = np.stack([low_level["invalid_action"], low_level["fallen"], clearance < t["wall_contact_m"],
                      carrying & (state["tilt_rad"] > t["object_tilt_rad"]),
                      carrying & ~prefix & (state["bottom_m"] < t["object_floor_clearance_m"]),
                      ~carrying & (pushed_m > t["object_push_m"])])
    failed = flags.any(axis=0)
    arrived = ((state["position_error"] <= g["position_tolerance_m"]) &
               (np.abs(state["yaw_error"]) <= np.radians(g["yaw_tolerance_deg"])) & (speed <= g["stop_speed_mps"]))
    success = ~failed & ~prefix & arrived
    outcome = np.where(failed, np.argmax(flags, axis=0), np.where(success, OUTCOMES.index("success"), -1))
    near = state["position_error"] < 0.5
    shaping = (w["progress"]*(previous_remaining - state["remaining_m"])/decision_dt
               - w["pose"]*near*(state["position_error"] + np.abs(state["yaw_error"]))
               - w["wall"]*np.clip(1.0 - clearance/w["safe_clearance_m"], 0.0, 1.0)
               - w["action_rate"]*np.sum((actions - last_actions)**2, axis=1)
               - w["overspeed"]*np.maximum(speed - w["speed_limit_mps"], 0.0) - w["time"])
    reward = np.where(prefix, 0.0, shaping) - w["failure"]*failed + w["success"]*success
    return reward, failed | success, outcome


def bound_arrays(bounds):
    import numpy as np
    return tuple(np.array([bounds[k][i] for k in ACTION_KEYS], dtype=float) for i in (0, 1))


def action_commands(actions, bounds):
    """Map normalized actions in [-1, 1] to GR00T commands [vx, vy, yaw rate]."""
    import numpy as np
    low, high = bound_arrays(bounds)
    return low + (np.clip(np.asarray(actions, dtype=float), -1.0, 1.0) + 1.0)/2*(high - low)


def normalized_action(values, bounds):
    import numpy as np
    low, high = bound_arrays(bounds)
    return 2*(np.asarray(values, dtype=float) - low)/(high - low) - 1.0


def docking_actions(policy_obs, config):
    """Scripted baseline: follow the path at 0.2 m/s toward the 0.6 m look-ahead point, then within 0.5 m of the goal
    correct position with forward/lateral steps and heading with yaw, and stop (Balance) inside 80% of the tolerance.
    Corrections are at least 0.06 in command norm, because GR00T switches to Balance at 0.05 and the robot would stall."""
    import numpy as np
    g, b = config["goal"], config["action_bounds"]
    dx, dy = policy_obs[:, 0], policy_obs[:, 1]
    yaw_error = np.arctan2(policy_obs[:, 2], policy_obs[:, 3])
    bearing = np.arctan2(policy_obs[:, 7], policy_obs[:, 6])
    distance = np.hypot(dx, dy)
    far = distance > 0.5
    command = np.where(far[:, None],
                       np.c_[0.2*np.clip(np.cos(bearing), 0.0, 1.0), np.zeros(len(dx)), np.clip(bearing, *b["yaw_rate_rps"])],
                       np.c_[np.clip(0.8*dx, *b["forward_mps"]), np.clip(0.8*dy, *b["lateral_mps"]),
                             np.clip(1.2*yaw_error, *b["yaw_rate_rps"])])
    norm = np.linalg.norm(command, axis=1)
    small = ~far & (norm > 1e-9) & (norm < 0.06)
    command[small] *= (0.06/norm[small])[:, None]
    done = (distance < 0.8*g["position_tolerance_m"]) & (np.abs(yaw_error) < 0.8*np.radians(g["yaw_tolerance_deg"]))
    command[done] = 0.0
    return normalized_action(np.clip(command, *bound_arrays(b)), b)


def scripted_actor(config):
    """Scripted docking as an actor: a zero residual when the task adds the docking action itself, else the action."""
    import numpy as np
    import torch

    def act(obs):
        policy = obs["policy"].detach().cpu().numpy()
        action = np.zeros((len(policy), len(ACTION_KEYS))) if config["residual_scale"] > 0 else docking_actions(policy, config)
        return torch.as_tensor(action, dtype=torch.float32)
    return act


class RoomTaskEnv:
    """RSL-RL vectorized environment over the prepared scene with per-environment rooms (approach or carry)."""

    def __init__(self, scene, policy, contract, profile, asset, runtime, config, rooms, masses, object_in_root, arms,
                 observations):
        import numpy as np
        from g1_wbc import BatchHistory
        self.s, self.policy, self.profile, self.asset, self.config = scene, policy, profile, asset, config
        self.rooms, self.masses, self.object_in_root, self.observations = rooms, np.asarray(masses), object_in_root, observations
        self.padded = rg.pad_rooms(rooms, max(1, config["room"]["obstacles"][1]))
        self.arm_ids, self.arm_start, self.arm_lift = arms
        self.carrying = config["task"] == "carry"
        self.reach = float(object_in_root[0, 3])
        self.num_envs, self.num_actions, self.device, self.cfg = len(scene.origins), len(ACTION_KEYS), runtime["device"], config
        control_dt = asset["control_decimation"]*asset["physics_dt"]
        self.decision_dt = config["decision_control_steps"]*control_dt
        self.max_episode_length = int(round(config["episode_seconds"]/self.decision_dt))
        pre = config["lift_prefix_seconds"]
        self.settle_steps, self.lift_steps = (int(round(pre[k]/control_dt)) for k in ("settle", "lift"))
        prefix = sum(pre.values()) if self.carrying else pre["settle"]
        self.prefix_decisions = int(round(prefix/self.decision_dt))
        self.history = BatchHistory(contract, self.num_envs)
        self.targets = np.tile(scene.defaults, (self.num_envs, 1))
        n = self.num_envs
        self.last_action, self.last_command = np.zeros((n, 3)), np.zeros((n, 3))
        self.length, self.since_reset = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        self.goals, self.paths, self.spawned_xy = np.zeros((n, 3)), [None]*n, np.zeros((n, 2))
        self.previous_remaining, self.start_remaining = np.zeros(n), np.zeros(n)
        self.rng = np.random.default_rng(config["seed"])
        self.recorder, self.last_completed, self.last_task, self.reset_on_done = None, {}, {}, True
        self.reset_envs(np.arange(n))
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

    def episode(self, i, rng):
        """Episode layout for environment i (up to 5 x 200 sampling tries)."""
        for _ in range(5):
            episode = sample_episode(self.config["task"], self.rooms[i], self.config["room"], self.config["goal"], self.reach,
                                     self.config["object"]["size_m"], rng)
            if episode is not None:
                return episode
        raise RuntimeError(f"No valid episode layout in room {i}")

    def reset_envs(self, ids, rng=None):
        """Lay out each environment's episode; the robot starts at its crouch-start height at the start pose."""
        import numpy as np
        from g1_wbc import axis_rotation, matrix_quat_xyzw, pose_matrix
        s, o = self.s, self.config["object"]
        rng = self.rng if rng is None else rng
        n, env_ids = len(ids), ids.tolist()
        robot, cabinet = np.zeros((n, 7)), np.zeros((n, 7))
        for j, i in enumerate(ids):
            episode = self.episode(i, rng)
            root = pose_matrix(s.root[i])
            root[:3, :3] = axis_rotation([0, 0, 1], episode["start"][2]) @ root[:3, :3]
            root[:2, 3] = s.origins[i, :2] + episode["start"][:2]
            robot[j] = np.r_[root[:3, 3], matrix_quat_xyzw(root[:3, :3])]
            if self.carrying:
                held = root @ self.object_in_root
                cabinet[j] = np.r_[held[:3, 3], matrix_quat_xyzw(held[:3, :3])]
            else:
                yaw = episode["cabinet"][2]
                cabinet[j] = np.r_[s.origins[i, :2] + episode["cabinet"][:2], o["size_m"][2]/2 + 1e-3, 0.0, 0.0,
                                   np.sin(yaw/2), np.cos(yaw/2)]
            self.goals[i], self.paths[i] = episode["goal"], episode["path"]
            self.spawned_xy[i] = cabinet[j, :2] - s.origins[i, :2]
            self.start_remaining[i] = self.previous_remaining[i] = rg.path_lookup(episode["path"], episode["start"][:2], [0.0])[0]
        s.scene.reset(env_ids)
        s.robot.write_root_pose_to_sim_index(root_pose=s.device(robot), env_ids=env_ids)
        s.robot.write_root_velocity_to_sim_index(root_velocity=s.device(np.zeros((n, 6))), env_ids=env_ids)
        s.robot.write_joint_position_to_sim_index(position=s.device(np.tile(s.defaults, (n, 1))), env_ids=env_ids)
        s.robot.write_joint_velocity_to_sim_index(velocity=s.device(np.zeros((n, 29))), env_ids=env_ids)
        s.box.write_root_pose_to_sim_index(root_pose=s.device(cabinet), env_ids=env_ids)
        s.box.write_root_velocity_to_sim_index(root_velocity=s.device(np.zeros((n, 6))), env_ids=env_ids)
        self.history.reset(ids)
        self.targets[ids] = s.defaults
        s.robot.set_joint_position_target_index(target=s.device(self.targets))
        self.last_action[ids] = 0.0
        self.length[ids] = 0
        self.since_reset[ids] = 0

    def task_state(self):
        from g1_wbc import finite
        s, n = self.s, self.num_envs
        robot = finite(s.batch(s.robot.data.root_link_pose_w), (n, 7))
        cabinet = finite(s.batch(s.box.data.root_link_pose_w), (n, 7))
        bodies = finite(s.batch(s.robot.data.body_link_pose_w), (n, len(s.bodies), 7))[..., :3]
        velocity = finite(s.batch(s.robot.data.root_link_vel_w), (n, 6))
        return robot, room_state(robot, velocity, cabinet, bodies, s.origins, self.padded, self.goals, self.paths,
                                 self.config["object"]["size_m"], self.config["termination"]["robot_link_radius_m"])

    def observe(self):
        import numpy as np
        import torch
        from g1_wbc import quat_xyzw_matrices
        robot, state = self.task_state()
        gravity = -(quat_xyzw_matrices(robot[:, 3:]) @ self.s.rotation)[:, 2, :]
        task = task_observation(state, np.full(self.num_envs, self.carrying), gravity, self.last_action,
                                self.length[:, None]/self.max_episode_length, (self.since_reset < self.prefix_decisions)[:, None])
        low, high = self.config["mass_kg"]
        privileged = ((self.masses - low)/(high - low))[:, None].astype(np.float32)

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        return self.observations({"policy": tensor(task), "critic": tensor(np.c_[task, privileged])})

    def control(self, commands, rpys, arms=None):
        """One control step: physics substeps, then GR00T lower-body targets for `commands` (and arm targets if given).

        Returns the root pose and velocity read after the substeps and per-environment invalid-action and fallen flags.
        """
        import numpy as np
        from g1_wbc import finite, quat_xyzw_matrices
        s, n = self.s, self.num_envs
        for _ in range(self.asset["control_decimation"]):
            s.scene.write_data_to_sim()
            s.sim.step(render=False)
            s.scene.update(self.asset["physics_dt"])
        pose = finite(s.batch(s.robot.data.root_link_pose_w), (n, 7))
        velocity = finite(s.batch(s.robot.data.root_link_vel_w), (n, 6))
        observation = self.history.observe(s.batch(s.robot.data.joint_pos)[:, s.indices],
                                           s.batch(s.robot.data.joint_vel)[:, s.indices], pose[:, 3:],
                                           velocity[:, 3:], s.rotation, commands, rpys)
        lower_targets, _, invalid = self.history.targets(self.policy.batch(observation, commands), s.limits)
        self.targets[:, s.lower_indices] = lower_targets
        if arms is not None:
            self.targets[:, self.arm_ids] = arms
        s.robot.set_joint_position_target_index(target=s.device(self.targets))
        gravity = -(quat_xyzw_matrices(pose[:, 3:]) @ s.rotation)[:, 2, :]
        fallen = ((pose[:, 2] < self.profile["minimum_base_height_m"]) |
                  (np.arccos(np.clip(-gravity[:, 2], -1, 1)) > self.profile["maximum_tilt_rad"]))
        return pose, velocity, invalid, fallen

    def step(self, actions):
        import numpy as np
        import torch
        n = self.num_envs
        action = np.clip(np.asarray(actions.detach().cpu().numpy(), dtype=float), -1.0, 1.0)
        if self.config["residual_scale"] > 0:
            # The policy corrects the scripted docking action computed from the same observation.
            action = np.clip(docking_actions(self.obs["policy"].detach().cpu().numpy(), self.config) +
                             self.config["residual_scale"]*action, -1.0, 1.0)
        prefix = self.since_reset < self.prefix_decisions
        commands = action_commands(action, self.config["action_bounds"])
        commands[prefix] = 0.0
        self.last_command = commands.copy()
        rpys = np.zeros((n, 3))
        low_level = {"invalid_action": np.zeros(n, dtype=bool), "fallen": np.zeros(n, dtype=bool)}
        steps, speed = self.config["decision_control_steps"], np.zeros(n)
        for k in range(steps):
            arms = None
            if self.carrying:
                fraction = np.clip((self.since_reset*steps + k + 1 - self.settle_steps)/self.lift_steps, 0.0, 1.0)[:, None]
                arms = self.arm_start + fraction*(self.arm_lift - self.arm_start)
            _, velocity, invalid, fallen = self.control(commands, rpys, arms)
            speed += np.linalg.norm(velocity[:, :2], axis=1)/steps
            low_level["invalid_action"] |= invalid
            low_level["fallen"] |= fallen
        state = self.task_state()[1]
        pushed = np.linalg.norm(state["object_xy"] - self.spawned_xy, axis=1)
        carrying = np.full(n, self.carrying)
        reward, terminated, outcome = task_step(state, self.previous_remaining, prefix, carrying, pushed, speed, action,
                                                self.last_action, low_level, self.config, self.decision_dt)
        self.last_task = {"position_error_m": state["position_error"], "yaw_error_rad": state["yaw_error"],
                          "clearance_m": np.where(carrying, np.minimum(state["robot_clearance_m"], state["object_clearance_m"]),
                                                  state["robot_clearance_m"]),
                          "pushed_m": pushed, "speed_mps": speed, "active": ~prefix}
        self.previous_remaining = state["remaining_m"].copy()
        self.last_action = action
        self.length += 1
        self.since_reset += 1
        time_out = ~terminated & (self.length >= self.max_episode_length)
        ids = np.flatnonzero(terminated | time_out)
        self.last_completed = {int(i): {"outcome": OUTCOMES[outcome[i]] if terminated[i] else "time_out",
                                        "progress_m": float(self.start_remaining[i] - state["remaining_m"][i]),
                                        "seconds": float(self.length[i]*self.decision_dt),
                                        "position_error_m": float(state["position_error"][i]),
                                        "yaw_error_deg": float(np.degrees(abs(state["yaw_error"][i])))}
                               for i in ids}
        extras = {"time_outs": torch.as_tensor(time_out, device=self.device)}
        if len(ids):
            records = list(self.last_completed.values())
            extras["log"] = {f"/episode/{name}": torch.as_tensor([r["outcome"] == name for r in records], dtype=torch.float32)
                             for name in ("success", "wall_contact", "object_pushed", "time_out")}
            if self.reset_on_done:
                self.reset_envs(ids)
        if self.recorder is not None:
            self.recorder(self.last_completed, n)
        self.obs = self.observe()
        return (self.obs, torch.as_tensor(reward, dtype=torch.float32, device=self.device),
                torch.as_tensor(terminated | time_out, dtype=torch.long, device=self.device), extras)


def evaluate(env, act, seed):
    """One episode per environment from seeded layouts; final pose errors, clearance, speed and commands per episode."""
    import numpy as np
    n = env.num_envs
    env.reset_envs(np.arange(n), np.random.default_rng(seed))
    env.obs = env.observe()
    first, commands, speeds = {}, [[] for _ in range(n)], [[] for _ in range(n)]
    clearance, pushed = np.full(n, np.inf), np.zeros(n)
    for _ in range(env.max_episode_length):
        active = [i for i in range(n) if i not in first]
        env.step(act(env.obs))
        for i in active:
            clearance[i] = min(clearance[i], env.last_task["clearance_m"][i])
            pushed[i] = max(pushed[i], env.last_task["pushed_m"][i])
            if env.last_task["active"][i]:
                commands[i].append(env.last_command[i])
                speeds[i].append(env.last_task["speed_mps"][i])
        for i, record in env.last_completed.items():
            first.setdefault(i, record)
        if len(first) == n:
            break
    records = []
    for i in range(n):
        c = np.asarray(commands[i], dtype=float).reshape(-1, 3)
        records.append({"obstacles": len(env.rooms[i]["boxes"]), "mass_kg": float(env.masses[i]), **first[i],
                        "min_clearance_m": float(clearance[i]), "max_pushed_m": float(pushed[i]),
                        "speed_mps": {"mean": float(np.mean(speeds[i])), "max": float(np.max(speeds[i]))} if speeds[i] else None,
                        "commands": {"decisions": len(c),
                                     **({"forward_mean_mps": float(c[:, 0].mean()), "lateral_abs_mean_mps": float(np.abs(c[:, 1]).mean()),
                                         "yaw_rate_abs_mean_rps": float(np.abs(c[:, 2]).mean()),
                                         "balance_fraction": float((np.linalg.norm(c, axis=1) <= 0.05).mean())} if len(c) else {})}})
    successes = [r for r in records if r["outcome"] == "success"]
    samples = np.concatenate([np.asarray(v, dtype=float) for v in speeds if v]) if any(speeds) else np.zeros(0)
    return {**ppo.episode_summary(records, OUTCOMES), "seed": seed,
            "final_error_at_success": {"position_m_median": float(np.median([r["position_error_m"] for r in successes])),
                                       "yaw_deg_median": float(np.median([r["yaw_error_deg"] for r in successes]))} if successes else None,
            "walking_speed_mps": {"mean": float(samples.mean()), "p95": float(np.percentile(samples, 95)),
                                  "max": float(samples.max())} if len(samples) else None,
            "per_env": records}


def run_worker(config, profile, asset, runtime, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp
    from g1_wbc import finite, matrix_quat_xyzw, pose_matrix

    contract, reference = trial_contract(profile)
    policy = ppo.gr00t_policy(contract, model_paths)
    count, o, carrying = config["num_envs"], config["object"], config["task"] == "carry"
    torch.manual_seed(config["seed"])
    wp.config.enable_backward = False
    rng = np.random.default_rng(config["seed"])
    rooms = [feasible_room(config, rng) for _ in range(count)]
    masses = rng.uniform(*config["mass_kg"], count)
    from isaaclab.app import launch_simulation

    scene_profile, built = carry.carry_scene(config, profile, asset, runtime, contract, reference, masses, num_envs=count,
                                             env_spacing=config["env_spacing_m"], with_camera=False)
    base = lift.base_pose(reference, asset, contract)
    frames = lift.attachment_frames(reference, base, o, contract["height_command"])
    joints = [n for side in lift.SIDES for n in lift.arm_joints(reference, side)]
    lifted = {n: v for side in lift.SIDES for n, v in lift.arm_pose(reference, base, side, config["arm_lift_m"]).items()}
    evidence = directory / "room-worker.json"
    result = {"status": "running", "task": config["task"], "num_envs": count, "masses_kg": masses.tolist(),
              "rooms": [{"width_m": r["width_m"], "depth_m": r["depth_m"], "boxes": r["boxes"].tolist()} for r in rooms]}
    write_json(evidence, result)
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "device": runtime["device"]}):
        try:
            authored = {}

            def edits(stage, robot, box, env):
                i = int(env.rsplit("_", 1)[1])
                # Both tasks filter hand/wrist collisions with the cabinet (the palms meet its side faces at the pre-grasp
                # pose); the joints are enabled only when carrying.
                lift.apply_hand_attachment(stage, robot, box, env, frames, carrying)
                authored[env] = carry.apply_corridor(stage, env, rg.wall_boxes(rooms[i], config["room"]["wall_thickness_m"],
                                                                                config["room"]["wall_height_m"]))

            s = rl.prepare_push_scene(built, count, scene_profile, asset, runtime, contract, reference, directory, result,
                                      evidence, box_properties=(masses, np.full(count, float(o["friction"]))),
                                      extra_edits=edits, check_box_start=False)
            result["authored_rooms"] = len(authored)
            # Cabinet pose that both palm joints expect at the crouch start, relative to the robot root (both tasks use it:
            # carry to hold the cabinet, approach to define the pre-grasp pose).
            palm_ids = {side: s.bodies.index(f"{side}_hand_palm_link") for side in lift.SIDES}
            poses = finite(s.batch(s.robot.data.body_link_pose_w), (count, len(s.bodies), 7))
            held = pose_matrix(poses[0, palm_ids["left"]]) @ frames["palm_object"]["left"]
            object_in_root = np.linalg.inv(pose_matrix(s.root[0])) @ held
            result["object_in_root"] = object_in_root.tolist()
            write_json(evidence, result)

            from tensordict import TensorDict
            arms = ([s.names.index(n) for n in joints], np.zeros(len(joints)), np.array([lifted[n] for n in joints]))
            env = RoomTaskEnv(s, policy, contract, profile, asset, runtime, config, rooms, masses, object_in_root, arms,
                              lambda groups: TensorDict(groups, batch_size=[count]))
            result["scripted_eval"] = evaluate(env, scripted_actor(config), config["seed"] + 1)
            write_json(evidence, result)

            from rsl_rl.runners import OnPolicyRunner
            p = config["ppo"]
            env.recorder = ppo.ProgressLog(directory / "room-progress.jsonl", p["num_steps_per_env"],
                                           config["decision_control_steps"], OUTCOMES)
            runner = OnPolicyRunner(env, ppo.train_config(config), log_dir=str(directory / "train"), device=runtime["device"])
            if config["init_checkpoint"] is not None:
                runner.load(str(ROOT / config["init_checkpoint"]), map_location=runtime["device"])
                result["init_checkpoint"] = {"path": config["init_checkpoint"], "iteration": runner.current_learning_iteration}
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
            result["status"] = "passed_room_training"
            write_json(evidence, result)
        except BaseException as error:
            result.update(status="failed", error=str(error) or repr(error), error_type=type(error).__name__)
            write_json(evidence, result)
            (directory / "room-error.log").write_text(traceback.format_exc())
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
        import numpy as np
        p = config["ppo"]
        rng = np.random.default_rng(config["seed"])
        room = rg.sample_room(config["room"], rng)
        episode = sample_episode(config["task"], room, config["room"], config["goal"], 0.411, config["object"]["size_m"], rng)
        print(json.dumps({"status": "local_plan_only", "task": config["task"], "num_envs": config["num_envs"],
                          "iterations": p["max_iterations"], "decisions": config["num_envs"]*p["num_steps_per_env"]*p["max_iterations"],
                          "example_room": {"width_m": room["width_m"], "depth_m": room["depth_m"], "boxes": len(room["boxes"])},
                          "example_path_points": None if episode is None else len(episode["path"]), "training": True}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    if config["init_checkpoint"] is not None and not (ROOT / config["init_checkpoint"]).is_file():
        parser.error(f"Missing init_checkpoint: {config['init_checkpoint']}")
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(runtime))
        run_worker(config, profile, asset, runtime, ppo.gr00t_models(contract), args.run_dir)
        return 0
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%S.%fZ-room-{config['task']}")
    directory.mkdir(parents=True)
    for name, value in (("room-config.json", config), ("profile.json", profile), ("runtime-config.json", runtime),
                        ("controller-contract.json", contract)):
        write_json(directory / name, value)
    ppo.gr00t_models(contract)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--worker", "--run-dir", str(directory)]
    code = run_stage(command, directory / "room.log", config["stage_timeout_seconds"])
    path = directory / "room-worker.json"
    evidence = json.loads(path.read_text()) if path.exists() else {"status": "missing"}

    def brief(name):
        e = evidence.get(name) or {}
        return {k: e.get(k) for k in ("success_rate", "outcome_rates", "median_success_seconds", "final_error_at_success",
                                      "walking_speed_mps")} if e else None

    summary = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code, "run_dir": str(directory),
               "scripted_eval": brief("scripted_eval"), "learned_eval": brief("learned_eval"),
               "training": evidence.get("training"), "error": evidence.get("error")}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "passed_room_training" else 1


if __name__ == "__main__":
    raise SystemExit(main())
