"""High-level PPO policy that carries an attached cabinet through an L-shaped corridor over frozen GR00T Walk/Balance.

Each environment has its own corridor (width, corner distance, left or right turn), authored as static collider walls
before reset, and its own cabinet mass. Every episode starts from the crouch start with the cabinet attached to the
palms (the lift probe's grasp abstraction); GR00T Balance settles, arm IK lifts the cabinet, and then every 0.1 s the
policy chooses the walk forward speed and yaw rate. An episode succeeds when the cabinet centre crosses the goal line
in the second leg. It fails on a fall, a GR00T action beyond the transfer bound, wall contact (a cabinet corner, or a
robot link origin inflated by a radius, within wall_contact_m of a wall; the walls are colliders, so physical contact
holds that distance near zero), a tipped cabinet or a cabinet back on the floor. A pure-pursuit scripted command is
evaluated on the same seeded starts before training, and the learned policy after it.
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

import isaac_g1_lift as lift
import isaac_g1_ppo as ppo
import isaac_g1_rl as rl
from isaac_g1_wbc import ROOT, load_plan, trial_contract
from isaac_preflight import check_versions, run_stage, write_json


DEFAULT_CONFIG = ROOT / "configs/isaac_g1_carry.json"
ACTION_KEYS = ("forward_mps", "yaw_rate_rps")
FAILURES = ("invalid_action", "fallen", "wall_contact", "object_tipped", "object_grounded")
OUTCOMES = (*FAILURES, "success", "time_out")
LOOKAHEAD_M = (0.5, 1.0, 1.5)
OBSERVATIONS = 25


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"push_profile", "object", "mass_kg", "arm_lift_m", "lift_prefix_seconds", "corridor", "start",
                       "num_envs", "env_spacing_m", "decision_control_steps", "episode_seconds", "action_bounds",
                       "termination", "reward", "scripted", "ppo", "seed", "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed carry configuration keys")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    def interval(value, low, high):
        return isinstance(value, list) and len(value) == 2 and all(number(v, low, high) for v in value) and value[0] < value[1]

    for key, (low, high) in {"num_envs": (1, 4096), "decision_control_steps": (1, 50), "seed": (0, 2**31-1),
                             "stage_timeout_seconds": (60, 10800)}.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid {key}")
    if not number(config["env_spacing_m"], 4.0, 20.0) or not number(config["episode_seconds"], 5.0, 120.0):
        raise ValueError("Invalid environment spacing or episode length")
    profile, asset, runtime = load_plan(ROOT / config["push_profile"])
    if (profile.get("functional_case") != "push" or profile.get("arm_hold") != "implicit" or
            profile.get("initialization", {}).get("mode") != "collision_clearance"):
        raise ValueError("Carrying starts from the reviewed implicit-arm crouch-start push profile")
    o = config["object"]
    if (set(o) != {"size_m", "friction", "start_clearance_m", "near_face_ahead_of_palm_m"} or
            not isinstance(o["size_m"], list) or len(o["size_m"]) != 3 or not all(number(v, 0.1, 1.5) for v in o["size_m"]) or
            not number(o["friction"], 0.05, 1.5) or not number(o["start_clearance_m"], 0.0, 0.02) or
            not number(o["near_face_ahead_of_palm_m"], 0.0, 0.1)):
        raise ValueError("Invalid carried object")
    # Lift sweep 20261031T184440 carried up to 4 kg without feedforward and failed at 5 kg.
    if not interval(config["mass_kg"], 0.5, 4.0) or not number(config["arm_lift_m"], 0.04, 0.12):
        raise ValueError("Masses must stay within the measured 4 kg capacity and the arm lift within 4-12 cm")
    pre = config["lift_prefix_seconds"]
    if set(pre) != {"settle", "lift", "hold"} or not all(number(v, 0.1, 5.0) for v in pre.values()):
        raise ValueError("Invalid lift prefix")
    c = config["corridor"]
    if (set(c) != {"width_m", "corner_m", "goal_past_corner_m", "leg_two_m", "back_m", "wall_thickness_m", "wall_height_m"} or
            not interval(c["width_m"], 0.8, 3.0) or not interval(c["corner_m"], 0.8, 5.0) or
            not number(c["goal_past_corner_m"], 0.3, 3.0) or not number(c["leg_two_m"], 0.8, 6.0) or
            c["leg_two_m"] < c["goal_past_corner_m"] + 0.5 or not number(c["back_m"], 0.3, 2.0) or
            not number(c["wall_thickness_m"], 0.02, 0.5) or not number(c["wall_height_m"], 0.5, 3.0)):
        raise ValueError("Invalid corridor")
    # Neighbouring corridors must not overlap on the environment grid (either turn direction).
    t, widest = c["wall_thickness_m"], c["width_m"][1]
    if (c["back_m"] + c["corner_m"][1] + widest/2 + 2*t > config["env_spacing_m"] or
            2*(c["leg_two_m"] + widest/2 + t) > config["env_spacing_m"]):
        raise ValueError("Corridors overlap at this environment spacing")
    s = config["start"]
    if set(s) != {"lateral_m", "yaw_rad"} or not interval(s["lateral_m"], -0.3, 0.3) or not interval(s["yaw_rad"], -0.5, 0.5):
        raise ValueError("Invalid start randomization")
    b = config["action_bounds"]
    # Zero speed is allowed: a command norm of at most 0.05 hands the robot to GR00T Balance (source rule).
    if set(b) != set(ACTION_KEYS) or not interval(b["forward_mps"], 0.0, 0.4) or not interval(b["yaw_rate_rps"], -0.4, 0.4):
        raise ValueError("Invalid action bounds")
    term = config["termination"]
    if (set(term) != {"robot_link_radius_m", "wall_contact_m", "object_tilt_rad", "object_floor_clearance_m"} or
            not number(term["robot_link_radius_m"], 0.0, 0.3) or not number(term["wall_contact_m"], 0.0, 0.05) or
            not number(term["object_tilt_rad"], 0.05, 1.0) or not number(term["object_floor_clearance_m"], 0.0, 0.05)):
        raise ValueError("Invalid termination limits")
    w = config["reward"]
    if (set(w) != {"progress", "wall", "safe_clearance_m", "action_rate", "failure", "success"} or
            not all(number(v, 0.0, 100.0) for v in w.values()) or not number(w["safe_clearance_m"], 0.01, 0.5)):
        raise ValueError("Invalid reward weights")
    sc = config["scripted"]
    if set(sc) != {"forward_mps", "yaw_gain"} or not number(sc["forward_mps"], *b["forward_mps"]) or not number(sc["yaw_gain"], 0.0, 5.0):
        raise ValueError("Scripted command must lie inside the action bounds")
    if not ppo.valid_ppo_settings(config["ppo"]):
        raise ValueError("Invalid PPO settings")
    return config, profile, asset, runtime


def sample_corridors(settings, count, rng):
    """Per-environment corridor width, corner distance and turn (+1 left, -1 right, alternating by index)."""
    import numpy as np
    return {"width_m": rng.uniform(*settings["width_m"], count), "corner_m": rng.uniform(*settings["corner_m"], count),
            "turn": np.where(np.arange(count) % 2 == 0, 1.0, -1.0)}


def corridor_walls(width, corner, turn, settings):
    """Wall boxes (centre, size) in the environment frame. The corridor runs along +x from the start, then turns to
    +y (turn=1) or -y (turn=-1) at x=corner; the far end of the second leg is open."""
    h, t, height = width/2, settings["wall_thickness_m"], settings["wall_height_m"]
    back, end = -settings["back_m"], settings["leg_two_m"]
    rects = [(back - t, corner + h + t, -h - t, -h),          # first leg, outer side
             (back - t, back, -h - t, h + t),                 # behind the start
             (back - t, corner - h, h, h + t),                # first leg, inner side
             (corner + h, corner + h + t, -h - t, end),       # facing the first leg
             (corner - h - t, corner - h, h, end)]            # second leg, inner side
    walls = []
    for x0, x1, y0, y1 in rects:
        if turn < 0:
            y0, y1 = -y1, -y0
        walls.append(([(x0 + x1)/2, (y0 + y1)/2, height/2], [x1 - x0, y1 - y0, height]))
    return walls


def apply_corridor(stage, env, walls):
    """Author static collider walls (no rigid body) under one environment in the session layer before reset."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    paths = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, f"{env}/Corridor")
        for k, (centre, size) in enumerate(walls):
            cube = UsdGeom.Cube.Define(stage, f"{env}/Corridor/wall_{k}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.78, 0.74)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            paths.append(str(cube.GetPath()))
    return paths


def mirrored(xy, turn):
    """Environment-frame points (n, ..., 2) mirrored so that every corridor turns left (+y)."""
    import numpy as np
    turn = np.asarray(turn, dtype=float).reshape((-1,) + (1,)*(xy.ndim - 2))
    return np.stack([xy[..., 0], xy[..., 1]*turn], axis=-1)


def inside_distance(xy, width, corner, back):
    """Signed distance from mirrored points (n, k, 2) to the corridor walls, positive inside; the far end is open."""
    import numpy as np
    x, y = xy[..., 0], xy[..., 1]
    h, c = (np.asarray(width, dtype=float)/2)[:, None], np.asarray(corner, dtype=float)[:, None]
    first = np.minimum.reduce([x + back, c + h - x, y + h, h - y])
    second = np.minimum.reduce([x - (c - h), c + h - x, y + h])
    return np.maximum(first, second)


def path_progress(xy, corner):
    """Distance along the centre line (0,0)->(corner,0)->(corner,+inf), projected on the nearer leg (mirrored xy)."""
    import numpy as np
    x, y = xy[:, 0], xy[:, 1]
    along, rise = np.clip(x, 0.0, corner), np.maximum(y, 0.0)
    return np.where(np.hypot(x - along, y) <= np.hypot(x - corner, y - rise), along, corner + rise)


def path_point(progress, corner):
    import numpy as np
    return np.stack([np.minimum(progress, corner), np.maximum(progress - corner, 0.0)], axis=-1)


def box_corners(poses, size):
    """World corners (n, 8, 3) of boxes with poses [x, y, z, qx, qy, qz, qw]."""
    import numpy as np
    from g1_wbc import quat_xyzw_matrices
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return poses[:, None, :3] + (signs*np.asarray(size, dtype=float)/2) @ quat_xyzw_matrices(poses[:, 3:]).transpose(0, 2, 1)


def corridor_state(robot_pose, object_pose, body_xyz, origins, corridors, settings, object_size, link_radius):
    """Task geometry per environment: look-ahead centre-line points in the robot heading frame, cabinet progress along
    the centre line, wall clearances of the cabinet corners and of the robot link origins less link_radius, cabinet
    floor clearance and tilt."""
    import numpy as np
    from g1_wbc import quat_xyzw_matrices
    turn, corner = corridors["turn"], corridors["corner_m"]
    count = len(turn)

    def local_xy(points):
        return mirrored(points[..., :2] - origins[:, None, :2], turn)

    robot = local_xy(robot_pose[:, None])[:, 0]
    heading = turn*ppo.yaw_of(robot_pose[:, 3:])
    ahead = path_point(path_progress(robot, corner)[:, None] + np.array(LOOKAHEAD_M), corner[:, None]) - robot[:, None]
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    # Rotate into the mirrored heading frame, then mirror the lateral axis back to the real robot frame.
    lookahead = np.stack([c*ahead[..., 0] + s*ahead[..., 1], turn[:, None]*(-s*ahead[..., 0] + c*ahead[..., 1])], axis=-1)
    corners = box_corners(object_pose, object_size)
    rotation = quat_xyzw_matrices(object_pose[:, 3:])
    back = settings["back_m"]
    return {"lookahead": lookahead.reshape(count, -1),
            "progress_m": path_progress(local_xy(object_pose[:, None])[:, 0], corner),
            "goal_m": corner + settings["goal_past_corner_m"],
            "object_clearance_m": inside_distance(local_xy(corners), corridors["width_m"], corner, back).min(axis=1),
            "robot_clearance_m": inside_distance(local_xy(body_xyz), corridors["width_m"], corner, back).min(axis=1) - link_radius,
            "bottom_m": corners[..., 2].min(axis=1),
            "tilt_rad": np.arccos(np.clip(rotation[:, 2, 2], -1.0, 1.0)),
            "object_up": rotation[:, :, 2]}


def task_observation(state, robot_pose, robot_velocity, object_pose, width, gravity_b, last_action, time_fraction,
                     prefix):
    """Policy input; world velocities are [linear, angular] and planar vectors are given in the robot heading frame."""
    import numpy as np
    heading = ppo.yaw_of(robot_pose[:, 3:])
    c, s = np.cos(heading), np.sin(heading)

    def local(v):
        return np.c_[c*v[:, 0] + s*v[:, 1], -s*v[:, 0] + c*v[:, 1]]

    return np.c_[state["lookahead"], state["goal_m"] - state["progress_m"], state["object_clearance_m"],
                 state["robot_clearance_m"], width, local(object_pose[:, :2] - robot_pose[:, :2]), state["bottom_m"],
                 local(state["object_up"]), local(robot_velocity[:, :2]), robot_velocity[:, 5], gravity_b, last_action,
                 time_fraction, prefix].astype(np.float32)


def task_step(state, previous_progress, prefix, actions, last_actions, low_level, config, decision_dt):
    """Reward, termination and outcome index (into OUTCOMES, -1 while running) for one high-level decision.

    During the lift prefix the policy has no control: failures still end the episode, the shaping terms are zero and
    the cabinet may rest on the floor.
    """
    import numpy as np
    t, w = config["termination"], config["reward"]
    clearance = np.minimum(state["object_clearance_m"], state["robot_clearance_m"])
    flags = np.stack([low_level["invalid_action"], low_level["fallen"], clearance < t["wall_contact_m"],
                      state["tilt_rad"] > t["object_tilt_rad"],
                      ~prefix & (state["bottom_m"] < t["object_floor_clearance_m"])])
    failed = flags.any(axis=0)
    success = ~failed & ~prefix & (state["progress_m"] >= state["goal_m"])
    outcome = np.where(failed, np.argmax(flags, axis=0), np.where(success, OUTCOMES.index("success"), -1))
    shaping = (w["progress"]*(state["progress_m"] - previous_progress)/decision_dt
               - w["wall"]*np.clip(1.0 - clearance/w["safe_clearance_m"], 0.0, 1.0)
               - w["action_rate"]*np.sum((actions - last_actions)**2, axis=1))
    reward = np.where(prefix, 0.0, shaping) - w["failure"]*failed + w["success"]*success
    return reward, failed | success, outcome


def bound_arrays(bounds):
    import numpy as np
    return tuple(np.array([bounds[k][i] for k in ACTION_KEYS], dtype=float) for i in (0, 1))


def action_commands(actions, bounds):
    """Map normalized actions in [-1, 1] to GR00T commands [vx, 0, yaw rate]."""
    import numpy as np
    low, high = bound_arrays(bounds)
    values = low + (np.clip(np.asarray(actions, dtype=float), -1.0, 1.0) + 1.0)/2*(high - low)
    return np.c_[values[:, 0], np.zeros(len(values)), values[:, 1]]


def normalized_action(values, bounds):
    import numpy as np
    low, high = bound_arrays(bounds)
    return 2*(np.asarray(values, dtype=float) - low)/(high - low) - 1.0


def pursuit_actions(policy_obs, config):
    """Scripted baseline: constant forward speed, yaw rate proportional to the bearing of the 1.0 m look-ahead point."""
    import numpy as np
    s, b = config["scripted"], config["action_bounds"]
    yaw = np.clip(s["yaw_gain"]*np.arctan2(policy_obs[:, 3], policy_obs[:, 2]), *b["yaw_rate_rps"])
    return normalized_action(np.c_[np.full(len(yaw), s["forward_mps"]), yaw], b)


class CarryTaskEnv:
    """RSL-RL vectorized environment over the prepared parallel scene with attached cabinets and corridor walls.

    object_start: cabinet pose relative to the crouch-start robot root (4x4). arms: (joint indices, start targets,
    lift targets) for the 14 arm joints.
    """

    def __init__(self, scene, policy, contract, profile, asset, runtime, config, corridors, masses, object_start, arms,
                 observations):
        import numpy as np
        from g1_wbc import BatchHistory
        self.s, self.policy, self.profile, self.asset, self.config = scene, policy, profile, asset, config
        self.corridors, self.masses, self.object_start, self.observations = corridors, np.asarray(masses), object_start, observations
        self.arm_ids, self.arm_start, self.arm_lift = arms
        self.num_envs, self.num_actions, self.device, self.cfg = len(scene.origins), len(ACTION_KEYS), runtime["device"], config
        control_dt = asset["control_decimation"]*asset["physics_dt"]
        self.decision_dt = config["decision_control_steps"]*control_dt
        self.max_episode_length = int(round(config["episode_seconds"]/self.decision_dt))
        pre = config["lift_prefix_seconds"]
        self.settle_steps, self.lift_steps = (int(round(pre[k]/control_dt)) for k in ("settle", "lift"))
        self.prefix_decisions = int(round(sum(pre.values())/self.decision_dt))
        self.history = BatchHistory(contract, self.num_envs)
        self.targets = np.tile(scene.defaults, (self.num_envs, 1))
        self.last_action = np.zeros((self.num_envs, self.num_actions))
        self.last_command = np.zeros((self.num_envs, self.num_actions))
        # Episode length (randomized by the runner at the start of training) and decisions since the last reset, which
        # drives the lift prefix.
        self.length = np.zeros(self.num_envs, dtype=int)
        self.since_reset = np.zeros(self.num_envs, dtype=int)
        self.start_progress, self.previous_progress = np.zeros(self.num_envs), np.zeros(self.num_envs)
        self.rng = np.random.default_rng(config["seed"])
        self.recorder, self.last_completed, self.last_task, self.reset_on_done = None, {}, {}, True
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
        """Crouch start with a random lateral offset and heading; the cabinet follows the root so the joints hold."""
        import numpy as np
        from g1_wbc import axis_rotation, matrix_quat_xyzw, pose_matrix
        s, start = self.s, self.config["start"]
        rng = self.rng if rng is None else rng
        n, env_ids = len(ids), ids.tolist()
        lateral, yaw = rng.uniform(*start["lateral_m"], n), rng.uniform(*start["yaw_rad"], n)
        robot, cabinet = np.zeros((n, 7)), np.zeros((n, 7))
        for j, i in enumerate(ids):
            root = pose_matrix(s.root[i])
            root[:3, :3] = axis_rotation([0, 0, 1], yaw[j]) @ root[:3, :3]
            root[1, 3] += lateral[j]
            held = root @ self.object_start
            robot[j] = np.r_[root[:3, 3], matrix_quat_xyzw(root[:3, :3])]
            cabinet[j] = np.r_[held[:3, 3], matrix_quat_xyzw(held[:3, :3])]
        s.scene.reset(env_ids)
        s.robot.write_root_pose_to_sim_index(root_pose=s.device(robot), env_ids=env_ids)
        s.robot.write_root_velocity_to_sim_index(root_velocity=s.device(np.zeros((n, 6))), env_ids=env_ids)
        s.robot.write_joint_position_to_sim_index(position=s.device(np.tile(s.defaults, (n, 1))), env_ids=env_ids)
        s.robot.write_joint_velocity_to_sim_index(velocity=s.device(np.zeros((n, 29))), env_ids=env_ids)
        s.box.write_root_pose_to_sim_index(root_pose=s.device(cabinet), env_ids=env_ids)
        s.box.write_root_velocity_to_sim_index(root_velocity=s.device(np.zeros((n, 6))), env_ids=env_ids)
        # Source schedule restarts: default targets for the first physics interval and an empty history.
        self.history.reset(ids)
        self.targets[ids] = s.defaults
        s.robot.set_joint_position_target_index(target=s.device(self.targets))
        self.last_action[ids] = 0.0
        self.length[ids] = 0
        self.since_reset[ids] = 0
        xy = mirrored(cabinet[:, None, :2] - s.origins[ids, None, :2], self.corridors["turn"][ids])[:, 0]
        self.start_progress[ids] = self.previous_progress[ids] = path_progress(xy, self.corridors["corner_m"][ids])

    def task_state(self):
        from g1_wbc import finite
        s, n = self.s, self.num_envs
        robot = finite(s.batch(s.robot.data.root_link_pose_w), (n, 7))
        cabinet = finite(s.batch(s.box.data.root_link_pose_w), (n, 7))
        bodies = finite(s.batch(s.robot.data.body_link_pose_w), (n, len(s.bodies), 7))[..., :3]
        return robot, cabinet, corridor_state(robot, cabinet, bodies, s.origins, self.corridors, self.config["corridor"],
                                              self.config["object"]["size_m"],
                                              self.config["termination"]["robot_link_radius_m"])

    def observe(self):
        import numpy as np
        import torch
        from g1_wbc import finite, quat_xyzw_matrices
        s, n = self.s, self.num_envs
        robot, cabinet, state = self.task_state()
        gravity = -(quat_xyzw_matrices(robot[:, 3:]) @ s.rotation)[:, 2, :]
        task = task_observation(state, robot, finite(s.batch(s.robot.data.root_link_vel_w), (n, 6)), cabinet,
                                self.corridors["width_m"], gravity, self.last_action,
                                self.length[:, None]/self.max_episode_length,
                                (self.since_reset < self.prefix_decisions)[:, None])
        low, high = self.config["mass_kg"]
        # The critic also sees the per-environment cabinet mass, normalized to its range.
        privileged = ((self.masses - low)/(high - low))[:, None].astype(np.float32)

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        return self.observations({"policy": tensor(task), "critic": tensor(np.c_[task, privileged])})

    def step(self, actions):
        import numpy as np
        import torch
        from g1_wbc import finite, quat_xyzw_matrices
        s, n, profile = self.s, self.num_envs, self.profile
        action = np.clip(np.asarray(actions.detach().cpu().numpy(), dtype=float), -1.0, 1.0)
        prefix = self.since_reset < self.prefix_decisions
        commands = action_commands(action, self.config["action_bounds"])
        # Zero command during the lift prefix, which hands the robot to GR00T Balance under the source rule.
        commands[prefix] = 0.0
        self.last_command = commands[:, [0, 2]]
        rpys = np.zeros((n, 3))
        low_level = {"invalid_action": np.zeros(n, dtype=bool), "fallen": np.zeros(n, dtype=bool)}
        steps = self.config["decision_control_steps"]
        for k in range(steps):
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
            # Arms hold the start pose while settling, then ramp to the lift pose over the lift interval.
            fraction = np.clip((self.since_reset*steps + k + 1 - self.settle_steps)/self.lift_steps, 0.0, 1.0)[:, None]
            self.targets[:, self.arm_ids] = self.arm_start + fraction*(self.arm_lift - self.arm_start)
            s.robot.set_joint_position_target_index(target=s.device(self.targets))
            gravity = -(quat_xyzw_matrices(pose[:, 3:]) @ s.rotation)[:, 2, :]
            low_level["invalid_action"] |= invalid
            low_level["fallen"] |= ((pose[:, 2] < profile["minimum_base_height_m"]) |
                                    (np.arccos(np.clip(-gravity[:, 2], -1, 1)) > profile["maximum_tilt_rad"]))
        state = self.task_state()[2]
        reward, terminated, outcome = task_step(state, self.previous_progress, prefix, action, self.last_action,
                                                low_level, self.config, self.decision_dt)
        self.last_task = {"clearance_m": np.minimum(state["object_clearance_m"], state["robot_clearance_m"]),
                          "tilt_rad": state["tilt_rad"], "active": ~prefix}
        self.previous_progress = state["progress_m"].copy()
        self.last_action = action
        self.length += 1
        self.since_reset += 1
        time_out = ~terminated & (self.length >= self.max_episode_length)
        ids = np.flatnonzero(terminated | time_out)
        self.last_completed = {int(i): {"outcome": OUTCOMES[outcome[i]] if terminated[i] else "time_out",
                                        "progress_m": float(state["progress_m"][i] - self.start_progress[i]),
                                        "seconds": float(self.length[i]*self.decision_dt)}
                               for i in ids}
        extras = {"time_outs": torch.as_tensor(time_out, device=self.device)}
        if len(ids):
            records = list(self.last_completed.values())
            extras["log"] = {f"/episode/{name}": torch.as_tensor([r["outcome"] == name for r in records], dtype=torch.float32)
                             for name in ("success", "wall_contact", "fallen", "time_out")}
            extras["log"]["/episode/progress_m"] = torch.as_tensor([r["progress_m"] for r in records], dtype=torch.float32)
            if self.reset_on_done:
                self.reset_envs(ids)
        if self.recorder is not None:
            self.recorder(self.last_completed, n)
        self.obs = self.observe()
        return (self.obs, torch.as_tensor(reward, dtype=torch.float32, device=self.device),
                torch.as_tensor(terminated | time_out, dtype=torch.long, device=self.device), extras)


def evaluate(env, act, seed):
    """One episode per environment from seeded starts; outcomes are also grouped by corridor width, turn and mass.

    Commands are sampled once per decision after the lift prefix until each environment's episode ends.
    """
    import numpy as np
    n = env.num_envs
    env.reset_envs(np.arange(n), np.random.default_rng(seed))
    env.obs = env.observe()
    first, commands = {}, [[] for _ in range(n)]
    clearance, tilt = np.full(n, np.inf), np.zeros(n)
    for _ in range(env.max_episode_length):
        active = [i for i in range(n) if i not in first]
        env.step(act(env.obs))
        for i in active:
            clearance[i] = min(clearance[i], env.last_task["clearance_m"][i])
            tilt[i] = max(tilt[i], env.last_task["tilt_rad"][i])
            if env.last_task["active"][i]:
                commands[i].append(env.last_command[i])
        for i, record in env.last_completed.items():
            first.setdefault(i, record)
        if len(first) == n:
            break
    records = []
    for i in range(n):
        c = np.asarray(commands[i], dtype=float).reshape(-1, 2)
        balance = np.hypot(c[:, 0], c[:, 1]) <= 0.05
        records.append({"width_m": float(env.corridors["width_m"][i]), "corner_m": float(env.corridors["corner_m"][i]),
                        "turn": "left" if env.corridors["turn"][i] > 0 else "right", "mass_kg": float(env.masses[i]),
                        **first[i], "min_wall_clearance_m": float(clearance[i]),
                        "max_object_tilt_deg": float(np.degrees(tilt[i])),
                        "commands": {"decisions": len(c),
                                     "forward_mean_mps": float(c[:, 0].mean()) if len(c) else None,
                                     "forward_max_mps": float(c[:, 0].max()) if len(c) else None,
                                     "yaw_rate_abs_mean_rps": float(np.abs(c[:, 1]).mean()) if len(c) else None,
                                     "balance_fraction": float(balance.mean()) if len(c) else None}})
    width, mass = sum(env.config["corridor"]["width_m"])/2, sum(env.config["mass_kg"])/2
    groups = {"narrow": [r for r in records if r["width_m"] < width], "wide": [r for r in records if r["width_m"] >= width],
              "left": [r for r in records if r["turn"] == "left"], "right": [r for r in records if r["turn"] == "right"],
              "light": [r for r in records if r["mass_kg"] < mass], "heavy": [r for r in records if r["mass_kg"] >= mass]}
    return {**ppo.episode_summary(records, OUTCOMES), "seed": seed,
            "groups": {k: ppo.episode_summary(v, OUTCOMES) for k, v in groups.items()}, "per_env": records}


def run_worker(config, profile, asset, runtime, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp
    from g1_wbc import finite, matrix_quat_xyzw, pose_matrix

    contract, reference = trial_contract(profile)
    policy = ppo.gr00t_policy(contract, model_paths)
    count, o = config["num_envs"], config["object"]
    torch.manual_seed(config["seed"])
    wp.config.enable_backward = False
    rng = np.random.default_rng(config["seed"])
    corridors = sample_corridors(config["corridor"], count, rng)
    masses = rng.uniform(*config["mass_kg"], count)
    from isaaclab.app import launch_simulation

    # The cabinet replaces the push box in the shared scene; each environment authors its own mass.
    scene_profile = {**profile, "push_object": {"size_m": o["size_m"], "mass_kg": float(masses[0]), "friction": o["friction"],
                                                "gap_m": profile["push_object"]["gap_m"]}}
    built = rl.scene_config(scene_profile, asset, runtime, contract, reference, with_camera=False, num_envs=count,
                            env_spacing=config["env_spacing_m"], replicate_physics=False)
    base = lift.base_pose(reference, asset, contract)
    frames = lift.attachment_frames(reference, base, o, contract["height_command"])
    joints = [n for side in lift.SIDES for n in lift.arm_joints(reference, side)]
    lifted = {n: v for side in lift.SIDES for n, v in lift.arm_pose(reference, base, side, config["arm_lift_m"]).items()}
    evidence = directory / "carry-worker.json"
    result = {"status": "running", "num_envs": count, "masses_kg": masses.tolist(),
              "corridors": {k: v.tolist() for k, v in corridors.items()}, "arm_lift_rad": lifted}
    write_json(evidence, result)
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "device": runtime["device"]}):
        try:
            attachments, walls = {}, {}

            def edits(stage, robot, box, env):
                i = int(env.rsplit("_", 1)[1])
                attachments[env] = lift.apply_hand_attachment(stage, robot, box, env, frames)
                walls[env] = apply_corridor(stage, env, corridor_walls(corridors["width_m"][i], corridors["corner_m"][i],
                                                                       corridors["turn"][i], config["corridor"]))

            s = rl.prepare_push_scene(built, count, scene_profile, asset, runtime, contract, reference, directory, result,
                                      evidence, box_properties=(masses, np.full(count, float(o["friction"]))),
                                      extra_edits=edits, check_box_start=False)
            result["authored"] = {"attachment_env0": attachments["/World/envs/env_0"],
                                  "environments": sorted(set(attachments) & set(walls), key=lambda e: int(e.rsplit("_", 1)[1])),
                                  "walls_per_environment": len(walls["/World/envs/env_0"])}
            # Cabinet pose that both palm joints expect at the crouch start, relative to the robot root.
            palm_ids = {side: s.bodies.index(f"{side}_hand_palm_link") for side in lift.SIDES}
            poses = finite(s.batch(s.robot.data.body_link_pose_w), (count, len(s.bodies), 7))
            relative, mismatch, bottoms = [], [], []
            for i in range(count):
                held = {side: pose_matrix(poses[i, k]) @ frames["palm_object"][side] for side, k in palm_ids.items()}
                mismatch.append(float(np.linalg.norm(held["left"][:3, 3] - held["right"][:3, 3])))
                relative.append(np.linalg.inv(pose_matrix(s.root[i])) @ held["left"])
                bottoms.append(lift.lowest_point_z(np.r_[held["left"][:3, 3], matrix_quat_xyzw(held["left"][:3, :3])], o["size_m"]))
            spread = float(max(np.abs(r - relative[0]).max() for r in relative))
            expected = o["start_clearance_m"] + result["initialization"]["lift_m"]
            result["start"] = {"palm_frame_mismatch_max_m": max(mismatch), "relative_pose_spread": spread,
                               "object_bottom_m": [min(bottoms), max(bottoms)], "expected_bottom_m": expected,
                               "object_in_root": relative[0].tolist()}
            write_json(evidence, result)
            if max(mismatch) > 0.002 or spread > 0.001 or max(abs(b - expected) for b in bottoms) > 0.01:
                raise RuntimeError("Attached cabinet start pose is inconsistent with the palms or the standing height")

            from tensordict import TensorDict
            arms = ([s.names.index(n) for n in joints], np.zeros(len(joints)), np.array([lifted[n] for n in joints]))
            env = CarryTaskEnv(s, policy, contract, profile, asset, runtime, config, corridors, masses, relative[0], arms,
                               lambda groups: TensorDict(groups, batch_size=[count]))
            # Body poses refresh only through the simulation, so update kinematics before the spawn check.
            s.sim.forward()
            s.scene.update(0.0)
            state = env.task_state()[2]
            clearance = np.minimum(state["object_clearance_m"], state["robot_clearance_m"])
            result["start"]["wall_clearance_min_m"] = float(clearance.min())
            write_json(evidence, result)
            if clearance.min() <= config["termination"]["wall_contact_m"]:
                raise RuntimeError("A robot or cabinet starts in contact with its corridor walls")

            result["scripted_eval"] = evaluate(
                env, lambda obs: torch.as_tensor(pursuit_actions(obs["policy"].detach().cpu().numpy(), config),
                                                 dtype=torch.float32), config["seed"] + 1)
            write_json(evidence, result)

            from rsl_rl.runners import OnPolicyRunner
            p = config["ppo"]
            env.recorder = ppo.ProgressLog(directory / "carry-progress.jsonl", p["num_steps_per_env"],
                                           config["decision_control_steps"], OUTCOMES)
            runner = OnPolicyRunner(env, ppo.train_config(config), log_dir=str(directory / "train"), device=runtime["device"])
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
            result["status"] = "passed_carry_training"
            write_json(evidence, result)
        except BaseException as error:
            result.update(status="failed", error=str(error) or repr(error), error_type=type(error).__name__)
            write_json(evidence, result)
            (directory / "carry-error.log").write_text(traceback.format_exc())
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
                          "corridor": config["corridor"], "mass_kg": config["mass_kg"],
                          "action_bounds": config["action_bounds"], "training": True}, indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(runtime))
        run_worker(config, profile, asset, runtime, ppo.gr00t_models(contract), args.run_dir)
        return 0
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-carry")
    directory.mkdir(parents=True)
    for name, value in (("carry-config.json", config), ("profile.json", profile), ("runtime-config.json", runtime),
                        ("controller-contract.json", contract)):
        write_json(directory / name, value)
    ppo.gr00t_models(contract)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--worker", "--run-dir", str(directory)]
    code = run_stage(command, directory / "carry.log", config["stage_timeout_seconds"])
    path = directory / "carry-worker.json"
    evidence = json.loads(path.read_text()) if path.exists() else {"status": "missing"}

    def brief(name):
        e = evidence.get(name) or {}
        return ({k: e.get(k) for k in ("success_rate", "outcome_rates", "progress_m", "median_success_seconds")} |
                {"group_success": {g: v["success_rate"] for g, v in e.get("groups", {}).items()}}) if e else None

    summary = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code, "run_dir": str(directory),
               "start": evidence.get("start"), "scripted_eval": brief("scripted_eval"),
               "learned_eval": brief("learned_eval"), "training": evidence.get("training"), "error": evidence.get("error")}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "passed_carry_training" else 1


if __name__ == "__main__":
    raise SystemExit(main())
