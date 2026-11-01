"""Furniture-moving rehearsal in one room: G1 walks to the chosen cabinet, grasps it, lifts it, carries it to the chosen
pose and sets it down, with GR00T Walk/Balance as the low-level controller.

The room comes from a scene spec (scripts/scene_from_image.py: SAM 3.1 detections + Muse Spark layout) and a request
naming the target, the user's corrections (size and mass) and the destination. Walls surround the spec's width x depth,
every other furniture item is a static axis-aligned box, and the target is the dynamic cabinet the room policies were
trained with. Phases: approach (room approach checkpoint, or scripted docking: the zero residual) -> reach (arm IK to the grasp points) ->
grasp (palm joints re-anchored to the current palm pose and enabled) -> lift -> hold -> carry (room carry checkpoint, or
scripted docking) -> lower -> release -> withdraw -> back away. A failed approach stops the run; a failed carry still
sets the cabinet down unless the robot fell. One camera records every decision, and every decision logs the robot link
poses and the cabinet pose for the WebGL replay. The placement error against the request is measured at the end.

Scene-spec frame: origin at the back-left floor corner, x along the back wall, y toward the camera; a furniture yaw of 0
faces +y and 90 faces +x. With z up this frame is left-handed, so the right-handed simulation frame flips y: the room is
centred on the environment origin with x to the right and y toward the back wall (away from the camera); spec x = sim
x + W/2 and spec y = D/2 - sim y (run 20261101T103451 used the spec axes directly and rendered the room mirrored). The
robot faces the cabinet's front to grasp it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys
import traceback

import isaac_g1_carry as carry
import isaac_g1_lift as lift
import isaac_g1_ppo as ppo
import isaac_g1_ppo_eval as ppo_eval
import isaac_g1_rl as rl
import isaac_g1_room as room
import room_geometry as rg
from isaac_g1_wbc import ROOT, trial_contract
from isaac_preflight import check_versions, run_stage, write_json
from isaac_video import encode_recording, require_ffmpeg

DEFAULT_CONFIG = ROOT / "configs/isaac_g1_rehearsal.json"
TIMELINE = ("reach", "grasp", "lift", "hold", "lower", "release", "withdraw", "back_away")


def repo_path(value, prefix, suffix):
    """A repository-relative path under `prefix` ending in `suffix`, or a ValueError."""
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith(suffix) or ".." in Path(value).parts:
        raise ValueError(f"Expected a {suffix} path under {prefix}: {value}")
    return ROOT / value


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"approach_training", "carry_training", "approach_checkpoint", "carry_checkpoint", "request",
                       "start_inside_door_m", "timeline_seconds", "back_away_mps", "camera", "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed rehearsal configuration keys")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    trainings = {}
    for task in ("approach", "carry"):
        trainings[task] = room.load_config(repo_path(config[f"{task}_training"], "configs/", ".json"))
        if trainings[task][0]["task"] != task:
            raise ValueError(f"{task}_training must be a {task} configuration")
        if config[f"{task}_checkpoint"] is not None:
            repo_path(config[f"{task}_checkpoint"], "outputs/isaac/", ".pt")
    a, c = trainings["approach"][0], trainings["carry"][0]
    if (a["object"] != c["object"] or a["push_profile"] != c["push_profile"] or a["arm_lift_m"] != c["arm_lift_m"] or
            a["decision_control_steps"] != c["decision_control_steps"]):
        raise ValueError("Approach and carry must share the cabinet, profile, arm lift and decision rate")
    repo_path(config["request"], "configs/", ".json")
    t = config["timeline_seconds"]
    if set(t) != set(TIMELINE) or not all(number(v, 0.1, 5.0) for v in t.values()):
        raise ValueError("Invalid rehearsal timeline")
    cam = config["camera"]
    if (set(cam) != {"eye_from_front_m", "target_height_m"} or not isinstance(cam["eye_from_front_m"], list) or
            len(cam["eye_from_front_m"]) != 3 or not all(number(v, -8.0, 8.0) for v in cam["eye_from_front_m"]) or
            not number(cam["target_height_m"], 0.0, 2.0)):
        raise ValueError("Invalid rehearsal camera")
    if (not number(config["start_inside_door_m"], 0.3, 1.5) or not number(config["back_away_mps"], 0.0, 0.15) or
            type(config["stage_timeout_seconds"]) is not int or not 60 <= config["stage_timeout_seconds"] <= 3600):
        raise ValueError("Invalid start distance, back-away speed or stage timeout")
    return config, trainings


def load_request(path):
    request = json.loads(Path(path).read_text())
    if (set(request) != {"scene", "target", "corrections", "destination"} or not isinstance(request["target"], str) or
            set(request["corrections"]) != {"size_m", "mass_kg"} or set(request["destination"]) != {"center_m", "yaw_deg"}):
        raise ValueError("A request needs scene, target, corrections {size_m, mass_kg} and destination {center_m, yaw_deg}")
    scene = json.loads(repo_path(request["scene"], "configs/scenes/", ".json").read_text())
    return request, scene


def facing_yaw(yaw_deg):
    """Simulation yaw of a cabinet whose front faces the spec yaw (0 = spec +y, toward the camera = sim -y; 90 = +x): the
    robot stands in front of it facing back along that direction, so the cabinet's x axis points away from the robot."""
    import numpy as np
    theta = np.radians(yaw_deg)
    return float(np.arctan2(np.cos(theta), -np.sin(theta)))


def rehearsal_layout(scene, request, trainings, inside_door, reach):
    """Room, start pose, cabinet pose, goals and paths for one request (simulation frame, centred on the room).

    Raises ValueError when the corrected target is outside the carry tier, the cabinet or its destination overlaps other
    furniture, the door start is blocked, or no path exists.
    """
    import numpy as np
    a, c = trainings["approach"][0], trainings["carry"][0]
    spec, o = scene["scene"], a["object"]
    width, depth = spec["room"]["width_m"], spec["room"]["depth_m"]
    items = {f["id"]: f for f in spec["furniture"]}
    if request["target"] not in items:
        raise ValueError(f"Unknown target {request['target']}; furniture: {sorted(items)}")
    fix = request["corrections"]
    if not np.allclose(fix["size_m"], o["size_m"], atol=0.02) or not c["mass_kg"][0] <= fix["mass_kg"] <= c["mass_kg"][1]:
        raise ValueError(f"The carry tier covers a {o['size_m']} m cabinet of {c['mass_kg'][0]}-{c['mass_kg'][1]} kg; "
                         "this target needs the push branch")

    def centred(xy):
        return np.array([xy[0] - width/2, depth/2 - xy[1]])

    boxes = []
    for f in spec["furniture"]:
        if f["id"] == request["target"]:
            continue
        theta = np.radians(f["yaw_deg"])
        sx = abs(np.cos(theta))*f["size_m"][0] + abs(np.sin(theta))*f["size_m"][1]
        sy = abs(np.sin(theta))*f["size_m"][0] + abs(np.cos(theta))*f["size_m"][1]
        x0, y0 = np.maximum(centred(f["center_m"]) - [sx/2, sy/2], [-width/2, -depth/2])
        x1, y1 = np.minimum(centred(f["center_m"]) + [sx/2, sy/2], [width/2, depth/2])
        if x1 > x0 and y1 > y0:
            boxes.append([(x0 + x1)/2, (y0 + y1)/2, x1 - x0, y1 - y0, f["size_m"][2]])
    layout_room = {"width_m": float(width), "depth_m": float(depth), "boxes": np.array(boxes, dtype=float).reshape(-1, 5)}
    target = items[request["target"]]
    cabinet = np.r_[centred(target["center_m"]), facing_yaw(target["yaw_deg"])]
    placed = np.r_[centred(request["destination"]["center_m"]), facing_yaw(request["destination"]["yaw_deg"])]
    size_xy = o["size_m"][:2]
    if not room.footprint_clear(layout_room, cabinet[:2], cabinet[2], size_xy, 0.0):
        raise ValueError("The corrected cabinet overlaps a wall or other furniture; correct its position")
    if not room.footprint_clear(layout_room, placed[:2], placed[2], size_xy, c["goal"]["object_margin_m"]):
        raise ValueError("The destination overlaps a wall or other furniture")
    door = spec["door"]
    # Side-wall doors are measured from the back wall, back-wall doors from the left wall; the robot faces into the room.
    start = {"right": [width/2 - inside_door, depth/2 - door["center_m"], np.pi],
             "left": [-width/2 + inside_door, depth/2 - door["center_m"], 0.0],
             "back": [door["center_m"] - width/2, depth/2 - inside_door, -np.pi/2]}[door["wall"]]
    start = np.array(start, dtype=float)
    approach_goal = rg.pregrasp_pose(cabinet[:2], cabinet[2], reach)
    carry_goal = rg.pregrasp_pose(placed[:2], placed[2], reach)
    for name, point, inflation in (("door start", start, a["room"]["start_clearance_m"]),
                                   ("approach goal", approach_goal, a["room"]["inflation_m"]),
                                   ("carry goal", carry_goal, c["room"]["inflation_m"])):
        if float(rg.obstacle_distance(point[:2], layout_room)) < inflation:
            raise ValueError(f"The {name} is within {inflation} m of a wall or furniture")
    approach_grid, final_grid = (rg.occupancy(layout_room, a["room"]["grid_m"], a["room"]["inflation_m"],
                                              (cabinet[:2], size_xy, cabinet[2], inflation))
                                 for inflation in (a["room"]["inflation_m"], a["goal"]["object_inflation_m"]))
    approach_path = rg.staged_path(approach_grid, start[:2], approach_goal, a["goal"]["staging_m"], final_grid)
    carry_path = rg.staged_path(rg.occupancy(layout_room, c["room"]["grid_m"], c["room"]["inflation_m"]),
                                approach_goal[:2], carry_goal, c["goal"]["staging_m"])
    if approach_path is None or carry_path is None:
        raise ValueError("No collision-free path for the approach or the carry")
    return {"room": layout_room, "start": start, "cabinet": cabinet, "placed": placed, "approach_goal": approach_goal,
            "approach_path": approach_path, "carry_goal": carry_goal, "carry_path": carry_path,
            "spec_transform": {"x": [1.0, width/2], "y": [-1.0, depth/2]}}


def to_spec(xy, transform):
    """Simulation (x, y) -> scene-spec (x, y): spec = sign * sim + offset per axis."""
    return [transform[axis][0]*float(v) + transform[axis][1] for axis, v in zip(("x", "y"), xy)]


def placement_error(cabinet_pose_xyzw, placed):
    """Position (m) and yaw (deg) error of the set-down cabinet against the requested pose (simulation frame)."""
    import numpy as np
    yaw = float(ppo.yaw_of(np.asarray(cabinet_pose_xyzw, dtype=float)[None, 3:])[0])
    return {"position_m": float(np.linalg.norm(np.asarray(cabinet_pose_xyzw[:2]) - placed[:2])),
            "yaw_deg": float(np.degrees(abs(room.wrap(yaw - placed[2])))), "yaw_rad": yaw}


class RehearsalEnv(room.RoomTaskEnv):
    """RoomTaskEnv over one fixed room in which the approach and carry phases share the robot without resets."""

    def __init__(self, layout, *args):
        self.layout = layout
        super().__init__(*args)
        self.reset_on_done = False

    def episode(self, i, rng):
        L = self.layout
        return {"start": L["start"], "cabinet": L["cabinet"], "goal": L["approach_goal"], "path": L["approach_path"]}

    def begin(self, task_config, goal, path, prefix_seconds):
        """Switch to `task_config`'s task toward `goal` along `path`; the episode clock starts where training hands control
        to the policy (after its prefix), and no prefix is replayed."""
        import numpy as np
        from g1_wbc import finite
        s = self.s
        self.config, self.carrying = task_config, task_config["task"] == "carry"
        self.goals[0], self.paths[0] = goal, path
        self.max_episode_length = int(round(task_config["episode_seconds"]/self.decision_dt))
        self.length[:] = int(round(prefix_seconds/self.decision_dt))
        self.since_reset[:] = 10**6
        xy = finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0, :2] - s.origins[0, :2]
        self.previous_remaining[0] = self.start_remaining[0] = rg.path_lookup(path, xy, [0.0])[0]
        self.spawned_xy[0] = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0, :2] - s.origins[0, :2]
        self.last_action[:] = 0.0
        self.obs = self.observe()

    def hold(self, begin, end, decisions, command=(0.0, 0.0, 0.0)):
        """Scripted decisions: a fixed GR00T command while the arm targets blend from begin to end per control step."""
        import numpy as np
        steps, total = self.config["decision_control_steps"], decisions*self.config["decision_control_steps"]
        commands, rpys = np.array([command], dtype=float), np.zeros((1, 3))
        flags = {"invalid_action": False, "fallen": False}
        for d in range(decisions):
            for k in range(steps):
                _, _, invalid, fallen = self.control(commands, rpys, begin + (d*steps + k + 1)/total*(end - begin))
                flags["invalid_action"] |= bool(invalid[0])
                flags["fallen"] |= bool(fallen[0])
            yield flags


def run_worker(config, trainings, request, scene, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp
    from PIL import Image
    from g1_wbc import finite, pose_matrix, quat_xyzw_matrices

    approach, profile, asset, runtime = trainings["approach"]
    carry_config = trainings["carry"][0]
    contract, reference = trial_contract(profile)
    policy = ppo.gr00t_policy(contract, model_paths)
    encoder = require_ffmpeg(directory)
    o, mass = approach["object"], float(request["corrections"]["mass_kg"])
    torch.manual_seed(approach["seed"])
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation

    scene_profile, built = carry.carry_scene(approach, profile, asset, runtime, contract, reference, np.array([mass]),
                                             num_envs=1, env_spacing=approach["env_spacing_m"], with_camera=True)
    base = lift.base_pose(reference, asset, contract)
    frames = lift.attachment_frames(reference, base, o, contract["height_command"])
    joints = [n for side in lift.SIDES for n in lift.arm_joints(reference, side)]
    angles = {name: {n: v for side in lift.SIDES for n, v in lift.arm_pose(reference, base, side, dz).items()}
              for name, dz in (("start", 0.0), ("lift", approach["arm_lift_m"]), ("place", -0.02))}
    arm = {name: np.array([[pose[n] for n in joints]]) for name, pose in angles.items()}
    evidence = directory / "rehearsal-worker.json"
    result = {"status": "running", "request": request, "mass_kg": mass,
              "checkpoints": {task: config[f"{task}_checkpoint"] for task in ("approach", "carry")}, "phases": {}}
    write_json(evidence, result)
    env_path = "/World/envs/env_0"
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "enable_cameras": True, "device": runtime["device"]}):
        try:
            def edits(stage, robot, box, env):
                lift.apply_hand_attachment(stage, robot, box, env, frames, False)
                result["walls"] = len(carry.apply_corridor(stage, env, rg.wall_boxes(
                    layout["room"], approach["room"]["wall_thickness_m"], approach["room"]["wall_height_m"])))

            # The pre-grasp reach comes from the nominal grasp at the crouch start; it is known only after the scene
            # exists, but the room (walls) must be authored before it. The room depends on the reach only through goals
            # and paths, so the layout is built with the training default and re-checked below.
            layout = rehearsal_layout(scene, request, trainings, config["start_inside_door_m"], 0.411)
            s = rl.prepare_push_scene(built, 1, scene_profile, asset, runtime, contract, reference, directory, result,
                                      evidence, box_properties=(np.array([mass]), np.array([float(o["friction"])])),
                                      extra_edits=edits, check_box_start=False)
            palm_ids = {side: s.bodies.index(f"{side}_hand_palm_link") for side in lift.SIDES}

            def palm_poses():
                poses = finite(s.batch(s.robot.data.body_link_pose_w), (1, len(s.bodies), 7))
                return {side: [pose_matrix(poses[0, k])] for side, k in palm_ids.items()}

            # Cabinet pose both palm joints expect at the crouch start, relative to the root (defines the pre-grasp pose).
            object_in_root = np.linalg.inv(pose_matrix(s.root[0])) @ palm_poses()["left"][0] @ frames["palm_object"]["left"]
            reach = float(object_in_root[0, 3])
            layout = rehearsal_layout(scene, request, trainings, config["start_inside_door_m"], reach)
            result["layout"] = {"reach_m": reach, "room": {**layout["room"], "boxes": layout["room"]["boxes"].tolist()},
                                **{k: np.asarray(layout[k]).tolist() for k in ("start", "cabinet", "placed", "approach_goal",
                                                                               "carry_goal")},
                                "spec_transform": layout["spec_transform"]}
            write_json(evidence, result)

            from tensordict import TensorDict
            ids = [s.names.index(n) for n in joints]
            room_config = {**approach, "num_envs": 1,
                           "room": {**approach["room"], "obstacles": [0, max(1, len(layout["room"]["boxes"]))]}}
            env = RehearsalEnv(layout, s, policy, contract, profile, asset, runtime, room_config, [layout["room"]],
                               np.array([mass]), object_in_root, (ids, arm["start"][0], arm["lift"][0]),
                               lambda groups: TensorDict(groups, batch_size=[1]))
            actors = {}
            for task, training in (("approach", approach), ("carry", carry_config)):
                if config[f"{task}_checkpoint"] is None:
                    actors[task] = room.scripted_actor(training)
                else:
                    actors[task], result["phases"][f"{task}_checkpoint_iteration"] = ppo_eval.load_policy(
                        env, training, ROOT / config[f"{task}_checkpoint"], runtime["device"])

            camera, (width, height) = s.scene["camera"], (profile["camera"]["width"], profile["camera"]["height"])
            cam, w, d = config["camera"], layout["room"]["width_m"], layout["room"]["depth_m"]
            # The camera stands in front of the room (the photo's side, sim -y) and looks toward the back wall.
            eye = s.origins[0] + [cam["eye_from_front_m"][0], -d/2 - cam["eye_from_front_m"][1], cam["eye_from_front_m"][2]]
            camera.set_world_poses_from_view(s.device(eye[None]), s.device((s.origins[0] + [0, 0, cam["target_height_m"]])[None]))
            case = directory / "rehearsal"
            (case / "frames").mkdir(parents=True)
            replay, count = [], 0

            def record(phase, command=(0.0, 0.0, 0.0)):
                nonlocal count
                for _ in range(2):
                    s.sim.render()
                    camera.update(asset["physics_dt"], force_recompute=True)
                rgb = s.batch(camera.data.output["rgb"])[0, ..., :3]
                if rgb.shape != (height, width, 3) or float(rgb.std()) < 1:
                    raise RuntimeError("Invalid rehearsal camera frame")
                Image.fromarray(rgb.astype(np.uint8)).save(case / "frames" / f"{count:06d}.png")
                count += 1
                bodies = finite(s.batch(s.robot.data.body_link_pose_w), (1, len(s.bodies), 7))[0]
                cabinet = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                offset = np.r_[s.origins[0], np.zeros(4)]
                replay.append({"t": round(len(replay)*env.decision_dt, 3), "phase": phase,
                               "command": [round(float(v), 4) for v in command],
                               "bodies": np.round(bodies - offset, 4).tolist(), "cabinet": np.round(cabinet - offset, 4).tolist()})

            def drive(task, act):
                """Policy decisions until the task ends (success, failure or its episode limit)."""
                while True:
                    env.step(act(env.obs))
                    record(task, env.last_command[0])
                    if 0 in env.last_completed:
                        return env.last_completed[0]

            # Station keeping while setting the load down: plain Balance stepped the robot back 18 cm and dragged the
            # cabinet 10 cm in run 20261101T103451 (14 cm in the lift probes), so these phases dock to the carry goal.
            keep = {**carry_config, "goal": {**carry_config["goal"], "position_tolerance_m": 0.03, "yaw_tolerance_deg": 5.0}}

            def scripted(name, begin, end, command=(0.0, 0.0, 0.0), dock=False):
                """Scripted decisions with arm targets blended from begin to end; dock=True replaces the fixed command by
                scripted docking to the current goal (3 cm / 5 deg) at every decision."""
                decisions = int(round(config["timeline_seconds"][name]/env.decision_dt))
                flags = {"invalid_action": False, "fallen": False}
                for d in range(decisions):
                    if dock:
                        command = tuple(room.action_commands(room.docking_actions(
                            env.observe()["policy"].detach().cpu().numpy(), keep), carry_config["action_bounds"])[0])
                    for step in env.hold(begin + d/decisions*(end - begin), begin + (d + 1)/decisions*(end - begin), 1, command):
                        flags = {k: flags[k] or step[k] for k in flags}
                    record(name, command)
                result["phases"][name] = flags
                return not any(flags.values())

            record("start")
            result["phases"]["approach"] = drive("approach", actors["approach"])
            write_json(evidence, result)
            status = "failed_approach"
            if result["phases"]["approach"]["outcome"] == "success":
                # Reach: each palm to its grasp point on the cabinet where it stands; then grasp where the hands are.
                pelvis = quat_xyzw_matrices(finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[:, 3:]) @ s.rotation
                cabinet = pose_matrix(finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0])
                arm["reach"], result["reach"] = lift.reach_plan(reference, base, 0.0, joints, frames["object_palm"],
                                                                palm_poses(), [cabinet], pelvis)
                scripted("reach", arm["start"], arm["reach"])
                palms = palm_poses()
                cabinet = pose_matrix(finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0])
                gap = max(float(np.linalg.norm((cabinet @ frames["object_palm"][side])[:3, 3] - palms[side][0][:3, 3]))
                          for side in lift.SIDES)
                result["grasp"] = {"palm_gap_m": gap, "authored": lift.set_attachment(
                    s.sim.stage, env_path, True, {side: np.linalg.inv(cabinet) @ palms[side][0] for side in lift.SIDES})}
                ok = scripted("grasp", arm["reach"], arm["reach"])
                result["grasp"]["object_shift_m"] = float(np.linalg.norm(
                    finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0, :3] - cabinet[:3, 3]))
                ok = ok and scripted("lift", arm["reach"], arm["lift"]) and scripted("hold", arm["lift"], arm["lift"])
                status = "failed_lift"
                if ok:
                    xy = finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0, :2] - s.origins[0, :2]
                    path = rg.staged_path(rg.occupancy(layout["room"], carry_config["room"]["grid_m"],
                                                       carry_config["room"]["inflation_m"]), xy, layout["carry_goal"],
                                          carry_config["goal"]["staging_m"])
                    env.arm_start = env.arm_lift = arm["lift"][0]
                    env.begin(carry_config, layout["carry_goal"], layout["carry_path"] if path is None else path,
                              sum(carry_config["lift_prefix_seconds"].values()))
                    result["phases"]["carry"] = drive("carry", actors["carry"])
                    write_json(evidence, result)
                    status = "failed_carry"
                    if result["phases"]["carry"]["outcome"] != "fallen":
                        # Dock only near the goal; a carry that failed on the way sets the load down where it is.
                        near = result["phases"]["carry"]["position_error_m"] <= 0.5
                        scripted("lower", arm["lift"], arm["place"], dock=near)
                        result["release_authored"] = lift.set_attachment(s.sim.stage, env_path, False)
                        if (scripted("release", arm["place"], arm["place"], dock=near) and
                                scripted("withdraw", arm["place"], arm["start"]) and
                                scripted("back_away", arm["start"], arm["start"], (-config["back_away_mps"], 0.0, 0.0))):
                            cabinet = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                            local = np.r_[cabinet[:3] - s.origins[0], cabinet[3:]]
                            error = placement_error(local, layout["placed"])
                            tolerance = carry_config["goal"]
                            result["placement"] = {
                                **error, "cabinet_spec_xy_m": to_spec(local[:2], layout["spec_transform"]),
                                "tilt_deg": lift.tilt_deg(cabinet),
                                "within_tolerance": bool(error["position_m"] <= tolerance["position_tolerance_m"] and
                                                         error["yaw_deg"] <= tolerance["yaw_tolerance_deg"])}
                            if result["phases"]["carry"]["outcome"] == "success":
                                status = "completed_rehearsal"
            result["video"] = encode_recording(case, binary=encoder, count=count, fps=1/Fraction(str(env.decision_dt)),
                                               width=width, height=height, require_motion=False)
            write_json(directory / "replay.json", {"bodies": list(s.bodies), "decision_dt": env.decision_dt,
                                                   "frame": "simulation room centred on the origin, x right, y toward the back wall; "
                                                            "spec = sign * sim + offset per axis (spec_transform)",
                                                   "spec_transform": layout["spec_transform"], "cabinet_size_m": o["size_m"],
                                                   "room": result["layout"]["room"], "frames": replay})
            result["status"] = status
            write_json(evidence, result)
        except BaseException as error:
            result.update(status="failed", error=str(error) or repr(error), error_type=type(error).__name__)
            write_json(evidence, result)
            (directory / "rehearsal-error.log").write_text(traceback.format_exc())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--request", type=str, help="configs/... request JSON (default: the config's request)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isaac")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config, trainings = load_config(args.config)
    if args.request:
        config = {**config, "request": args.request}
    request, scene = load_request(repo_path(config["request"], "configs/", ".json"))
    layout = rehearsal_layout(scene, request, trainings, config["start_inside_door_m"], 0.411)
    if args.dry_run:
        import numpy as np

        def length(path):
            return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

        print(json.dumps({"status": "local_plan_only", "target": request["target"], "mass_kg": request["corrections"]["mass_kg"],
                          "room_m": [layout["room"]["width_m"], layout["room"]["depth_m"]], "obstacles": len(layout["room"]["boxes"]),
                          **{k: np.round(layout[k], 3).tolist() for k in ("start", "cabinet", "approach_goal", "placed", "carry_goal")},
                          "approach_path_m": length(layout["approach_path"]), "carry_path_m": length(layout["carry_path"]),
                          "checkpoints": {t: config[f"{t}_checkpoint"] or "scripted docking" for t in ("approach", "carry")}},
                         indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    for task in ("approach", "carry"):
        if config[f"{task}_checkpoint"] is not None and not (ROOT / config[f"{task}_checkpoint"]).is_file():
            parser.error(f"Missing checkpoint: {config[f'{task}_checkpoint']}")
    profile = trainings["approach"][1]
    contract, _ = trial_contract(profile)
    if args.worker:
        write_json(args.run_dir / "versions.json", check_versions(trainings["approach"][3]))
        run_worker(config, trainings, request, scene, ppo.gr00t_models(contract), args.run_dir)
        return 0
    directory = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-rehearsal")
    directory.mkdir(parents=True)
    for name, value in (("rehearsal-config.json", config), ("request.json", request), ("scene.json", scene),
                        ("controller-contract.json", contract)):
        write_json(directory / name, value)
    ppo.gr00t_models(contract)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--request", config["request"], "--worker", "--run-dir", str(directory)]
    code = run_stage(command, directory / "rehearsal.log", config["stage_timeout_seconds"])
    path = directory / "rehearsal-worker.json"
    evidence = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
    summary = {"status": evidence.get("status") if code == 0 else "failed", "exit_code": code, "run_dir": str(directory),
               "approach": evidence.get("phases", {}).get("approach"), "grasp": evidence.get("grasp"),
               "carry": evidence.get("phases", {}).get("carry"), "placement": evidence.get("placement"),
               "video": (evidence.get("video") or {}).get("status"), "error": evidence.get("error")}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "completed_rehearsal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
