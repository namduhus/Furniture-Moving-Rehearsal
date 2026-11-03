"""Furniture-moving rehearsal in one room: G1 walks to the chosen cabinet, grasps it, lifts it, carries it to the chosen
pose and sets it down, with GR00T Walk/Balance as the low-level controller.

The room comes from a scene spec (scripts/scene_from_image.py: SAM 3.1 detections + Muse Spark layout) and a request
naming the target, the user's corrections (size [width along its front, depth, height] and mass), the destination and
the strategy: carry, push, or auto (walk up and try to lift it; if it does not leave the floor, set it down, let Muse
Spark diagnose the attempt through scripts/rehearsal_advisor.py and push it if that is the decision). A push is a
straight push along one side of the item on furniture sliders, with the push policy over GR00T driven through a
course-frame adapter (the policy was trained pushing along +x from a standing start in front of the box). Walls surround the spec's width x depth,
every other furniture item is a static axis-aligned box, and the target is the dynamic cabinet the room policies were
trained with. Phases: approach (room approach checkpoint, or scripted docking: the zero residual) -> reach (arm IK to the grasp points) ->
grasp (palm joints re-anchored to the current palm pose and enabled) -> lift -> hold -> carry (room carry checkpoint, or
scripted docking) -> align (scripted docking to the root pose that puts the cabinet, as actually held, at the requested
pose) -> lower -> release -> withdraw -> back away. A failed approach stops the run; a failed carry still
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
import shutil
import sys
import traceback

import isaac_g1_carry as carry
import isaac_g1_lift as lift
import isaac_g1_ppo as ppo
import isaac_g1_ppo_eval as ppo_eval
import isaac_g1_rl as rl
import isaac_g1_room as room
import meta_models
import rehearsal_advisor as advisor
import room_geometry as rg
from isaac_g1_wbc import ROOT, trial_contract
from isaac_preflight import check_versions, run_stage, write_json
from isaac_video import encode_recording, require_ffmpeg

DEFAULT_CONFIG = ROOT / "configs/isaac_g1_rehearsal.json"
TIMELINE = ("square_up", "reach", "grasp", "lift", "hold", "align", "lower", "release", "withdraw", "step_back", "back_away")


def repo_path(value, prefix, suffix):
    """A repository-relative path under `prefix` ending in `suffix`, or a ValueError."""
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith(suffix) or ".." in Path(value).parts:
        raise ValueError(f"Expected a {suffix} path under {prefix}: {value}")
    return ROOT / value


def load_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {"approach_training", "carry_training", "approach_checkpoint", "carry_checkpoint", "push", "request",
                       "start_inside_door_m", "timeline_seconds", "back_away_mps", "set_down", "carry_min_command", "camera",
                       "stage_timeout_seconds", "scope"}:
        raise ValueError("Unreviewed rehearsal configuration keys")

    def number(value, low, high):
        return type(value) in (int, float) and low <= value <= high

    trainings = {}
    for task in ("approach", "carry"):
        trainings[task] = room.load_config(repo_path(config[f"{task}_training"], "configs/", ".json"))
        if trainings[task][0]["task"] != task:
            raise ValueError(f"{task}_training must be a {task} configuration")
        if config[f"{task}_checkpoint"] is not None:
            repo_path(config[f"{task}_checkpoint"], ("models/", "outputs/isaac/"), ".pt")
    a, c = trainings["approach"][0], trainings["carry"][0]
    if (a["object"] != c["object"] or a["push_profile"] != c["push_profile"] or a["arm_lift_m"] != c["arm_lift_m"] or
            a["decision_control_steps"] != c["decision_control_steps"]):
        raise ValueError("Approach and carry must share the cabinet, profile, arm lift and decision rate")
    push = config["push"]
    if (set(push) != {"training", "checkpoint", "slider_friction", "gap_m", "distance_m", "face_alignment_deg"} or
            not number(push["slider_friction"], 0.05, 0.15) or not number(push["gap_m"], 0.03, 0.15) or
            not isinstance(push["distance_m"], list) or len(push["distance_m"]) != 2 or
            not all(number(v, 0.2, 2.0) for v in push["distance_m"]) or not number(push["face_alignment_deg"], 1.0, 20.0)):
        raise ValueError("Invalid push settings")
    trainings["push"] = ppo.load_config(repo_path(push["training"], "configs/", ".json"))
    if push["checkpoint"] is not None:
        repo_path(push["checkpoint"], ("models/", "outputs/isaac/"), ".pt")
    if trainings["push"][0]["push_profile"] != a["push_profile"]:
        raise ValueError("The push policy must share the room tasks' profile")
    repo_path(config["request"], "configs/", ".json")
    t = config["timeline_seconds"]
    if set(t) != set(TIMELINE) or not all(number(v, 0.1, 5.0) for v in t.values()):
        raise ValueError("Invalid rehearsal timeline")
    down = config["set_down"]
    if (set(down) != {"touchdown_m", "position_tolerance_m", "yaw_tolerance_deg", "handover_m", "handover_s"} or
            not number(down["touchdown_m"], 0.001, 0.03) or not number(down["position_tolerance_m"], 0.01, 0.1) or
            not number(down["yaw_tolerance_deg"], 1.0, 10.0) or not number(down["handover_m"], 0.1, 0.5) or
            not number(down["handover_s"], 0.2, 5.0)):
        raise ValueError("Invalid set-down settings")
    cam = config["camera"]
    if (set(cam) != {"eye_from_front_m", "target_height_m"} or not isinstance(cam["eye_from_front_m"], list) or
            len(cam["eye_from_front_m"]) != 3 or not all(number(v, -8.0, 8.0) for v in cam["eye_from_front_m"]) or
            not number(cam["target_height_m"], 0.0, 2.0)):
        raise ValueError("Invalid rehearsal camera")
    if (not number(config["start_inside_door_m"], 0.3, 1.5) or not number(config["back_away_mps"], 0.0, 0.15) or
            not number(config["carry_min_command"], 0.0, 0.15) or
            type(config["stage_timeout_seconds"]) is not int or not 60 <= config["stage_timeout_seconds"] <= 3600):
        raise ValueError("Invalid start distance, back-away speed, carry minimum command or stage timeout")
    return config, trainings


def load_request(path):
    """Request and its scene spec. Optional keys continue in a room an earlier move left (scripts/rehearsal_sequence.py):
    `start` is the robot's root pose [x, y, yaw] in the simulation frame (default: inside the door) and `furniture` maps
    item ids to center_m / yaw_deg / size_m values that replace the spec's (items already moved or corrected)."""
    request = json.loads(Path(path).read_text())
    required = {"scene", "target", "corrections", "destination", "strategy"}
    if (not required <= set(request) <= required | {"start", "furniture"} or not isinstance(request["target"], str) or
            set(request["corrections"]) != {"size_m", "mass_kg"} or set(request["destination"]) != {"center_m", "yaw_deg"} or
            request["strategy"] not in ("carry", "push", "auto")):
        raise ValueError("A request needs scene, target, corrections {size_m, mass_kg}, destination {center_m, yaw_deg} "
                         "and strategy carry, push or auto (optional: start, furniture)")
    start = request.get("start")
    if start is not None and (not isinstance(start, list) or len(start) != 3 or
                              not all(type(v) in (int, float) for v in start)):
        raise ValueError("start must be the root pose [x, y, yaw] in the simulation frame")
    scene = json.loads(repo_path(request["scene"], "configs/scenes/", ".json").read_text())
    return request, with_furniture(scene, request.get("furniture", {}))


def with_furniture(scene, moved):
    """The scene spec with per-item center_m / yaw_deg / size_m overrides (a request's `furniture`)."""
    ids = {f["id"] for f in scene["scene"]["furniture"]}
    if not isinstance(moved, dict) or not set(moved) <= ids or not all(
            isinstance(v, dict) and set(v) <= {"center_m", "yaw_deg", "size_m"} for v in moved.values()):
        raise ValueError(f"furniture overrides need known ids ({sorted(ids)}) and center_m, yaw_deg or size_m")
    return {**scene, "scene": {**scene["scene"], "furniture": [{**f, **moved.get(f["id"], {})}
                                                               for f in scene["scene"]["furniture"]]}}


def facing_yaw(yaw_deg):
    """Simulation yaw of a cabinet whose front faces the spec yaw (0 = spec +y, toward the camera = sim -y; 90 = +x): the
    robot stands in front of it facing back along that direction, so the cabinet's x axis points away from the robot."""
    import numpy as np
    theta = np.radians(yaw_deg)
    return float(np.arctan2(np.cos(theta), -np.sin(theta)))


def sim_size(size_m):
    """Spec size [width along its front, depth, height] -> simulation box size [x, y, z] in the object frame, whose x
    axis points from the front into the item (depth) and y along the front (width)."""
    return [float(size_m[1]), float(size_m[0]), float(size_m[2])]


def spec_to_sim(xy, width, depth):
    """Scene-spec floor point (origin at the back-left corner, y toward the camera) -> centred simulation point."""
    import numpy as np
    return np.array([xy[0] - width/2, depth/2 - xy[1]])


def spec_room(scene, exclude=None):
    """The simulation room of a scene spec and the ids of its boxes: every furniture item except `exclude` as an
    axis-aligned box [cx, cy, sx, sy, sz] bounding its rotated footprint, clipped to the room."""
    import numpy as np
    spec = scene["scene"]
    width, depth = spec["room"]["width_m"], spec["room"]["depth_m"]
    boxes, ids = [], []
    for f in spec["furniture"]:
        if f["id"] == exclude:
            continue
        theta = np.radians(f["yaw_deg"])
        sx = abs(np.cos(theta))*f["size_m"][0] + abs(np.sin(theta))*f["size_m"][1]
        sy = abs(np.sin(theta))*f["size_m"][0] + abs(np.cos(theta))*f["size_m"][1]
        centre = spec_to_sim(f["center_m"], width, depth)
        x0, y0 = np.maximum(centre - [sx/2, sy/2], [-width/2, -depth/2])
        x1, y1 = np.minimum(centre + [sx/2, sy/2], [width/2, depth/2])
        if x1 > x0 and y1 > y0:
            boxes.append([(x0 + x1)/2, (y0 + y1)/2, x1 - x0, y1 - y0, f["size_m"][2]])
            ids.append(f["id"])
    return {"width_m": float(width), "depth_m": float(depth), "boxes": np.array(boxes, dtype=float).reshape(-1, 5)}, ids


def door_start(scene, inside_door):
    """Robot start pose (x, y, yaw) `inside_door` metres into the room from the spec's door, facing into the room.
    Side-wall doors are measured from the back wall, back-wall doors from the left wall."""
    import numpy as np
    spec = scene["scene"]
    width, depth, door = spec["room"]["width_m"], spec["room"]["depth_m"], spec["door"]
    return np.array({"right": [width/2 - inside_door, depth/2 - door["center_m"], np.pi],
                     "left": [-width/2 + inside_door, depth/2 - door["center_m"], 0.0],
                     "back": [door["center_m"] - width/2, depth/2 - inside_door, -np.pi/2]}[door["wall"]], dtype=float)


def entry_start(scene, inside_door, clearance):
    """Robot start inside the door with `clearance` from every wall and furniture item (the target included): the door
    start, else the first pose shifted along the door (up to 0.4 m, half the 0.8 m door width) or up to 0.4 m further in.
    In the edited pedestal room a pedestal stands 0.45 m from the door start. The door start when none is clear."""
    import numpy as np
    room, _ = spec_room(scene)
    nominal = door_start(scene, inside_door)
    inward = np.array([np.cos(nominal[2]), np.sin(nominal[2])])
    along = np.array([-inward[1], inward[0]])
    for deeper in (0.0, 0.2, 0.4):
        for shift in (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4):
            xy = nominal[:2] + deeper*inward + shift*along
            if float(rg.obstacle_distance(xy, room)) >= clearance:
                return np.r_[xy, nominal[2]]
    return nominal


def push_geometry(layout_room, item, size, placed, push_settings, hand_reach, inflation, margin):
    """Straight push of an item (x, y, yaw, box size [x, y, z]) to `placed`: push pose (palms gap_m from the pushed face,
    hand_reach ahead of the root) and course. Raises ValueError unless the destination is the item moved distance_m
    straight along one of its sides with the same heading, the pushed face is wide enough, the swept footprint is clear
    and the robot's line behind it keeps `inflation` from obstacles."""
    import numpy as np
    p = push_settings
    offset = np.asarray(placed[:2]) - item[:2]
    distance = float(np.linalg.norm(offset))
    tolerance = np.radians(p["face_alignment_deg"])
    if not p["distance_m"][0] <= distance <= p["distance_m"][1]:
        raise ValueError(f"A push moves the item {p['distance_m'][0]}-{p['distance_m'][1]} m, not {distance:.2f} m")
    # Headings equal up to a quarter turn: a box turned by 90 degrees is the same box.
    if abs(room.wrap(placed[2] - item[2] - round(float(room.wrap(placed[2] - item[2]))/(np.pi/2))*np.pi/2)) > tolerance:
        raise ValueError("A push keeps the item's heading")
    theta = float(np.arctan2(offset[1], offset[0]))
    side = round(float(room.wrap(theta - item[2]))/(np.pi/2))
    if abs(room.wrap(theta - item[2] - side*np.pi/2)) > tolerance:
        raise ValueError("Push straight along one side of the item")
    along, face = (size[0], size[1]) if side % 2 == 0 else (size[1], size[0])
    if face < 0.36:
        raise ValueError(f"The pushed face ({face} m) is narrower than the 0.36 m the palms need")
    u = offset/distance
    goal = np.r_[item[:2] - (along/2 + p["gap_m"] + hand_reach)*u, theta]
    for t in np.linspace(0.0, distance, int(np.ceil(distance/0.05)) + 1):
        if not room.footprint_clear(layout_room, item[:2] + t*u, item[2], size[:2], margin):
            raise ValueError("The push path of the item is blocked")
    behind = goal[:2] + np.linspace(0.0, distance, 20)[:, None]*u
    if float(rg.obstacle_distance(behind, layout_room).min()) < inflation:
        raise ValueError("No room for the robot to walk behind the item while pushing")
    return goal, {"origin": item[:2].tolist(), "theta": theta, "distance_m": distance, "along_m": along, "face_m": face}


def approach_path(layout_room, start, item, size, goal, settings):
    """Staged approach path to `goal` around the item (room inflation), the last straight segment with the smaller
    object inflation. A start inside the inflation (e.g. next to the item after a lift attempt) first steps to the
    nearest free cell."""
    import numpy as np
    grids = [rg.occupancy(layout_room, settings["room"]["grid_m"], settings["room"]["inflation_m"],
                          (item[:2], size[:2], item[2], inflation))
             for inflation in (settings["room"]["inflation_m"], settings["goal"]["object_inflation_m"])]
    entry = rg.nearest_free(grids[0], start[:2])
    path = None if entry is None else rg.staged_path(grids[0], entry, goal, settings["goal"]["staging_m"], grids[1])
    return path if path is None or np.allclose(entry, start[:2]) else np.vstack([np.asarray(start[:2], dtype=float), path])


def carry_side(layout_room, start, cabinet, size, reach, settings):
    """Grip side of a carry-tier item: its front when the robot can reach it, else the reachable side with the shortest
    approach; the tier's 0.30 x 0.28 m footprint is gripped the same from any side (the trio pedestals' spec fronts face
    the wall). Returns the item pose turned to that side, its pre-grasp goal and the approach path, or None."""
    import numpy as np
    options = []
    for turn in (0.0, np.pi/2, np.pi, -np.pi/2):
        item = np.r_[cabinet[:2], room.wrap(cabinet[2] + turn)]
        goal = rg.pregrasp_pose(item[:2], item[2], reach)
        if float(rg.obstacle_distance(goal[:2], layout_room)) < settings["room"]["inflation_m"]:
            continue
        path = approach_path(layout_room, start, item, size, goal, settings)
        if path is not None:
            options.append((turn != 0.0, float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), item, goal, path))
    return min(options, key=lambda option: option[:2])[2:] if options else None


def rehearsal_layout(scene, request, trainings, inside_door, reach, push_settings, hand_reach=0.415):
    """Room, start pose, item pose, goals and paths for one request (simulation frame, centred on the room).

    carry: the approach goal is the pre-grasp pose (`reach`) of the side chosen by carry_side, and the destination
    heading is turned the same way; the carry goal is the destination's pre-grasp pose. push: the
    approach goal is the push pose (see push_geometry). auto: the item must be grippable across one dimension (the
    trained cabinet's width, from the end facing the shorter approach) and pushable to the destination; the approach goal
    is the pre-grasp pose of that grip, and the push pose is planned too. The item is described as a box whose x axis
    points away from the robot during the grip. Raises ValueError when the request does not fit the strategy, the item or
    its destination overlaps other furniture, the door start is blocked, or no path exists.
    """
    import numpy as np
    a, c = trainings["approach"][0], trainings["carry"][0]
    spec, o, fix = scene["scene"], a["object"], request["corrections"]
    width, depth = spec["room"]["width_m"], spec["room"]["depth_m"]
    items = {f["id"]: f for f in spec["furniture"]}
    if request["target"] not in items:
        raise ValueError(f"Unknown target {request['target']}; furniture: {sorted(items)}")
    strategy, size = request["strategy"], sim_size(fix["size_m"])
    if strategy == "carry":
        if not np.allclose(size, o["size_m"], atol=0.02) or not c["mass_kg"][0] <= fix["mass_kg"] <= c["mass_kg"][1]:
            raise ValueError(f"The carry tier covers a {o['size_m']} m cabinet of {c['mass_kg'][0]}-{c['mass_kg'][1]} kg; "
                             "this target needs the push branch")
        size = o["size_m"]
    else:
        masses = trainings["push"][0]["randomization"]["mass_kg"]
        if not masses[0] <= fix["mass_kg"] <= masses[1] or size[2] < 0.9:
            raise ValueError(f"The push tier covers {masses[0]}-{masses[1]} kg items at least 0.9 m tall (palm height)")

    layout_room, _ = spec_room(scene, request["target"])
    target = items[request["target"]]
    cabinet = np.r_[spec_to_sim(target["center_m"], width, depth), facing_yaw(target["yaw_deg"])]
    placed = np.r_[spec_to_sim(request["destination"]["center_m"], width, depth), facing_yaw(request["destination"]["yaw_deg"])]
    if not room.footprint_clear(layout_room, cabinet[:2], cabinet[2], size[:2], 0.0):
        raise ValueError("The corrected cabinet overlaps a wall or other furniture; correct its position")
    if not room.footprint_clear(layout_room, placed[:2], placed[2], size[:2], c["goal"]["object_margin_m"]):
        raise ValueError("The destination overlaps a wall or other furniture")
    # A start carried over from an earlier move is where the robot already stands; it needs the planner's inflation.
    start = (np.asarray(request["start"], dtype=float) if "start" in request else
             entry_start(scene, inside_door, a["room"]["start_clearance_m"]))
    clearance = a["room"]["inflation_m"] if "start" in request else a["room"]["start_clearance_m"]
    if float(rg.obstacle_distance(start[:2], layout_room)) < clearance:
        raise ValueError(f"The {'start' if 'start' in request else 'door start'} is within {clearance} m of a wall or furniture")
    layout = {"strategy": strategy, "room": layout_room, "start": start, "placed": placed,
              "spec_transform": {"x": [1.0, width/2], "y": [-1.0, depth/2]}}

    def reachable(goal, inflation, name):
        if float(rg.obstacle_distance(goal[:2], layout_room)) < inflation:
            raise ValueError(f"The {name} is within {inflation} m of a wall or furniture")

    if strategy == "auto":
        # Grip across whichever horizontal dimension matches the trained cabinet's width; the box frame then points
        # its x axis away from the robot (from an end of the item when gripping across its depth).
        if abs(size[0] - o["size_m"][1]) <= 0.03:
            size, yaws = [size[1], size[0], size[2]], [cabinet[2] + np.pi/2, cabinet[2] - np.pi/2]
        elif abs(size[1] - o["size_m"][1]) <= 0.03:
            yaws = [cabinet[2]]
        else:
            raise ValueError(f"Neither horizontal side of the item matches the {o['size_m'][1]} m grip")
        grip = reach + (size[0] - o["size_m"][0])/2
        options = []
        for yaw in yaws:
            item = np.r_[cabinet[:2], room.wrap(yaw)]
            goal = rg.pregrasp_pose(item[:2], item[2], grip)
            path = approach_path(layout_room, start, item, size, goal, a)
            if float(rg.obstacle_distance(goal[:2], layout_room)) >= a["room"]["inflation_m"] and path is not None:
                options.append((float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), item, goal, path))
        if not options:
            raise ValueError("No end of the item can be reached for the lift attempt")
        facing = cabinet[2]
        _, cabinet, approach_goal, path = min(options, key=lambda option: option[0])
        # The destination heading in the same (grip) box frame; run 20261101T145650 compared the item-front heading
        # with the grip-frame box and reported a 98 deg error for a 7.9 deg one.
        placed = np.r_[placed[:2], room.wrap(placed[2] + cabinet[2] - facing)]
        layout["placed"], layout["grip_reach_m"] = placed, grip
        layout["push_goal"], layout["push"] = push_geometry(layout_room, cabinet, size, placed, push_settings, hand_reach,
                                                            a["room"]["inflation_m"], c["goal"]["object_margin_m"])
        reachable(layout["push_goal"], a["room"]["inflation_m"], "push pose")
    elif strategy == "carry":
        side = carry_side(layout_room, start, cabinet, size, reach, a)
        if side is None:
            raise ValueError("No side of the item can be reached for the grasp")
        facing = cabinet[2]
        cabinet, approach_goal, _ = side
        placed = np.r_[placed[:2], room.wrap(placed[2] + cabinet[2] - facing)]
        layout["placed"] = placed
        layout["carry_goal"] = rg.pregrasp_pose(placed[:2], placed[2], reach)
        reachable(layout["carry_goal"], c["room"]["inflation_m"], "carry goal")
    else:
        approach_goal, layout["push"] = push_geometry(layout_room, cabinet, size, placed, push_settings, hand_reach,
                                                      a["room"]["inflation_m"], c["goal"]["object_margin_m"])
    reachable(approach_goal, a["room"]["inflation_m"], "approach goal")
    layout.update(size_m=list(size), cabinet=cabinet, approach_goal=approach_goal,
                  approach_path=approach_path(layout_room, start, cabinet, size, approach_goal, a))
    if strategy == "carry":
        layout["carry_path"] = rg.staged_path(rg.occupancy(layout_room, c["room"]["grid_m"], c["room"]["inflation_m"]),
                                              approach_goal[:2], layout["carry_goal"], c["goal"]["staging_m"])
    if layout["approach_path"] is None or layout.get("carry_path", 0) is None:
        raise ValueError(f"No collision-free path for the approach or the {strategy}")
    return layout


def carry_status(carry, placement):
    """Rehearsal status after a carry and its set-down. A carry handed over near its goal, or one that timed out still
    holding the cabinet, counts when the alignment finished it: run 20261102T002432 hovered 15-18 cm short beside the TV
    bench for 40 s, and the alignment then set the pedestal down 2.8 cm / 1.1 deg off. Any other carry failure stays a
    failure."""
    finished = carry == "success" or (carry in ("near_goal", "time_out") and bool(placement and placement["within_tolerance"]))
    return "completed_rehearsal" if finished else "failed_carry"


def keep_walking(commands, minimum):
    """Commands [vx, vy, yaw rate] with every non-zero one shorter than `minimum` scaled up to it (zero stays zero)."""
    import numpy as np
    commands = np.array(commands, dtype=float)
    norm = np.linalg.norm(commands, axis=1)
    small = (norm > 1e-9) & (norm < minimum)
    commands[small] *= (minimum/norm[small])[:, None]
    return commands


def turn_z(quat, angle):
    """Quaternion [x, y, z, w] rotated by `angle` about the world z axis."""
    import numpy as np
    qx, qy, qz, qw = 0.0, 0.0, np.sin(angle/2), np.cos(angle/2)
    x, y, z, w = quat
    return np.array([qw*x + qx*w + qy*z - qz*y, qw*y - qx*z + qy*w + qz*x, qw*z + qx*y - qy*x + qz*w,
                     qw*w - qx*x - qy*y - qz*z])


def course_frame(pose, velocity, origin, theta):
    """World pose [x, y, z, qx, qy, qz, qw] and velocity [linear, angular] (env-local) -> the push course frame, whose
    origin is the item's start and whose x axis is the push direction (the frame the push policy was trained in)."""
    import numpy as np
    c, s = np.cos(theta), np.sin(theta)
    rotate = np.array([[c, s], [-s, c]])
    pose, velocity = np.asarray(pose, dtype=float), np.asarray(velocity, dtype=float)
    moved = np.r_[rotate @ (pose[:2] - origin), pose[2], turn_z(pose[3:], -theta)]
    turned = np.r_[rotate @ velocity[:2], velocity[2], rotate @ velocity[3:5], velocity[5]]
    return moved, turned


def square_yaw(pose):
    """The pose turned about z by a multiple of 90 degrees so its yaw lies in [-45, 45): a box turned by 90 degrees
    presents the same faces, and the push policy only saw boxes within a few degrees of its course. Tilt is kept."""
    import numpy as np
    yaw = float(ppo.yaw_of(np.asarray(pose)[None, 3:])[0])
    return np.r_[pose[:3], turn_z(pose[3:], -round(yaw/(np.pi/2))*np.pi/2)]


def to_spec(xy, transform):
    """Simulation (x, y) -> scene-spec (x, y): spec = sign * sim + offset per axis."""
    return [transform[axis][0]*float(v) + transform[axis][1] for axis, v in zip(("x", "y"), xy)]


def placement_goal(cabinet_in_root, placed):
    """Root pose (x, y, yaw) that puts a cabinet held at `cabinet_in_root` (4x4 in the root frame) at the placed pose."""
    import numpy as np
    yaw = placed[2] - np.arctan2(cabinet_in_root[1, 0], cabinet_in_root[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = cabinet_in_root[:2, 3]
    return np.array([placed[0] - (c*dx - s*dy), placed[1] - (s*dx + c*dy), float(room.wrap(yaw))])


def placement_error(cabinet_pose_xyzw, placed):
    """Position (m) and yaw (deg) error of the set-down cabinet against the requested pose (simulation frame)."""
    import numpy as np
    yaw = float(ppo.yaw_of(np.asarray(cabinet_pose_xyzw, dtype=float)[None, 3:])[0])
    return {"position_m": float(np.linalg.norm(np.asarray(cabinet_pose_xyzw[:2]) - placed[:2])),
            "yaw_deg": float(np.degrees(abs(room.wrap(yaw - placed[2])))), "yaw_rad": yaw}


class RehearsalEnv(room.RoomTaskEnv):
    """RoomTaskEnv over one fixed room in which the approach and carry phases share the robot without resets."""

    def __init__(self, layout, *args):
        self.layout, self.min_command = layout, 0.0
        super().__init__(*args)
        self.reset_on_done = False

    def control(self, commands, rpys, arms=None):
        """With min_command set, small non-zero commands are scaled up to it (keep_walking): under GR00T's 0.05 switch
        norm Balance takes over, and its stance lowered the hands 1.5 cm and grounded the pedestal mid-carry in run
        20261102T005739 (move 2, twice, both while the carry policy commanded 0.04-0.046)."""
        if self.min_command > 0:
            commands = keep_walking(commands, self.min_command)
            self.last_command = commands.copy()
        return super().control(commands, rpys, arms)

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


class PolicyShape:
    """Observation and action shapes of a policy's training environment, enough to rebuild the policy for inference."""

    def __init__(self, observations, actions, device, cfg):
        self.num_envs, self.num_actions, self.device, self.cfg, self.max_episode_length = 1, actions, device, cfg, 1
        self.observations = observations

    def get_observations(self):
        return self.observations


def run_worker(config, trainings, request, scene, model_paths, directory):
    import numpy as np
    import torch
    import warp as wp
    from PIL import Image
    from g1_wbc import finite, pose_matrix, quat_xyzw_matrices

    approach, profile, asset, runtime = trainings["approach"]
    carry_config, push_config = trainings["carry"][0], trainings["push"][0]
    pushing, auto = request["strategy"] == "push", request["strategy"] == "auto"
    contract, reference = trial_contract(profile)
    policy = ppo.gr00t_policy(contract, model_paths)
    encoder = require_ffmpeg(directory)
    mass = float(request["corrections"]["mass_kg"])
    # The room depends on the reach only through goals and paths, so the walls are authored from a layout with the
    # training defaults; the layout is re-checked with the measured reaches once the scene exists.
    layout = rehearsal_layout(scene, request, trainings, config["start_inside_door_m"], 0.411, config["push"])
    # A carried item is the trained cabinet; a pushed or auto item has its corrected size (in the grip frame for auto)
    # and stands on furniture sliders (they do not change lifting).
    o = approach["object"] if request["strategy"] == "carry" else {**approach["object"], "size_m": layout["size_m"]}
    friction = o["friction"] if request["strategy"] == "carry" else config["push"]["slider_friction"]
    torch.manual_seed(approach["seed"])
    wp.config.enable_backward = False
    from isaaclab.app import launch_simulation

    scene_profile, built = carry.carry_scene({**approach, "object": o}, profile, asset, runtime, contract, reference,
                                             np.array([mass]), num_envs=1, env_spacing=approach["env_spacing_m"],
                                             with_camera=True)
    base = lift.base_pose(reference, asset, contract)
    trained = lift.attachment_frames(reference, base, approach["object"], contract["height_command"])
    frames = lift.attachment_frames(reference, base, o, contract["height_command"]) if auto else trained
    joints = [n for side in lift.SIDES for n in lift.arm_joints(reference, side)]
    angles = {name: {n: v for side in lift.SIDES for n, v in lift.arm_pose(reference, base, side, dz).items()}
              for name, dz in (("start", 0.0), ("lift", approach["arm_lift_m"]), ("place", -0.02))}
    arm = {name: np.array([[pose[n] for n in joints]]) for name, pose in angles.items()}
    evidence = directory / "rehearsal-worker.json"
    result = {"status": "running", "request": request, "mass_kg": mass, "friction": friction,
              "checkpoints": {"approach": config["approach_checkpoint"],
                              **({} if pushing else {"carry": config["carry_checkpoint"]}),
                              **({"push": config["push"]["checkpoint"]} if pushing or auto else {})},
              "phases": {}}
    write_json(evidence, result)
    env_path = "/World/envs/env_0"
    with launch_simulation(built.cfg, {**rl.LAUNCH_SETTINGS, "enable_cameras": True, "device": runtime["device"]}):
        try:
            def edits(stage, robot, box, env):
                # Carrying filters hand collisions with the cabinet (the palms meet its sides); pushing needs them, so
                # an auto item (grasped, then perhaps pushed) keeps them, its grip faces flush with the palms.
                if not pushing:
                    lift.apply_hand_attachment(stage, robot, box, env, frames, False, filter_hands=not auto)
                result["walls"] = len(carry.apply_corridor(stage, env, rg.wall_boxes(
                    layout["room"], approach["room"]["wall_thickness_m"], approach["room"]["wall_height_m"])))

            s = rl.prepare_push_scene(built, 1, scene_profile, asset, runtime, contract, reference, directory, result,
                                      evidence, box_properties=(np.array([mass]), np.array([float(friction)])),
                                      extra_edits=edits, check_box_start=False)
            palm_ids = {side: s.bodies.index(f"{side}_hand_palm_link") for side in lift.SIDES}

            def palm_poses():
                poses = finite(s.batch(s.robot.data.body_link_pose_w), (1, len(s.bodies), 7))
                return {side: [pose_matrix(poses[0, k])] for side, k in palm_ids.items()}

            # Trained cabinet pose both palm joints expect at the crouch start, relative to the root (pre-grasp reach).
            object_in_root = np.linalg.inv(pose_matrix(s.root[0])) @ palm_poses()["left"][0] @ trained["palm_object"]["left"]
            reach = float(object_in_root[0, 3])
            layout = rehearsal_layout(scene, request, trainings, config["start_inside_door_m"], reach, config["push"],
                                      float(s.hand_reach))
            result["layout"] = {"reach_m": reach, "hand_reach_m": float(s.hand_reach), "size_m": layout["size_m"],
                                "room": {**layout["room"], "boxes": layout["room"]["boxes"].tolist()},
                                **{k: np.asarray(layout[k]).tolist() for k in ("start", "cabinet", "placed", "approach_goal",
                                                                               "carry_goal") if k in layout},
                                "push": layout.get("push"), "spec_transform": layout["spec_transform"]}
            write_json(evidence, result)

            from tensordict import TensorDict
            ids = [s.names.index(n) for n in joints]
            room_config = {**approach, "num_envs": 1, "object": o,
                           "room": {**approach["room"], "obstacles": [0, max(1, len(layout["room"]["boxes"]))]}}
            env = RehearsalEnv(layout, s, policy, contract, profile, asset, runtime, room_config, [layout["room"]],
                               np.array([mass]), object_in_root, (ids, arm["start"][0], arm["lift"][0]),
                               lambda groups: TensorDict(groups, batch_size=[1]))
            actors = {}
            for task, training in (("approach", approach),) + ((("carry", carry_config),) if not pushing else ()):
                if config[f"{task}_checkpoint"] is None:
                    actors[task] = room.scripted_actor(training)
                else:
                    actors[task], result["phases"][f"{task}_checkpoint_iteration"] = ppo_eval.load_policy(
                        env, training, ROOT / config[f"{task}_checkpoint"], runtime["device"])
            if pushing or auto:
                # The push policy's own observation (course frame) and critic (plus normalized mass and friction).
                r = push_config["randomization"]
                privileged = np.array([[(mass - r["mass_kg"][0])/(r["mass_kg"][1] - r["mass_kg"][0]),
                                        (friction - r["friction"][0])/(r["friction"][1] - r["friction"][0])]])
                identity = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
                size = len(ppo.task_observation(identity, np.zeros((1, 6)), identity, np.zeros((1, 6)), np.zeros((1, 3)),
                                                np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 1)))[0])

                def push_obs(task):
                    return TensorDict({"policy": torch.as_tensor(task, dtype=torch.float32, device=runtime["device"]),
                                       "critic": torch.as_tensor(np.c_[task, privileged], dtype=torch.float32,
                                                                 device=runtime["device"])}, batch_size=[1])

                if config["push"]["checkpoint"] is None:
                    scripted = ppo.normalized_action([push_config["scripted_action"][k] for k in ppo.ACTION_KEYS],
                                                     push_config["action_bounds"])
                    actors["push"] = lambda obs: torch.as_tensor(scripted[None], dtype=torch.float32)
                else:
                    actors["push"], result["phases"]["push_checkpoint_iteration"] = ppo_eval.load_policy(
                        PolicyShape(push_obs(np.zeros((1, size), dtype=np.float32)), len(ppo.ACTION_KEYS),
                                    runtime["device"], push_config),
                        push_config, ROOT / config["push"]["checkpoint"], runtime["device"])

            camera, (width, height) = s.scene["camera"], (profile["camera"]["width"], profile["camera"]["height"])
            cam, w, d = config["camera"], layout["room"]["width_m"], layout["room"]["depth_m"]
            # The camera stands in front of the room (the photo's side, sim -y) and looks at the middle of the item's start
            # and destination (the room centre left the bookcase at the top edge in the 5.8 m deep room of run 141513).
            focus = (np.asarray(layout["cabinet"][:2]) + np.asarray(layout["placed"][:2]))/2
            eye = s.origins[0] + [focus[0] + cam["eye_from_front_m"][0], -d/2 - cam["eye_from_front_m"][1], cam["eye_from_front_m"][2]]
            target = s.origins[0] + [focus[0], focus[1], cam["target_height_m"]]
            camera.set_world_poses_from_view(s.device(eye[None]), s.device(target[None]))
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

            def drive(task, act, handover=None):
                """Policy decisions until the task ends (success, failure or its episode limit), or, with `handover`
                (metres, decisions), as "near_goal" once the root has stayed that close to its goal that long."""
                near = 0
                while True:
                    env.step(act(env.obs))
                    record(task, env.last_command[0])
                    if 0 in env.last_completed:
                        return env.last_completed[0]
                    error = float(env.last_task["position_error_m"][0])
                    near = near + 1 if handover and error <= handover[0] else 0
                    if handover and near >= handover[1]:
                        return {"outcome": "near_goal", "progress_m": float(env.start_remaining[0] - env.previous_remaining[0]),
                                "seconds": float(env.length[0]*env.decision_dt), "position_error_m": error,
                                "yaw_error_deg": float(np.degrees(abs(env.last_task["yaw_error_rad"][0])))}

            # Station keeping while setting the load down: plain Balance stepped the robot back 18 cm and dragged the
            # cabinet 10 cm in run 20261101T103451 (14 cm in the lift probes), so these phases dock to the carry goal.
            down = config["set_down"]
            keep = {**carry_config, "goal": {**carry_config["goal"], "position_tolerance_m": down["position_tolerance_m"],
                                             "yaw_tolerance_deg": down["yaw_tolerance_deg"]}}

            def aim():
                """Root goal that puts the cabinet, as held now, at the requested pose."""
                root = pose_matrix(finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0])
                held = np.linalg.inv(root) @ pose_matrix(finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0])
                return placement_goal(held, layout["placed"])

            def landed():
                item = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                return bool(lift.lowest_point_z(item, o["size_m"]) - s.origins[0, 2] <= down["touchdown_m"])

            def scripted(name, begin, end, command=(0.0, 0.0, 0.0), dock=False, follow=False, until=None):
                """Scripted decisions with arm targets blended from begin to end; dock=True replaces the fixed command by
                scripted docking to the current goal (set_down tolerance) at every decision, follow=True first re-aims
                that goal at the cabinet as held now, and `until` ends the phase early once it returns True. The arm
                target the phase ended at is kept as arm["held"]."""
                decisions = int(round(config["timeline_seconds"][name]/env.decision_dt))
                flags, target = {"invalid_action": False, "fallen": False}, begin
                for d in range(decisions):
                    if follow:
                        env.goals[0] = aim()
                    if dock:
                        command = tuple(room.action_commands(room.docking_actions(
                            env.observe()["policy"].detach().cpu().numpy(), keep), carry_config["action_bounds"])[0])
                    target = begin + (d + 1)/decisions*(end - begin)
                    for step in env.hold(begin + d/decisions*(end - begin), target, 1, command):
                        flags = {k: flags[k] or step[k] for k in flags}
                    record(name, command)
                    if until is not None and until():
                        flags["stopped_s"] = round((d + 1)*env.decision_dt, 3)
                        break
                arm["held"] = target
                result["phases"][name] = flags
                return not (flags["invalid_action"] or flags["fallen"])

            def push(course):
                """Push policy decisions in the course frame until the push succeeds, fails or reaches its time limit."""
                cfg = {**push_config, "goal_distance_m": course["distance_m"]}
                origin, theta = np.asarray(course["origin"]), course["theta"]
                limit = int(round(cfg["episode_seconds"]/env.decision_dt))
                last, previous, zeros = np.zeros((1, len(ppo.ACTION_KEYS))), np.zeros(1), np.zeros((1, 3))

                def state():
                    robot = finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0]
                    box = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                    robot_c, robot_v = course_frame(np.r_[robot[:3] - s.origins[0], robot[3:]],
                                                    finite(s.batch(s.robot.data.root_link_vel_w), (1, 6))[0], origin, theta)
                    box_c, box_v = course_frame(np.r_[box[:3] - s.origins[0], box[3:]],
                                                finite(s.batch(s.box.data.root_link_vel_w), (1, 6))[0], origin, theta)
                    return robot, robot_c[None], robot_v[None], square_yaw(box_c)[None], box_v[None]

                for k in range(limit):
                    robot, robot_c, robot_v, box_c, box_v = state()
                    gravity = -(quat_xyzw_matrices(robot[None, 3:]) @ s.rotation)[:, 2, :]
                    task = ppo.task_observation(robot_c, robot_v, box_c, box_v, zeros, np.array([cfg["goal_distance_m"]]),
                                                gravity, last, np.array([[k/limit]]))
                    action = np.clip(np.asarray(actors["push"](push_obs(task)).detach().cpu().numpy(), dtype=float), -1.0, 1.0)
                    commands, rpys = ppo.action_commands(action, cfg["action_bounds"])
                    low = {"invalid_action": np.zeros(1, dtype=bool), "fallen": np.zeros(1, dtype=bool)}
                    for _ in range(cfg["decision_control_steps"]):
                        _, _, invalid, fallen = env.control(commands, rpys)
                        low["invalid_action"] |= invalid
                        low["fallen"] |= fallen
                    _, robot_c, _, box_c, box_v = state()
                    _, done, outcome, progress = ppo.task_step(box_c, robot_c, zeros, np.zeros(1), previous, action, last, low,
                                                               cfg, env.decision_dt)
                    previous, last = box_c[:, 0], action
                    record("push", commands[0])
                    if done[0]:
                        return {"outcome": ppo.OUTCOMES[int(outcome[0])], "progress_m": float(progress[0]),
                                "seconds": (k + 1)*env.decision_dt, "lateral_m": float(abs(box_c[0, 1]))}
                    # Stop once the item's coast would reach the goal: one more decision of pushing, then sliding friction
                    # on the sliders. Run 20261101T145650 stopped 1.8 cm short at 0.2 m/s and the bookcase slid 4.3 cm on.
                    speed = max(float(box_v[0, 0]), 0.0)
                    coast = speed*env.decision_dt + speed**2/(2*friction*9.81)
                    if previous[0] + coast >= cfg["goal_distance_m"]:
                        return {"outcome": "success", "progress_m": float(previous[0]), "coast_m": coast,
                                "seconds": (k + 1)*env.decision_dt, "lateral_m": float(abs(box_c[0, 1]))}
                return {"outcome": "time_out", "progress_m": float(previous[0]), "seconds": limit*env.decision_dt,
                        "lateral_m": float(abs(box_c[0, 1]))}

            def measure():
                """Placement error of the item against the requested pose (carry tolerance)."""
                item = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                local = np.r_[item[:3] - s.origins[0], item[3:]]
                error = placement_error(local, layout["placed"])
                tolerance = carry_config["goal"]
                result["placement"] = {
                    **error, "cabinet_spec_xy_m": to_spec(local[:2], layout["spec_transform"]), "tilt_deg": lift.tilt_deg(item),
                    "within_tolerance": bool(error["position_m"] <= tolerance["position_tolerance_m"] and
                                             error["yaw_deg"] <= tolerance["yaw_tolerance_deg"])}

            def step_back():
                """Walk backwards until the root keeps the room inflation plus 5 cm from the item (at most the step_back
                time), so the robot can turn without sweeping it: run 20261101T141513 stepped back only 5 cm in 1 s and
                its arms pushed the bookcase while turning toward the push pose."""
                command = (-config["back_away_mps"], 0.0, 0.0)
                flags, clearance = {"invalid_action": False, "fallen": False}, 0.0
                for _ in range(int(round(config["timeline_seconds"]["step_back"]/env.decision_dt))):
                    robot = finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0]
                    item = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                    clearance = float(rg.rotated_distance(robot[:2] - s.origins[0, :2], item[:2] - s.origins[0, :2],
                                                          layout["size_m"][:2], ppo.yaw_of(item[None, 3:])[0]))
                    behind = float(rg.obstacle_distance(robot[:2] - s.origins[0, :2], layout["room"]))
                    if clearance >= approach["room"]["inflation_m"] + 0.05 or behind < 0.35:
                        break
                    for step in env.hold(arm["start"], arm["start"], 1, command):
                        flags = {k: flags[k] or step[k] for k in flags}
                    record("step_back", command)
                result["phases"]["step_back"] = {**flags, "clearance_m": clearance}

            def diagnose():
                """Muse Spark diagnoses the lift attempt from the measured outcome and the current frame; the capability
                rules accept or override its strategy. Without the Meta Model API (no key or network) the rules decide,
                and the error is recorded."""
                settings = json.loads(advisor.DEFAULT_CONFIG.read_text())
                phases = [frame["phase"] for frame in replay]
                held = len(phases) - 1 - phases[::-1].index("hold")
                frames = [case / "frames" / f"{k:06d}.png" for k in (held, count - 1)]     # end of the attempt, now
                try:
                    return advisor.advise(meta_models.MetaModels(), settings, request, scene,
                                          {**result, "status": "lift_attempt"}, frames)
                except Exception as error:
                    return {"mode": "rules_only", "error": f"{type(error).__name__}: {error}",
                            "decision": advisor.decide({"strategy": "push"}, request["corrections"]["size_m"], mass, settings)}

            def reposition_and_push():
                """Walk from where the lift attempt left the robot to the push pose of the item as it stands now, then
                push it to the destination; returns the rehearsal status."""
                robot = finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0]
                item = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                here = np.r_[item[:2] - s.origins[0, :2], ppo.yaw_of(item[None, 3:])[0]]
                try:
                    goal, course = push_geometry(layout["room"], here, layout["size_m"], layout["placed"], config["push"],
                                                 float(s.hand_reach), approach["room"]["inflation_m"],
                                                 carry_config["goal"]["object_margin_m"])
                except ValueError as error:
                    result["push_plan_error"] = str(error)
                    return "failed_push_plan"
                path = approach_path(layout["room"], robot[:2] - s.origins[0, :2], here, layout["size_m"], goal, approach)
                if path is None:
                    result["push_plan_error"] = "no collision-free path to the push pose"
                    return "failed_push_plan"
                result["layout"].update(push=course, push_goal=goal.tolist())
                env.begin(room_config, goal, path, approach["lift_prefix_seconds"]["settle"])
                result["phases"]["reposition"] = drive("reposition", actors["approach"])
                write_json(evidence, result)
                if result["phases"]["reposition"]["outcome"] != "success":
                    return "failed_reposition"
                result["phases"]["push"] = push(course)
                if (result["phases"]["push"]["outcome"] != "fallen" and
                        scripted("back_away", arm["start"], arm["start"], (-config["back_away_mps"], 0.0, 0.0))):
                    measure()
                return "completed_rehearsal" if result["phases"]["push"]["outcome"] == "success" else "failed_push"

            record("start")
            result["phases"]["approach"] = drive("approach", actors["approach"])
            write_json(evidence, result)
            status = "failed_approach"
            if result["phases"]["approach"]["outcome"] == "success" and pushing:
                result["phases"]["push"] = push(layout["push"])
                write_json(evidence, result)
                status = "failed_push"
                if (result["phases"]["push"]["outcome"] != "fallen" and
                        scripted("back_away", arm["start"], arm["start"], (-config["back_away_mps"], 0.0, 0.0))):
                    measure()
                    if result["phases"]["push"]["outcome"] == "success":
                        status = "completed_rehearsal"
            elif result["phases"]["approach"]["outcome"] == "success":
                # Square up first: the approach succeeds within 6 cm / 8 deg, but the reach corrects at most 5 cm sideways.
                # Run 20261101T175925 (trio, pedestal_3) stopped 6.9 cm to the side, the right palm met the pedestal 2.8 cm
                # off its face and the lift tilted it onto one edge.
                scripted("square_up", arm["start"], arm["start"], dock=True)
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
                item = finite(s.batch(s.box.data.root_link_pose_w), (1, 7))[0]
                bottom = lift.lowest_point_z(item, o["size_m"]) - s.origins[0, 2]
                result["lift"] = {"lifted": bool(bottom >= carry_config["termination"]["object_floor_clearance_m"]),
                                  "bottom_m": float(bottom), "mass_kg": mass}
                in_tier = advisor.carry_rule(request["corrections"]["size_m"], mass,
                                             json.loads(advisor.DEFAULT_CONFIG.read_text()))[0]
                status = "failed_lift"
                if ok and auto and not (result["lift"]["lifted"] and in_tier):
                    # The item did not leave the floor (or would leave the carry tier): set it down, let go, step back,
                    # and let Meta diagnose the attempt.
                    scripted("lower", arm["lift"], arm["place"])
                    result["release_authored"] = lift.set_attachment(s.sim.stage, env_path, False)
                    scripted("release", arm["place"], arm["place"])
                    scripted("withdraw", arm["place"], arm["start"])
                    step_back()
                    result["advice"] = diagnose()
                    write_json(evidence, result)
                    strategy = result["advice"]["decision"]["strategy"]
                    status = reposition_and_push() if strategy == "push" else f"advised_{strategy}"
                elif ok and result["lift"]["lifted"]:
                    xy = finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0, :2] - s.origins[0, :2]
                    path = rg.staged_path(rg.occupancy(layout["room"], carry_config["room"]["grid_m"],
                                                       carry_config["room"]["inflation_m"]), xy, layout["carry_goal"],
                                          carry_config["goal"]["staging_m"])
                    env.arm_start = env.arm_lift = arm["lift"][0]
                    env.begin(carry_config, layout["carry_goal"], layout["carry_path"] if path is None else path,
                              sum(carry_config["lift_prefix_seconds"].values()))
                    # Hand over to the alignment near the goal: beside furniture the carry policy hovered 15-18 cm short
                    # until its time limit (runs 20261102T002432/003755), and in the second run the tilted pedestal's
                    # corner dipped under the 1 cm floor clearance while hovering.
                    env.min_command = config["carry_min_command"]
                    result["phases"]["carry"] = drive("carry", actors["carry"],
                                                      (down["handover_m"], int(round(down["handover_s"]/env.decision_dt))))
                    env.min_command = 0.0
                    write_json(evidence, result)
                    status = "failed_carry"
                    if result["phases"]["carry"]["outcome"] != "fallen":
                        # Dock only near the goal; a carry that failed on the way sets the load down where it is.
                        near = result["phases"]["carry"]["position_error_m"] <= 0.5
                        if near:
                            # Precision placement: the carry stops within 10 cm of its root goal and the grasp holds the
                            # cabinet a few cm off nominal (14 cm placement error in run 20261101T105656), so dock to the
                            # root pose that puts the cabinet, as held now, at the requested pose, re-aimed every decision.
                            root = finite(s.batch(s.robot.data.root_link_pose_w), (1, 7))[0]
                            goal = aim()
                            env.goals[0], env.paths[0] = goal, np.array([root[:2] - s.origins[0, :2], goal[:2]])
                            scripted("align", arm["lift"], arm["lift"], dock=True, follow=True)
                        # Lower toward the grasp pose (the arms that held it resting on the floor) and stop at touchdown,
                        # still re-aiming: lowering to the nominal place pose, 3 cm under the grasp, pressed the cabinet
                        # into the floor in run 20261101T142242; the reaction walked the robot back 10 cm against the
                        # docking and dragged the cabinet 5.6 cm off the 2.4 cm the alignment had reached.
                        scripted("lower", arm["lift"], arm["reach"], dock=near, follow=near, until=landed)
                        if near:
                            result["placement_goal"] = np.asarray(env.goals[0]).tolist()
                        result["release_authored"] = lift.set_attachment(s.sim.stage, env_path, False)
                        if (scripted("release", arm["held"], arm["held"], dock=near) and
                                scripted("withdraw", arm["held"], arm["start"]) and
                                scripted("back_away", arm["start"], arm["start"], (-config["back_away_mps"], 0.0, 0.0))):
                            # Leave room to turn: the next move of a sequence starts here.
                            step_back()
                            measure()
                            status = carry_status(result["phases"]["carry"]["outcome"], result["placement"])
                            result["phases"]["carry"]["finished_by_align"] = (result["phases"]["carry"]["outcome"] != "success"
                                                                              and status == "completed_rehearsal")
            result["video"] = encode_recording(case, binary=encoder, count=count, fps=1/Fraction(str(env.decision_dt)),
                                               width=width, height=height, require_motion=False)
            # Key frames for the Meta model diagnosis (the frames folder is not downloaded): the end of each walking
            # phase and the final scene.
            phases = [frame["phase"] for frame in replay]
            keys = sorted({len(phases) - 1 - phases[::-1].index(p) for p in ("approach", "hold", "carry", "push") if p in phases} |
                          {len(phases) - 1})
            (case / "key_frames").mkdir()
            for k in keys:
                shutil.copyfile(case / "frames" / f"{k:06d}.png", case / "key_frames" / f"{k:06d}_{phases[k]}.png")
            result["key_frames"] = [f"{k:06d}_{phases[k]}.png" for k in keys]
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
    layout = rehearsal_layout(scene, request, trainings, config["start_inside_door_m"], 0.411, config["push"])
    if args.dry_run:
        import numpy as np

        def length(path):
            return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

        pushing = request["strategy"] != "carry"
        print(json.dumps({"status": "local_plan_only", "strategy": request["strategy"], "target": request["target"],
                          "mass_kg": request["corrections"]["mass_kg"], "size_m": layout["size_m"],
                          "room_m": [layout["room"]["width_m"], layout["room"]["depth_m"]], "obstacles": len(layout["room"]["boxes"]),
                          **{k: np.round(layout[k], 3).tolist() for k in ("start", "cabinet", "approach_goal", "placed", "carry_goal",
                                                                          "push_goal") if k in layout},
                          "approach_path_m": length(layout["approach_path"]),
                          **({"push": layout["push"]} if pushing else {"carry_path_m": length(layout["carry_path"])}),
                          "checkpoints": {"approach": config["approach_checkpoint"] or "scripted docking",
                                          **({"carry": config["carry_checkpoint"] or "scripted docking"}
                                             if request["strategy"] != "push" else {}),
                                          **({"push": config["push"]["checkpoint"] or "scripted push"} if pushing else {})}},
                         indent=2))
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("Execute in the existing pinned Isaac image; use --dry-run on Mac")
    for path in (config["approach_checkpoint"], config["carry_checkpoint"], config["push"]["checkpoint"]):
        if path is not None and not (ROOT / path).is_file():
            parser.error(f"Missing checkpoint: {path}")
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
               "strategy": request["strategy"], "approach": evidence.get("phases", {}).get("approach"),
               "grasp": evidence.get("grasp"), "carry": evidence.get("phases", {}).get("carry"),
               "push": evidence.get("phases", {}).get("push"), "placement": evidence.get("placement"),
               "video": (evidence.get("video") or {}).get("status"), "error": evidence.get("error")}
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "completed_rehearsal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
