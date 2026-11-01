"""Room geometry for the furniture rehearsal: random rooms, obstacle distances, grid paths and look-ahead points.

Rooms are centred on their environment origin: interior x in [-W/2, W/2], y in [-D/2, D/2], walls outside it. Furniture
obstacles are axis-aligned boxes; the carried object may be rotated. Paths come from 8-connected A* on an occupancy grid
whose obstacles are inflated by a clearance radius. Everything is plain NumPy so it runs in tests and in the simulator.
"""

from __future__ import annotations

import heapq

import numpy as np


def sample_room(settings, rng):
    """Room size and 1-3 non-overlapping axis-aligned furniture boxes [cx, cy, sx, sy, sz] that leave a connected floor."""
    for _ in range(100):
        width, depth = rng.uniform(*settings["width_m"]), rng.uniform(*settings["depth_m"])
        boxes = []
        for _ in range(int(rng.integers(settings["obstacles"][0], settings["obstacles"][1] + 1))):
            size = np.array([rng.uniform(*settings["obstacle_size_m"][k]) for k in range(3)])
            if rng.random() < 0.5:
                size[[0, 1]] = size[[1, 0]]
            for _ in range(50):
                centre = rng.uniform([-width/2 + size[0]/2, -depth/2 + size[1]/2], [width/2 - size[0]/2, depth/2 - size[1]/2])
                if all(abs(centre[0] - b[0]) > (size[0] + b[2])/2 + settings["obstacle_gap_m"] or
                       abs(centre[1] - b[1]) > (size[1] + b[3])/2 + settings["obstacle_gap_m"] for b in boxes):
                    boxes.append([*centre, *size])
                    break
        room = {"width_m": float(width), "depth_m": float(depth), "boxes": np.array(boxes, dtype=float).reshape(-1, 5)}
        grid = occupancy(room, settings["grid_m"], settings["inflation_m"])
        if largest_component(grid["free"]).mean() >= 0.5*grid["free"].mean() > 0:
            return room
    raise RuntimeError("Could not sample a connected room")


def wall_boxes(room, thickness, height):
    """Static wall boxes (centre, size) around the room interior, in the environment frame."""
    w, d, t = room["width_m"]/2, room["depth_m"]/2, thickness
    rects = [(-w - t, w + t, -d - t, -d), (-w - t, w + t, d, d + t), (-w - t, -w, -d, d), (w, w + t, -d, d)]
    walls = [([(x0 + x1)/2, (y0 + y1)/2, height/2], [x1 - x0, y1 - y0, height]) for x0, x1, y0, y1 in rects]
    furniture = [([b[0], b[1], b[4]/2], [b[2], b[3], b[4]]) for b in room["boxes"]]
    return walls + furniture


def rect_distance(points, rects):
    """Signed distance from points (..., 2) to the nearest axis-aligned rectangle [cx, cy, sx, sy]; negative inside."""
    points = np.asarray(points, dtype=float)
    if len(rects) == 0:
        return np.full(points.shape[:-1], np.inf)
    rects = np.asarray(rects, dtype=float)
    q = np.abs(points[..., None, :] - rects[:, :2]) - rects[:, 2:4]/2
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return np.min(outside + inside, axis=-1)


def rotated_distance(points, centre, size_xy, yaw):
    """Signed distance from points (..., 2) to one rectangle of size (sx, sy) rotated by yaw about its centre."""
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(points, dtype=float) - centre
    local = np.stack([c*d[..., 0] + s*d[..., 1], -s*d[..., 0] + c*d[..., 1]], axis=-1)
    q = np.abs(local) - np.asarray(size_xy)/2
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def obstacle_distance(points, room):
    """Distance from points to the walls (room outside) and furniture boxes."""
    points = np.asarray(points, dtype=float)
    w, d = room["width_m"]/2, room["depth_m"]/2
    to_walls = np.minimum(np.minimum(points[..., 0] + w, w - points[..., 0]), np.minimum(points[..., 1] + d, d - points[..., 1]))
    return np.minimum(to_walls, rect_distance(points, room["boxes"][:, :4]))


def occupancy(room, resolution, inflation, extra=None):
    """Free-cell grid over the interior: a cell is free when its centre is at least `inflation` from every obstacle.

    extra: optional (centre, size_xy, yaw, inflation) rotated rectangle added as an obstacle (e.g. the target object).
    """
    nx, ny = int(np.ceil(room["width_m"]/resolution)), int(np.ceil(room["depth_m"]/resolution))
    xs = -room["width_m"]/2 + (np.arange(nx) + 0.5)*resolution
    ys = -room["depth_m"]/2 + (np.arange(ny) + 0.5)*resolution
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    free = obstacle_distance(centres, room) >= inflation
    if extra is not None:
        centre, size_xy, yaw, extra_inflation = extra
        free &= rotated_distance(centres, centre, size_xy, yaw) >= extra_inflation
    return {"free": free, "xs": xs, "ys": ys, "resolution": resolution}


def largest_component(free):
    """Mask of the largest 4-connected free region."""
    labels = np.zeros(free.shape, dtype=int)
    best, best_size, label = np.zeros_like(free), 0, 0
    for start in zip(*np.nonzero(free)):
        if labels[start]:
            continue
        label += 1
        stack, members = [start], []
        labels[start] = label
        while stack:
            i, j = stack.pop()
            members.append((i, j))
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and not labels[a, b]:
                    labels[a, b] = label
                    stack.append((a, b))
        if len(members) > best_size:
            best_size, best = len(members), labels == label
    return best


def cell_of(grid, point):
    i = int(np.clip(np.floor((point[0] - grid["xs"][0])/grid["resolution"] + 0.5), 0, len(grid["xs"]) - 1))
    j = int(np.clip(np.floor((point[1] - grid["ys"][0])/grid["resolution"] + 0.5), 0, len(grid["ys"]) - 1))
    return i, j


def plan_path(grid, start, goal):
    """8-connected A* between the free cells holding start and goal; the polyline (k, 2) runs from start to goal.

    Returns None when either end is blocked or no path exists.
    """
    free = grid["free"]
    a, b = cell_of(grid, start), cell_of(grid, goal)
    if not free[a] or not free[b]:
        return None
    steps = [(di, dj, np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    best, parent, frontier = {a: 0.0}, {}, [(np.hypot(b[0] - a[0], b[1] - a[1]), 0.0, a)]
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell == b:
            break
        if cost > best[cell]:
            continue
        for di, dj, length in steps:
            nxt = (cell[0] + di, cell[1] + dj)
            if not (0 <= nxt[0] < free.shape[0] and 0 <= nxt[1] < free.shape[1]) or not free[nxt]:
                continue
            # No corner cutting between two blocked neighbours.
            if di and dj and not (free[cell[0] + di, cell[1]] and free[cell[0], cell[1] + dj]):
                continue
            new = cost + length
            if new < best.get(nxt, np.inf):
                best[nxt], parent[nxt] = new, cell
                heapq.heappush(frontier, (new + np.hypot(b[0] - nxt[0], b[1] - nxt[1]), new, nxt))
    else:
        return None
    if b not in best:
        return None
    cells = [b]
    while cells[-1] != a:
        cells.append(parent[cells[-1]])
    points = np.array([[grid["xs"][i], grid["ys"][j]] for i, j in reversed(cells)])
    return np.vstack([np.asarray(start, dtype=float)[None], points[1:-1], np.asarray(goal, dtype=float)[None]])


def staged_path(grid, start, goal, staging):
    """A* path from start to a staging point `staging` metres behind the goal pose (x, y, yaw), then straight to the goal,
    so the robot arrives along the goal heading. None when the straight segment, the staging point or the path is blocked.
    """
    stage = goal[:2] - staging*np.array([np.cos(goal[2]), np.sin(goal[2])])
    segment = stage + np.linspace(0.0, 1.0, max(2, int(np.ceil(staging/grid["resolution"])) + 1))[:, None]*(goal[:2] - stage)
    if not all(grid["free"][cell_of(grid, point)] for point in segment):
        return None
    path = plan_path(grid, start, stage)
    return None if path is None else np.vstack([path, np.asarray(goal[:2], dtype=float)[None]])


def path_lookup(path, point, ahead):
    """Remaining arc length from the path point closest to `point`, and the points `ahead` metres further on."""
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    if len(segments) == 0:
        return 0.0, np.tile(path[-1], (len(ahead), 1))
    d = point - path[:-1]
    t = np.clip(np.sum(d*segments, axis=1)/np.maximum(lengths**2, 1e-12), 0.0, 1.0)
    projected = path[:-1] + t[:, None]*segments
    k = int(np.argmin(np.linalg.norm(projected - point, axis=1)))
    here = cumulative[k] + t[k]*lengths[k]
    targets = np.minimum(here + np.asarray(ahead, dtype=float), cumulative[-1])
    index = np.clip(np.searchsorted(cumulative, targets, side="right") - 1, 0, len(segments) - 1)
    fraction = (targets - cumulative[index])/np.maximum(lengths[index], 1e-12)
    return float(cumulative[-1] - here), path[index] + fraction[:, None]*segments[index]


def pregrasp_pose(object_xy, object_yaw, reach):
    """Robot root pose (x, y, yaw) that holds an object at `reach` metres straight ahead, facing it."""
    return np.array([object_xy[0] - reach*np.cos(object_yaw), object_xy[1] - reach*np.sin(object_yaw), object_yaw])


def pad_rooms(rooms, count):
    """Room widths (n,), depths (n,) and furniture boxes (n, count, 4) padded with far-away dummies for batching."""
    boxes = np.tile([1e3, 1e3, 0.0, 0.0], (len(rooms), count, 1))
    for i, room in enumerate(rooms):
        boxes[i, :len(room["boxes"])] = room["boxes"][:, :4]
    return (np.array([r["width_m"] for r in rooms]), np.array([r["depth_m"] for r in rooms]), boxes)


def batched_distance(points, widths, depths, boxes):
    """obstacle_distance for many rooms at once: points (n, k, 2), padded boxes (n, m, 4) -> (n, k)."""
    w, d = (widths/2)[:, None], (depths/2)[:, None]
    to_walls = np.minimum(np.minimum(points[..., 0] + w, w - points[..., 0]), np.minimum(points[..., 1] + d, d - points[..., 1]))
    q = np.abs(points[:, :, None, :] - boxes[:, None, :, :2]) - boxes[:, None, :, 2:4]/2
    to_boxes = np.min(np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0), axis=-1)
    return np.minimum(to_walls, to_boxes)
