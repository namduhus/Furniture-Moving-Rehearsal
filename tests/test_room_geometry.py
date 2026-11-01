from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import room_geometry as rg

SETTINGS = {"width_m": [3.0, 5.0], "depth_m": [3.0, 5.0], "obstacles": [1, 3],
            "obstacle_size_m": [[0.4, 1.6], [0.4, 0.9], [0.4, 0.9]], "obstacle_gap_m": 0.7, "grid_m": 0.1, "inflation_m": 0.35}


class RoomGeometryTests(unittest.TestCase):
    def setUp(self):
        self.room = {"width_m": 4.0, "depth_m": 3.0, "boxes": np.array([[0.5, 0.0, 1.0, 0.6, 0.8]])}

    def test_distances_to_walls_furniture_and_a_rotated_object(self):
        points = np.array([[0.0, 1.0], [1.5, 0.0], [0.5, 0.0], [-1.9, -1.4]])
        # (0, 1): 0.5 m above the box top edge (y = 0.3 is 0.7 away) and 0.5 m below the wall -> 0.5.
        np.testing.assert_allclose(rg.obstacle_distance(points, self.room), [0.5, 0.5, -0.3, 0.1], atol=1e-12)
        self.assertAlmostEqual(float(rg.rect_distance([2.0, 0.0], [[0.0, 0.0, 2.0, 2.0]])), 1.0)
        # A 1.0 x 0.2 rectangle turned 90 degrees reaches 0.5 m along y.
        self.assertAlmostEqual(float(rg.rotated_distance([0.0, 0.8], np.zeros(2), (1.0, 0.2), np.pi/2)), 0.3, places=12)
        walls = rg.wall_boxes(self.room, 0.1, 1.2)
        self.assertEqual(len(walls), 5)
        self.assertEqual(walls[-1], ([0.5, 0.0, 0.4], [1.0, 0.6, 0.8]))

    def test_paths_avoid_inflated_obstacles_and_feed_look_ahead_points(self):
        grid = rg.occupancy(self.room, 0.1, 0.3)
        path = rg.plan_path(grid, np.array([-1.5, 0.0]), np.array([1.6, 0.0]))
        self.assertIsNotNone(path)
        np.testing.assert_allclose((path[0], path[-1]), ([-1.5, 0.0], [1.6, 0.0]))
        # Interior path points keep the inflation radius from every obstacle (start/goal are given poses).
        self.assertGreaterEqual(float(rg.obstacle_distance(path[1:-1], self.room).min()), 0.3 - 1e-9)
        remaining, ahead = rg.path_lookup(path, np.array([-1.5, 0.0]), [0.3, 100.0])
        self.assertAlmostEqual(remaining, float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
        np.testing.assert_allclose(ahead[1], [1.6, 0.0], atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(ahead[0] - path[0])), 0.3, delta=0.05)
        # A blocked goal (inside the box) has no path; a goal pose next to the box with a smaller object inflation does.
        self.assertIsNone(rg.plan_path(grid, np.array([-1.5, 0.0]), np.array([0.5, 0.0])))
        pose = rg.pregrasp_pose([0.5, 1.0], -np.pi/2, 0.41)
        np.testing.assert_allclose(pose, [0.5, 1.41, -np.pi/2], atol=1e-12)

    def test_batched_distances_match_the_per_room_distances(self):
        rooms = [self.room, {"width_m": 3.0, "depth_m": 4.0, "boxes": np.zeros((0, 5))}]
        points = np.random.default_rng(0).uniform(-2.0, 2.0, (2, 7, 2))
        widths, depths, boxes = rg.pad_rooms(rooms, 3)
        expected = np.stack([rg.obstacle_distance(points[i], rooms[i]) for i in range(2)])
        np.testing.assert_allclose(rg.batched_distance(points, widths, depths, boxes), expected, atol=1e-12)

    def test_sampled_rooms_are_connected_and_reproducible(self):
        rooms = [rg.sample_room(SETTINGS, np.random.default_rng(seed)) for seed in range(5)]
        again = rg.sample_room(SETTINGS, np.random.default_rng(3))
        np.testing.assert_allclose(rooms[3]["boxes"], again["boxes"])
        for room in rooms:
            self.assertTrue(1 <= len(room["boxes"]) <= 3 and 3.0 <= room["width_m"] <= 5.0)
            grid = rg.occupancy(room, 0.1, 0.35)
            self.assertGreaterEqual(rg.largest_component(grid["free"]).sum(), 0.5*grid["free"].sum())
            for box in room["boxes"]:
                self.assertLessEqual(abs(box[0]) + box[2]/2, room["width_m"]/2 + 1e-9)


if __name__ == "__main__":
    unittest.main()
