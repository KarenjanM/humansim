from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import label

from arena_humansim.utils.types import Pose2D, Segments

REACHABLE_MAX_RESOLUTION = 0.1  # coarser grids rasterize metre-wide doors shut [m]


def world_to_grid(origin: Pose2D, resolution: float, wx: float, wy: float) -> tuple[int, int]:
    col = int(round((wx - origin.x) / resolution))
    row = int(round((wy - origin.y) / resolution))
    return row, col


def grid_to_world(origin: Pose2D, resolution: float, row: int, col: int) -> Pose2D:
    return Pose2D(x=col * resolution + origin.x, y=row * resolution + origin.y)


def label_free_cells(grid: np.ndarray) -> np.ndarray:
    return label(grid == 0, structure=np.ones((3, 3), dtype=bool))[0]  # pyright: ignore[reportIndexIssue]


def nearest_reachable_cell(labels: np.ndarray, origin: Pose2D, resolution: float, start: tuple[int, int], target: Pose2D) -> Pose2D:
    """Centre of the free cell connected to the free cell start that lies closest to target."""
    rows, cols = np.nonzero(labels == labels[start])
    k = int(np.argmin((cols * resolution + origin.x - target.x) ** 2 + (rows * resolution + origin.y - target.y) ** 2))
    return Pose2D(x=cols[k] * resolution + origin.x, y=rows[k] * resolution + origin.y, theta=target.theta)


def line_of_sight(grid: np.ndarray, origin: Pose2D, resolution: float, p1: Pose2D, p2: Pose2D) -> bool:
    r1, c1 = world_to_grid(origin, resolution, p1.x, p1.y)
    r2, c2 = world_to_grid(origin, resolution, p2.x, p2.y)
    rows, cols = grid.shape
    # Manhattan step count so thin diagonal walls aren't skipped.
    steps = abs(r2 - r1) + abs(c2 - c1)
    if steps == 0:
        return True
    for i in range(steps + 1):
        t = i / steps
        r = int(round(r1 + t * (r2 - r1)))
        c = int(round(c1 + t * (c2 - c1)))
        # Out-of-grid cells are free: the grid only spans walls + margin.
        if 0 <= r < rows and 0 <= c < cols and grid[r, c] != 0:
            return False
    return True


def simplify_with_los(
    grid: np.ndarray,
    origin: Pose2D,
    resolution: float,
    waypoints: list[Pose2D],
) -> list[Pose2D]:
    if len(waypoints) <= 2:
        return waypoints
    result = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        farthest = i + 1
        for j in range(len(waypoints) - 1, i + 1, -1):
            if line_of_sight(grid, origin, resolution, waypoints[i], waypoints[j]):
                farthest = j
                break
        result.append(waypoints[farthest])
        i = farthest
    return result


def push_from_walls(
    wall_segments: Segments,
    inflation_radius: float,
    waypoints: list[Pose2D],
) -> list[Pose2D]:
    if len(waypoints) <= 2:
        return waypoints
    margin = inflation_radius
    margin_sq = margin * margin
    result = [waypoints[0]]
    for wp in waypoints[1:-1]:
        px, py = wp.x, wp.y
        push_x, push_y = 0.0, 0.0
        for (x1, y1), (x2, y2) in wall_segments:
            sx, sy = x2 - x1, y2 - y1
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < 1e-12:
                cx, cy = x1, y1
            else:
                t = max(0.0, min(1.0, ((px - x1) * sx + (py - y1) * sy) / seg_len_sq))
                cx, cy = x1 + t * sx, y1 + t * sy
            dx, dy = px - cx, py - cy
            dist_sq = dx * dx + dy * dy
            if dist_sq < margin_sq and dist_sq > 1e-12:
                dist = math.sqrt(dist_sq)
                nx, ny = dx / dist, dy / dist
                push_x += nx * (margin - dist)
                push_y += ny * (margin - dist)
        result.append(Pose2D(x=px + push_x, y=py + push_y))
    result.append(waypoints[-1])
    return result


def min_distances_to_paths(positions: np.ndarray, paths: Sequence[np.ndarray]) -> np.ndarray:
    lengths = np.fromiter((len(p) for p in paths), dtype=np.intp, count=len(paths))
    starts = np.zeros(len(paths), dtype=np.intp)
    np.cumsum(lengths[:-1], out=starts[1:])
    b = np.concatenate(paths)
    prev = np.arange(len(b), dtype=np.intp) - 1
    prev[starts] = starts
    a = b[prev]
    dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    p = np.repeat(positions, lengths, axis=0)
    rx, ry = p[:, 0] - a[:, 0], p[:, 1] - a[:, 1]
    span = dx * dx + dy * dy
    t = np.zeros_like(span)
    np.divide(rx * dx + ry * dy, span, out=t, where=span != 0.0)
    np.clip(t, 0.0, 1.0, out=t)
    return np.fmin.reduceat(np.hypot(rx - t * dx, ry - t * dy), starts)


def next_waypoint(waypoints: Sequence[Pose2D], idx: int) -> Pose2D:
    target = idx + 1
    if target < len(waypoints):
        return waypoints[target]
    return waypoints[-1]
