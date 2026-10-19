from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from arena_humansim.utils.types import Pose2D, Segments

from . import GlobalPlanner, PlanRequest
from ._grid import (
    REACHABLE_MAX_RESOLUTION,
    grid_to_world,
    label_free_cells,
    nearest_reachable_cell,
    push_from_walls,
    simplify_with_los,
    world_to_grid,
)

_SQRT2 = math.sqrt(2)


def _nearest_free_cell(
    grid: np.ndarray,
    row: int,
    col: int,
    max_radius: int = 200,
) -> tuple[int, int] | None:
    rows, cols = grid.shape
    if 0 <= row < rows and 0 <= col < cols and grid[row, col] == 0:
        return (row, col)
    for r in range(1, max_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    return (nr, nc)
    return None


def _build_grid_graph(grid: np.ndarray) -> csr_matrix:
    rows, cols = grid.shape
    free = grid == 0
    n_cells = rows * cols

    src = []
    dst = []
    weights = []

    for dr, dc, cost in (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, _SQRT2),
        (-1, 1, _SQRT2),
        (1, -1, _SQRT2),
        (1, 1, _SQRT2),
    ):
        # build index arrays for the valid region of the shift
        r_src = slice(max(-dr, 0), rows + min(-dr, 0))
        c_src = slice(max(-dc, 0), cols + min(-dc, 0))
        r_dst = slice(max(dr, 0), rows + min(dr, 0))
        c_dst = slice(max(dc, 0), cols + min(dc, 0))

        src_free = free[r_src, c_src]
        dst_free = free[r_dst, c_dst]
        both_free = src_free & dst_free

        r_idx, c_idx = np.where(both_free)

        src_r = r_idx + max(-dr, 0)
        src_c = c_idx + max(-dc, 0)
        dst_r = r_idx + max(dr, 0)
        dst_c = c_idx + max(dc, 0)

        src_flat = src_r * cols + src_c
        dst_flat = dst_r * cols + dst_c

        src.append(src_flat)
        dst.append(dst_flat)
        weights.append(np.full(src_flat.shape[0], cost, dtype=np.float32))

    src = np.concatenate(src)
    dst = np.concatenate(dst)
    weights = np.concatenate(weights)

    return csr_matrix((weights, (src, dst)), shape=(n_cells, n_cells))


def _dijkstra_path(
    graph: csr_matrix,
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    rows, cols = grid.shape

    actual_start = _nearest_free_cell(grid, start[0], start[1])
    actual_goal = _nearest_free_cell(grid, goal[0], goal[1])
    if actual_start is None or actual_goal is None:
        return None

    prepend_start = actual_start != start
    sr, sc = actual_start
    gr, gc = actual_goal

    if (sr, sc) == (gr, gc):
        path = [(sr, sc)]
        if prepend_start:
            path.insert(0, start)
        return path

    goal_flat = gr * cols + gc
    start_flat = sr * cols + sc

    # straight-line distance as lower bound; allow 4x detour before giving up
    straight = math.sqrt((sr - gr) ** 2 + (sc - gc) ** 2)
    limit = max(straight * 4.0, 200.0)

    dist, predecessors = dijkstra(
        graph,
        directed=False,
        indices=goal_flat,
        return_predecessors=True,
        limit=limit,
    )

    if np.isinf(dist[start_flat]):
        # retry without limit in case path requires long detour
        dist, predecessors = dijkstra(
            graph,
            directed=False,
            indices=goal_flat,
            return_predecessors=True,
        )

    if np.isinf(dist[start_flat]):
        return None

    # trace back from start to goal
    path_flat = []
    node = start_flat
    while node != goal_flat:
        path_flat.append(node)
        node = predecessors[node]
        if node < 0:
            return None
    path_flat.append(goal_flat)

    path = [(idx // cols, idx % cols) for idx in path_flat]

    if prepend_start:
        path.insert(0, start)
    return path


class DijkstraPlanner(GlobalPlanner):
    def __init__(
        self,
        replan_distance: float = 1.0,
        inflation_radius: float = 0.38,
        resolution: float = 0.2,
    ):
        super().__init__(replan_distance)
        self._inflation_radius = inflation_radius
        self._resolution = resolution

        self._occupancy_grid: np.ndarray | None = None
        self._grid_graph: csr_matrix | None = None
        self._free_labels: np.ndarray | None = None
        self._origin: Pose2D = Pose2D()

    def configure(self, *, inflation_radius: float, resolution: float, comfort_radius: float) -> None:
        if (inflation_radius, resolution) == (self._inflation_radius, self._resolution):
            return
        self._inflation_radius = inflation_radius
        self._resolution = resolution
        self.set_walls(self._wall_segments)

    def set_walls(self, segments: Segments) -> None:
        self._forget_paths()
        self._grid_graph = None
        self._free_labels = None
        self._wall_segments = list(segments)
        if not segments:
            self._occupancy_grid = None
            return

        arr = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
        all_points = arr.reshape(-1, 2)
        margin = self._inflation_radius + self._resolution * 2
        x_min = float(all_points[:, 0].min()) - margin
        y_min = float(all_points[:, 1].min()) - margin
        x_max = float(all_points[:, 0].max()) + margin
        y_max = float(all_points[:, 1].max()) + margin

        self._origin = Pose2D(x=x_min, y=y_min)
        res = self._resolution
        cols = int(math.ceil((x_max - x_min) / res)) + 1
        rows = int(math.ceil((y_max - y_min) / res)) + 1
        grid = np.zeros((rows, cols), dtype=np.uint8)

        for (x1, y1), (x2, y2) in segments:
            c1, r1 = (x1 - x_min) / res, (y1 - y_min) / res
            c2, r2 = (x2 - x_min) / res, (y2 - y_min) / res
            n = int(max(abs(c2 - c1), abs(r2 - r1))) + 1
            for t in np.linspace(0.0, 1.0, n):
                c = int(round(c1 + t * (c2 - c1)))
                r = int(round(r1 + t * (r2 - r1)))
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = 1

        radius_cells = int(math.ceil(self._inflation_radius / res))
        if radius_cells > 0:
            y, x = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
            kernel = (x * x + y * y) <= radius_cells * radius_cells
            grid = binary_dilation(grid, structure=kernel).astype(np.uint8)

        self._occupancy_grid = grid
        self._grid_graph = _build_grid_graph(grid)
        self._logger.info(f"Walls rasterized: {cols}x{rows} ({cols * rows} cells), res={res}m, {len(segments)} segment(s), inflation={self._inflation_radius}m ({radius_cells} cells), graph edges={self._grid_graph.nnz}")

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        if self._occupancy_grid is None:
            return pose
        rows, cols = self._occupancy_grid.shape
        rc = world_to_grid(self._origin, self._resolution, pose.x, pose.y)
        if not (0 <= rc[0] < rows and 0 <= rc[1] < cols):
            return pose
        snapped = _nearest_free_cell(self._occupancy_grid, rc[0], rc[1])
        if snapped is None or snapped == rc:
            return pose
        cell = grid_to_world(self._origin, self._resolution, snapped[0], snapped[1])
        return Pose2D(x=cell.x, y=cell.y, theta=pose.theta)

    def _nearest_reachable(self, start: Pose2D, target: Pose2D) -> Pose2D | None:
        if self._occupancy_grid is None or self._resolution > REACHABLE_MAX_RESOLUTION:
            return None
        if self._free_labels is None:
            self._free_labels = label_free_cells(self._occupancy_grid)
        rc = world_to_grid(self._origin, self._resolution, start.x, start.y)
        cell = _nearest_free_cell(self._occupancy_grid, rc[0], rc[1])
        if cell is None:
            return None
        return nearest_reachable_cell(self._free_labels, self._origin, self._resolution, cell, target)

    def _has_map(self) -> bool:
        return self._occupancy_grid is not None and self._grid_graph is not None

    def _plan(self, requests: list[PlanRequest]) -> dict[int, list[Pose2D] | None]:
        results: dict[int, list[Pose2D] | None] = {}
        assert self._grid_graph is not None and self._occupancy_grid is not None
        for agent_id, agent_pos, target in requests:
            start_rc = world_to_grid(self._origin, self._resolution, agent_pos.x, agent_pos.y)
            goal_rc = world_to_grid(self._origin, self._resolution, target.x, target.y)

            raw_path = _dijkstra_path(self._grid_graph, self._occupancy_grid, start_rc, goal_rc)

            if raw_path is None:
                results[agent_id] = None
                continue

            waypoints = [grid_to_world(self._origin, self._resolution, r, c) for r, c in raw_path]
            waypoints[0] = Pose2D(x=agent_pos.x, y=agent_pos.y)
            waypoints[-1] = self.snap_terminal(target)
            waypoints = simplify_with_los(self._occupancy_grid, self._origin, self._resolution, waypoints)
            if self._wall_segments:
                waypoints = push_from_walls(self._wall_segments, self._inflation_radius, waypoints)
            results[agent_id] = waypoints
        return results
