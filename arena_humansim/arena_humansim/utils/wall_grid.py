from __future__ import annotations

import math

import numpy as np
from numba import njit

from arena_humansim.utils.types import Segments


class WallGrid:
    def __init__(self, segments: Segments | np.ndarray, cell: float = 1.0) -> None:
        seg = np.ascontiguousarray(np.asarray(segments, dtype=np.float64).reshape(-1, 4))
        self.segments = seg
        self.cell = cell
        if len(seg) == 0:
            self.origin_x = 0.0
            self.origin_y = 0.0
            self.nx = 1
            self.ny = 1
            self.cell_start = np.zeros(2, dtype=np.int64)
            self.cell_walls = np.empty(0, dtype=np.int64)
            self.wall_cx0 = np.empty(0, dtype=np.int64)
            self.wall_cy0 = np.empty(0, dtype=np.int64)
            return

        lo = np.minimum(seg[:, :2], seg[:, 2:])
        hi = np.maximum(seg[:, :2], seg[:, 2:])
        self.origin_x = float(lo[:, 0].min())
        self.origin_y = float(lo[:, 1].min())
        cx0 = ((lo[:, 0] - self.origin_x) / cell).astype(np.int64)
        cy0 = ((lo[:, 1] - self.origin_y) / cell).astype(np.int64)
        cx1 = ((hi[:, 0] - self.origin_x) / cell).astype(np.int64)
        cy1 = ((hi[:, 1] - self.origin_y) / cell).astype(np.int64)
        self.nx = int(cx1.max()) + 1
        self.ny = int(cy1.max()) + 1

        cells: list[int] = []
        walls: list[int] = []
        for w in range(len(seg)):
            for cy in range(int(cy0[w]), int(cy1[w]) + 1):
                for cx in range(int(cx0[w]), int(cx1[w]) + 1):
                    cells.append(cy * self.nx + cx)
                    walls.append(w)
        cell_ids = np.asarray(cells, dtype=np.int64)
        order = np.argsort(cell_ids, kind="stable")
        self.cell_walls = np.asarray(walls, dtype=np.int64)[order]
        self.cell_start = np.zeros(self.nx * self.ny + 1, dtype=np.int64)
        np.cumsum(np.bincount(cell_ids, minlength=self.nx * self.ny), out=self.cell_start[1:])
        self.wall_cx0 = cx0
        self.wall_cy0 = cy0


@njit(cache=True)
def query_walls(
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    x: float,
    y: float,
    radius: float,
    out: np.ndarray,
) -> int:
    """Writes a superset of the walls within `radius` of (x, y) once each in ascending id and returns the count, which exceeds len(out) when the buffer was too small."""
    qx0 = max(int(math.floor((x - radius - origin_x) / cell)), 0)
    qy0 = max(int(math.floor((y - radius - origin_y) / cell)), 0)
    qx1 = min(int(math.floor((x + radius - origin_x) / cell)), nx - 1)
    qy1 = min(int(math.floor((y + radius - origin_y) / cell)), ny - 1)
    k = 0
    for cy in range(qy0, qy1 + 1):
        for cx in range(qx0, qx1 + 1):
            c = cy * nx + cx
            for t in range(cell_start[c], cell_start[c + 1]):
                w = cell_walls[t]
                if cx != max(qx0, wall_cx0[w]) or cy != max(qy0, wall_cy0[w]):
                    continue
                if k < out.shape[0]:
                    j = k
                    while j > 0 and out[j - 1] > w:
                        out[j] = out[j - 1]
                        j -= 1
                    out[j] = w
                k += 1
    return k


@njit(cache=True)
def cast_one(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    cap: float,
    seg: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    buf: np.ndarray,
) -> float:
    """Ray parameter of the first wall hit along (ox, oy) + t * (dx, dy), capped at cap, with buf holding one slot per wall."""
    n_walls = seg.shape[0]
    best = cap
    half = 0.5 * cap * math.hypot(dx, dy)
    span = 2.0 * half / cell + 2.0
    if span * span >= n_walls:
        k = n_walls
        for w in range(n_walls):
            buf[w] = w
    else:
        k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, ox + 0.5 * cap * dx, oy + 0.5 * cap * dy, half, buf)
    for j in range(k):
        w = buf[j]
        ax = seg[w, 0]
        ay = seg[w, 1]
        ex = seg[w, 2] - ax
        ey = seg[w, 3] - ay
        wx = ax - ox
        wy = ay - oy
        denom = dx * ey - dy * ex
        if abs(denom) <= 1e-12:
            continue
        t = (wx * ey - wy * ex) / denom
        u = (wx * dy - wy * dx) / denom
        if 0.0 <= t < best and 0.0 <= u <= 1.0:
            best = t
    return best


@njit(cache=True)
def _cast_kernel(
    origins: np.ndarray,
    dirs: np.ndarray,
    cap: np.ndarray,
    seg: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    out: np.ndarray,
) -> None:
    buf = np.empty(seg.shape[0], dtype=np.int64)
    for i in range(origins.shape[0]):
        out[i] = cast_one(origins[i, 0], origins[i, 1], dirs[i, 0], dirs[i, 1], cap[i], seg, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, buf)


def cast_rays(grid: WallGrid, origins: np.ndarray, dirs: np.ndarray, max_dist: float | np.ndarray) -> np.ndarray:
    """Ray parameter of the first wall hit along each origin + t * dir, capped at max_dist."""
    o = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 2)
    d = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1, 2)
    cap = np.ascontiguousarray(np.broadcast_to(np.asarray(max_dist, dtype=np.float64), (len(o),)))
    out = cap.copy()
    if len(grid.segments) and len(o):
        _cast_kernel(o, d, cap, grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, out)
    return out
