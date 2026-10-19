from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.ndimage import distance_transform_edt

from arena_humansim.utils.types import Segments

from . import Occluder

_RESOLUTION = 0.05
_MARGIN = _RESOLUTION * math.sqrt(2.0)
_MIN_STEP = _RESOLUTION / math.sqrt(2.0)


@njit(cache=True, parallel=True)
def _march_kernel(p_a: np.ndarray, p_b: np.ndarray, dist: np.ndarray, ox: float, oy: float, out: np.ndarray) -> None:
    height, width = dist.shape
    for i in prange(p_a.shape[0]):
        out[i] = True
        ax = p_a[i, 0]
        ay = p_a[i, 1]
        dx = p_b[i, 0] - ax
        dy = p_b[i, 1] - ay
        length = math.hypot(dx, dy)
        if not length > 1e-9:
            continue
        inv_len = 1.0 / length
        s = 0.0
        while True:
            col = int((ax + s * dx - ox) / _RESOLUTION)
            row = int((ay + s * dy - oy) / _RESOLUTION)
            in_bounds = col >= 0 and col < width and row >= 0 and row < height
            d = dist[min(max(row, 0), height - 1), min(max(col, 0), width - 1)]
            if in_bounds and d == 0.0:
                out[i] = False
                break
            if not s < 1.0:
                break
            s = min(s + max(d - _MARGIN, _MIN_STEP) * inv_len, 1.0)


class BitmapOccluder(Occluder):
    def __init__(self) -> None:
        self._grid: np.ndarray | None = None
        self._dist: np.ndarray | None = None
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._warmup()

    def _warmup(self) -> None:
        p_a = np.zeros((2, 2), dtype=np.float64)
        p_b = np.ones((2, 2), dtype=np.float64)
        _march_kernel(p_a, p_b, np.ones((2, 2), dtype=np.float64), 0.0, 0.0, np.empty(2, dtype=np.bool_))

    def set_walls(self, segments: Segments) -> None:
        if not segments:
            self._grid = None
            self._dist = None
            return

        xs = [x for (x, _), (ex, _) in segments for x in (x, ex)]
        ys = [y for (_, y), (_, ey) in segments for y in (y, ey)]
        xmin = min(xs) - 1.0
        ymin = min(ys) - 1.0
        xmax = max(xs) + 1.0
        ymax = max(ys) + 1.0

        self._origin_x = xmin
        self._origin_y = ymin

        width = math.ceil((xmax - xmin) / _RESOLUTION)
        height = math.ceil((ymax - ymin) / _RESOLUTION)
        grid = np.zeros((height, width), dtype=np.bool_)

        for (sx, sy), (ex, ey) in segments:
            c0 = int(math.floor((sx - xmin) / _RESOLUTION))
            r0 = int(math.floor((sy - ymin) / _RESOLUTION))
            c1 = int(math.floor((ex - xmin) / _RESOLUTION))
            r1 = int(math.floor((ey - ymin) / _RESOLUTION))

            dc = abs(c1 - c0)
            dr = abs(r1 - r0)
            sc = 1 if c1 > c0 else -1
            sr = 1 if r1 > r0 else -1
            err = dc - dr

            c, r = c0, r0
            while True:
                cc = max(0, min(c, width - 1))
                rc = max(0, min(r, height - 1))
                grid[rc, cc] = True
                if c == c1 and r == r1:
                    break
                e2 = 2 * err
                if e2 > -dr:
                    err -= dr
                    c += sc
                if e2 < dc:
                    err += dc
                    r += sr

        # 4-neighbor dilation closes the off-diagonal gaps in 8-connected lines that thin walls would otherwise leak rays through.
        dilated = grid.copy()
        dilated[1:, :] |= grid[:-1, :]
        dilated[:-1, :] |= grid[1:, :]
        dilated[:, 1:] |= grid[:, :-1]
        dilated[:, :-1] |= grid[:, 1:]
        self._grid = dilated
        self._dist = np.ascontiguousarray(distance_transform_edt(~dilated) * _RESOLUTION, dtype=np.float64)  # pyright: ignore[reportOperatorIssue]

    def clear(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        result = np.ones(len(p_a), dtype=np.bool_)
        if self._dist is None:
            return result
        _march_kernel(np.ascontiguousarray(p_a, dtype=np.float64), np.ascontiguousarray(p_b, dtype=np.float64), self._dist, self._origin_x, self._origin_y, result)
        return result
