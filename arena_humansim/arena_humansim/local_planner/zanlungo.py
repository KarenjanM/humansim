"""Zanlungo, Ikeda, Kanda 2011 social force with explicit collision prediction, after Menge's AgtZanlungo."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numba import njit, prange

from arena_humansim.core.agents.types import ParamDist

from .force import GOAL, INF, SOCIAL, WALL, ForcePlanner, WallGridArgs, near_walls, ray_circle_ttc, wall_point

_EPS = 1e-6
_WALL_QUERY_RANGE = 5.0

_TAU, _MASS, _A, _A_WALL, _B = range(5)


@njit(cache=True)
def _wall_ttc(seg: np.ndarray, w: int, x: float, y: float, vx: float, vy: float, radius: float) -> float:
    speed = math.hypot(vx, vy)
    if speed < _EPS:
        return INF
    fx = vx / speed
    fy = vy / speed
    ax0 = seg[w, 0] - x
    ay0 = seg[w, 1] - y
    bx0 = seg[w, 2] - x
    by0 = seg[w, 3] - y
    ax = ax0 * fx + ay0 * fy
    ay = ay0 * fx - ax0 * fy
    bx = bx0 * fx + by0 * fy
    by = by0 * fx - bx0 * fy
    length = math.hypot(bx - ax, by - ay)
    if length < _EPS:
        return INF
    dx = (bx - ax) / length
    dy = (by - ay) / length
    c = -(dy * ax - dx * ay)
    if c < 0.0:
        ax, ay, bx, by = bx, by, ax, ay
        dx = -dx
        dy = -dy
        c = -c
    rad_sq = radius * radius
    if c < radius:
        t = -(dx * ax + dy * ay)
        if -radius <= t <= length + radius and (0.0 <= t <= length or (t < 0.0 and ax * ax + ay * ay < rad_sq) or (t > length and bx * bx + by * by < rad_sq)):
            return 0.0
    a2x = ax + dy * radius
    a2y = ay - dx * radius
    b2y = by - dx * radius
    if (a2y < 0.0) != (b2y < 0.0):
        hit = a2x + dx * (-a2y / dy)
        return hit / speed if hit > 0.0 else INF
    t_min = INF
    for px, py in ((ax, ay), (bx, by)):
        if py * py < rad_sq:
            hit = px - math.sqrt(rad_sq - py * py)
            if hit > 0.0:
                t_min = min(t_min, hit / speed)
    return t_min


@njit(cache=True)
def _crosses(x0: float, y0: float, x1: float, y1: float, seg: np.ndarray, w: int) -> bool:
    ax = seg[w, 0]
    ay = seg[w, 1]
    bx = seg[w, 2]
    by = seg[w, 3]
    d0 = (bx - ax) * (y0 - ay) - (by - ay) * (x0 - ax)
    d1 = (bx - ax) * (y1 - ay) - (by - ay) * (x1 - ax)
    e0 = (x1 - x0) * (ay - y0) - (y1 - y0) * (ax - x0)
    e1 = (x1 - x0) * (by - y0) - (y1 - y0) * (bx - x0)
    return d0 * d1 < 0.0 and e0 * e1 < 0.0


@njit(cache=True, parallel=True)
def _zanlungo_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
    pref: np.ndarray,
    max_acc: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    params: np.ndarray,
    seg: np.ndarray,
    grid: WallGridArgs,
    dt: float,
    out: np.ndarray,
) -> None:
    t_floor = max(dt, _EPS)
    for i in prange(pos.shape[0]):
        x = pos[i, 0]
        y = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        r = radius[i]
        tau = params[i, _TAU]
        mass = params[i, _MASS]
        a = params[i, _A]
        a_wall = params[i, _A_WALL]
        b = params[i, _B]
        speed = math.hypot(vx, vy)

        out[i, GOAL, 0] = (pref[i, 0] - vx) / tau
        out[i, GOAL, 1] = (pref[i, 1] - vy) / tau

        walls = near_walls(grid, x, y, _WALL_QUERY_RANGE) if speed >= _EPS else np.empty(0, dtype=np.int64)
        k = 0
        for w in walls:
            px, py = wall_point(seg, w, x, y)  # pyright: ignore[reportArgumentType]
            if (x - px) ** 2 + (y - py) ** 2 <= _WALL_QUERY_RANGE * _WALL_QUERY_RANGE:
                walls[k] = w
                k += 1
        walls = walls[:k]

        t_col = INF
        t_i = INF
        for s in range(indptr[i], indptr[i + 1]):
            j = indices[s]
            rvx = vx - vel[j, 0]
            rvy = vy - vel[j, 1]
            rpx = x - pos[j, 0]
            rpy = y - pos[j, 1]
            dp = -(rpx * rvx + rpy * rvy)
            r_ij = r + radius[j]
            if dp <= 0.0 and rpx * rpx + rpy * rpy >= r_ij * r_ij:
                continue
            contact = ray_circle_ttc(rvx, rvy, -rpx, -rpy, r_ij)
            if contact < INF:
                t_col = min(t_col, contact)
            elif dp > 0.0:
                t_i = min(t_i, dp / (rvx * rvx + rvy * rvy))
        for w in walls:
            t_i = min(t_i, _wall_ttc(seg, w, x, y, vx, vy, r))
        if t_col < INF:
            t_i = t_col
        if t_i == INF:
            continue
        t_i = max(t_i, t_floor)

        fx = 0.0
        fy = 0.0
        for s in range(indptr[i], indptr[i + 1]):
            j = indices[s]
            rvx = vx - vel[j, 0]
            rvy = vy - vel[j, 1]
            ox = x - pos[j, 0] + rvx * t_i
            oy = y - pos[j, 1] + rvy * t_i
            if ox * rvx + oy * rvy > 0.0:
                continue
            d = math.hypot(ox, oy)
            if d < _EPS:
                continue
            mag = a * math.hypot(rvx, rvy) / t_i * math.exp(-(d - r - radius[j]) / b)
            fx += mag * ox / d
            fy += mag * oy / d
        out[i, SOCIAL, 0] = fx / mass
        out[i, SOCIAL, 1] = fy / mass

        fut_x = x + vx * t_i
        fut_y = y + vy * t_i
        wall_mag = a_wall * speed / t_i
        fx = 0.0
        fy = 0.0
        for w in walls:
            px, py = wall_point(seg, w, fut_x, fut_y)
            ox = fut_x - px
            oy = fut_y - py
            d = math.hypot(ox, oy)
            if d < _EPS:
                continue
            if _crosses(x, y, fut_x, fut_y, seg, w):
                ox = -ox
                oy = -oy
            mag = wall_mag * math.exp(-(d - r) / b)
            fx += mag * ox / d
            fy += mag * oy / d
        out[i, WALL, 0] = fx / mass
        out[i, WALL, 1] = fy / mass


class ZanlungoPlanner(ForcePlanner):
    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        "relaxation_time": ParamDist(0.5, 0.0, clip_low=0.05),
        "mass": ParamDist(80.0, 0.0, clip_low=1.0),
        "agent_scale": ParamDist(2000.0, 0.0, clip_low=0.0),
        "obstacle_scale": ParamDist(2000.0, 0.0, clip_low=0.0),
        "force_distance": ParamDist(0.08, 0.0, clip_low=0.01),
    }

    _kernel = staticmethod(_zanlungo_kernel)
