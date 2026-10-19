"""Chraibi, Seyfried, Schadschneider 2010 generalized centrifugal force model, after Menge's AgtGCF."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numba import njit, prange

from arena_humansim.core.agents.types import ParamDist

from .force import GOAL, SOCIAL, WALL, ForcePlanner, WallGridArgs, near_walls

_EPS = 1e-6
_PARALLEL_EPS = 1e-10
_SWAY_SPEED = 1.3
_MIN_AXIS = 0.01
_MIN_WALL_LENGTH = 0.1

(
    _TAU,
    _NU,
    _MAX_DIST,
    _MAX_FORCE,
    _INTERP,
    _A_MIN,
    _A_RATE,
    _B_MAX,
    _B_GROWTH,
    _NU_WALL,
    _MAX_WALL_DIST,
    _MAX_WALL_FORCE,
    _WALL_INTERP,
) = range(13)


@njit(cache=True)
def ellipse_axes(a_min: float, a_rate: float, b_max: float, b_growth: float, speed: float) -> tuple[float, float]:
    return a_min + a_rate * speed, max(b_max - b_growth * speed / _SWAY_SPEED, _MIN_AXIS)


@njit(cache=True)
def polar_radius(a: float, b: float, c: float, s: float, dx: float, dy: float) -> float:
    u = (dx * c + dy * s) / a
    v = (dy * c - dx * s) / b
    return 1.0 / math.sqrt(u * u + v * v)


@njit(cache=True)
def _support(a: float, b: float, c: float, s: float, dx: float, dy: float) -> float:
    u = a * (dx * c + dy * s)
    v = b * (dy * c - dx * s)
    return math.sqrt(u * u + v * v)


@njit(cache=True, error_model="numpy")
def _cubic_max_root(a: float, b: float, c: float) -> float:
    p = b - a * a / 3.0
    q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
    disc = 0.25 * q * q + p * p * p / 27.0
    if disc > 0.0:
        big = 0.5 * abs(q) + math.sqrt(disc)
        u = -math.copysign(big ** (1.0 / 3.0), q)
        t = u - p / (3.0 * u)
    else:
        r = math.sqrt(max(-p / 3.0, 0.0))
        t = 0.0 if r == 0.0 else 2.0 * r * math.cos(math.acos(min(max(-0.5 * q / (r * r * r), -1.0), 1.0)) / 3.0)
    return t - a / 3.0


@njit(cache=True, error_model="numpy")
def _quartic_positive_root(alpha: float, beta: float, gamma: float, shift: float) -> float:
    best = -1.0
    loose = -1.0
    if beta == 0.0:
        disc = math.sqrt(max(alpha * alpha - 4.0 * gamma, 0.0))
        for z in (0.5 * (disc - alpha), -0.5 * (disc + alpha)):
            x = math.sqrt(max(z, 0.0))
            for q in (shift + x, shift - x):
                if q > 0.0:
                    if z >= 0.0:
                        best = max(best, q)
                    else:
                        loose = max(loose, q)
    else:
        m = _cubic_max_root(alpha, 0.25 * alpha * alpha - gamma, -0.125 * beta * beta)
        if not m > 0.0:
            return -1.0
        sm = math.sqrt(2.0 * m)
        for s1 in (1.0, -1.0):
            inner = -(2.0 * alpha + 2.0 * m + s1 * 2.0 * beta / sm)
            sq = math.sqrt(max(inner, 0.0))
            for q in (shift + 0.5 * (s1 * sm + sq), shift + 0.5 * (s1 * sm - sq)):
                if q > 0.0:
                    if inner >= 0.0:
                        best = max(best, q)
                    else:
                        loose = max(loose, q)
    return best if best > 0.0 else loose


@njit(cache=True, error_model="numpy")
def _contact_distance(a1: float, b1: float, c1: float, s1: float, a2: float, b2: float, c2: float, s2: float, dx: float, dy: float) -> float:
    if a1 < b1:
        a1, b1 = b1, a1
        c1, s1 = -s1, c1
    if a2 < b2:
        a2, b2 = b2, a2
        c2, s2 = -s2, c2
    k12 = c1 * c2 + s1 * s2
    if k12 < 0.0:
        c2 = -c2
        s2 = -s2
        k12 = -k12
    qa1 = a1 * a1
    qb1 = b1 * b1
    eps1 = 1.0 - qb1 / qa1
    eps2 = 1.0 - (b2 * b2) / (a2 * a2)
    k1d = c1 * dx + s1 * dy
    k2d = c2 * dx + s2 * dy
    qk1d = k1d * k1d
    qk12 = k12 * k12
    nu = a1 / b1 - 1.0
    nn = nu * (2.0 + nu)
    ratio = qb1 / (b2 * b2)
    ap11 = ratio * (1.0 + 0.5 * (1.0 + k12) * (nn - eps2 * (1.0 + nu * k12) ** 2))
    ap22 = ratio * (1.0 + 0.5 * (1.0 - k12) * (nn - eps2 * (1.0 - nu * k12) ** 2))
    ap12 = ratio * 0.5 * math.sqrt(max(1.0 - qk12, 0.0)) * (nn + eps2 * (1.0 - nu * nu * qk12))
    half_diff = 0.5 * (ap11 - ap22)
    root = math.sqrt(half_diff * half_diff + ap12 * ap12)
    lp = 0.5 * (ap11 + ap22) + root
    lm = 0.5 * (ap11 + ap22) - root
    sl = math.sqrt(lp)
    bp2 = 1.0 / sl
    ap2 = 1.0 / math.sqrt(lm)
    delta = lp / lm - 1.0
    den = 1.0 - eps1 * qk1d
    if delta <= 0.0:
        return (1.0 + ap2) * b1 / math.sqrt(den)
    if k12 > 1.0 - _PARALLEL_EPS:
        cosphi = qb1 / qa1 * qk1d / den if ap11 > ap22 else (1.0 - qk1d) / den
    else:
        t4 = b1 / a1 * k1d
        t3 = k2d + (b1 / a1 - 1.0) * k1d * k12
        t7 = ap12 / math.sqrt(1.0 + k12) * (t4 + t3) + (lp - ap11) / math.sqrt(1.0 - k12) * (t4 - t3)
        cosphi = t7 * t7 / (2.0 * (ap12 * ap12 + (lp - ap11) ** 2) * den)
    if cosphi == 0.0:
        return (1.0 + ap2) * b1 / math.sqrt(den)
    tan2 = 1.0 / cosphi - 1.0
    tt = 1.0 + tan2
    d1 = 1.0 + delta
    inv = -1.0 / (tt * lp)
    qb = -2.0 * (tt + delta) * sl * inv
    qc = (-tan2 - d1 * d1 + (1.0 + d1 * tan2) * lp) * inv
    qd = 2.0 * tt * d1 * sl * inv
    qe = (tt + delta) * d1 * inv
    sqb = qb * qb
    alpha = qc - 0.375 * sqb
    beta = 0.125 * sqb * qb - 0.5 * qb * qc + qd
    gamma = -3.0 * sqb * sqb / 256.0 + qc * sqb / 16.0 - 0.25 * qb * qd + qe
    best = _quartic_positive_root(alpha, beta, gamma, -0.25 * qb)
    w = (best * best - 1.0) / delta
    f1 = 1.0 + bp2 * d1 / best
    f2 = 1.0 + bp2 / best
    return math.sqrt(max(w * f1 * f1 + (1.0 - w) * f2 * f2, 0.0)) * b1 / math.sqrt(den)


@njit(cache=True)
def closest_approach(a1: float, b1: float, c1: float, s1: float, a2: float, b2: float, c2: float, s2: float, dx: float, dy: float) -> float:
    res = _contact_distance(a1, b1, c1, s1, a2, b2, c2, s2, dx, dy)
    lower = polar_radius(a1, b1, c1, s1, dx, dy) + polar_radius(a2, b2, c2, s2, dx, dy)
    if not res >= lower:
        return lower
    return min(res, max(a1, b1) + max(a2, b2))


@njit(cache=True)
def _hermite(t: float, x1: float, x2: float, y1: float, y2: float, dy1: float, dy2: float) -> float:
    scale = x2 - x1
    t = (t - x1) / scale
    t2 = t * t
    t3 = t2 * t
    return y1 * (2.0 * t3 - 3.0 * t2 + 1.0) + dy1 * (t3 - 2.0 * t2 + t) * scale + y2 * (-2.0 * t3 + 3.0 * t2) + dy2 * (t3 - t2) * scale


@njit(cache=True)
def distance_response(eff: float, max_dist: float, width: float, max_force: float) -> float:
    if eff >= max_dist:
        return 0.0
    peak = 3.0 * max_force
    if width < 1.0 / peak:
        width = 1.5 / peak
    if eff <= 0.0:
        return peak
    right = max_dist - width
    if eff > right:
        f = 1.0 / right
        return _hermite(eff, right, max_dist, f, 0.0, -f * f, 0.0)
    if eff > width:
        return 1.0 / eff
    f = 1.0 / width
    return _hermite(eff, 0.0, width, peak, f, 0.0, -f * f)


@njit(cache=True, parallel=True)
def _gcf_kernel(
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
    n = pos.shape[0]
    axis_a = np.empty(n)
    axis_b = np.empty(n)
    cos_t = np.empty(n)
    sin_t = np.empty(n)
    for i in prange(n):
        axis_a[i], axis_b[i] = ellipse_axes(params[i, _A_MIN], params[i, _A_RATE], params[i, _B_MAX], params[i, _B_GROWTH], math.hypot(vel[i, 0], vel[i, 1]))
        cos_t[i] = math.cos(theta[i])
        sin_t[i] = math.sin(theta[i])

    for i in prange(n):
        x = pos[i, 0]
        y = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        tau = params[i, _TAU]
        nu = params[i, _NU]
        max_dist = params[i, _MAX_DIST]
        max_force = params[i, _MAX_FORCE]
        width = params[i, _INTERP]
        v0 = math.hypot(pref[i, 0], pref[i, 1])
        ci = cos_t[i]
        si = sin_t[i]
        ai = axis_a[i]
        bi = axis_b[i]
        reach_i = max(ai, bi)
        speed = math.hypot(vx, vy)

        out[i, GOAL, 0] = (pref[i, 0] - vx) / tau
        out[i, GOAL, 1] = (pref[i, 1] - vy) / tau

        fx = 0.0
        fy = 0.0
        for t in range(indptr[i], indptr[i + 1]):
            j = indices[t]
            ox = x - pos[j, 0]
            oy = y - pos[j, 1]
            d = math.hypot(ox, oy)
            if d < _EPS:
                continue
            aj = axis_a[j]
            bj = axis_b[j]
            if d - reach_i - max(aj, bj) >= max_dist:
                continue
            nx = ox / d
            ny = oy / d
            cj = cos_t[j]
            sj = sin_t[j]
            if d - _support(ai, bi, ci, si, nx, ny) - _support(aj, bj, cj, sj, nx, ny) >= max_dist:  # pyright: ignore[reportArgumentType]
                continue
            eff = d - closest_approach(ai, bi, ci, si, aj, bj, cj, sj, -nx, -ny)  # pyright: ignore[reportArgumentType]
            if eff >= max_dist:
                continue
            fov = max(-(vx * nx + vy * ny), 0.0) / speed if speed > _EPS else 0.0
            v_ij = max(-((vx - vel[j, 0]) * nx + (vy - vel[j, 1]) * ny), 0.0)
            scale = nu * v0 + v_ij
            mag = fov * distance_response(eff, max_dist, width, max_force) * scale * scale
            fx += mag * nx
            fy += mag * ny
        out[i, SOCIAL, 0] = fx
        out[i, SOCIAL, 1] = fy

        nu_w = params[i, _NU_WALL]
        max_wall_dist = params[i, _MAX_WALL_DIST]
        max_wall_force = params[i, _MAX_WALL_FORCE]
        wall_width = params[i, _WALL_INTERP]
        fx = 0.0
        fy = 0.0
        for w in near_walls(grid, x, y, max_wall_dist + reach_i):
            ax = seg[w, 0]
            ay = seg[w, 1]
            sx = seg[w, 2] - ax
            sy = seg[w, 3] - ay
            length = math.hypot(sx, sy)
            if length < _MIN_WALL_LENGTH:
                continue
            tm = min(max(((x - ax) * sx + (y - ay) * sy) / (length * length), 0.0), 1.0)
            ex = ax + tm * sx - x
            ey = ay + tm * sy - y
            dm = math.hypot(ex, ey)
            if dm < _EPS:
                continue
            ex /= dm
            ey /= dm
            fov = max(vx * ex + vy * ey, 0.0) / speed if speed > _EPS else 0.0
            scale = nu_w * v0 + max(vx * ex + vy * ey, 0.0)
            gain = fov * scale * scale
            step = bi / length
            for tk in (tm - step, tm, tm + step):
                if tk < 0.0 or tk > 1.0:
                    continue
                px = ax + tk * sx - x
                py = ay + tk * sy - y
                dk = math.hypot(px, py)
                if dk < _EPS:
                    continue
                px /= dk
                py /= dk
                eff = dk - polar_radius(ai, bi, ci, si, px, py)
                mag = gain * distance_response(eff, max_wall_dist, wall_width, max_wall_force)
                fx -= mag * px
                fy -= mag * py
        out[i, WALL, 0] = fx
        out[i, WALL, 1] = fy


class GCFPlanner(ForcePlanner):
    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        "relaxation_time": ParamDist(0.5, 0.0, clip_low=0.05),
        "nu_agent": ParamDist(0.3, 0.0, clip_low=0.0),
        "max_agent_dist": ParamDist(2.0, 0.0, clip_low=1.0),
        "max_agent_force": ParamDist(3.0, 0.0, clip_low=1.0),
        "agent_interp_width": ParamDist(0.12, 0.0, clip_low=0.01, clip_high=0.4),
        "a_min": ParamDist(0.18, 0.0, clip_low=0.05),
        "a_rate": ParamDist(0.53, 0.0, clip_low=0.0),
        "b_max": ParamDist(0.25, 0.0, clip_low=0.05),
        "b_growth": ParamDist(0.05, 0.0, clip_low=0.0),
        "nu_wall": ParamDist(0.3, 0.0, clip_low=0.0),
        "max_wall_dist": ParamDist(2.0, 0.0, clip_low=1.0),
        "max_wall_force": ParamDist(3.0, 0.0, clip_low=1.0),
        "wall_interp_width": ParamDist(0.12, 0.0, clip_low=0.01, clip_high=0.4),
    }

    _kernel = staticmethod(_gcf_kernel)
