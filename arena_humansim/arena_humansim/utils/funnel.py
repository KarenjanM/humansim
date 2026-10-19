from __future__ import annotations

import math
from collections import deque
from typing import Literal, overload

import numpy as np

ARC_STEP = math.pi / 8  # max turn per arc waypoint
TURN_TOL = 0.1  # accepted turn against the side of a corner [rad]

Vec = tuple[float, float]
Corner = tuple[float, float, float, float]
Link = tuple[float, float, float, float, Vec]


def _tangent(a: Corner | Link, b: Corner | Link) -> Vec:
    """Unit direction of the path leg from corner a to corner b. Corner = (x, y, sigma, rho), sigma -1 left, +1 right, 0 point."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy)
    if d < 1e-12:
        return (1.0, 0.0)
    cx, cy = dx / d, dy / d
    k = b[2] * b[3] - a[2] * a[3]
    if k == 0.0:
        return (cx, cy)
    q = max(-1.0, min(1.0, k / d))
    c = math.sqrt(1.0 - q * q)
    return (cx * c - cy * q, cx * q + cy * c)


def _beyond(a: Corner | Link, p: Corner | Link, u: Vec, g: Corner) -> bool:
    """True when corner a projects past the end point g of the leg that leaves p with direction u."""
    return (a[0] - p[0]) * u[0] + (a[1] - p[1]) * u[1] > (g[0] - p[0]) * u[0] + (g[1] - p[1]) * u[1]


def _reach(a: Corner | Link, b: Corner | Link) -> float:
    """Length of the tangent leg from corner a to corner b."""
    k = b[2] * b[3] - a[2] * a[3]
    return math.sqrt(max((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 - k * k, 0.0))


def funnel(s: np.ndarray, g: np.ndarray, sides: list[int], ids: list[int], xs: list[float], ys: list[float], forced: list[bool], rho: list[float], nearest_first: bool = False) -> list[Corner | Link]:
    """Corner sequence of the taut path through a channel given as left/right vertex events, event e kept at distance rho[e].

    A forced event must become a corner: the funnel collapses onto it and restarts from there.
    With nearest_first an event that closes the funnel is forced when its disc is touched before the opposite one.
    """
    apex = (s[0], s[1], 0.0, 0.0)
    path: list[Corner | Link] = [apex]
    chains: dict[int, deque[Link]] = {-1: deque(), 1: deque()}
    last = {-1: -1, 1: -1}
    cur: dict[int, tuple[Corner, int] | None] = {-1: None, 1: None}
    n = len(sides)
    for e in range(n + 1):
        goal = e == n
        if goal:
            side, v = -1, (g[0], g[1], 0.0, 0.0)
        else:
            side = sides[e]
            if ids[e] == last[side]:
                continue
            last[side] = ids[e]
            v = (xs[e], ys[e], float(side), rho[e])
            cur[side] = (v, ids[e])
        own, opp = chains[side], chains[-side]
        pin = not goal and forced[e]
        # side -1 is the left chain, kept turning left. Mirror the sign for the right chain.
        sg = -side
        while own:
            a = own[-1]
            u1 = a[4]
            u2 = _tangent(a, v) if goal or a[3] != v[3] else (v[0] - a[0], v[1] - a[1])
            if (u1[0] * u2[1] - u1[1] * u2[0]) * sg > 0:
                if not goal:
                    break
                b = own[-2] if len(own) > 1 else apex
                if not _beyond(a, b, _tangent(b, v), v):
                    break
            own.pop()
        if own:
            own.append((v[0], v[1], v[2], v[3], _tangent(own[-1], v)))
        else:
            while opp and not pin:
                a = opp[0]
                u2 = _tangent(apex, v)
                u1 = a[4]
                if (u1[0] * u2[1] - u1[1] * u2[0]) * sg >= 0 or (goal and _beyond(a, apex, u2, v)):
                    break
                if nearest_first and not goal and _reach(apex, v) < _reach(apex, a):
                    pin = True
                    break
                opp.popleft()
                apex = a
                path.append(a)
            own.append((v[0], v[1], v[2], v[3], _tangent(apex, v)))
        if pin:
            path.extend(own)
            apex = own[-1]
            own.clear()
            opp.clear()
            last[-side] = -1
            pending = cur[-side]
            if pending is not None:
                c, cid = pending
                opp.append((c[0], c[1], c[2], c[3], _tangent(apex, c)))
                last[-side] = cid
    path.extend(chains[-1])
    return path


@overload
def polyline(corners: list[Corner | Link], strict: Literal[False] = False) -> np.ndarray: ...
@overload
def polyline(corners: list[Corner | Link], strict: bool) -> np.ndarray | None: ...
def polyline(corners: list[Corner | Link], strict: bool = False) -> np.ndarray | None:
    """Tangent legs plus arc waypoints. Waypoint radius balances overshoot and chord sag, both below 1 % of the corner radius.

    A corner that turns against its side gets no arc. With strict the result is None when such a turn exceeds TURN_TOL
    or when two discs overlap so that a leg has no tangent.
    """
    out = [(corners[0][0], corners[0][1])]
    if strict and any(math.hypot(b[0] - a[0], b[1] - a[1]) < abs(b[2] * b[3] - a[2] * a[3]) * (1 - 1e-9) for a, b in zip(corners[:-1], corners[1:], strict=True)):
        return None
    dirs = [_tangent(corners[i], corners[i + 1]) for i in range(len(corners) - 1)]
    for i in range(1, len(corners) - 1):
        x, y, sg = corners[i][0], corners[i][1], corners[i][2]
        t_in = math.atan2(dirs[i - 1][1], dirs[i - 1][0])
        dl = (math.atan2(dirs[i][1], dirs[i][0]) - t_in + math.pi) % (2 * math.pi) - math.pi
        if dl * sg > 0:
            if strict and dl * sg > TURN_TOL:
                return None
            dl = 0.0
        k = max(1, math.ceil(abs(dl) / ARC_STEP))
        rho = sg * corners[i][3] * 2 / (1 + math.cos(dl / (2 * k)))
        for j in range(k):
            th = t_in + (j + 0.5) * dl / k
            out.append((x - rho * math.sin(th), y + rho * math.cos(th)))
    out.append((corners[-1][0], corners[-1][1]))
    return np.array(out)
