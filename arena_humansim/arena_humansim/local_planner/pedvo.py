"""Pedestrian velocity obstacles (Curtis and Manocha), after Menge's PedVO."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.wall_grid import query_walls

from .orca import _RVO_EPS, ORCAPlanner, _agent_line, _det, _dist_sq_point_segment, _obstacle_line, _push_line

_DENSITY_AREA = 1.5
_DENSITY_NORM = 1.0 / (_DENSITY_AREA * math.sqrt(2.0 * math.pi))
_DENSITY_AREA_INV = 1.0 / (2.0 * _DENSITY_AREA * _DENSITY_AREA)
_DENSITY_OBST_AREA_INV = 1.0 / (2.0 * 0.75 * 0.75)
_DENSITY_LATERAL = 2.5
_DENSITY_WALL_RANGE = 3.0
_AGENT_WIDTH = 0.48
_STRIDE_LEN = 1.0


@njit(cache=True)
def _linear_program1_turning(lines: np.ndarray, line_no: int, radius: float, opt_x: float, opt_y: float, direction_opt: bool, turn_bias: float) -> tuple[bool, float, float]:
    px = lines[line_no, 0]
    py = lines[line_no, 1]
    dx = lines[line_no, 2]
    dy = lines[line_no, 3]
    dot = px * dx + py * dy
    if turn_bias != 1.0:
        ty = py * turn_bias
        tdy = dy * turn_bias
        real_dot = (px * dx + ty * tdy) / math.sqrt(dx * dx + tdy * tdy)
        if real_dot * real_dot + radius * radius - (px * px + ty * ty) < 0.0:
            return False, 0.0, 0.0
    discriminant = dot * dot + radius * radius - (px * px + py * py)
    if discriminant < 0.0:
        return False, 0.0, 0.0

    sqrt_disc = math.sqrt(discriminant)
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc

    for i in range(line_no):
        denominator = _det(dx, dy, lines[i, 2], lines[i, 3])
        numerator = _det(lines[i, 2], lines[i, 3], px - lines[i, 0], py - lines[i, 1])
        if abs(denominator) <= _RVO_EPS:
            if numerator < 0.0:
                return False, 0.0, 0.0
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, 0.0, 0.0

    if direction_opt:
        t = t_right if opt_x * dx + opt_y * dy > 0.0 else t_left
    else:
        t = min(max(dx * (opt_x - px) + dy * (opt_y - py), t_left), t_right)
    return True, px + t * dx, py + t * dy


@njit(cache=True)
def _linear_program2_turning(lines: np.ndarray, n_lines: int, radius: float, opt_x: float, opt_y: float, direction_opt: bool, turn_bias: float) -> tuple[int, float, float]:
    opt_sq = opt_x * opt_x + opt_y * opt_y
    if direction_opt:
        rx = opt_x * radius
        ry = opt_y * radius
    elif opt_sq > radius * radius:
        opt_len = math.sqrt(opt_sq)
        rx = opt_x / opt_len * radius
        ry = opt_y / opt_len * radius
    else:
        rx = opt_x
        ry = opt_y

    for i in range(n_lines):
        if _det(lines[i, 2], lines[i, 3], lines[i, 0] - rx, lines[i, 1] - ry) > 0.0:
            ok, nx, ny = _linear_program1_turning(lines, i, radius, opt_x, opt_y, direction_opt, turn_bias)
            if not ok:
                return i, rx, ry
            rx = nx
            ry = ny
    return n_lines, rx, ry


@njit(cache=True)
def _linear_program3_turning(lines: np.ndarray, n_lines: int, n_obst_lines: int, begin_line: int, radius: float, rx: float, ry: float, turn_bias: float) -> tuple[float, float]:
    proj = np.empty((n_lines, 4), dtype=np.float64)
    distance = 0.0
    for i in range(begin_line, n_lines):
        pix = lines[i, 0]
        piy = lines[i, 1]
        dix = lines[i, 2]
        diy = lines[i, 3]
        if _det(dix, diy, pix - rx, piy - ry) <= distance:
            continue

        proj[:n_obst_lines] = lines[:n_obst_lines]
        n_proj = n_obst_lines
        for j in range(n_obst_lines, i):
            pjx = lines[j, 0]
            pjy = lines[j, 1]
            djx = lines[j, 2]
            djy = lines[j, 3]
            determinant = _det(dix, diy, djx, djy)
            if abs(determinant) <= _RVO_EPS:
                if dix * djx + diy * djy > 0.0:
                    continue
                qx = 0.5 * (pix + pjx)
                qy = 0.5 * (piy + pjy)
            else:
                s = _det(djx, djy, pix - pjx, piy - pjy) / determinant
                qx = pix + s * dix
                qy = piy + s * diy
            ex = djx - dix
            ey = djy - diy
            e_len = math.sqrt(ex * ex + ey * ey)
            n_proj = _push_line(proj, n_proj, qx, qy, ex / e_len, ey / e_len)

        fail, nx, ny = _linear_program2_turning(proj, n_proj, radius, -diy, dix, True, turn_bias)
        if fail >= n_proj:
            rx = nx
            ry = ny
        distance = _det(dix, diy, pix - rx, piy - ry)
    return rx, ry


@njit(cache=True, parallel=True)
def _pedvo_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    max_speed: np.ndarray,
    desired_speed: np.ndarray,
    goal: np.ndarray,
    has_goal: np.ndarray,
    nbr_idx: np.ndarray,
    max_neighbors: int,
    edges: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    goal_radius: float,
    time_horizon: float,
    time_horizon_obst: float,
    wall_clearance: float,
    turn_bias: float,
    dense_aware: bool,
    speed_const: float,
    inv_dt: float,
    out: np.ndarray,
) -> None:
    n = pos.shape[0]
    n_walls = edges.shape[0] // 2
    n_agent_lines = min(max_neighbors, nbr_idx.shape[1])
    inv_tau = 1.0 / time_horizon
    inv_tau_obst = 1.0 / time_horizon_obst
    inv_turn = 1.0 / turn_bias
    for i in prange(n):
        out[i, 0] = 0.0
        out[i, 1] = 0.0
        if not has_goal[i]:
            continue

        x = pos[i, 0]
        y = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        ra = radius[i]
        vmax = max_speed[i]

        gx = goal[i, 0] - x
        gy = goal[i, 1] - y
        dist = math.sqrt(gx * gx + gy * gy)
        pdx = 0.0
        pdy = 0.0
        pref_speed = 0.0
        if dist >= goal_radius:
            pdx = gx / dist
            pdy = gy / dist
            pref_speed = desired_speed[i]

        walls = np.empty(64, dtype=np.int64)

        if dense_aware and pref_speed > 0.0:
            cx = x + _STRIDE_LEN * pdx
            cy = y + _STRIDE_LEN * pdy
            density = 0.0
            n_found = 0
            for c in range(nbr_idx.shape[1]):
                if n_found >= n_agent_lines:
                    break
                j = nbr_idx[i, c]
                if j >= n:
                    break
                if j == i:
                    continue
                n_found += 1
                ddx = pos[j, 0] - cx
                ddy = pos[j, 1] - cy
                along = ddx * pdx + ddy * pdy
                sx = (ddx - along * pdx) * _DENSITY_LATERAL + along * pdx
                sy = (ddy - along * pdy) * _DENSITY_LATERAL + along * pdy
                density += _DENSITY_NORM * math.exp(-(sx * sx + sy * sy) * _DENSITY_AREA_INV)

            n_hit = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, cx, cy, _DENSITY_WALL_RANGE, walls)
            if n_hit > walls.shape[0]:
                walls = np.empty(n_hit, dtype=np.int64)
                n_hit = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, cx, cy, _DENSITY_WALL_RANGE, walls)
            for h in range(n_hit):
                w = walls[h]
                ax = edges[w, 0]
                ay = edges[w, 1]
                abx = edges[w, 2] - ax
                aby = edges[w, 3] - ay
                r = min(max(((cx - ax) * abx + (cy - ay) * aby) / (abx * abx + aby * aby), 0.0), 1.0)
                qx = ax + r * abx
                qy = ay + r * aby
                if (qx - x) * pdx + (qy - y) * pdy < 0.0:
                    continue
                d_sq = (cx - qx) * (cx - qx) + (cy - qy) * (cy - qy)
                density += _DENSITY_NORM * math.exp(-d_sq * _DENSITY_OBST_AREA_INV)

            avail = 100.0 if density < 0.001 else _AGENT_WIDTH / density
            cap = speed_const * avail * avail
            if cap < pref_speed:
                pref_speed = cap

        ra_obst = ra + wall_clearance
        range_sq = (time_horizon_obst * vmax + ra_obst) ** 2
        query_r = math.sqrt(range_sq)
        n_hit = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, walls)
        if n_hit > walls.shape[0]:
            walls = np.empty(n_hit, dtype=np.int64)
            n_hit = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, query_r, walls)
        cand = np.empty(2 * n_hit, dtype=np.int64)
        cand_d = np.empty(2 * n_hit, dtype=np.float64)
        n_cand = 0
        for h in range(n_hit):
            for side in range(2):
                e = walls[h] + side * n_walls
                ax = edges[e, 0]
                ay = edges[e, 1]
                bx = edges[e, 2]
                by = edges[e, 3]
                if _det(ax - x, ay - y, bx - ax, by - ay) >= 0.0:
                    continue
                d = _dist_sq_point_segment(ax, ay, bx, by, x, y)
                if d >= range_sq:
                    continue
                k = n_cand
                while k > 0 and (d < cand_d[k - 1] or (d == cand_d[k - 1] and e < cand[k - 1])):
                    cand[k] = cand[k - 1]
                    cand_d[k] = cand_d[k - 1]
                    k -= 1
                cand[k] = e
                cand_d[k] = d
                n_cand += 1

        lines = np.empty((n_cand + n_agent_lines, 4), dtype=np.float64)
        n_lines = 0
        for c in range(n_cand):
            n_lines = _obstacle_line(lines, n_lines, edges[cand[c]], x, y, vx, vy, ra_obst, inv_tau_obst)
        n_obst_lines = n_lines

        n_found = 0
        for c in range(nbr_idx.shape[1]):
            if n_found >= n_agent_lines:
                break
            j = nbr_idx[i, c]
            if j >= n:
                break
            if j == i:
                continue
            n_found += 1
            n_lines = _agent_line(lines, n_lines, pos[j, 0] - x, pos[j, 1] - y, vx - vel[j, 0], vy - vel[j, 1], vx, vy, ra + radius[j], inv_tau, inv_dt)

        turning = turn_bias != 1.0 and pref_speed > _RVO_EPS
        if turn_bias == 1.0:
            opt_x = pdx * pref_speed
            opt_y = pdy * pref_speed
        else:
            opt_x = pref_speed
            opt_y = 0.0
        if turning:
            for k in range(n_lines):
                lpx = lines[k, 0]
                lpy = lines[k, 1]
                ldx = lines[k, 2]
                ldy = lines[k, 3]
                tdx = ldx * pdx + ldy * pdy
                tdy = (ldy * pdx - ldx * pdy) * inv_turn
                t_len = math.sqrt(tdx * tdx + tdy * tdy)
                lines[k, 0] = lpx * pdx + lpy * pdy
                lines[k, 1] = (lpy * pdx - lpx * pdy) * inv_turn
                lines[k, 2] = tdx / t_len
                lines[k, 3] = tdy / t_len

        fail, rx, ry = _linear_program2_turning(lines, n_lines, vmax, opt_x, opt_y, False, turn_bias)
        if fail < n_lines:
            rx, ry = _linear_program3_turning(lines, n_lines, n_obst_lines, fail, vmax, rx, ry, turn_bias)
        if turning:
            back_y = ry * turn_bias
            ox = rx * pdx - back_y * pdy
            oy = rx * pdy + back_y * pdx
            speed = math.sqrt(ox * ox + oy * oy)
            if speed > vmax:
                ox *= vmax / speed
                oy *= vmax / speed
            rx = ox
            ry = oy
        out[i, 0] = rx
        out[i, 1] = ry


class PedVOPlanner(ORCAPlanner):
    def __init__(
        self,
        time_horizon: float = 2.5,
        max_neighbors: int = 10,
        neighbor_dist: float = 5.0,
        goal_radius: float = 1e-6,
        time_horizon_obst: float = 0.15,
        wall_clearance: float = 0.05,
        wall_grid_cell: float = 1.0,
        turning_bias: float = 1.0,
        stride_factor: float = 1.57,
        stride_buffer: float = 0.9,
        dense_aware: bool = True,
    ):
        self.turning_bias = turning_bias
        self.stride_factor = stride_factor
        self.stride_buffer = stride_buffer
        self.dense_aware = dense_aware
        super().__init__(
            time_horizon=time_horizon,
            max_neighbors=max_neighbors,
            neighbor_dist=neighbor_dist,
            goal_radius=goal_radius,
            time_horizon_obst=time_horizon_obst,
            wall_clearance=wall_clearance,
            wall_grid_cell=wall_grid_cell,
        )

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            return
        pos = pool.pos[:n]
        k = min(self.max_neighbors + 1, n)
        _, idx = cKDTree(pos).query(pos, k=k, distance_upper_bound=self.neighbor_dist)
        idx = np.ascontiguousarray(idx.reshape(n, k), dtype=np.int64)
        out = np.empty((n, 2), dtype=np.float64)
        stride_const = 0.5 * (1.0 + self.stride_buffer) / self.stride_factor
        grid = self._grid
        _pedvo_kernel(
            pos,
            pool.prev_vel[:n],
            pool.agent_radius[:n],
            pool.max_velocity[:n],
            pool.desired_vel[:n],
            pool.goal_pos[:n],
            pool.has_goal[:n],
            idx,
            self.max_neighbors,
            self._edges,
            grid.cell_start,
            grid.cell_walls,
            grid.wall_cx0,
            grid.wall_cy0,
            grid.origin_x,
            grid.origin_y,
            grid.cell,
            grid.nx,
            grid.ny,
            self.goal_radius,
            self.time_horizon,
            self.time_horizon_obst,
            self.wall_clearance,
            self.turning_bias,
            self.dense_aware,
            1.0 / (stride_const * stride_const),
            1.0 / dt,
            out,
        )
        pool.vel[:n] = out
