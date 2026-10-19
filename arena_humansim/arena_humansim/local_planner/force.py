from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import njit

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Pose2D, Segments
from arena_humansim.utils.wall_grid import WallGrid, query_walls

from . import LocalPlanner
from .sfm import SFMPlanner

if TYPE_CHECKING:
    from arena_humansim.core.viz import MarkerPublisher

_EPS = 1e-6
_WALL_QUERY_CAPACITY = 64

GOAL = 0
SOCIAL = 1
WALL = 2

INF = math.inf

WallGridArgs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, int, int]


@njit(cache=True)
def near_walls(grid: WallGridArgs, x: float, y: float, radius: float) -> np.ndarray:
    cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny = grid
    buf = np.empty(_WALL_QUERY_CAPACITY, dtype=np.int64)
    k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, radius, buf)
    if k > buf.shape[0]:
        buf = np.empty(k, dtype=np.int64)
        k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, radius, buf)
    return buf[:k]


@njit(cache=True)
def wall_point(seg: np.ndarray, w: int, x: float, y: float) -> tuple[float, float]:
    ax = seg[w, 0]
    ay = seg[w, 1]
    dx = seg[w, 2] - ax
    dy = seg[w, 3] - ay
    len_sq = dx * dx + dy * dy
    t = 0.0 if len_sq <= 0.0 else min(max(((x - ax) * dx + (y - ay) * dy) / len_sq, 0.0), 1.0)
    return ax + t * dx, ay + t * dy


@njit(cache=True)
def ray_circle_ttc(dir_x: float, dir_y: float, cx: float, cy: float, radius: float) -> float:
    a = dir_x * dir_x + dir_y * dir_y
    if a <= 0.0:
        return 0.0 if cx * cx + cy * cy < radius * radius else INF
    b = -2.0 * (dir_x * cx + dir_y * cy)
    c = cx * cx + cy * cy - radius * radius
    discr = b * b - 4.0 * a * c
    if discr < 0.0:
        return INF
    sq = math.sqrt(discr)
    t0 = (-b - sq) / (2.0 * a)
    t1 = (-b + sq) / (2.0 * a)
    if (t0 < 0.0 < t1) or (t1 < 0.0 < t0):
        return 0.0
    if 0.0 < t0 < t1:
        return t0
    if t1 > 0.0:
        return t1
    return INF


def _preferred_velocity(pool: AgentPool, n: int) -> tuple[np.ndarray, np.ndarray]:
    d = pool.goal_pos[:n] - pool.pos[:n]
    dist = np.hypot(d[:, 0], d[:, 1])
    moving = pool.has_goal[:n] & (dist >= _EPS)
    pref = d / np.maximum(dist, _EPS)[:, None] * pool.desired_vel[:n, None]
    pref[~moving] = 0.0
    return np.ascontiguousarray(pref), moving


class ForcePlanner(LocalPlanner):
    supports_pool: bool = True

    _kernel: ClassVar[Callable[..., None]]

    def __init__(self, wall_grid_cell: float = 1.0) -> None:
        self.wall_grid_cell = wall_grid_cell
        self._defaults = np.array([d.mean for d in self.PARAM_DEFAULTS.values()], dtype=np.float64)
        self._params = np.empty((0, len(self._defaults)), dtype=np.float64)
        self._seg = np.empty((0, 4), dtype=np.float64)
        self._grid = WallGrid(self._seg, cell=wall_grid_cell)
        self._last_force_arrays: tuple[np.ndarray, ...] | None = None
        self._warmup()

    def _warmup(self) -> None:
        self.set_walls([((-1.0, -1.0), (2.0, -1.0))])
        pool = AgentPool(capacity=2)
        pool.n = 2
        pool.pos[1] = (1.0, 0.0)
        pool.goal_pos[:2] = (3.0, 0.0)
        pool.has_goal[:2] = True
        pool.desired_vel[:2] = 1.0
        pool.max_velocity[:2] = 1.5
        pool.max_acceleration[:2] = 1.5
        pool.agent_radius[:2] = 0.25
        pool.set_neighbor_csr(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32))
        self._step(pool, np.tile(self._defaults, (2, 1)), 0.05, store_forces=False)
        self.set_walls([])

    def _row(self, agent: BaseAgent) -> np.ndarray:
        lp = agent.params.local_planner_params
        return np.array([float(lp.get(k, float(d))) for k, d in zip(self.PARAM_DEFAULTS, self._defaults, strict=True)], dtype=np.float64)

    def attach(self, pool: AgentPool) -> None:
        self._params = np.tile(self._defaults, (pool.capacity, 1))
        pool.register_extension(self)

    def on_pool_grow(self, new_capacity: int, old_capacity: int) -> None:
        params = np.tile(self._defaults, (new_capacity, 1))
        params[:old_capacity] = self._params[:old_capacity]
        self._params = params

    def on_pool_add(self, idx: int, agent: BaseAgent) -> None:
        self._params[idx] = self._row(agent)

    def on_pool_update(self, idx: int, agent: BaseAgent) -> None:
        self.on_pool_add(idx, agent)

    def on_pool_swap(self, idx: int, last: int) -> None:
        self._params[idx] = self._params[last]

    def set_walls(self, segments: Segments) -> None:
        self._seg = np.ascontiguousarray(np.asarray(segments, dtype=np.float64).reshape(-1, 4))
        self._grid = WallGrid(self._seg, cell=self.wall_grid_cell)

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            self._last_force_arrays = None
            return
        if self._params.shape[0] < n:
            raise RuntimeError(f"{type(self).__name__}: parameter arrays hold {self._params.shape[0]} rows for {n} pooled agents, this planner was never attached to the pool (AgentPool.attach_late)")
        self._step(pool, self._params[:n], dt, store_forces)

    def _step(self, pool: AgentPool, params: np.ndarray, dt: float, store_forces: bool) -> None:
        n = pool.n
        vel = np.ascontiguousarray(pool.prev_vel[:n])
        pref, moving = _preferred_velocity(pool, n)
        indptr = pool.neighbor_indptr
        indices = pool.neighbor_indices
        if indptr.shape[0] != n + 1:
            indptr = np.zeros(n + 1, dtype=np.int32)
        grid = self._grid
        out = np.zeros((n, 3, 2), dtype=np.float64)
        type(self)._kernel(
            np.ascontiguousarray(pool.pos[:n]),
            vel,
            np.ascontiguousarray(pool.theta[:n]),
            np.ascontiguousarray(pool.agent_radius[:n]),
            pref,
            np.ascontiguousarray(pool.max_acceleration[:n]),
            np.ascontiguousarray(indptr, dtype=np.int64),
            np.ascontiguousarray(indices, dtype=np.int64),
            np.ascontiguousarray(params),
            self._seg,
            (grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny),
            float(dt),
            out,
        )
        new_vel = vel + out.sum(axis=1) * dt
        speed = np.hypot(new_vel[:, 0], new_vel[:, 1])
        max_v = pool.max_velocity[:n]
        too_fast = speed > max_v
        new_vel[too_fast] *= (max_v[too_fast] / speed[too_fast])[:, None]
        new_vel[~moving] = 0.0
        pool.vel[:n] = new_vel
        if store_forces:
            self._last_force_arrays = (pool.agent_ids[:n].copy(), pool.pos[:n].copy(), out)
        else:
            self._last_force_arrays = None

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            return {}
        pool = AgentPool(capacity=len(agents))
        for agent in agents:
            pool.add_agent(agent)
        pool.set_goals(global_goals)
        ids = {a.state.agent_id for a in agents}
        indptr = [0]
        indices: list[int] = []
        for agent in agents:
            if agent.belief is not None:
                for other in agent.belief.observed_agents:
                    if other.agent_id != agent.state.agent_id and other.agent_id in ids:
                        indices.append(pool.idx(other.agent_id))
            indptr.append(len(indices))
        pool.set_neighbor_csr(np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32))
        self._step(pool, np.array([self._row(a) for a in agents]), dt, store_forces=False)
        return {int(pool.agent_ids[i]): (float(pool.vel[i, 0]), float(pool.vel[i, 1])) for i in range(pool.n)}

    def publish_markers(self, pub: MarkerPublisher) -> None:
        from visualization_msgs.msg import Marker

        from arena_humansim.core.viz import rgba

        if self._last_force_arrays is None:
            return
        ids, pos, out = self._last_force_arrays
        views = (
            (pub.view("f_goal", Marker.ARROW), rgba(0.2, 0.9, 0.2, 0.7)),
            (pub.view("f_social", Marker.ARROW), rgba(1.0, 0.2, 0.2, 0.7)),
            (pub.view("f_obstacle", Marker.ARROW), rgba(1.0, 0.6, 0.0, 0.7)),
        )
        for i in range(len(ids)):
            x, y = float(pos[i, 0]), float(pos[i, 1])
            for c, (view, color) in enumerate(views):
                SFMPlanner._emit_force(view, int(ids[i]), x, y, (float(out[i, c, 0]), float(out[i, c, 1])), 0.3, color)
