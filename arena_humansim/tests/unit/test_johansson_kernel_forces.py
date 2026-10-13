from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import GOAL, SOCIAL, WALL, _preferred_velocity
from arena_humansim.local_planner.johansson import JohanssonPlanner
from arena_humansim.utils.types import Pose2D

_TAU, _A, _A_WALL, _B, _STRIDE, _W = range(6)


def _pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: JohanssonPlanner,
    states: list[tuple[float, float, float, float, float]],
    neighbors: list[list[int]],
) -> AgentPool:
    pool = pool_empty(capacity=len(states))
    planner.attach(pool)
    for k, (x, y, theta, vx, vy) in enumerate(states):
        pool.add_agent(agent_factory(agent_id=k + 1, x=x, y=y))
        pool.theta[k] = theta
        pool.vel[k] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({k + 1: Pose2D(x=20.0, y=0.0) for k in range(len(states))})
    indptr = np.cumsum([0] + [len(row) for row in neighbors]).astype(np.int32)
    pool.set_neighbor_csr(indptr, np.array([j for row in neighbors for j in row], dtype=np.int32))
    return pool


def _forces(planner: JohanssonPlanner, pool: AgentPool, dt: float = 0.05) -> np.ndarray:
    n = pool.n
    pref, _ = _preferred_velocity(pool, n)
    grid = planner._grid
    out = np.zeros((n, 3, 2))
    type(planner)._kernel.py_func(
        pool.pos[:n].copy(),
        pool.prev_vel[:n].copy(),
        pool.theta[:n].copy(),
        pool.agent_radius[:n].copy(),
        pref,
        pool.max_acceleration[:n].copy(),
        pool.neighbor_indptr.astype(np.int64),
        pool.neighbor_indices.astype(np.int64),
        planner._params[:n].copy(),
        planner._seg,
        (grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny),
        dt,
        out,
    )
    planner.compute_pool(pool, store_forces=True, dt=dt)
    assert planner._last_force_arrays is not None
    assert out == pytest.approx(planner._last_force_arrays[2], rel=1e-12, abs=1e-12)
    return out


def test_goal_term_relaxes_toward_preferred_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.5, 0.5)], [[]])
    out = _forces(planner, pool)
    tau = planner._defaults[_TAU]
    assert out[0, GOAL] == pytest.approx(((pool.desired_vel[0] - 0.5) / tau, -0.5 / tau))


def test_static_neighbor_ahead_gets_full_weight_and_behind_gets_fov_weight(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0)], [[1], [0]])
    out = _forces(planner, pool)
    p = planner._defaults
    base = p[_A] * math.exp(-1.0 / p[_B])
    assert out[0, SOCIAL] == pytest.approx((-base, 0.0))
    assert out[1, SOCIAL] == pytest.approx((base * p[_W], 0.0))


def test_neighbor_sitting_on_its_own_stride_point_or_between_exerts_nothing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    stride = planner._defaults[_STRIDE]
    states = [(0.0, 0.0, 0.0, 0.0, 0.0), (-stride, 0.0, 0.0, 1.0, 0.0), (-0.5 * stride, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0)]
    pool = _pool(pool_empty, agent_factory, planner, states, [[1, 2, 3], [], [], []])
    out = _forces(planner, pool)
    assert out[0, SOCIAL].tolist() == [0.0, 0.0]


def test_moving_neighbor_matches_elliptical_formula(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    p = planner._defaults
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.3, 0.0, 0.0), (1.0, 0.5, 0.0, -0.8, 0.2)], [[1], []])
    out = _forces(planner, pool)
    r = np.array([-1.0, -0.5])
    s = np.array([-0.8, 0.2]) * p[_STRIDE]
    o = r - s
    d, od = np.linalg.norm(r), np.linalg.norm(o)
    b = 0.5 * math.sqrt((d + od) ** 2 - s @ s)
    cos_theta = (r / d) @ (math.cos(0.3), math.sin(0.3))
    mag = p[_A] * (p[_W] + (1.0 - p[_W]) * (1.0 - cos_theta) * 0.5) * (d + od) / (2.0 * b) * math.exp(-b / p[_B])
    assert out[0, SOCIAL] == pytest.approx(mag * 0.5 * (r / d + o / od))


def test_wall_force_is_exponential_in_distance_with_fov_weight(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    planner.set_walls([((-6.0, -1.0), (6.0, -1.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0, 0.0), (3.0, 0.0, -math.pi / 2, 0.0, 0.0)], [[], []])
    out = _forces(planner, pool)
    p = planner._defaults
    side = p[_A_WALL] * (p[_W] + (1.0 - p[_W]) * 0.5) * math.exp(-1.0 / p[_B])
    assert out[0, WALL] == pytest.approx((0.0, side))
    assert out[1, WALL] == pytest.approx((0.0, p[_A_WALL] * math.exp(-1.0 / p[_B])))


def test_walls_beyond_neighbor_distance_or_under_the_agent_are_ignored(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = JohanssonPlanner()
    planner.set_walls([((-10.0, -5.5), (10.0, -5.5)), ((-10.0, 0.0), (10.0, 0.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0, 0.0), (0.0, -0.1, 0.0, 0.0, 0.0)], [[], []])
    out = _forces(planner, pool)
    assert out[0, WALL].tolist() == [0.0, 0.0]
    p = planner._defaults
    w = p[_A_WALL] * (p[_W] + (1.0 - p[_W]) * 0.5)
    assert out[1, WALL] == pytest.approx((0.0, -w * math.exp(-0.1 / p[_B])))
