from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import GOAL, SOCIAL, WALL, _preferred_velocity
from arena_humansim.local_planner.helbing import HelbingPlanner
from arena_humansim.utils.types import Pose2D

_TAU, _MASS, _A, _A_WALL, _B, _K, _KAPPA = range(7)


def _pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: HelbingPlanner,
    states: list[tuple[float, float, float, float]],
    neighbors: list[list[int]],
    goals: list[tuple[float, float] | None],
) -> AgentPool:
    pool = pool_empty(capacity=len(states))
    planner.attach(pool)
    for k, (x, y, vx, vy) in enumerate(states):
        pool.add_agent(agent_factory(agent_id=k + 1, x=x, y=y))
        pool.vel[k] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({k + 1: Pose2D(x=g[0], y=g[1]) for k, g in enumerate(goals) if g is not None})
    indptr = np.cumsum([0] + [len(row) for row in neighbors]).astype(np.int32)
    pool.set_neighbor_csr(indptr, np.array([j for row in neighbors for j in row], dtype=np.int32))
    return pool


def _forces(planner: HelbingPlanner, pool: AgentPool, dt: float = 0.05) -> np.ndarray:
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


def test_goal_term_is_velocity_error_over_relaxation_time(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.4, -0.2), (5.0, 5.0, 1.0, 0.0)], [[], []], [(0.0, 3.0), None])
    out = _forces(planner, pool)
    p = planner._defaults
    assert out[0, GOAL] == pytest.approx(((0.0 - 0.4) / p[_TAU], (pool.desired_vel[0] + 0.2) / p[_TAU]))
    assert out[1, GOAL] == pytest.approx((-1.0 / p[_TAU], 0.0))


def test_separated_pair_feels_equal_and_opposite_exponential_repulsion(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0), (0.6, 0.8, 0.0, 0.0)], [[1], [0]], [None, None])
    out = _forces(planner, pool)
    p = planner._defaults
    mag = p[_A] * math.exp((0.5 - 1.0) / p[_B]) / p[_MASS]
    assert out[0, SOCIAL] == pytest.approx((-0.6 * mag, -0.8 * mag))
    assert out[1, SOCIAL] == pytest.approx(-out[0, SOCIAL])


def test_overlapping_pair_adds_body_force_and_sliding_friction(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0), (0.48, 0.0, 0.0, 0.5)], [[1], [0]], [None, None])
    out = _forces(planner, pool)
    p = planner._defaults
    g = 0.02
    assert out[0, SOCIAL, 0] == pytest.approx(-(p[_A] * math.exp(g / p[_B]) + p[_K] * g) / p[_MASS])
    assert out[0, SOCIAL, 1] == pytest.approx(p[_KAPPA] * g * 0.5 / p[_MASS])
    assert out[1, SOCIAL] == pytest.approx(-out[0, SOCIAL])


def test_wall_contact_pushes_out_and_brakes_sliding_along_the_wall(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    planner.set_walls([((-5.0, 0.0), (5.0, 0.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.2, 1.0, 0.0), (2.0, 0.6, 1.0, 0.0)], [[], []], [None, None])
    out = _forces(planner, pool)
    p = planner._defaults
    g = 0.05
    assert out[0, WALL] == pytest.approx((-p[_KAPPA] * g / p[_MASS], (p[_A_WALL] * math.exp(g / p[_B]) + p[_K] * g) / p[_MASS]))
    assert out[1, WALL] == pytest.approx((0.0, p[_A_WALL] * math.exp((0.25 - 0.6) / p[_B]) / p[_MASS]))


def test_coincident_neighbor_and_wall_under_the_agent_exert_nothing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    planner.set_walls([((-5.0, 0.0), (5.0, 0.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(1.0, 0.0, 0.3, 0.0), (1.0, 0.0, 0.0, 0.0)], [[1], [0]], [None, None])
    out = _forces(planner, pool)
    assert out[:, SOCIAL].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert out[:, WALL].tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_python_kernel_matches_compiled_in_crowded_corridor(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HelbingPlanner()
    planner.set_walls([((-4.0, -1.0), (4.0, -1.0)), ((-4.0, 1.0), (4.0, 1.0))])
    rng = np.random.default_rng(8)
    states = [(float(x), float(y), float(vx), float(vy)) for x, y, vx, vy in np.column_stack([rng.uniform(-3, 3, 8), rng.uniform(-0.9, 0.9, 8), rng.normal(0, 1, (8, 2))])]
    neighbors = [[j for j in range(8) if j != i] for i in range(8)]
    pool = _pool(pool_empty, agent_factory, planner, states, neighbors, [(4.0, 0.0)] * 8)
    out = _forces(planner, pool)
    assert np.all(np.isfinite(out))
    assert np.abs(out[:, SOCIAL]).max() > 0.0
