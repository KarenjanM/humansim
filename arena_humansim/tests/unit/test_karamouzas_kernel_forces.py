from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import GOAL, SOCIAL, WALL, _preferred_velocity
from arena_humansim.local_planner.karamouzas import KaramouzasPlanner, _magnitude
from arena_humansim.utils.types import Pose2D


def _pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: KaramouzasPlanner,
    others: list[tuple[float, float, float, float]],
    seen: list[int],
    **params: float,
) -> AgentPool:
    pool = pool_empty(capacity=len(others) + 1)
    planner.attach(pool)
    first = agent_factory(agent_id=1)
    first.params = attrs.evolve(first.params, local_planner_params=params)
    first.state.velocity = (1.0, 0.0)
    first.state.desired_velocity = 1.0
    pool.add_agent(first)
    for k, (x, y, vx, vy) in enumerate(others):
        pool.add_agent(agent_factory(agent_id=k + 2, x=x, y=y))
        pool.vel[k + 1] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({1: Pose2D(x=20.0, y=0.0)})
    indptr = np.full(pool.n + 1, len(seen), dtype=np.int32)
    indptr[0] = 0
    pool.set_neighbor_csr(indptr, np.array(seen, dtype=np.int32))
    return pool


def _forces(planner: KaramouzasPlanner, pool: AgentPool, dt: float = 0.05) -> np.ndarray:
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


@pytest.mark.parametrize(("d", "expected"), [(0.5, 6.0), (1.0, 3.0), (5.0, 3.0), (8.0, 3.0), (9.0, 1.5), (10.0, 0.0), (12.0, 0.0)])
def test_magnitude_is_inverse_then_flat_then_linear_then_zero(d: float, expected: float) -> None:
    assert _magnitude.py_func(d, 1.0, 8.0, 10.0, 3.0) == pytest.approx(expected)
    assert _magnitude.py_func(d, 1.0, 8.0, 10.0, 3.0) == _magnitude(d, 1.0, 8.0, 10.0, 3.0)


def test_magnitude_is_continuous_at_the_breakpoints() -> None:
    for d in (1.0, 8.0, 10.0):
        assert _magnitude.py_func(d - 1e-9, 1.0, 8.0, 10.0, 3.0) == pytest.approx(_magnitude.py_func(d + 1e-9, 1.0, 8.0, 10.0, 3.0), abs=1e-7)


def test_goal_term_is_zero_at_preferred_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    out = _forces(planner, _pool(pool_empty, agent_factory, planner, [], []))
    assert out[0].tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


def test_wall_force_inside_safe_distance_only(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    planner.set_walls([((-5.0, -1.0), (5.0, -1.0)), ((-5.0, -3.0), (5.0, -3.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(3.0, -1.0, 0.0, 0.0)], [])
    out = _forces(planner, pool)
    assert out[0, WALL] == pytest.approx((0.0, 1.25 / 0.75**2))
    assert out[1, WALL] == pytest.approx((0.0, 0.25 / 1.75**2))


def test_colliding_neighbor_force_is_capped_and_masks_predictions(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (3.0, 0.0, -1.0, 0.0)], [1, 2, 3])
    out = _forces(planner, pool)
    assert out[0, SOCIAL] == pytest.approx((-1.5, 0.0))


def test_colliding_neighbor_below_cap_follows_inverse_magnitude(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 1.2, 0.0, 0.0)], [1], agent_force=0.5)
    out = _forces(planner, pool)
    assert out[0, SOCIAL] == pytest.approx((0.0, -0.5 / 0.7))


def test_predicted_collision_pushes_with_flat_magnitude(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(4.0, 0.0, 0.0, 0.0)], [1], agent_force=0.5)
    out = _forces(planner, pool)
    assert out[0, GOAL].tolist() == [0.0, 0.0]
    assert out[0, SOCIAL] == pytest.approx((-0.5, 0.0))


def test_only_most_imminent_predictions_count_with_decaying_weight(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    others = [(x, 0.0, 0.0, 0.0) for x in (3.5, 2.0, 3.0, 2.5, 4.0)]
    pool = _pool(pool_empty, agent_factory, planner, others, [1, 2, 3, 4, 5], colliding_count=2.0, d_mid=1.6, d_max=3.5, agent_force=0.5)
    out = _forces(planner, pool)
    assert out[0, SOCIAL] == pytest.approx((-0.5 * (1.0 + 0.8 * 1.5 / 1.9), 0.0))


@pytest.mark.parametrize(
    ("other", "params"),
    [
        ((-2.0, 0.0, 3.0, 0.0), {}),
        ((8.0, 0.0, 0.0, 0.0), {}),
        ((2.0, 0.0, 0.0, 0.0), {"d_max": 1.2}),
    ],
    ids=["outside_fov", "beyond_anticipation", "gap_beyond_d_max"],
)
def test_neighbors_outside_view_horizon_or_range_exert_nothing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], other: tuple[float, float, float, float], params: dict[str, float]) -> None:
    planner = KaramouzasPlanner()
    out = _forces(planner, _pool(pool_empty, agent_factory, planner, [other], [1], **params))
    assert out[0, SOCIAL].tolist() == [0.0, 0.0]


def test_predicted_exact_overlap_of_point_neighbor_is_skipped(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(2.0, 0.0, 0.0, 0.0)], [1], personal_space=0.0)
    pool.agent_radius[1] = 0.0
    out = _forces(planner, pool)
    assert out[0, SOCIAL].tolist() == [0.0, 0.0]


def test_python_kernel_matches_compiled_in_crowd(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = KaramouzasPlanner()
    planner.set_walls([((-4.0, -2.0), (4.0, -2.0)), ((-4.0, 2.0), (4.0, 2.0))])
    spots = [(1.1, 0.2), (2.0, -0.4), (0.4, 1.3), (-1.2, 0.6), (3.1, 0.9), (1.6, -1.5), (-0.8, -1.1)]
    others = [(x, y, math.cos(k), math.sin(2 * k)) for k, (x, y) in enumerate(spots)]
    pool = _pool(pool_empty, agent_factory, planner, others, list(range(1, 8)))
    pool.theta[: pool.n] = 0.7 * np.arange(pool.n)
    n = pool.n
    indices = [j for i in range(n) for j in range(n) if j != i]
    pool.set_neighbor_csr(np.arange(0, n * (n - 1) + 1, n - 1, dtype=np.int32), np.array(indices, dtype=np.int32))
    out = _forces(planner, pool)
    assert np.all(np.isfinite(out))
    assert np.all(np.hypot(out[:, SOCIAL, 0], out[:, SOCIAL, 1]) <= pool.max_acceleration[:n] + 1e-12)
