from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import INF, SOCIAL, WALL, _preferred_velocity
from arena_humansim.local_planner.zanlungo import ZanlungoPlanner, _crosses, _wall_ttc
from arena_humansim.utils.types import Pose2D

_TAU, _MASS, _A, _A_WALL, _B = range(5)


def _pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: ZanlungoPlanner,
    states: list[tuple[float, float, float, float]],
    neighbors: list[list[int]],
) -> AgentPool:
    pool = pool_empty(capacity=len(states))
    planner.attach(pool)
    for k, (x, y, vx, vy) in enumerate(states):
        pool.add_agent(agent_factory(agent_id=k + 1, x=x, y=y))
        pool.vel[k] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({k + 1: Pose2D(x=x + 10.0 * vx, y=y + 10.0 * vy) for k, (x, y, vx, vy) in enumerate(states)})
    indptr = np.cumsum([0] + [len(row) for row in neighbors]).astype(np.int32)
    pool.set_neighbor_csr(indptr, np.array([j for row in neighbors for j in row], dtype=np.int32))
    return pool


def _forces(planner: ZanlungoPlanner, pool: AgentPool, dt: float = 0.05) -> np.ndarray:
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


def _ttc(a: tuple[float, float], b: tuple[float, float], velocity: tuple[float, float], radius: float = 0.5) -> float:
    seg = np.array([[*a, *b]], dtype=np.float64)
    got = _wall_ttc.py_func(seg, 0, 0.0, 0.0, *velocity, radius)
    assert got == _wall_ttc(seg, 0, 0.0, 0.0, *velocity, radius)
    return got


@pytest.mark.parametrize(
    ("a", "b", "velocity", "expected"),
    [
        ((3.0, -1.0), (3.0, 1.0), (2.0, 0.0), 1.25),
        ((3.0, 1.0), (3.0, -1.0), (2.0, 0.0), 1.25),
        ((-3.0, -1.0), (-3.0, 1.0), (1.0, 0.0), INF),
        ((-1.0, 2.0), (1.0, 2.0), (1.0, 0.0), INF),
        ((1.0, 0.3), (3.0, 0.3), (2.0, 0.0), 0.3),
        ((3.0, 0.3), (1.0, 0.3), (2.0, 0.0), 0.3),
        ((-1.0, 0.3), (1.0, 0.3), (1.0, 0.0), 0.0),
        ((0.2, 0.3), (2.0, 0.3), (1.0, 0.0), 0.0),
        ((-2.0, 0.3), (-0.2, 0.3), (1.0, 0.0), 0.0),
        ((1.0, 1.0), (1.0, 1.0), (1.0, 0.0), INF),
        ((3.0, -1.0), (3.0, 1.0), (0.0, 0.0), INF),
        ((-1.0, 3.0), (1.0, 3.0), (0.0, 2.0), 1.25),
        ((-3.0, 0.3), (-1.0, 0.3), (1.0, 0.0), INF),
    ],
    ids=["ahead", "ahead_reversed", "behind", "parallel_clear", "endpoint_cap", "endpoint_cap_reversed", "overlapping", "overlapping_start_cap", "overlapping_end_cap", "zero_length", "resting", "rotated", "behind_parallel"],
)
def test_wall_ttc_known_cases(a: tuple[float, float], b: tuple[float, float], velocity: tuple[float, float], expected: float) -> None:
    assert _ttc(a, b, velocity) == pytest.approx(expected)


def test_crosses_detects_proper_segment_intersection_only() -> None:
    seg = np.array([[1.0, -1.0, 1.0, 1.0]])
    assert _crosses.py_func(0.0, 0.0, 2.0, 0.0, seg, 0)
    assert not _crosses.py_func(0.0, 0.0, 0.5, 0.0, seg, 0)
    assert not _crosses.py_func(0.0, 2.0, 2.0, 2.0, seg, 0)
    assert not _crosses.py_func(0.0, -2.0, 0.0, 2.0, seg, 0)
    assert _crosses(0.0, 0.0, 2.0, 0.0, seg, 0)


def test_head_on_pair_force_uses_time_to_contact(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 1.0, 0.0), (2.0, 0.0, -1.0, 0.0)], [[1], [0]])
    out = _forces(planner, pool)
    p = planner._defaults
    assert out[0, SOCIAL] == pytest.approx((-p[_A] * 2.0 / 0.75 / p[_MASS], 0.0))
    assert out[1, SOCIAL] == pytest.approx(-out[0, SOCIAL])


def test_passing_neighbor_force_uses_time_of_closest_approach(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 1.0, 0.0), (2.0, 1.0, 0.0, 0.0)], [[1], []])
    out = _forces(planner, pool)
    p = planner._defaults
    assert out[0, SOCIAL] == pytest.approx((0.0, -p[_A] * 1.0 / 2.0 * math.exp(-0.5 / p[_B]) / p[_MASS]))


def test_resting_or_diverging_agents_feel_no_social_or_wall_force(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((-5.0, -0.5), (5.0, -0.5))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 0.0), (3.0, 0.0, 1.0, 0.0), (1.5, 0.0, -0.5, 0.0)], [[], [2], [1]])
    out = _forces(planner, pool)
    assert out[:, SOCIAL].tolist() == [[0.0, 0.0]] * 3
    assert out[:, WALL].tolist() == [[0.0, 0.0]] * 3


def test_already_overlapping_neighbor_is_floored_at_one_step(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 1.0, 0.0), (0.1, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)], [[1, 2], [], []])
    out = _forces(planner, pool, dt=0.1)
    assert out[0, SOCIAL].tolist() == [0.0, 0.0]


def test_wall_ahead_force_uses_wall_time_to_contact(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((2.0, -1.0), (2.0, 1.0)), ((-10.0, -5.5), (10.0, -5.5))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 1.0, 0.0)], [[]])
    out = _forces(planner, pool)
    p = planner._defaults
    assert out[0, WALL] == pytest.approx((-p[_A_WALL] / 1.75 / p[_MASS], 0.0))


def test_wall_between_agent_and_predicted_position_still_pushes_back(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((1.0, -1.0), (1.0, 1.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 1.0, 0.0), (4.0, 0.0, 0.0, 0.0)], [[1], []])
    out = _forces(planner, pool)
    p = planner._defaults
    assert out[0, WALL] == pytest.approx((-p[_A_WALL] / 3.5 * math.exp(-(2.5 - 0.25) / p[_B]) / p[_MASS], 0.0), rel=1e-9, abs=0.0)


def test_predicted_position_on_the_wall_gets_no_wall_force(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((1.0, -1.0), (1.0, 1.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 1.0, 0.0), (1.5, 0.0, 0.0, 0.0)], [[1], []])
    out = _forces(planner, pool)
    assert out[0, WALL].tolist() == [0.0, 0.0]


def test_python_kernel_matches_compiled_in_crowded_corridor(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = ZanlungoPlanner()
    planner.set_walls([((-4.0, -1.0), (4.0, -1.0)), ((-4.0, 1.0), (4.0, 1.0))])
    rng = np.random.default_rng(9)
    states = [(float(x), float(y), float(vx), float(vy)) for x, y, vx, vy in np.column_stack([rng.uniform(-3, 3, 8), rng.uniform(-0.9, 0.9, 8), rng.normal(0, 1, (8, 2))])]
    pool = _pool(pool_empty, agent_factory, planner, states, [[j for j in range(8) if j != i] for i in range(8)])
    out = _forces(planner, pool)
    assert np.all(np.isfinite(out))
    assert np.abs(out[:, SOCIAL]).max() > 0.0
    assert np.abs(out[:, WALL]).max() > 0.0
