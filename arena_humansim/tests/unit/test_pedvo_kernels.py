from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.spatial import cKDTree

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.orca import _linear_program1, _linear_program2, _linear_program3
from arena_humansim.local_planner.pedvo import (
    _AGENT_WIDTH,
    _DENSITY_NORM,
    PedVOPlanner,
    _linear_program1_turning,
    _linear_program2_turning,
    _linear_program3_turning,
    _pedvo_kernel,
)

_DT = 0.05


def _lines(*rows: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def _random_lines(rng: np.random.Generator, n: int) -> np.ndarray:
    ang = rng.uniform(-math.pi, math.pi, n)
    return np.column_stack([rng.uniform(-2.0, 2.0, (n, 2)), np.cos(ang), np.sin(ang)])


def _pool(agent_factory: Callable[..., BaseAgent], specs: list[tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float]]]) -> AgentPool:
    pool = AgentPool(capacity=max(len(specs), 1))
    for i, (pos, goal, vel) in enumerate(specs):
        agent = agent_factory(agent_id=i + 1, x=pos[0], y=pos[1])
        agent.state.velocity = vel
        idx = pool.add_agent(agent)
        if goal is not None:
            pool.goal_pos[idx] = goal
            pool.has_goal[idx] = True
    return pool


def _kernel_both(planner: PedVOPlanner, pool: AgentPool, nbr_idx: np.ndarray | None = None) -> np.ndarray:
    n = pool.n
    pos = pool.pos[:n]
    if nbr_idx is None:
        k = min(planner.max_neighbors + 1, n)
        _, idx = cKDTree(pos).query(pos, k=k, distance_upper_bound=planner.neighbor_dist)
        nbr_idx = np.ascontiguousarray(idx.reshape(n, k), dtype=np.int64)
    stride_const = 0.5 * (1.0 + planner.stride_buffer) / planner.stride_factor
    grid = planner._grid
    args = (
        pos.copy(),
        pool.prev_vel[:n].copy(),
        pool.agent_radius[:n].copy(),
        pool.max_velocity[:n].copy(),
        pool.desired_vel[:n].copy(),
        pool.goal_pos[:n].copy(),
        pool.has_goal[:n].copy(),
        nbr_idx,
        planner.max_neighbors,
        planner._edges,
        grid.cell_start,
        grid.cell_walls,
        grid.wall_cx0,
        grid.wall_cy0,
        grid.origin_x,
        grid.origin_y,
        grid.cell,
        grid.nx,
        grid.ny,
        planner.goal_radius,
        planner.time_horizon,
        planner.time_horizon_obst,
        planner.wall_clearance,
        planner.turning_bias,
        planner.dense_aware,
        1.0 / (stride_const * stride_const),
        1.0 / _DT,
    )
    py_out = np.full((n, 2), np.nan)
    jit_out = np.full((n, 2), np.nan)
    _pedvo_kernel.py_func(*args, py_out)
    _pedvo_kernel(*args, jit_out)
    assert py_out == pytest.approx(jit_out, rel=1e-12, abs=1e-12)
    return py_out


def test_turning_lp1_rejects_lines_missing_the_real_speed_circle() -> None:
    lines = _lines((0.0, 0.8, 1.0, 0.0))
    assert _linear_program1_turning.py_func(lines, 0, 1.0, 0.0, 0.0, False, 2.0) == (False, 0.0, 0.0)
    ok, rx, ry = _linear_program1_turning.py_func(lines, 0, 1.0, 0.0, 0.0, False, 0.5)
    assert ok
    assert (rx, ry) == pytest.approx((0.0, 0.8))


def test_turning_lps_with_unit_bias_match_python_orca() -> None:
    rng = np.random.default_rng(41)
    cases = [
        (_lines((0.0, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, -1.0)), 0.6, 0.0),
        (_lines((0.0, 10.0, 0.0, -1.0), (0.5, 0.0, 0.0, -1.0)), 0.5, 0.5),
        (_lines((0.0, 0.3, 1.0, 0.0), (0.0, -0.3, -1.0, 0.0), (0.0, 0.6, 1.0, 0.0)), 0.0, 0.0),
        (_lines((5.0, 0.0, 0.0, 1.0)), 0.0, 0.0),
    ]
    cases += [(_random_lines(rng, int(rng.integers(1, 7))), *(float(v) for v in rng.uniform(-2.5, 2.5, 2))) for _ in range(150)]
    for lines, ox, oy in cases:
        n = len(lines)
        for line_no in range(n):
            for direction_opt in (False, True):
                assert _linear_program1_turning.py_func(lines, line_no, 1.2, ox, oy, direction_opt, 1.0) == _linear_program1.py_func(lines, line_no, 1.2, ox, oy, direction_opt)
        fail, rx, ry = _linear_program2_turning.py_func(lines, n, 1.2, ox, oy, False, 1.0)
        assert (fail, rx, ry) == _linear_program2.py_func(lines, n, 1.2, ox, oy, False)
        if fail < n:
            n_obst = int(rng.integers(0, fail + 1))
            assert _linear_program3_turning.py_func(lines, n, n_obst, fail, 1.2, rx, ry, 1.0) == _linear_program3.py_func(lines, n, n_obst, fail, 1.2, rx, ry)


@pytest.mark.parametrize("turn_bias", [0.4, 2.5])
def test_turning_lps_python_matches_compiled_for_biased_turning(turn_bias: float) -> None:
    rng = np.random.default_rng(43)
    infeasible = 0
    for _ in range(200):
        n = int(rng.integers(1, 7))
        lines = _random_lines(rng, n)
        fail, rx, ry = _linear_program2_turning.py_func(lines, n, 1.5, 1.2, 0.0, False, turn_bias)
        assert (fail, rx, ry) == pytest.approx(_linear_program2_turning(lines, n, 1.5, 1.2, 0.0, False, turn_bias), abs=1e-12)
        if fail < n:
            infeasible += 1
            got = _linear_program3_turning.py_func(lines, n, 0, fail, 1.5, rx, ry, turn_bias)
            assert got == pytest.approx(_linear_program3_turning(lines, n, 0, fail, 1.5, rx, ry, turn_bias), abs=1e-12)
            assert all(math.isfinite(v) for v in got)
    assert infeasible > 0


def test_turning_lp2_direction_mode_starts_from_the_scaled_direction() -> None:
    assert _linear_program2_turning.py_func(_lines(), 0, 1.5, 0.6, 0.8, True, 2.0) == pytest.approx((0, 0.9, 1.2))
    lines = _lines((0.0, -0.5, -1.0, 0.0))
    assert _linear_program2_turning.py_func(lines, 1, 2.0, 1.0, 0.0, True, 1.0) == _linear_program2.py_func(lines, 1, 2.0, 1.0, 0.0, True)


def test_turning_lp3_with_contradicting_obstacle_lines_keeps_the_last_result() -> None:
    lines = _lines((0.5, 0.0, 0.0, -1.0), (-0.5, 0.0, 0.0, 1.0), (0.0, 0.3, 1.0, 0.0))
    fail, rx, ry = _linear_program2_turning.py_func(lines, 3, 1.0, 0.0, 0.0, False, 1.0)
    assert fail == 1
    assert _linear_program3_turning.py_func(lines, 3, 2, fail, 1.0, rx, ry, 1.0) == (rx, ry)


@pytest.mark.parametrize("turning_bias", [1.0, 2.5])
def test_lone_agent_walks_at_preferred_velocity(agent_factory: Callable[..., BaseAgent], turning_bias: float) -> None:
    planner = PedVOPlanner(turning_bias=turning_bias)
    pool = _pool(agent_factory, [((0.0, 0.0), (3.0, 4.0), (0.0, 0.0))])
    out = _kernel_both(planner, pool)
    assert out[0] == pytest.approx((0.6 * pool.desired_vel[0], 0.8 * pool.desired_vel[0]))


def test_stacked_walls_at_the_stride_point_cap_speed_by_density_and_walls_behind_do_not(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = PedVOPlanner()
    front = [((1.0, -1.0), (1.0, 1.0))] * 5
    planner.set_walls(front)
    pool = _pool(agent_factory, [((0.0, 0.0), (5.0, 0.0), (0.0, 0.0))])
    capped = _kernel_both(planner, pool)
    stride_const = 0.5 * (1.0 + planner.stride_buffer) / planner.stride_factor
    expected = (_AGENT_WIDTH / (5.0 * _DENSITY_NORM)) ** 2 / stride_const**2
    assert capped[0] == pytest.approx((expected, 0.0))
    planner.set_walls([*front, ((-1.0, -1.0), (-1.0, 1.0))])
    assert np.array_equal(_kernel_both(planner, pool), capped)
    planner.dense_aware = False
    assert _kernel_both(planner, pool)[0] == pytest.approx((pool.desired_vel[0], 0.0))


def test_density_scan_respects_neighbor_cap_missing_entries_and_self(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = PedVOPlanner(max_neighbors=1)
    specs = [((0.0, 0.0), (5.0, 0.0), (0.0, 0.0)), ((1.0, 0.0), None, (0.0, 0.0)), ((1.0, 0.1), None, (0.0, 0.0))]
    pool = _pool(agent_factory, specs)
    capped = _kernel_both(planner, pool, np.array([[1, 2], [3, 3], [3, 3]], dtype=np.int64))
    first_only = _kernel_both(planner, pool, np.array([[1, 3], [3, 3], [3, 3]], dtype=np.int64))
    assert capped[0] == pytest.approx(first_only[0], abs=1e-12)
    alone = _kernel_both(planner, pool, np.array([[0, 3], [3, 3], [3, 3]], dtype=np.int64))
    assert alone[0] == pytest.approx((pool.desired_vel[0], 0.0))
    assert capped[1:].tolist() == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("turning_bias", [1.0, 3.0])
def test_kernel_python_matches_compiled_in_dense_walled_crowd(agent_factory: Callable[..., BaseAgent], turning_bias: float) -> None:
    planner = PedVOPlanner(turning_bias=turning_bias, time_horizon_obst=1.0)
    shorts = [((-2.0 + 0.05 * k, 0.9), (-1.9 + 0.05 * k, 0.9)) for k in range(70)]
    planner.set_walls([*shorts, ((-4.0, -0.9), (4.0, -0.9)), ((4.0, -0.9), (-4.0, -0.9)), ((-4.0, -2.5), (4.0, -2.5))])
    rng = np.random.default_rng(int(10 * turning_bias))
    xs, ys = rng.uniform(-3.0, 3.0, 12), rng.uniform(-0.6, 0.6, 12)
    vel = rng.normal(0.0, 0.8, (12, 2))
    specs = [((float(xs[k]), float(ys[k])), (float(-4.0 * np.sign(xs[k])), 0.0), (float(vel[k, 0]), float(vel[k, 1]))) for k in range(12)]
    specs.append(((0.0, 0.0), None, (0.0, 0.0)))
    specs.append(((-0.3, 0.3), (-0.3, 0.3), (0.0, 0.0)))
    pool = _pool(agent_factory, specs)
    out = _kernel_both(planner, pool)
    planner.compute_pool(pool, dt=_DT)
    assert out == pytest.approx(pool.vel[: pool.n], abs=1e-12)
    assert np.all(np.hypot(out[:, 0], out[:, 1]) <= pool.max_velocity[: pool.n] + 1e-9)
    assert out[12].tolist() == [0.0, 0.0]


def test_biased_turning_result_is_clamped_to_max_speed(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = PedVOPlanner(turning_bias=4.0)
    ring = [((0.6 * math.cos(a), 0.6 * math.sin(a)), (0.0, 0.0), (-math.cos(a), -math.sin(a))) for a in np.linspace(0.3, 2.0 * math.pi + 0.3, 5, endpoint=False)]
    pool = _pool(agent_factory, [((0.0, 0.0), (5.0, 0.0), (1.0, 0.0)), *ring])
    out = _kernel_both(planner, pool)
    assert np.all(np.hypot(out[:, 0], out[:, 1]) <= pool.max_velocity[: pool.n] + 1e-9)


def test_compute_pool_on_empty_pool_leaves_it_untouched() -> None:
    planner = PedVOPlanner()
    pool = AgentPool(capacity=2)
    pool.vel[:] = 7.0
    planner.compute_pool(pool)
    assert np.all(pool.vel == 7.0)
