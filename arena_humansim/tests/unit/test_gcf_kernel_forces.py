from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.force import GOAL, SOCIAL, WALL, _preferred_velocity
from arena_humansim.local_planner.gcf import (
    GCFPlanner,
    _contact_distance,
    _cubic_max_root,
    _hermite,
    _quartic_positive_root,
    _support,
    closest_approach,
    distance_response,
    ellipse_axes,
    polar_radius,
)
from arena_humansim.utils.types import Pose2D

_CIRCLES = {"a_min": 0.25, "a_rate": 0.0, "b_max": 0.25, "b_growth": 0.0}


def _support_reference(a1: float, b1: float, th1: float, a2: float, b2: float, th2: float, psi: float) -> float:
    phi = np.linspace(-math.pi, math.pi, 40001)
    nx = np.cos(phi)
    ny = np.sin(phi)

    def support(a: float, b: float, th: float) -> np.ndarray:
        u = nx * math.cos(th) + ny * math.sin(th)
        v = ny * math.cos(th) - nx * math.sin(th)
        return np.sqrt(a * a * u * u + b * b * v * v)

    nd = nx * math.cos(psi) + ny * math.sin(psi)
    front = nd > 1e-9
    return float(np.min((support(a1, b1, th1) + support(a2, b2, th2))[front] / nd[front]))


def _pool(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    planner: GCFPlanner,
    states: list[tuple[float, float, float, float, float]],
    neighbors: list[list[int]],
    goals: list[tuple[float, float] | None],
    **params: float,
) -> AgentPool:
    pool = pool_empty(capacity=len(states))
    planner.attach(pool)
    for k, (x, y, theta, vx, vy) in enumerate(states):
        agent = agent_factory(agent_id=k + 1, x=x, y=y)
        agent.params = attrs.evolve(agent.params, local_planner_params=params)
        pool.add_agent(agent)
        pool.theta[k] = theta
        pool.vel[k] = (vx, vy)
    pool.store_prev_vel()
    pool.set_goals({k + 1: Pose2D(x=g[0], y=g[1]) for k, g in enumerate(goals) if g is not None})
    indptr = np.cumsum([0] + [len(row) for row in neighbors]).astype(np.int32)
    pool.set_neighbor_csr(indptr, np.array([j for row in neighbors for j in row], dtype=np.int32))
    return pool


def _forces(planner: GCFPlanner, pool: AgentPool, dt: float = 0.05) -> np.ndarray:
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


def test_ellipse_axes_grow_along_motion_and_clamp_the_sway_axis() -> None:
    assert ellipse_axes.py_func(0.18, 0.53, 0.25, 0.05, 1.3) == pytest.approx((0.18 + 0.53 * 1.3, 0.2))
    assert ellipse_axes.py_func(0.18, 0.53, 0.25, 0.05, 100.0)[1] == 0.01


def test_polar_radius_and_support_on_the_axes_and_off_them() -> None:
    a, b, th = 0.6, 0.25, 0.4
    c, s = math.cos(th), math.sin(th)
    assert polar_radius.py_func(a, b, c, s, c, s) == pytest.approx(a)
    assert polar_radius.py_func(a, b, c, s, -s, c) == pytest.approx(b)
    assert _support.py_func(a, b, c, s, c, s) == pytest.approx(a)
    assert _support.py_func(a, b, c, s, -s, c) == pytest.approx(b)
    for psi in np.linspace(-math.pi, math.pi, 13):
        dx, dy = math.cos(psi), math.sin(psi)
        r = polar_radius.py_func(a, b, c, s, dx, dy)
        assert ((dx * c + dy * s) * r / a) ** 2 + ((dy * c - dx * s) * r / b) ** 2 == pytest.approx(1.0)
        assert b - 1e-12 <= r <= _support.py_func(a, b, c, s, dx, dy) + 1e-12


@pytest.mark.parametrize(
    ("coeffs", "expected"),
    [((-6.0, 11.0, -6.0), 3.0), ((0.0, 0.0, -1.0), 1.0), ((3.0, 3.0, 1.0), -1.0), ((0.0, 1.0, 0.0), 0.0), ((-1.0, -4.0, 4.0), 2.0)],
    ids=["three_real", "one_real", "triple", "one_real_with_complex_pair", "three_real_mixed_sign"],
)
def test_cubic_max_root_finds_the_largest_real_root(coeffs: tuple[float, float, float], expected: float) -> None:
    got = _cubic_max_root.py_func(*coeffs)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got == pytest.approx(_cubic_max_root(*coeffs), abs=1e-12)


@pytest.mark.parametrize(
    ("coeffs", "shift", "expected"),
    [
        ((-5.0, 0.0, 4.0), 0.0, 2.0),
        ((-5.0, 0.0, 4.0), -3.0, -1.0),
        ((5.0, 0.0, 4.0), 0.5, 0.5),
        ((-4.75, -0.75, 4.5), 0.0, 2.0),
        ((-4.75, -0.75, 4.5), 0.5, 2.5),
        ((3.0, 6.0, 10.0), 0.0, 1.0),
        ((1.0, math.nan, 1.0), 0.0, -1.0),
    ],
    ids=["biquadratic", "biquadratic_all_negative", "biquadratic_complex", "ferrari", "ferrari_shifted", "ferrari_complex", "nan"],
)
def test_quartic_positive_root_known_cases(coeffs: tuple[float, float, float], shift: float, expected: float) -> None:
    assert _quartic_positive_root.py_func(*coeffs, shift) == pytest.approx(expected, abs=1e-9)


def test_hermite_interpolates_values_and_slopes() -> None:
    assert _hermite.py_func(1.0, 1.0, 3.0, 2.0, 5.0, -1.0, 0.5) == pytest.approx(2.0)
    assert _hermite.py_func(3.0, 1.0, 3.0, 2.0, 5.0, -1.0, 0.5) == pytest.approx(5.0)
    h = 1e-6
    assert (_hermite.py_func(1.0 + h, 1.0, 3.0, 2.0, 5.0, -1.0, 0.5) - 2.0) / h == pytest.approx(-1.0, abs=1e-4)
    assert (5.0 - _hermite.py_func(3.0 - h, 1.0, 3.0, 2.0, 5.0, -1.0, 0.5)) / h == pytest.approx(0.5, abs=1e-4)


def test_distance_response_branches_are_continuous_and_decreasing() -> None:
    max_dist, width, max_force = 2.0, 0.12, 3.0
    peak = 3.0 * max_force
    assert distance_response.py_func(-0.1, max_dist, width, max_force) == peak
    assert distance_response.py_func(0.0, max_dist, width, max_force) == peak
    assert distance_response.py_func(width, max_dist, width, max_force) == pytest.approx(1.0 / width)
    assert distance_response.py_func(1.0, max_dist, width, max_force) == pytest.approx(1.0)
    assert distance_response.py_func(max_dist - width, max_dist, width, max_force) == pytest.approx(1.0 / (max_dist - width))
    assert distance_response.py_func(max_dist, max_dist, width, max_force) == 0.0
    samples = [distance_response.py_func(x, max_dist, width, max_force) for x in np.linspace(width, max_dist, 501)]
    assert np.all(np.diff(samples) <= 1e-12)
    assert samples[-2] < 0.01


def test_distance_response_widens_a_too_narrow_core() -> None:
    narrow = [distance_response.py_func(x, 2.0, 0.01, 3.0) for x in (0.0, 0.05, 1.5 / 9.0, 0.5)]
    assert narrow[0] == 9.0
    assert narrow[2] == pytest.approx(9.0 / 1.5)
    assert narrow[3] == pytest.approx(2.0)
    assert narrow[0] > narrow[1] > narrow[2]


@pytest.mark.parametrize(
    ("a1", "b1", "th1", "a2", "b2", "th2", "psi"),
    [
        (0.3, 0.3, 0.0, 0.2, 0.2, 1.0, 0.7),
        (0.2, 0.5, 0.3, 0.6, 0.25, 1.2, 2.0),
        (0.6, 0.25, 0.3, 0.2, 0.5, 1.2, -1.0),
        (0.6, 0.25, 0.0, 0.5, 0.2, 2.5, 0.4),
        (0.8, 0.2, 0.4, 0.5, 0.3, 0.4, 1.1),
        (0.5, 0.3, 0.4, 0.8, 0.2, 0.4, 1.1),
        (0.7, 0.3, -0.9, 0.4, 0.35, 0.6, -2.8),
    ],
    ids=["circles", "first_axes_swapped", "second_axes_swapped", "obtuse_orientations", "parallel_first_longer", "parallel_second_longer", "generic"],
)
def test_contact_distance_matches_support_function_reference(a1: float, b1: float, th1: float, a2: float, b2: float, th2: float, psi: float) -> None:
    args = (a1, b1, math.cos(th1), math.sin(th1), a2, b2, math.cos(th2), math.sin(th2), math.cos(psi), math.sin(psi))
    ref = _support_reference(a1, b1, th1, a2, b2, th2, psi)
    assert _contact_distance.py_func(*args) == pytest.approx(ref, abs=2e-3)
    assert closest_approach.py_func(*args) == pytest.approx(ref, abs=2e-3)


def test_contact_distance_of_aligned_ellipses_along_shared_axes_adds_semi_axes() -> None:
    assert _contact_distance.py_func(0.8, 0.2, 1.0, 0.0, 0.5, 0.3, 1.0, 0.0, 0.0, 1.0) == pytest.approx(0.5)
    assert _contact_distance.py_func(0.5, 0.3, 1.0, 0.0, 0.8, 0.2, 1.0, 0.0, 1.0, 0.0) == pytest.approx(1.3)


def test_closest_approach_never_exceeds_the_sum_of_major_axes_or_falls_below_polar_sum() -> None:
    rng = np.random.default_rng(21)
    for _ in range(200):
        a1, a2 = rng.uniform(0.15, 1.2, 2)
        b1 = a1 * (1.0 + rng.normal(0.0, 1e-6))
        b2 = rng.uniform(0.15, 0.4)
        th1, th2, psi = rng.uniform(-math.pi, math.pi, 3)
        args = (a1, b1, math.cos(th1), math.sin(th1), a2, b2, math.cos(th2), math.sin(th2), math.cos(psi), math.sin(psi))
        got = closest_approach.py_func(*args)
        lower = polar_radius(a1, b1, args[2], args[3], args[8], args[9]) + polar_radius(a2, b2, args[6], args[7], args[8], args[9])
        assert lower - 1e-12 <= got <= max(a1, b1) + max(a2, b2) + 1e-12
        assert got == pytest.approx(_support_reference(a1, b1, th1, a2, b2, th2, psi), abs=2e-3)


@pytest.mark.parametrize("kind", ["near_identical", "near_circles", "near_antiparallel"])
def test_closest_approach_of_near_degenerate_pairs_stays_within_bounds(kind: str) -> None:
    rng = np.random.default_rng(31)
    for _ in range(150):
        a1, b1 = rng.uniform(0.3, 1.0), rng.uniform(0.15, 0.3)
        th1, psi = rng.uniform(-math.pi, math.pi, 2)
        if kind == "near_identical":
            a2, b2, th2 = a1 * (1.0 + rng.normal(0.0, 1e-7)), b1, th1 + rng.normal(0.0, 1e-9)
        elif kind == "near_circles":
            b1 = a1 * (1.0 + rng.normal(0.0, 1e-9))
            a2, th2 = rng.uniform(0.15, 0.6), rng.uniform(-math.pi, math.pi)
            b2 = a2 * (1.0 + rng.normal(0.0, 1e-9))
        else:
            a2, b2, th2 = rng.uniform(0.3, 1.0), rng.uniform(0.15, 0.3), th1 + math.pi + rng.normal(0.0, 1e-9)
        args = (a1, b1, math.cos(th1), math.sin(th1), a2, b2, math.cos(th2), math.sin(th2), math.cos(psi), math.sin(psi))
        got = closest_approach.py_func(*args)
        lower = polar_radius(a1, b1, args[2], args[3], args[8], args[9]) + polar_radius(a2, b2, args[6], args[7], args[8], args[9])
        assert lower - 1e-12 <= got <= max(a1, b1) + max(a2, b2) + 1e-12
        assert got == pytest.approx(_support_reference(a1, b1, th1, a2, b2, th2, psi), abs=2e-3)


def test_head_on_circles_follow_the_velocity_squared_law(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 1.0, 0.0), (1.5, 0.0, math.pi, -1.0, 0.0)], [[1], [0]], [(10.0, 0.0), (-10.0, 0.0)], **_CIRCLES)
    out = _forces(planner, pool)
    scale = 0.3 * pool.desired_vel[0] + 2.0
    assert out[0, SOCIAL] == pytest.approx((-(scale**2), 0.0))
    assert out[1, SOCIAL] == pytest.approx(-out[0, SOCIAL])
    assert out[0, GOAL] == pytest.approx(((pool.desired_vel[0] - 1.0) / 0.5, 0.0))


def test_resting_coincident_or_far_neighbors_exert_nothing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    states = [(0.0, 0.0, 0.0, 1.3, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0), (-6.0, 0.0, 0.0, 0.0, 0.0), (-5.0, 0.3, 0.0, 0.0, 0.0)]
    pool = _pool(pool_empty, agent_factory, planner, states, [[1, 2], [], [], [2]], [None] * 4)
    out = _forces(planner, pool)
    assert out[:, SOCIAL].tolist() == [[0.0, 0.0]] * 4


def test_side_by_side_walkers_are_culled_by_support_distance(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 1.3, 0.0), (0.0, -3.0, 0.0, 1.3, 0.0)], [[1], []], [(10.0, 0.0), (10.0, -3.0)])
    out = _forces(planner, pool)
    assert out[0, SOCIAL].tolist() == [0.0, 0.0]


def test_diagonal_walker_beyond_contact_range_is_culled(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    d = 3.0 / math.sqrt(2.0)
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, 0.0, 1.3, 0.0), (-d, -d, 0.0, 1.3, 0.0)], [[1], []], [(10.0, 0.0), (10.0, -d)])
    out = _forces(planner, pool)
    assert out[0, SOCIAL].tolist() == [0.0, 0.0]


def test_wall_ahead_pushes_back_through_three_samples(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((-5.0, -1.0), (5.0, -1.0))])
    pool = _pool(pool_empty, agent_factory, planner, [(0.0, 0.0, -math.pi / 2, 0.0, -1.0)], [[]], [None], **_CIRCLES)
    out = _forces(planner, pool)
    side = math.hypot(0.025 * 10.0, 1.0)
    expected = 1.0 / 0.75 + 2.0 * (1.0 / (side - 0.25)) / side
    assert out[0, WALL] == pytest.approx((0.0, expected))


def test_short_walls_walls_underfoot_and_samples_past_endpoints(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((0.0, -1.0), (0.05, -1.0)), ((-5.0, 0.0), (5.0, 0.0)), ((1.0, 2.0), (6.0, 2.0))])
    states = [(0.0, 1.0, 0.0, 0.0, -1.0), (2.0, 0.0, 0.0, 1.0, 0.0), (0.0, 2.0, 0.0, 1.0, 0.0)]
    pool = _pool(pool_empty, agent_factory, planner, states, [[], [], []], [None] * 3, **_CIRCLES)
    out = _forces(planner, pool)
    assert out[0, WALL, 1] > 0.0
    assert out[0, WALL, 0] == pytest.approx(0.0, abs=1e-12)
    assert out[1, WALL].tolist() == [0.0, 0.0]
    assert out[2, WALL, 0] < 0.0


def test_python_kernel_matches_compiled_in_crowded_corridor(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = GCFPlanner()
    planner.set_walls([((-4.0, -1.0), (4.0, -1.0)), ((-4.0, 1.0), (4.0, 1.0))])
    rng = np.random.default_rng(12)
    xs, ys, ths = rng.uniform(-3, 3, 8), rng.uniform(-0.9, 0.9, 8), rng.uniform(-math.pi, math.pi, 8)
    states = [(float(xs[k]), float(ys[k]), float(ths[k]), 1.1 * math.cos(ths[k]), 1.1 * math.sin(ths[k])) for k in range(8)]
    pool = _pool(pool_empty, agent_factory, planner, states, [[j for j in range(8) if j != i] for i in range(8)], [(4.0, 0.0)] * 8)
    out = _forces(planner, pool)
    assert np.all(np.isfinite(out))
    assert np.abs(out[:, SOCIAL]).max() > 0.0
    assert np.abs(out[:, WALL]).max() > 0.0
